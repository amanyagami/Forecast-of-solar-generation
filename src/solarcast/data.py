"""Load, clean and align the NITK generation and irradiation workbooks.

The raw workbooks have a peculiar layout that this module documents and handles:

* Each sheet holds 12 side-by-side ``(Date, value)`` column pairs, one pair per month.
* Timestamps are a mix of real datetimes and ``dd/mm/yyyy HH:MM`` strings. Excel parsed the
  days 1-12 of each month as ``mm/dd`` datetimes (day and month swapped) and left days 13-31 as
  text. Datetimes therefore need their month and day swapped back.
* The generation column contains paired ``-X, +X`` spikes of about +-500,000 (a cumulative meter
  register leaking into the series) on roughly 0.7% of rows.
* Generation is missing 00:00-05:15 on the first day of each month and entirely on
  2019-08-28/29; irradiation has about 1.3% NaN, concentrated in late August.
"""

from __future__ import annotations

import io
import re
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FREQ = "15min"
STEPS_PER_DAY = 96
GEN_COL = "gen_kwh"
IRR_COL = "irr"

# A single 15-minute reading above this is not physical for this installation (observed
# genuine maximum is ~30.6 kWh); larger values are meter-register glitches.
MAX_GEN_KWH = 100.0
# Sensor readings above the solar constant plus enhancement are treated as glitches.
MAX_IRR = 1500.0
# Envelope level (kWh/15 min, ~5% of the observed ~30 kWh peak) below which the sun is too low
# for a stable clearness ratio.
ENV_FLOOR_KWH = 1.5
# Irradiation at or below this value counts as "dark" when filling missing night generation.
DARK_IRR = 1.0


def read_raw_sheet(path: str | Path) -> pd.DataFrame:
    """Read the first sheet of a workbook without headers.

    Some exports carry vendor ``extLst`` blocks inside ``styles.xml`` that crash openpyxl.
    Those blocks only hold cosmetic metadata, so they are stripped in memory before parsing.

    Args:
        path: Path to the ``.xlsx`` file.

    Returns:
        Raw frame with integer column labels and the header row as row 0.
    """
    src = zipfile.ZipFile(path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as dst:
        for info in src.infolist():
            payload = src.read(info.filename)
            if info.filename == "xl/styles.xml":
                payload = re.sub(rb"<extLst>.*?</extLst>", b"", payload, flags=re.DOTALL)
            dst.writestr(info, payload)
    buf.seek(0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unknown extension", category=UserWarning)
        return pd.read_excel(buf, header=None, engine="openpyxl")


def parse_timestamp(value: object) -> pd.Timestamp:
    """Convert one raw timestamp cell to a correct timestamp.

    Args:
        value: Either a ``datetime`` (whose month and day were swapped by Excel) or a
            ``dd/mm/yyyy HH:MM[:SS]`` string.

    Returns:
        The corrected timestamp.

    Raises:
        ValueError: If the value is neither a datetime nor a parseable string.
    """
    if isinstance(value, str):
        text = value.strip()
        fmt = "%d/%m/%Y %H:%M:%S" if text.count(":") == 2 else "%d/%m/%Y %H:%M"
        return pd.to_datetime(text, format=fmt)
    if hasattr(value, "year") and hasattr(value, "hour"):
        return pd.Timestamp(
            year=value.year,
            month=value.day,
            day=value.month,
            hour=value.hour,
            minute=value.minute,
        )
    raise ValueError(f"Unsupported timestamp cell: {value!r}")


def stack_month_pairs(raw: pd.DataFrame) -> pd.Series:
    """Stack the side-by-side monthly ``(Date, value)`` column pairs into one series.

    Args:
        raw: Output of :func:`read_raw_sheet` (header in row 0).

    Returns:
        Float series indexed by corrected timestamps, sorted ascending, without the
        duplicated timestamps (the first occurrence is kept).
    """
    parts: list[pd.Series] = []
    body = raw.iloc[1:]
    for k in range(raw.shape[1] // 2):
        stamps = body.iloc[:, 2 * k].dropna()
        if stamps.empty:
            continue
        values = pd.to_numeric(body.iloc[:, 2 * k + 1], errors="coerce").loc[stamps.index]
        index = pd.DatetimeIndex([parse_timestamp(s) for s in stamps])
        parts.append(pd.Series(values.to_numpy(dtype=float), index=index))
    series = pd.concat(parts).sort_index()
    return series[~series.index.duplicated(keep="first")]


def clean_generation(series: pd.Series, max_kwh: float = MAX_GEN_KWH) -> pd.Series:
    """Remove meter glitches from generation.

    Values outside ``[0, max_kwh]`` become NaN. Tiny negatives (> -1e-6) are clipped to zero.

    Args:
        series: Raw generation in kWh per 15 minutes.
        max_kwh: Largest physically plausible 15-minute energy.

    Returns:
        Cleaned series (NaN where the reading was rejected).
    """
    out = series.astype(float).copy()
    out[(out < 0) & (out > -1e-6)] = 0.0
    out[(out < 0) | (out > max_kwh)] = np.nan
    return out


def clean_irradiation(series: pd.Series, max_value: float = MAX_IRR) -> pd.Series:
    """Remove impossible irradiation readings.

    Args:
        series: Raw irradiation sensor values.
        max_value: Largest plausible reading.

    Returns:
        Series with negatives clipped to zero and values above ``max_value`` set to NaN.
    """
    out = series.astype(float).copy()
    out[out < 0] = 0.0
    out[out > max_value] = np.nan
    return out


def align(gen: pd.Series, irr: pd.Series, freq: str = FREQ) -> pd.DataFrame:
    """Align generation and irradiation onto one complete regular time grid.

    Missing generation is filled with zero only when irradiation is known and dark
    (night); every other gap stays NaN so that no fabricated daytime values are used.

    Args:
        gen: Cleaned generation series.
        irr: Cleaned irradiation series.
        freq: Grid frequency.

    Returns:
        Frame with columns ``gen_kwh`` and ``irr`` on a gap-free DatetimeIndex.
    """
    start = min(gen.index.min(), irr.index.min())
    end = max(gen.index.max(), irr.index.max())
    grid = pd.date_range(start.floor(freq), end.ceil(freq), freq=freq)
    frame = pd.DataFrame(
        {GEN_COL: gen.reindex(grid), IRR_COL: irr.reindex(grid)},
        index=grid,
    )
    dark = frame[IRR_COL].le(DARK_IRR)
    frame.loc[frame[GEN_COL].isna() & dark, GEN_COL] = 0.0
    frame.index.name = "timestamp"
    return frame


def load_dataset(
    generation_path: str | Path = "Generation data.xlsx",
    irradiation_path: str | Path = "Irradiation data.xlsx",
) -> pd.DataFrame:
    """Load both workbooks and return the cleaned, aligned frame.

    Args:
        generation_path: Generation workbook.
        irradiation_path: Irradiation workbook.

    Returns:
        Frame with columns ``gen_kwh`` and ``irr`` on a regular 15-minute grid.
    """
    gen = clean_generation(stack_month_pairs(read_raw_sheet(generation_path)))
    irr = clean_irradiation(stack_month_pairs(read_raw_sheet(irradiation_path)))
    return align(gen, irr)


@dataclass(frozen=True)
class DataReport:
    """Summary statistics describing data quality of a raw/aligned pair."""

    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    gen_nan_fraction: float
    irr_nan_fraction: float
    gen_spikes: int


def quality_report(gen_raw: pd.Series, irr_raw: pd.Series, frame: pd.DataFrame) -> DataReport:
    """Summarise data quality.

    Args:
        gen_raw: Raw generation series (before cleaning).
        irr_raw: Raw irradiation series (before cleaning), kept for symmetry and future checks.
        frame: Aligned frame from :func:`align`.

    Returns:
        A :class:`DataReport`.
    """
    del irr_raw
    spikes = int(((gen_raw < 0) | (gen_raw > MAX_GEN_KWH)).sum())
    return DataReport(
        rows=len(frame),
        start=frame.index.min(),
        end=frame.index.max(),
        gen_nan_fraction=float(frame[GEN_COL].isna().mean()),
        irr_nan_fraction=float(frame[IRR_COL].isna().mean()),
        gen_spikes=spikes,
    )
