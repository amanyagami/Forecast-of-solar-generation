import datetime as dt
import io
import re
import zipfile

import numpy as np
import openpyxl
import pandas as pd
import pytest

from solarcast.data import (
    GEN_COL,
    IRR_COL,
    align,
    clean_generation,
    clean_irradiation,
    parse_timestamp,
    read_raw_sheet,
    stack_month_pairs,
)


def test_parse_string_is_day_first() -> None:
    assert parse_timestamp("13/11/2019 05:45") == pd.Timestamp("2019-11-13 05:45")
    assert parse_timestamp("31/10/2019 23:45:00") == pd.Timestamp("2019-10-31 23:45")


def test_parse_datetime_swaps_month_and_day() -> None:
    # Excel read "01/11/2019" as 11 January; the true date is 1 November.
    assert parse_timestamp(dt.datetime(2019, 1, 11, 6, 0)) == pd.Timestamp("2019-11-01 06:00")


def test_parse_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_timestamp(3.14)


def _synthetic_workbook(path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Date", "v", "Date", "v"])
    ws.append([dt.datetime(2019, 1, 2, 0, 15), 1.0, "13/01/2019 00:15", 5.0])  # 2 Jan / 13 Jan
    ws.append([dt.datetime(2019, 1, 2, 0, 30), 2.0, "13/01/2019 00:30", 6.0])
    wb.save(path)


def test_stack_month_pairs_roundtrip(tmp_path) -> None:
    path = tmp_path / "a.xlsx"
    _synthetic_workbook(path)
    series = stack_month_pairs(read_raw_sheet(path))
    assert series.index.is_monotonic_increasing
    # The datetime cells were month/day swapped by "Excel": 2 Jan -> 1 Feb.
    assert pd.Timestamp("2019-02-01 00:15") in series.index
    assert pd.Timestamp("2019-01-02 00:15") not in series.index
    assert series.loc["2019-01-13 00:30"] == 6.0
    assert len(series) == 4


def test_read_raw_sheet_survives_vendor_extlst(tmp_path) -> None:
    path = tmp_path / "b.xlsx"
    _synthetic_workbook(path)
    buf = io.BytesIO()
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(buf, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "xl/styles.xml":
                text = data.decode()
                text, n = re.subn(
                    r"<patternFill([^>]*?)/>",
                    r"<patternFill\1><extLst><ext uri='x'/></extLst></patternFill>",
                    text,
                )
                assert n > 0
                data = text.encode()
            dst.writestr(info, data)
    path.write_bytes(buf.getvalue())
    with pytest.raises(TypeError):  # plain openpyxl cannot read this file
        pd.read_excel(path, header=None, engine="openpyxl")
    assert read_raw_sheet(path).shape == (3, 4)


def test_clean_generation_removes_paired_spikes() -> None:
    s = pd.Series([0.0, 5.0, -474592.0, 474592.0, 7.0, -1e-9])
    out = clean_generation(s)
    assert out.isna().tolist() == [False, False, True, True, False, False]
    assert out.iloc[-1] == 0.0


def test_clean_irradiation_bounds() -> None:
    out = clean_irradiation(pd.Series([-3.0, 500.0, 5000.0]))
    assert out.iloc[0] == 0.0 and out.iloc[1] == 500.0 and np.isnan(out.iloc[2])


def test_align_regular_grid_and_night_fill() -> None:
    gi = pd.date_range("2019-01-01 05:30", periods=4, freq="15min")
    ii = pd.date_range("2019-01-01 00:15", periods=30, freq="15min")
    gen = pd.Series([np.nan, 1.0, 2.0, np.nan], index=gi)
    irr = pd.Series(0.0, index=ii)
    irr.loc[gi[2:]] = [300.0, 400.0]
    frame = align(gen, irr)
    assert frame.index.freq is not None or pd.infer_freq(frame.index) == "15min"
    assert frame.loc[gi[0], GEN_COL] == 0.0  # dark -> filled
    assert np.isnan(frame.loc[gi[3], GEN_COL])  # daylight gap stays missing
    assert frame.loc["2019-01-01 01:00", GEN_COL] == 0.0  # night before gen starts
    assert IRR_COL in frame
