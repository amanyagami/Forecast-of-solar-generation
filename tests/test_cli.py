import pandas as pd

from solarcast.cli import format_table, main


def test_format_table_renders_regimes() -> None:
    metrics = pd.DataFrame(
        [
            {
                "protocol": "holdout",
                "model": m,
                "horizon_min": h,
                "regime": r,
                "n": 10,
                "rmse": 1.0,
                "mae": 0.5,
                "nrmse": 0.1,
                "skill_vs_persistence": 0.1,
            }
            for m in ("persistence", "lightgbm")
            for h in (15, 30)
            for r in ("clear", "cloudy", "all")
        ]
    )
    text = format_table(metrics, "holdout")
    assert "clear days" in text and "cloudy days" in text and "lightgbm" in text


def test_cli_requires_subcommand() -> None:
    import pytest

    with pytest.raises(SystemExit):
        main([])
