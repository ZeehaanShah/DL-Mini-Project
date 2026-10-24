"""
Test suite for NASA POWER dataset integrity.
Tests schema, data quality, temporal completeness, and parameter ranges.
"""
import pytest
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NASA_CSV = PROJECT_ROOT / "data" / "raw" / "nasa_power" / \
    "POWER_Point_Hourly_20230101_20231231_019d08N_072d88E_LST.csv"

pytestmark = pytest.mark.skipif(
    not NASA_CSV.exists(),
    reason="NASA POWER CSV not found"
)


@pytest.fixture(scope="module")
def nasa_df():
    """Load NASA POWER data once for all tests in this module."""
    import pandas as pd
    # Header is at line 12 (0-indexed), so skip 12 lines
    df = pd.read_csv(NASA_CSV, skiprows=12)
    return df


class TestNasaSchema:
    """Test NASA POWER file schema."""

    def test_file_exists(self):
        assert NASA_CSV.exists()

    def test_header_metadata(self):
        """File should have NASA POWER metadata header."""
        with open(NASA_CSV, 'r') as f:
            first_line = f.readline().strip()
        assert first_line == "-BEGIN HEADER-", f"Expected '-BEGIN HEADER-', got '{first_line}'"

    def test_expected_columns(self, nasa_df):
        expected = ["YEAR", "MO", "DY", "HR",
                    "ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS10M"]
        assert list(nasa_df.columns) == expected, \
            f"Columns mismatch: {list(nasa_df.columns)}"

    def test_column_types(self, nasa_df):
        assert nasa_df["YEAR"].dtype in ["int64", "int32"]
        assert nasa_df["MO"].dtype in ["int64", "int32"]
        assert nasa_df["DY"].dtype in ["int64", "int32"]
        assert nasa_df["HR"].dtype in ["int64", "int32"]
        assert nasa_df["ALLSKY_SFC_SW_DWN"].dtype == "float64"
        assert nasa_df["T2M"].dtype == "float64"
        assert nasa_df["RH2M"].dtype == "float64"
        assert nasa_df["WS10M"].dtype == "float64"


class TestNasaRowCount:
    """Test that NASA POWER has complete hourly coverage."""

    def test_row_count(self, nasa_df):
        """365 days * 24 hours = 8760 rows for non-leap year."""
        assert len(nasa_df) == 8760, f"Expected 8760 rows, got {len(nasa_df)}"

    def test_year_is_2023(self, nasa_df):
        assert (nasa_df["YEAR"] == 2023).all(), "Not all rows are year 2023"

    def test_all_months_present(self, nasa_df):
        months = sorted(nasa_df["MO"].unique())
        assert months == list(range(1, 13)), f"Missing months: {months}"

    def test_all_hours_present(self, nasa_df):
        hours = sorted(nasa_df["HR"].unique())
        assert hours == list(range(0, 24)), f"Missing hours: {hours}"


class TestNasaDataQuality:
    """Test data quality for NASA POWER."""

    def test_no_nan(self, nasa_df):
        total_nan = nasa_df.isna().sum().sum()
        assert total_nan == 0, f"Found {total_nan} NaN values"

    def test_no_sentinel_values(self, nasa_df):
        """Check that -999 sentinel values are not present."""
        for col in ["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS10M"]:
            count = (nasa_df[col] == -999).sum()
            assert count == 0, f"Column {col} has {count} sentinel (-999) values"

    def test_no_duplicates(self, nasa_df):
        dup_count = nasa_df.duplicated().sum()
        assert dup_count == 0, f"Found {dup_count} duplicate rows"

    def test_irradiance_non_negative(self, nasa_df):
        """Solar irradiance should never be negative."""
        assert (nasa_df["ALLSKY_SFC_SW_DWN"] >= 0).all(), "Negative irradiance found"

    def test_irradiance_plausible_max(self, nasa_df):
        """Max irradiance should be below ~1400 Wh/m^2 (theoretical max)."""
        assert nasa_df["ALLSKY_SFC_SW_DWN"].max() <= 1400, \
            f"Irradiance max {nasa_df['ALLSKY_SFC_SW_DWN'].max()} exceeds plausible limit"

    def test_temperature_plausible(self, nasa_df):
        """Mumbai temperatures should be between -10 and 55 C."""
        assert nasa_df["T2M"].min() >= -10, f"Temperature too low: {nasa_df['T2M'].min()}"
        assert nasa_df["T2M"].max() <= 55, f"Temperature too high: {nasa_df['T2M'].max()}"

    def test_humidity_range(self, nasa_df):
        """Relative humidity should be between 0 and 100%."""
        assert nasa_df["RH2M"].min() >= 0, "Humidity below 0%"
        assert nasa_df["RH2M"].max() <= 100, "Humidity above 100%"

    def test_wind_speed_non_negative(self, nasa_df):
        """Wind speed should be non-negative."""
        assert (nasa_df["WS10M"] >= 0).all(), "Negative wind speed found"

    def test_wind_speed_plausible(self, nasa_df):
        """Wind speed at 10m should be below ~50 m/s for normal conditions."""
        assert nasa_df["WS10M"].max() <= 50, \
            f"Wind speed {nasa_df['WS10M'].max()} m/s seems implausibly high"

    def test_nighttime_irradiance_zero(self, nasa_df):
        """
        Very late night hours (0-4) should have zero or near-zero irradiance.
        Mumbai is ~73E, so LST nighttime ~0-5 should have zero irradiance.
        """
        night = nasa_df[nasa_df["HR"].isin([0, 1, 2, 3, 4])]
        # Allow small tolerance for satellite measurement artifacts
        assert (night["ALLSKY_SFC_SW_DWN"] <= 1.0).all(), \
            "Non-zero irradiance found during deep nighttime hours"


class TestNasaTimestampParsing:
    """Test that timestamps can be parsed correctly."""

    def test_datetime_construction(self, nasa_df):
        """All rows should produce valid datetime objects."""
        import pandas as pd
        dt = pd.to_datetime(
            nasa_df[["YEAR", "MO", "DY"]].rename(
                columns={"YEAR": "year", "MO": "month", "DY": "day"}
            ).assign(hour=nasa_df["HR"])
        )
        assert len(dt) == len(nasa_df)
        assert dt.isna().sum() == 0, "Some datetimes could not be parsed"

    def test_timestamps_monotonic(self, nasa_df):
        """Timestamps should be monotonically increasing."""
        import pandas as pd
        dt = pd.to_datetime(
            nasa_df[["YEAR", "MO", "DY"]].rename(
                columns={"YEAR": "year", "MO": "month", "DY": "day"}
            ).assign(hour=nasa_df["HR"])
        )
        assert dt.is_monotonic_increasing, "Timestamps are not monotonically increasing"

    def test_no_missing_hours(self, nasa_df):
        """Check for gaps in hourly coverage."""
        import pandas as pd
        dt = pd.to_datetime(
            nasa_df[["YEAR", "MO", "DY"]].rename(
                columns={"YEAR": "year", "MO": "month", "DY": "day"}
            ).assign(hour=nasa_df["HR"])
        )
        diffs = dt.diff().dropna()
        expected_diff = pd.Timedelta(hours=1)
        gaps = diffs[diffs != expected_diff]
        assert len(gaps) == 0, f"Found {len(gaps)} gaps in hourly coverage"
