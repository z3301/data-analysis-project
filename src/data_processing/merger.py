"""
Data Merger and Cleaner
========================
Merges water quality, weather, and tide data into unified dataset.

Handles:
- Timestamp alignment (all to hourly granularity)
- Missing value imputation
- Outlier detection and handling
- Data validation
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import warnings


class DataMerger:
    """Merges and cleans multi-source environmental data."""

    def __init__(self):
        """Initialize the data merger."""
        self.merged_data = None
        self.merge_report = {}

    def merge_all(
        self,
        water_quality: pd.DataFrame,
        weather_station: pd.DataFrame,
        weather_external: pd.DataFrame,
        tide_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge all data sources on timestamp.

        Args:
            water_quality: Water quality sensor data
            weather_station: On-site weather station data
            weather_external: External weather API data
            tide_data: NOAA tide data

        Returns:
            Merged DataFrame with all parameters aligned by time
        """
        print("Starting data merge...")

        # Ensure all have 'time' column as datetime
        for df, name in [(water_quality, "water_quality"),
                         (weather_station, "weather_station"),
                         (weather_external, "weather_external"),
                         (tide_data, "tide_data")]:
            if "time" not in df.columns:
                raise ValueError(f"{name} must have 'time' column")
            df["time"] = pd.to_datetime(df["time"])

        # Start with water quality as base
        merged = water_quality.copy()
        self.merge_report["water_quality_records"] = len(merged)

        # Merge weather station data
        merged = merged.merge(
            weather_station,
            on="time",
            how="outer",
            suffixes=("", "_station")
        )
        self.merge_report["after_weather_station"] = len(merged)

        # Merge external weather data
        merged = merged.merge(
            weather_external,
            on="time",
            how="outer",
            suffixes=("", "_external")
        )
        self.merge_report["after_weather_external"] = len(merged)

        # Merge tide data
        merged = merged.merge(
            tide_data[["time", "tide_height"]],
            on="time",
            how="left"  # Left join since tide might have different granularity
        )
        self.merge_report["after_tide"] = len(merged)

        # Sort by time
        merged = merged.sort_values("time").reset_index(drop=True)

        # Handle duplicate columns (prefer original over suffixed)
        merged = self._resolve_duplicates(merged)

        print(f"Merge complete: {len(merged)} total records")
        self.merged_data = merged

        return merged

    def _resolve_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resolve duplicate columns from merges.

        Prefers non-suffixed columns, averages if both have data.
        """
        duplicate_bases = []

        # Find columns with _station or _external suffix
        for col in df.columns:
            if col.endswith("_station") or col.endswith("_external"):
                base = col.replace("_station", "").replace("_external", "")
                if base in df.columns:
                    duplicate_bases.append(base)

        duplicate_bases = list(set(duplicate_bases))

        for base in duplicate_bases:
            suffixed_cols = [c for c in df.columns if c.startswith(base + "_")]

            if len(suffixed_cols) > 0:
                # Average across all versions where available
                all_versions = [base] + suffixed_cols
                df[base] = df[all_versions].mean(axis=1)

                # Drop suffixed versions
                df = df.drop(columns=suffixed_cols)

        return df

    def clean_data(
        self,
        df: Optional[pd.DataFrame] = None,
        handle_missing: str = "interpolate",
        remove_outliers: bool = True
    ) -> pd.DataFrame:
        """
        Clean the merged dataset.

        Args:
            df: DataFrame to clean (uses self.merged_data if None)
            handle_missing: Strategy for missing values:
                - "interpolate": Linear interpolation
                - "forward_fill": Forward fill
                - "drop": Drop rows with missing values
            remove_outliers: Whether to remove statistical outliers

        Returns:
            Cleaned DataFrame
        """
        if df is None:
            if self.merged_data is None:
                raise ValueError("No data to clean. Run merge_all() first.")
            df = self.merged_data.copy()
        else:
            df = df.copy()

        print("Cleaning data...")
        initial_rows = len(df)

        # 1. Handle missing values
        df = self._handle_missing_values(df, strategy=handle_missing)

        # 2. Remove outliers
        if remove_outliers:
            df = self._remove_outliers(df)

        # 3. Validate physical constraints
        df = self._apply_physical_constraints(df)

        # 4. Add time-based features
        df = self._add_time_features(df)

        final_rows = len(df)
        print(f"Cleaning complete: {initial_rows} -> {final_rows} records")

        return df

    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "interpolate"
    ) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        missing_summary = df[numeric_cols].isnull().sum()
        if missing_summary.sum() > 0:
            print(f"\nMissing values found:")
            print(missing_summary[missing_summary > 0])

        if strategy == "interpolate":
            # Linear interpolation (simpler, doesn't require DatetimeIndex)
            df[numeric_cols] = df[numeric_cols].interpolate(
                method="linear",
                limit_direction="both"
            )

        elif strategy == "forward_fill":
            df[numeric_cols] = df[numeric_cols].fillna(method="ffill").fillna(method="bfill")

        elif strategy == "drop":
            df = df.dropna()

        return df

    def _remove_outliers(
        self,
        df: pd.DataFrame,
        z_threshold: float = 4.0
    ) -> pd.DataFrame:
        """
        Remove statistical outliers using z-score method.

        Args:
            df: DataFrame to clean
            z_threshold: Z-score threshold (default: 4.0 for conservative removal)

        Returns:
            DataFrame with outliers removed
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        initial_len = len(df)

        for col in numeric_cols:
            if col == "time":
                continue

            # Calculate z-scores
            mean = df[col].mean()
            std = df[col].std()

            if std > 0:  # Avoid division by zero
                z_scores = np.abs((df[col] - mean) / std)
                outliers = z_scores > z_threshold

                if outliers.sum() > 0:
                    print(f"  Removing {outliers.sum()} outliers from {col}")
                    df.loc[outliers, col] = np.nan

        # Interpolate the removed outliers
        df = self._handle_missing_values(df, strategy="interpolate")

        return df

    def _apply_physical_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply physical constraints to parameters.

        Ensures values are within physically realistic ranges.
        """
        constraints = {
            "water_temp": (0, 40),          # °C
            "air_temp": (-10, 45),          # °C
            "pH": (0, 14),                   # pH scale
            "dissolved_oxygen": (0, 20),     # mg/L
            "chlorophyll": (0, 150),         # µg/L
            "phycocyanin": (0, 100),         # µg/L
            "turbidity": (0, 300),           # NTU
            "nitrate": (0, 50),              # mg/L
            "humidity": (0, 100),            # %
            "cloud_cover": (0, 100),         # %
            "barometric_pressure": (950, 1050),  # hPa
            "wind_speed_avg": (0, 50),       # m/s
            "rain_accumulation": (0, 200),   # mm
            "solar_radiation": (0, 1200),    # W/m²
            "uv_index": (0, 15),             # Index
        }

        for col, (min_val, max_val) in constraints.items():
            if col in df.columns:
                original = df[col].copy()
                df[col] = df[col].clip(min_val, max_val)

                clipped = (original != df[col]).sum()
                if clipped > 0:
                    print(f"  Clipped {clipped} values in {col} to [{min_val}, {max_val}]")

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful time-based features."""
        df["hour"] = df["time"].dt.hour
        df["day_of_week"] = df["time"].dt.dayofweek
        df["month"] = df["time"].dt.month
        df["day_of_year"] = df["time"].dt.dayofyear

        # Is weekend?
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Season (for Florida: wet May-Oct, dry Nov-Apr)
        df["is_wet_season"] = df["month"].isin([5, 6, 7, 8, 9, 10]).astype(int)

        return df

    def get_merge_report(self) -> dict:
        """Get summary report of merge process."""
        return self.merge_report

    def save_merged_data(
        self,
        filepath: str,
        df: Optional[pd.DataFrame] = None
    ):
        """
        Save merged data to CSV.

        Args:
            filepath: Output file path
            df: DataFrame to save (uses self.merged_data if None)
        """
        if df is None:
            if self.merged_data is None:
                raise ValueError("No data to save.")
            df = self.merged_data

        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")


def main():
    """Example usage."""
    print("Example: Merging synthetic data")
    print("=" * 50)

    # Load synthetic data
    water_quality = pd.read_csv("../../data/raw/water_quality_synthetic.csv")
    weather_station = pd.read_csv("../../data/raw/weather_station_synthetic.csv")
    weather_external = pd.read_csv("../../data/raw/weather_external_synthetic.csv")

    # For this example, create simple tide data
    tide_data = weather_station[["time"]].copy()
    tide_data["tide_height"] = 0.5 + 0.3 * np.sin(
        2 * np.pi * np.arange(len(tide_data)) / 24.8
    )

    # Merge
    merger = DataMerger()
    merged = merger.merge_all(
        water_quality=water_quality,
        weather_station=weather_station,
        weather_external=weather_external,
        tide_data=tide_data
    )

    print("\n" + "=" * 50)
    print("Merge Report:")
    print(merger.get_merge_report())

    # Clean
    print("\n" + "=" * 50)
    cleaned = merger.clean_data(merged, handle_missing="interpolate")

    print("\n" + "=" * 50)
    print(f"Final dataset shape: {cleaned.shape}")
    print(f"\nColumns: {list(cleaned.columns)}")
    print(f"\nFirst few rows:")
    print(cleaned.head())

    # Save
    merger.save_merged_data("../../data/processed/merged_dataset.csv", cleaned)


if __name__ == "__main__":
    main()
