"""
Feature Engineering Pipeline for Newport Pier Data
===================================================
Creates time-series features for predictive modeling using Newport Pier parameters.

Available parameters:
- water_temp, pH, dissolved_oxygen_pct, dissolved_oxygen_mgl
- chlorophyll, salinity, conductivity, pressure, density

Features created:
- Lagged features (1h, 6h, 12h, 24h, 48h lookback)
- Rolling statistics (mean, std, min, max)
- Rate of change features
- Interaction features
- Cyclic time encoding
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineerNewport:
    """Creates predictive features from Newport Pier time-series data."""

    def __init__(self):
        """Initialize feature engineer."""
        self.feature_list = []

    def create_all_features(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create all features for predictive modeling.

        Args:
            df: Input DataFrame with time-series data
            target_col: Target column to exclude from feature creation

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        print("Creating engineered features for Newport Pier data...")

        # Ensure sorted by time
        if 'time' in df.columns:
            df = df.sort_values("time").reset_index(drop=True)

        # Add time-based columns if not present
        if 'time' in df.columns:
            df = self._add_time_columns(df)

        # 1. Lagged features
        df = self._create_lag_features(df, target_col)

        # 2. Rolling window statistics
        df = self._create_rolling_features(df, target_col)

        # 3. Rate of change features
        df = self._create_diff_features(df, target_col)

        # 4. Interaction features
        df = self._create_interaction_features(df)

        # 5. Time-based cyclic features
        df = self._create_cyclic_features(df)

        print(f"Created {len(self.feature_list)} new features")
        print(f"Final dataset shape: {df.shape}")

        return df

    def _add_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based columns from datetime."""
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['day_of_year'] = df['time'].dt.dayofyear
        df['month'] = df['time'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        return df

    def _create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        lags: List[int] = [1, 6, 12, 24, 48]
    ) -> pd.DataFrame:
        """
        Create lagged features for key parameters.

        Args:
            df: Input DataFrame
            target_col: Target column to exclude
            lags: List of lag periods (in hours)

        Returns:
            DataFrame with lag features
        """
        print(f"Creating lag features: {lags} hours")

        # Newport Pier parameters to lag
        lag_cols = [
            "water_temp", "pH", "dissolved_oxygen_pct", "dissolved_oxygen_mgl",
            "chlorophyll", "salinity", "conductivity", "pressure", "density"
        ]

        # Only lag columns that exist
        lag_cols = [c for c in lag_cols if c in df.columns and c != target_col]

        for col in lag_cols:
            for lag in lags:
                feature_name = f"{col}_lag_{lag}h"
                df[feature_name] = df[col].shift(lag)
                self.feature_list.append(feature_name)

        return df

    def _create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        windows: List[int] = [6, 12, 24, 48]
    ) -> pd.DataFrame:
        """
        Create rolling window statistics.

        Args:
            df: Input DataFrame
            target_col: Target column to exclude
            windows: List of window sizes (in hours)

        Returns:
            DataFrame with rolling features
        """
        print(f"Creating rolling features: {windows} hour windows")

        # Key parameters for rolling stats
        roll_cols = [
            "water_temp", "pH", "dissolved_oxygen_pct", "dissolved_oxygen_mgl",
            "chlorophyll", "salinity", "conductivity"
        ]

        roll_cols = [c for c in roll_cols if c in df.columns and c != target_col]

        for col in roll_cols:
            for window in windows:
                # Rolling mean
                feature_name = f"{col}_rolling_mean_{window}h"
                df[feature_name] = df[col].rolling(window=window, min_periods=1).mean()
                self.feature_list.append(feature_name)

                # Rolling std (for volatility) - only for larger windows
                if window >= 12:
                    feature_name = f"{col}_rolling_std_{window}h"
                    df[feature_name] = df[col].rolling(window=window, min_periods=1).std()
                    self.feature_list.append(feature_name)

                # Rolling min/max for key parameters
                if col in ["dissolved_oxygen_pct", "dissolved_oxygen_mgl", "pH", "chlorophyll"]:
                    feature_name = f"{col}_rolling_min_{window}h"
                    df[feature_name] = df[col].rolling(window=window, min_periods=1).min()
                    self.feature_list.append(feature_name)

                    feature_name = f"{col}_rolling_max_{window}h"
                    df[feature_name] = df[col].rolling(window=window, min_periods=1).max()
                    self.feature_list.append(feature_name)

        return df

    def _create_diff_features(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        periods: List[int] = [1, 6, 24]
    ) -> pd.DataFrame:
        """
        Create rate-of-change features.

        Args:
            df: Input DataFrame
            target_col: Target column to exclude
            periods: List of difference periods (in hours)

        Returns:
            DataFrame with difference features
        """
        print(f"Creating rate-of-change features: {periods} hour periods")

        # Parameters where change rate is important
        diff_cols = [
            "water_temp", "pH", "dissolved_oxygen_pct", "dissolved_oxygen_mgl",
            "chlorophyll", "salinity", "pressure"
        ]

        diff_cols = [c for c in diff_cols if c in df.columns and c != target_col]

        for col in diff_cols:
            for period in periods:
                feature_name = f"{col}_diff_{period}h"
                df[feature_name] = df[col].diff(period)
                self.feature_list.append(feature_name)

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between related variables.

        Returns:
            DataFrame with interaction features
        """
        print("Creating interaction features")

        # Temperature * Salinity (density proxy)
        if "water_temp" in df.columns and "salinity" in df.columns:
            df["temp_salinity_interaction"] = df["water_temp"] * df["salinity"]
            self.feature_list.append("temp_salinity_interaction")

        # Chlorophyll / DO (productivity indicator)
        if "chlorophyll" in df.columns and "dissolved_oxygen_pct" in df.columns:
            # Avoid division by zero
            df["chlor_do_ratio"] = df["chlorophyll"] / (df["dissolved_oxygen_pct"] + 1)
            self.feature_list.append("chlor_do_ratio")

        # Temperature * Chlorophyll (bloom conditions)
        if "chlorophyll" in df.columns and "water_temp" in df.columns:
            df["chlor_temp_interaction"] = df["chlorophyll"] * (df["water_temp"] / 20)
            self.feature_list.append("chlor_temp_interaction")

        # DO deviation from saturation (if we have both)
        if "dissolved_oxygen_pct" in df.columns:
            # Distance from 100% saturation
            df["do_saturation_deviation"] = abs(df["dissolved_oxygen_pct"] - 100)
            self.feature_list.append("do_saturation_deviation")

        # pH deviation from optimal (8.1 for seawater)
        if "pH" in df.columns:
            df["ph_deviation"] = abs(df["pH"] - 8.1)
            self.feature_list.append("ph_deviation")

        # Salinity anomaly (deviation from typical ~33.5 PSU)
        if "salinity" in df.columns:
            df["salinity_anomaly"] = df["salinity"] - 33.5
            self.feature_list.append("salinity_anomaly")

        return df

    def _create_cyclic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cyclic encoding of time features.

        Uses sine/cosine transformation to preserve cyclical nature.

        Returns:
            DataFrame with cyclic features
        """
        print("Creating cyclic time features")

        # Hour of day (24-hour cycle)
        if "hour" in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            self.feature_list.extend(["hour_sin", "hour_cos"])

        # Day of week (7-day cycle)
        if "day_of_week" in df.columns:
            df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
            self.feature_list.extend(["day_of_week_sin", "day_of_week_cos"])

        # Month (12-month cycle) - captures seasonality
        if "month" in df.columns:
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
            self.feature_list.extend(["month_sin", "month_cos"])

        # Day of year (365-day cycle)
        if "day_of_year" in df.columns:
            df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
            df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
            self.feature_list.extend(["day_of_year_sin", "day_of_year_cos"])

        return df

    def get_feature_names(self) -> List[str]:
        """
        Get list of created feature names.

        Returns:
            List of feature names
        """
        return self.feature_list.copy()


def main():
    """Example usage."""
    print("Example: Feature Engineering for Newport Pier")
    print("=" * 50)

    import sys
    sys.path.insert(0, '..')

    from data_collection.newport_data_loader import NewportDataLoader

    # Load Newport Pier data
    loader = NewportDataLoader(data_dir="../../data/real")
    df = loader.load()

    # Resample to hourly
    hourly = loader.resample_to_hourly(df)
    print(f"Input shape: {hourly.shape}")

    # Create features
    engineer = FeatureEngineerNewport()
    featured_df = engineer.create_all_features(hourly)

    print("\n" + "=" * 50)
    print(f"Output shape: {featured_df.shape}")
    print(f"\nNew features created: {len(engineer.get_feature_names())}")
    print(f"Sample features: {engineer.get_feature_names()[:10]}")


if __name__ == "__main__":
    main()
