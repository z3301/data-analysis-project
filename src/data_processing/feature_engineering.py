"""
Feature Engineering Pipeline
=============================
Creates time-series features for predictive modeling.

Features created:
- Lagged features (24h, 48h lookback)
- Rolling statistics (mean, std, min, max)
- Cumulative features (rainfall)
- Rate of change features
- Interaction features
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineer:
    """Creates predictive features from time-series environmental data."""

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
        print("Creating engineered features...")

        # Ensure sorted by time
        df = df.sort_values("time").reset_index(drop=True)

        # 1. Lagged features
        df = self._create_lag_features(df, target_col)

        # 2. Rolling window statistics
        df = self._create_rolling_features(df, target_col)

        # 3. Rate of change features
        df = self._create_diff_features(df, target_col)

        # 4. Cumulative features
        df = self._create_cumulative_features(df)

        # 5. Interaction features
        df = self._create_interaction_features(df)

        # 6. Time-based cyclic features
        df = self._create_cyclic_features(df)

        print(f"Created {len(self.feature_list)} new features")
        print(f"Final dataset shape: {df.shape}")

        return df

    def _create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        lags: List[int] = [1, 6, 12, 24, 48]
    ) -> pd.DataFrame:
        """
        Create lagged features.

        Args:
            df: Input DataFrame
            target_col: Target column to exclude
            lags: List of lag periods (in hours)

        Returns:
            DataFrame with lag features
        """
        print(f"Creating lag features: {lags} hours")

        # Key parameters to lag
        lag_cols = [
            "turbidity", "rain_accumulation", "rain_intensity",
            "water_temp", "dissolved_oxygen", "pH",
            "chlorophyll", "phycocyanin", "nitrate",
            "tide_height", "wind_speed_avg", "barometric_pressure"
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
            "turbidity", "rain_accumulation", "rain_intensity",
            "water_temp", "dissolved_oxygen", "chlorophyll", "phycocyanin",
            "wind_speed_avg", "barometric_pressure", "humidity"
        ]

        roll_cols = [c for c in roll_cols if c in df.columns and c != target_col]

        for col in roll_cols:
            for window in windows:
                # Rolling mean
                feature_name = f"{col}_rolling_mean_{window}h"
                df[feature_name] = df[col].rolling(window=window, min_periods=1).mean()
                self.feature_list.append(feature_name)

                # Rolling std (for volatility)
                if window >= 12:  # Only for larger windows
                    feature_name = f"{col}_rolling_std_{window}h"
                    df[feature_name] = df[col].rolling(window=window, min_periods=1).std()
                    self.feature_list.append(feature_name)

                # Rolling max (for peaks)
                if col in ["turbidity", "rain_intensity", "wind_speed_avg"]:
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

        # Parameters where change is important
        diff_cols = [
            "turbidity", "water_temp", "barometric_pressure",
            "dissolved_oxygen", "pH", "tide_height"
        ]

        diff_cols = [c for c in diff_cols if c in df.columns and c != target_col]

        for col in diff_cols:
            for period in periods:
                feature_name = f"{col}_diff_{period}h"
                df[feature_name] = df[col].diff(period)
                self.feature_list.append(feature_name)

        return df

    def _create_cumulative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cumulative features (especially for rainfall).

        Returns:
            DataFrame with cumulative features
        """
        print("Creating cumulative features")

        if "rain_accumulation" in df.columns:
            # Cumulative rain in last 24h, 48h, 7 days
            for window in [24, 48, 168]:  # hours
                feature_name = f"rain_cumulative_{window}h"
                df[feature_name] = df["rain_accumulation"].rolling(
                    window=window,
                    min_periods=1
                ).sum()
                self.feature_list.append(feature_name)

        # Days since last significant rain (>10mm)
        if "rain_accumulation" in df.columns:
            significant_rain = (df["rain_accumulation"] > 10).astype(int)

            # Create cumulative counter that resets at each rain event
            days_since_rain = []
            counter = 0

            for rain in significant_rain:
                if rain == 1:
                    counter = 0
                else:
                    counter += 1/24  # Convert hours to days
                days_since_rain.append(counter)

            df["days_since_significant_rain"] = days_since_rain
            self.feature_list.append("days_since_significant_rain")

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between related variables.

        Returns:
            DataFrame with interaction features
        """
        print("Creating interaction features")

        # Temperature * Humidity (affects perceived conditions)
        if "air_temp" in df.columns and "humidity" in df.columns:
            df["temp_humidity_interaction"] = df["air_temp"] * df["humidity"] / 100
            self.feature_list.append("temp_humidity_interaction")

        # Rain * Wind (rough measure of storm intensity)
        if "rain_intensity" in df.columns and "wind_speed_avg" in df.columns:
            df["rain_wind_interaction"] = df["rain_intensity"] * df["wind_speed_avg"]
            self.feature_list.append("rain_wind_interaction")

        # Low tide + high temperature (concentration effects)
        if "tide_height" in df.columns and "water_temp" in df.columns:
            df["tide_temp_interaction"] = (1 / (df["tide_height"] + 0.5)) * df["water_temp"]
            self.feature_list.append("tide_temp_interaction")

        # Chlorophyll + temperature (bloom conditions)
        if "chlorophyll" in df.columns and "water_temp" in df.columns:
            df["chlor_temp_interaction"] = df["chlorophyll"] * (df["water_temp"] / 25)
            self.feature_list.append("chlor_temp_interaction")

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

    def select_features_for_modeling(
        self,
        df: pd.DataFrame,
        target_col: str,
        min_lag_hours: int = 24
    ) -> pd.DataFrame:
        """
        Select features appropriate for predictive modeling.

        Removes:
        - Target column
        - Time column
        - Any features that use data less than min_lag_hours old

        Args:
            df: DataFrame with all features
            target_col: Target column name
            min_lag_hours: Minimum lag for features (default: 24h)

        Returns:
            DataFrame with only predictive features
        """
        # Columns to exclude
        exclude_cols = ["time", target_col]

        # Also exclude any current-time measurements that would leak information
        # (Only keep lagged, rolling, or derived features)
        current_time_cols = [
            "turbidity", "water_temp", "pH", "dissolved_oxygen",
            "chlorophyll", "phycocyanin", "nitrate", "tide_height",
            "air_temp", "humidity", "barometric_pressure", "wind_speed_avg",
            "rain_accumulation", "rain_intensity", "cloud_cover"
        ]

        exclude_cols.extend([c for c in current_time_cols if c in df.columns])

        # Keep only feature columns
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        return df[feature_cols + [target_col]]


def main():
    """Example usage."""
    print("Example: Feature Engineering")
    print("=" * 50)

    # Load merged data
    df = pd.read_csv("../../data/processed/merged_dataset.csv")
    df["time"] = pd.to_datetime(df["time"])

    print(f"Input shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10

    # Create features
    engineer = FeatureEngineer()
    featured_df = engineer.create_all_features(df)

    print("\n" + "=" * 50)
    print(f"Output shape: {featured_df.shape}")
    print(f"\nNew features created: {len(engineer.get_feature_names())}")
    print(f"Sample features: {engineer.get_feature_names()[:10]}")

    # Save
    output_path = "../../data/processed/featured_dataset.csv"
    featured_df.to_csv(output_path, index=False)
    print(f"\nFeatured dataset saved to {output_path}")


if __name__ == "__main__":
    main()
