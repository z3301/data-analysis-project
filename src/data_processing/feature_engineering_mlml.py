"""
Feature Engineering for MLML (Moss Landing) Data
=================================================
Creates derived features from MLML Monterey Bay oceanographic data.

Unique parameters available:
- Nitrate (µmol/L) - nutrient indicator
- Tide Height (m above MLLW) - tidal influence
- Beam Attenuation - turbidity proxy
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineerMLML:
    """Feature engineering for MLML oceanographic data."""

    def __init__(self):
        """Initialize feature engineer."""
        # Core parameters available at MLML
        self.core_params = [
            'water_temp',
            'pH',
            'dissolved_oxygen_pct',
            'dissolved_oxygen_umol',
            'nitrate',
            'salinity',
            'fluorescence',
            'beam_attenuation',
            'tide_height',
        ]

        # Rolling window sizes (in hours)
        self.windows = [6, 12, 24, 48]

        # Lag periods (in hours)
        self.lags = [1, 6, 12, 24]

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features.

        Args:
            df: DataFrame with hourly oceanographic data

        Returns:
            DataFrame with additional engineered features
        """
        df = df.copy()

        # Ensure time is datetime and sorted
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)

        # Get available parameters
        available = [p for p in self.core_params if p in df.columns]

        print(f"Engineering features for {len(available)} parameters...")

        # 1. Temporal features
        df = self._add_temporal_features(df)

        # 2. Rolling statistics
        df = self._add_rolling_features(df, available)

        # 3. Lag features
        df = self._add_lag_features(df, available)

        # 4. Rate of change
        df = self._add_rate_of_change(df, available)

        # 5. Anomaly features
        df = self._add_anomaly_features(df, available)

        # 6. Interaction features (MLML-specific)
        df = self._add_interaction_features(df)

        # 7. Tidal features (unique to MLML)
        df = self._add_tidal_features(df)

        print(f"Total features: {len(df.columns)}")

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'time' not in df.columns:
            return df

        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        df['day_of_year'] = df['time'].dt.dayofyear

        # Cyclical encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Cyclical encoding for month (seasonality)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Day/night indicator (approximate for Monterey Bay ~36.8°N)
        df['is_day'] = ((df['hour'] >= 7) & (df['hour'] <= 19)).astype(int)

        # Season
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,   # Winter
            3: 1, 4: 1, 5: 1,   # Spring (upwelling starts)
            6: 2, 7: 2, 8: 2,   # Summer (peak upwelling)
            9: 3, 10: 3, 11: 3  # Fall
        })

        return df

    def _add_rolling_features(self, df: pd.DataFrame, params: List[str]) -> pd.DataFrame:
        """Add rolling window statistics."""
        for param in params:
            if param not in df.columns:
                continue

            for window in self.windows:
                # Rolling mean
                df[f'{param}_rolling_mean_{window}h'] = (
                    df[param].rolling(window=window, min_periods=1).mean()
                )
                # Rolling std
                df[f'{param}_rolling_std_{window}h'] = (
                    df[param].rolling(window=window, min_periods=2).std()
                )
                # Rolling min
                df[f'{param}_rolling_min_{window}h'] = (
                    df[param].rolling(window=window, min_periods=1).min()
                )
                # Rolling max
                df[f'{param}_rolling_max_{window}h'] = (
                    df[param].rolling(window=window, min_periods=1).max()
                )

        return df

    def _add_lag_features(self, df: pd.DataFrame, params: List[str]) -> pd.DataFrame:
        """Add lagged features for time-series prediction."""
        for param in params:
            if param not in df.columns:
                continue

            for lag in self.lags:
                df[f'{param}_lag_{lag}h'] = df[param].shift(lag)

        return df

    def _add_rate_of_change(self, df: pd.DataFrame, params: List[str]) -> pd.DataFrame:
        """Add rate of change features."""
        for param in params:
            if param not in df.columns:
                continue

            # Hourly rate of change
            df[f'{param}_roc_1h'] = df[param].diff(1)

            # 6-hour rate of change
            df[f'{param}_roc_6h'] = df[param].diff(6)

            # 24-hour rate of change
            df[f'{param}_roc_24h'] = df[param].diff(24)

        return df

    def _add_anomaly_features(self, df: pd.DataFrame, params: List[str]) -> pd.DataFrame:
        """Add anomaly detection features (deviation from rolling mean)."""
        for param in params:
            if param not in df.columns:
                continue

            # 24-hour rolling mean and std
            rolling_mean = df[param].rolling(window=24, min_periods=6).mean()
            rolling_std = df[param].rolling(window=24, min_periods=6).std()

            # Z-score (how many std from rolling mean)
            df[f'{param}_zscore'] = (df[param] - rolling_mean) / rolling_std.replace(0, np.nan)

            # Simple anomaly (deviation from rolling mean)
            df[f'{param}_anomaly'] = df[param] - rolling_mean

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MLML-specific interaction features."""

        # Temperature-DO interaction (warmer water holds less oxygen)
        if 'water_temp' in df.columns and 'dissolved_oxygen_pct' in df.columns:
            df['temp_do_interaction'] = df['water_temp'] * df['dissolved_oxygen_pct']

        # pH-Temperature interaction
        if 'pH' in df.columns and 'water_temp' in df.columns:
            df['pH_temp_interaction'] = df['pH'] * df['water_temp']

        # Nitrate-Fluorescence ratio (nutrient uptake indicator)
        if 'nitrate' in df.columns and 'fluorescence' in df.columns:
            # Avoid division by zero
            df['nitrate_fluor_ratio'] = df['nitrate'] / df['fluorescence'].replace(0, np.nan)

        # Chlorophyll proxy (fluorescence) to DO ratio
        if 'fluorescence' in df.columns and 'dissolved_oxygen_pct' in df.columns:
            df['fluor_do_ratio'] = df['fluorescence'] / df['dissolved_oxygen_pct'].replace(0, np.nan)

        # Upwelling index proxy (low temp + high nitrate + low pH)
        if all(p in df.columns for p in ['water_temp', 'nitrate', 'pH']):
            # Normalize each component
            temp_norm = (df['water_temp'] - df['water_temp'].mean()) / df['water_temp'].std()
            nitrate_norm = (df['nitrate'] - df['nitrate'].mean()) / df['nitrate'].std()
            ph_norm = (df['pH'] - df['pH'].mean()) / df['pH'].std()

            # Upwelling: cold water, high nitrate, low pH
            df['upwelling_index'] = -temp_norm + nitrate_norm - ph_norm

        # Salinity-Temperature (water mass indicator)
        if 'salinity' in df.columns and 'water_temp' in df.columns:
            df['salinity_temp_product'] = df['salinity'] * df['water_temp']

        # Beam attenuation to fluorescence ratio (turbidity vs algae)
        if 'beam_attenuation' in df.columns and 'fluorescence' in df.columns:
            df['turbidity_algae_ratio'] = df['beam_attenuation'] / df['fluorescence'].replace(0, np.nan)

        return df

    def _add_tidal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add tidal influence features (unique to MLML)."""
        if 'tide_height' not in df.columns:
            return df

        # Tidal state indicator
        tide_mean = df['tide_height'].mean()
        df['tide_above_mean'] = (df['tide_height'] > tide_mean).astype(int)

        # Tidal rate of change (flood vs ebb)
        df['tide_change'] = df['tide_height'].diff(1)
        df['is_flooding'] = (df['tide_change'] > 0).astype(int)
        df['is_ebbing'] = (df['tide_change'] < 0).astype(int)

        # Extreme tide indicator
        tide_25 = df['tide_height'].quantile(0.25)
        tide_75 = df['tide_height'].quantile(0.75)
        df['is_low_tide'] = (df['tide_height'] < tide_25).astype(int)
        df['is_high_tide'] = (df['tide_height'] > tide_75).astype(int)

        # Tide-temperature interaction (tidal mixing)
        if 'water_temp' in df.columns:
            df['tide_temp_interaction'] = df['tide_height'] * df['water_temp']

        # Tide-nitrate interaction (upwelling during certain tidal phases)
        if 'nitrate' in df.columns:
            df['tide_nitrate_interaction'] = df['tide_height'] * df['nitrate']

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of all engineered feature names."""
        exclude = ['time', 'safety_label']
        return [c for c in df.columns if c not in exclude]


def main():
    """Test feature engineering."""
    import sys
    sys.path.insert(0, '..')

    from data_collection.mlml_data_loader import MLMLDataLoader
    from labeling.safety_classifier_mlml import SafetyClassifierMLML

    # Load data
    loader = MLMLDataLoader(data_dir="../../data/real")
    df = loader.load()

    # Resample to hourly
    hourly = loader.resample_to_hourly(df)

    # Engineer features
    engineer = FeatureEngineerMLML()
    featured = engineer.engineer_features(hourly)

    print(f"\nFeature engineering complete:")
    print(f"  Records: {len(featured):,}")
    print(f"  Features: {len(featured.columns)}")

    # Show sample
    print("\nSample features:")
    print(featured[['time', 'water_temp', 'nitrate', 'upwelling_index', 'tide_height']].head(10))


if __name__ == "__main__":
    main()
