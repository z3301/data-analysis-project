"""
Real Data Loader
=================
Loads and preprocesses real sensor data from Wahoo Bay and Pompano Beach.

Data Sources:
- Wahoo Bay Water Quality Sensors (wb_water_sensor.csv)
- Wahoo Bay Weather Station (wb_weather_sensor.csv)
- Pompano Beach Weather Station (pb_weather_sensor.csv)

All data from: https://www.sensestream.org/measurements
Date Range: Nov 30, 2024 - Nov 30, 2025 (1 year)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class RealDataLoader:
    """Loads and preprocesses real sensor data from CSV files."""

    def __init__(self, data_dir: str = "data/real"):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to directory containing real data CSV files
        """
        self.data_dir = Path(data_dir)

        # File paths
        self.water_quality_file = self.data_dir / "wb_water_sensor.csv"
        self.wb_weather_file = self.data_dir / "wb_weather_sensor.csv"
        self.pb_weather_file = self.data_dir / "pb_weather_sensor.csv"

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all three datasets.

        Returns:
            Tuple of (water_quality, wb_weather, pb_weather) DataFrames
        """
        print("Loading real data from SenseStream...")

        water_quality = self.load_water_quality()
        wb_weather = self.load_wahoo_bay_weather()
        pb_weather = self.load_pompano_beach_weather()

        print(f"\n✓ All data loaded successfully!")
        print(f"  Water Quality: {len(water_quality):,} records")
        print(f"  WB Weather:    {len(wb_weather):,} records")
        print(f"  PB Weather:    {len(pb_weather):,} records")

        return water_quality, wb_weather, pb_weather

    def load_water_quality(self) -> pd.DataFrame:
        """
        Load Wahoo Bay water quality sensor data.

        Original columns:
        - Timestamp (epoch ms)
        - Water Temperature Celsius
        - pH
        - DO Percent (Dissolved Oxygen as % saturation)
        - Chlorophyll (RFU) - Relative Fluorescence Units
        - Phycoerythrin (RFU) - Red algae pigment indicator
        - Turbidity (FNU) - Formazin Nephelometric Units
        - NO3-N mg/L - Nitrate as nitrogen
        - sPCond Milliseimens per Centimeter - Specific Conductance (salinity proxy)

        Returns:
            Cleaned DataFrame with standardized column names
        """
        print("\n[1/3] Loading water quality data...")

        df = pd.read_csv(self.water_quality_file)

        # Convert timestamp
        df['time'] = pd.to_datetime(df['Timestamp'], unit='ms')

        # Rename columns to standard names
        df = df.rename(columns={
            'Water Temperature Celsius': 'water_temp',
            'pH ': 'pH',
            'DO Percent': 'dissolved_oxygen_pct',  # Note: This is % saturation, not mg/L
            'Chlorophyll (RFU) ': 'chlorophyll_rfu',
            'Phycoerythrin (RFU) ': 'phycoerythrin_rfu',  # Similar to phycocyanin
            'Turbidity (FNU) ': 'turbidity',
            'NO3-N mg/L': 'nitrate',
            'sPCond Milliseimens per Centimeter': 'specific_conductance'
        })

        # Drop original timestamp
        df = df.drop(columns=['Timestamp'])

        # Reorder columns
        df = df[['time', 'water_temp', 'pH', 'dissolved_oxygen_pct',
                 'chlorophyll_rfu', 'phycoerythrin_rfu', 'turbidity',
                 'nitrate', 'specific_conductance']]

        # Basic cleaning - handle obvious sensor errors
        df = self._clean_water_quality(df)

        print(f"  ✓ Loaded {len(df):,} records")
        print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

        return df

    def _clean_water_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning rules to water quality data."""

        # Flag anomalies based on the raw data I observed
        # Some records have extremely high turbidity (millions) - likely sensor errors
        df.loc[df['turbidity'] > 1000, 'turbidity'] = np.nan

        # Negative nitrate values are sensor noise - set to 0
        df.loc[df['nitrate'] < 0, 'nitrate'] = 0

        # Very high nitrate (>1000) is likely sensor error
        df.loc[df['nitrate'] > 100, 'nitrate'] = np.nan

        # DO should be 0-200% saturation range
        df.loc[(df['dissolved_oxygen_pct'] < 0) | (df['dissolved_oxygen_pct'] > 200),
               'dissolved_oxygen_pct'] = np.nan

        # pH should be 0-14
        df.loc[(df['pH'] < 0) | (df['pH'] > 14), 'pH'] = np.nan

        # Water temp should be reasonable (10-40°C for Florida)
        df.loc[(df['water_temp'] < 10) | (df['water_temp'] > 40), 'water_temp'] = np.nan

        return df

    def load_wahoo_bay_weather(self) -> pd.DataFrame:
        """
        Load Wahoo Bay weather station data.

        Returns:
            Cleaned DataFrame with standardized column names
        """
        print("\n[2/3] Loading Wahoo Bay weather data...")

        df = pd.read_csv(self.wb_weather_file)

        # Convert timestamp
        df['time'] = pd.to_datetime(df['Timestamp'], unit='ms')

        # Rename columns
        df = df.rename(columns={
            'Barometric Pressure Hecto Pascals': 'barometric_pressure',
            'Wind direction minimum Degrees': 'wind_dir_min',
            'Wind direction average Degrees': 'wind_dir_avg',
            'Wind direction maximum Degrees': 'wind_dir_max',
            'Wind speed minimum m/s': 'wind_speed_min',
            'Wind speed average m/s': 'wind_speed_avg',
            'Wind speed maximum m/s': 'wind_speed_max',
            'Air temperature Celsius': 'air_temp',
            'Internal temperature Celsius': 'internal_temp',
            'Relative humidity  % RH': 'humidity',
            'Rain accumulation  Millimeters': 'rain_accumulation',
            'Rain duration Hours': 'rain_duration',
            'Rain intensity mm/h': 'rain_intensity',
            'Hail accumulation Count': 'hail_accumulation',
            'Hail duration ': 'hail_duration',
            'Hail intensity  -': 'hail_intensity',
            'Rain peak intensity mm/h': 'rain_peak_intensity',
            'Hail peak intensity ': 'hail_peak_intensity',
            'Water level Meters': 'water_level',
            'Solar radiation W/m^2': 'solar_radiation',
            'Heating temperature Celsius': 'heating_temp',
            'Heating voltage Volts': 'heating_voltage',
            'Supply voltage Volts': 'supply_voltage',
            '3.5V ref. voltage Volts': 'ref_voltage'
        })

        # Drop original timestamp
        df = df.drop(columns=['Timestamp'])

        # Select key columns
        key_cols = ['time', 'barometric_pressure', 'wind_dir_avg', 'wind_speed_avg',
                   'wind_speed_max', 'air_temp', 'humidity', 'rain_accumulation',
                   'rain_duration', 'rain_intensity', 'rain_peak_intensity',
                   'water_level', 'solar_radiation']

        df = df[[c for c in key_cols if c in df.columns]]

        # Clean
        df = self._clean_weather(df, source='wb')

        print(f"  ✓ Loaded {len(df):,} records")
        print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

        return df

    def load_pompano_beach_weather(self) -> pd.DataFrame:
        """
        Load Pompano Beach weather station data.

        Returns:
            Cleaned DataFrame with standardized column names
        """
        print("\n[3/3] Loading Pompano Beach weather data...")

        df = pd.read_csv(self.pb_weather_file)

        # Convert timestamp
        df['time'] = pd.to_datetime(df['Timestamp'], unit='ms')

        # Rename columns (same as WB weather)
        df = df.rename(columns={
            'Barometric Pressure Hecto Pascals': 'barometric_pressure',
            'Wind direction minimum Degrees': 'wind_dir_min',
            'Wind direction average Degrees': 'wind_dir_avg',
            'Wind direction maximum Degrees': 'wind_dir_max',
            'Wind speed minimum m/s': 'wind_speed_min',
            'Wind speed average m/s': 'wind_speed_avg',
            'Wind speed maximum m/s': 'wind_speed_max',
            'Air temperature Celsius': 'air_temp',
            'Internal temperature Celsius': 'internal_temp',
            'Relative humidity  % RH': 'humidity',
            'Rain accumulation  Millimeters': 'rain_accumulation',
            'Rain duration Hours': 'rain_duration',
            'Rain intensity mm/h': 'rain_intensity',
            'Hail accumulation Count': 'hail_accumulation',
            'Hail duration ': 'hail_duration',
            'Hail intensity  -': 'hail_intensity',
            'Rain peak intensity mm/h': 'rain_peak_intensity',
            'Hail peak intensity ': 'hail_peak_intensity',
            'Water level Meters': 'water_level',
            'Solar radiation W/m^2': 'solar_radiation',
            'Heating temperature Celsius': 'heating_temp',
            'Heating voltage Volts': 'heating_voltage',
            'Supply voltage Volts': 'supply_voltage',
            '3.5V ref. voltage Volts': 'ref_voltage'
        })

        # Drop original timestamp
        df = df.drop(columns=['Timestamp'])

        # Select key columns
        key_cols = ['time', 'barometric_pressure', 'wind_dir_avg', 'wind_speed_avg',
                   'wind_speed_max', 'air_temp', 'humidity', 'rain_accumulation',
                   'rain_duration', 'rain_intensity', 'rain_peak_intensity',
                   'water_level', 'solar_radiation']

        df = df[[c for c in key_cols if c in df.columns]]

        # Clean
        df = self._clean_weather(df, source='pb')

        print(f"  ✓ Loaded {len(df):,} records")
        print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

        return df

    def _clean_weather(self, df: pd.DataFrame, source: str = 'wb') -> pd.DataFrame:
        """Apply cleaning rules to weather data."""

        # Barometric pressure should be 950-1050 hPa
        if 'barometric_pressure' in df.columns:
            df.loc[(df['barometric_pressure'] < 950) | (df['barometric_pressure'] > 1050),
                   'barometric_pressure'] = np.nan

        # Wind speed should be 0-50 m/s
        for col in ['wind_speed_avg', 'wind_speed_max']:
            if col in df.columns:
                df.loc[(df[col] < 0) | (df[col] > 50), col] = np.nan

        # Air temp should be reasonable (-10 to 45°C)
        if 'air_temp' in df.columns:
            df.loc[(df['air_temp'] < -10) | (df['air_temp'] > 45), 'air_temp'] = np.nan

        # Humidity should be 0-100%
        if 'humidity' in df.columns:
            df.loc[(df['humidity'] < 0) | (df['humidity'] > 100), 'humidity'] = np.nan

        # Rain accumulation should be non-negative
        if 'rain_accumulation' in df.columns:
            df.loc[df['rain_accumulation'] < 0, 'rain_accumulation'] = 0

        # Solar radiation should be 0-1500 W/m²
        if 'solar_radiation' in df.columns:
            df.loc[(df['solar_radiation'] < 0) | (df['solar_radiation'] > 1500),
                   'solar_radiation'] = np.nan

        return df

    def resample_to_hourly(
        self,
        df: pd.DataFrame,
        agg_rules: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Resample data to hourly frequency.

        Args:
            df: DataFrame with 'time' column
            agg_rules: Custom aggregation rules per column

        Returns:
            Hourly resampled DataFrame
        """
        # Default aggregation rules
        default_agg = {
            'water_temp': 'mean',
            'pH': 'mean',
            'dissolved_oxygen_pct': 'mean',
            'chlorophyll_rfu': 'mean',
            'phycoerythrin_rfu': 'mean',
            'turbidity': 'mean',
            'nitrate': 'mean',
            'specific_conductance': 'mean',
            'barometric_pressure': 'mean',
            'wind_dir_avg': 'mean',
            'wind_speed_avg': 'mean',
            'wind_speed_max': 'max',
            'air_temp': 'mean',
            'humidity': 'mean',
            'rain_accumulation': 'sum',  # Sum rain over the hour
            'rain_intensity': 'mean',
            'rain_peak_intensity': 'max',
            'water_level': 'mean',
            'solar_radiation': 'mean',
        }

        if agg_rules:
            default_agg.update(agg_rules)

        # Set time as index
        df = df.set_index('time')

        # Only aggregate columns that exist
        agg_cols = {k: v for k, v in default_agg.items() if k in df.columns}

        # Resample
        hourly = df.resample('H').agg(agg_cols)

        # Reset index
        hourly = hourly.reset_index()

        return hourly

    def get_overlapping_period(
        self,
        water_quality: pd.DataFrame,
        wb_weather: pd.DataFrame,
        pb_weather: pd.DataFrame
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Find the overlapping time period across all three datasets.

        Returns:
            Tuple of (start_time, end_time)
        """
        start = max(
            water_quality['time'].min(),
            wb_weather['time'].min(),
            pb_weather['time'].min()
        )

        end = min(
            water_quality['time'].max(),
            wb_weather['time'].max(),
            pb_weather['time'].max()
        )

        return start, end


def main():
    """Test the data loader."""
    loader = RealDataLoader(data_dir="../../data/real")

    # Load all data
    water_quality, wb_weather, pb_weather = loader.load_all()

    # Show overlapping period
    start, end = loader.get_overlapping_period(water_quality, wb_weather, pb_weather)
    print(f"\n📅 Overlapping period: {start} to {end}")

    # Resample to hourly
    print("\n📊 Resampling to hourly frequency...")
    water_hourly = loader.resample_to_hourly(water_quality)
    wb_weather_hourly = loader.resample_to_hourly(wb_weather)
    pb_weather_hourly = loader.resample_to_hourly(pb_weather)

    print(f"  Water Quality (hourly): {len(water_hourly):,} records")
    print(f"  WB Weather (hourly):    {len(wb_weather_hourly):,} records")
    print(f"  PB Weather (hourly):    {len(pb_weather_hourly):,} records")

    # Show sample
    print("\n📈 Water Quality Sample:")
    print(water_hourly.head())

    print("\n📈 WB Weather Sample:")
    print(wb_weather_hourly.head())


if __name__ == "__main__":
    main()
