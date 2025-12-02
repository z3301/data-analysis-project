"""
Scripps Pier Data Loader
========================
Loads and preprocesses real sensor data from Scripps Pier Automated Shore Station.

Data Source: SCCOOS Scripps Pier station (Station ID: 120738)
Location: La Jolla, California (32.867°N, 117.257°W)
Files: NetCDF4/HDF5 format

Data spans 3 years:
- 2023: Jan 1 - Dec 31, 2023 (177,614 records)
- 2024: Jan 1 - Dec 31, 2024 (165,345 records)
- 2025: Jan 1 - Nov 30, 2025 (158,508 records)

Parameters:
- Water Temperature (CTD) - ~76% valid
- Salinity (CTD) - ~76% valid
- Chlorophyll (CTD) - ~76% valid
- pH (SeapHOx external) - ~21% valid
- Dissolved Oxygen % (SeapHOx) - ~21% valid
- Dissolved Oxygen mg/L (SeapHOx) - ~21% valid
- Turbidity (ECO) - 62% valid in 2025
- fDOM (ECO) - 62% valid in 2025
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict


class ScrippsDataLoader:
    """Loads and preprocesses Scripps Pier oceanographic data from NetCDF files."""

    # Mapping of internal variable IDs to standardized names
    # These IDs are consistent across the NetCDF files
    VARIABLE_MAPPING = {
        # CTD sensors (good coverage ~76%)
        'Water Temperature (CTD)': 'water_temp',
        'Salinity (CTD)': 'salinity',
        'Chlorophyll (CTD)': 'chlorophyll',
        'Conductivity (CTD)': 'conductivity',
        'Sea Water Pressure (CTD)': 'pressure',
        'Sea Water Sigma-t (CTD)': 'density',

        # SeapHOx sensors (partial coverage ~21%)
        'pH (SeapHOx external)': 'pH',
        'pH (SeapHOx internal)': 'pH_internal',
        'Oxygen Saturation (SeapHOx)': 'dissolved_oxygen_pct',
        'Dissolved Oxygen Concentration (SeapHOx)': 'dissolved_oxygen_mgl',
        'Water Temperature (SeapHOx)': 'water_temp_seaphox',
        'Salinity (SeapHOx)': 'salinity_seaphox',

        # ECO sensors (good coverage in 2025)
        'Turbidity (ECO)': 'turbidity',
        'Chlorophyll (ECO)': 'chlorophyll_eco',
        'Fluorescent Dissolved Organic Matter (fDOM) (ECO)': 'fdom',
    }

    def __init__(self, data_dir: str = "data/real"):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to directory containing NetCDF files
        """
        self.data_dir = Path(data_dir)

        # NetCDF file paths
        self.files = {
            2023: self.data_dir / "station_120738_split_20230101-20240101.nc",
            2024: self.data_dir / "station_120738_split_20240101-20250101.nc",
            2025: self.data_dir / "station_120738_split_20250101-20260101.nc",
        }

    def load(self, years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load Scripps Pier data from NetCDF files.

        Args:
            years: List of years to load (default: all available)

        Returns:
            Combined DataFrame with standardized column names
        """
        if years is None:
            years = [2023, 2024, 2025]

        print("Loading Scripps Pier data...")
        print(f"  Location: 32.867°N, 117.257°W (La Jolla, CA)")
        print(f"  Years: {years}")

        all_data = []

        for year in years:
            if year in self.files and self.files[year].exists():
                print(f"\n  Loading {year}...")
                df_year = self._load_single_file(self.files[year], year)
                all_data.append(df_year)
                print(f"    Records: {len(df_year):,}")
            else:
                print(f"  Warning: File for {year} not found")

        if not all_data:
            raise ValueError("No data files found")

        # Combine all years
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values('time').reset_index(drop=True)

        # Apply cleaning
        df = self._clean_data(df)

        print(f"\n  Total records: {len(df):,}")
        print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

        return df

    def _load_single_file(self, filepath: Path, year: int) -> pd.DataFrame:
        """Load a single NetCDF file."""

        f = h5py.File(filepath, 'r')

        # Get time data
        time_data = f['time'][:]
        time_units = f['time'].attrs.get('units', b'seconds since 1970-01-01T00:00:00+00:00')
        if isinstance(time_units, bytes):
            time_units = time_units.decode('utf-8')

        # Convert time to datetime
        # Time is in seconds since 1970-01-01
        timestamps = pd.to_datetime(time_data, unit='s', utc=True)

        # Create dataframe with time
        df = pd.DataFrame({'time': timestamps})

        # Load each variable
        for key in f.keys():
            if key.startswith('value_'):
                obj = f[key]
                long_name = obj.attrs.get('long_name', b'unknown')
                if isinstance(long_name, bytes):
                    long_name = long_name.decode('utf-8')

                # Map to standardized name
                if long_name in self.VARIABLE_MAPPING:
                    col_name = self.VARIABLE_MAPPING[long_name]
                    data = obj[:]

                    # Replace sentinel values (-9999) with NaN
                    data = np.where(data < -9000, np.nan, data)

                    df[col_name] = data

        # Load QC flags for key variables
        for key in f.keys():
            if key.startswith('qc_agg_'):
                obj = f[key]
                long_name = obj.attrs.get('long_name', b'unknown')
                if isinstance(long_name, bytes):
                    long_name = long_name.decode('utf-8')

                # Extract parameter name from QC long_name
                param_name = long_name.replace(' QC Aggregate Flag', '')
                if param_name in self.VARIABLE_MAPPING:
                    col_name = self.VARIABLE_MAPPING[param_name] + '_qc'
                    df[col_name] = obj[:]

        f.close()

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning rules based on QC flags and physical limits."""

        # Physical limits

        # Temperature: 5-30°C for California coastal waters
        if 'water_temp' in df.columns:
            df.loc[(df['water_temp'] < 5) | (df['water_temp'] > 35), 'water_temp'] = np.nan

        # pH: 6.5-9.0 for seawater
        if 'pH' in df.columns:
            df.loc[(df['pH'] < 6.5) | (df['pH'] > 9.5), 'pH'] = np.nan

        # DO %: 0-200% saturation
        if 'dissolved_oxygen_pct' in df.columns:
            df.loc[(df['dissolved_oxygen_pct'] < 0) | (df['dissolved_oxygen_pct'] > 200),
                   'dissolved_oxygen_pct'] = np.nan

        # DO mg/L: 0-15 mg/L
        if 'dissolved_oxygen_mgl' in df.columns:
            df.loc[(df['dissolved_oxygen_mgl'] < 0) | (df['dissolved_oxygen_mgl'] > 20),
                   'dissolved_oxygen_mgl'] = np.nan

        # Chlorophyll: -1 to 100 µg/L
        if 'chlorophyll' in df.columns:
            df.loc[df['chlorophyll'] < -1, 'chlorophyll'] = np.nan
            df.loc[df['chlorophyll'] > 200, 'chlorophyll'] = np.nan

        if 'chlorophyll_eco' in df.columns:
            df.loc[df['chlorophyll_eco'] < -1, 'chlorophyll_eco'] = np.nan
            df.loc[df['chlorophyll_eco'] > 200, 'chlorophyll_eco'] = np.nan

        # Salinity: 0-40 PSU
        if 'salinity' in df.columns:
            df.loc[(df['salinity'] < 0) | (df['salinity'] > 45), 'salinity'] = np.nan

        # Turbidity: 0-150 NTU
        if 'turbidity' in df.columns:
            df.loc[df['turbidity'] < -1, 'turbidity'] = np.nan
            df.loc[df['turbidity'] > 200, 'turbidity'] = np.nan

        # fDOM: reasonable range
        if 'fdom' in df.columns:
            df.loc[df['fdom'] < -5, 'fdom'] = np.nan
            df.loc[df['fdom'] > 500, 'fdom'] = np.nan

        # Conductivity: 0-60 mS/cm
        if 'conductivity' in df.columns:
            df.loc[(df['conductivity'] < 0) | (df['conductivity'] > 60), 'conductivity'] = np.nan

        return df

    def resample_to_hourly(self, df: pd.DataFrame, include_tides: bool = True) -> pd.DataFrame:
        """
        Resample data to hourly frequency and optionally merge tide data.

        Args:
            df: DataFrame with 'time' column
            include_tides: Whether to merge NOAA tide data

        Returns:
            Hourly resampled DataFrame
        """
        # Aggregation rules
        agg_rules = {
            'water_temp': 'mean',
            'pH': 'mean',
            'dissolved_oxygen_pct': 'mean',
            'dissolved_oxygen_mgl': 'mean',
            'chlorophyll': 'mean',
            'chlorophyll_eco': 'mean',
            'salinity': 'mean',
            'conductivity': 'mean',
            'pressure': 'mean',
            'density': 'mean',
            'turbidity': 'mean',
            'fdom': 'mean',
        }

        # Set time as index
        df_indexed = df.set_index('time')

        # Only aggregate columns that exist
        agg_cols = {k: v for k, v in agg_rules.items() if k in df_indexed.columns}

        # Resample using 'h' instead of deprecated 'H'
        hourly = df_indexed.resample('h').agg(agg_cols)

        # Reset index
        hourly = hourly.reset_index()

        # Merge tide data if requested
        if include_tides:
            hourly = self._merge_tide_data(hourly)

        return hourly

    def _merge_tide_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge NOAA tide data from scripps_tides.csv.

        Args:
            df: DataFrame with 'time' column (hourly)

        Returns:
            DataFrame with tide_height column added
        """
        tide_file = self.data_dir / "scripps_tides.csv"

        if not tide_file.exists():
            print(f"  Note: Tide file not found at {tide_file}")
            return df

        print("  Merging NOAA tide data (Station 9410230 - La Jolla)...")

        # Load tide data
        tides = pd.read_csv(tide_file)
        tides['time'] = pd.to_datetime(tides['time'])

        # Resample tide data to hourly (it's in 6-minute intervals)
        tides_indexed = tides.set_index('time')
        tides_hourly = tides_indexed[['tide_height']].resample('h').mean().reset_index()

        # Ensure both dataframes have timezone-aware timestamps for merging
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC')
        if tides_hourly['time'].dt.tz is None:
            tides_hourly['time'] = tides_hourly['time'].dt.tz_localize('UTC')

        # Merge on time
        merged = pd.merge(df, tides_hourly, on='time', how='left')

        tide_coverage = merged['tide_height'].notna().mean() * 100
        print(f"    Tide data coverage: {tide_coverage:.1f}%")

        return merged

    def get_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for all parameters.

        Args:
            df: DataFrame with oceanographic parameters

        Returns:
            DataFrame with summary statistics
        """
        params = ['water_temp', 'pH', 'dissolved_oxygen_pct', 'dissolved_oxygen_mgl',
                  'chlorophyll', 'chlorophyll_eco', 'salinity', 'conductivity',
                  'turbidity', 'fdom', 'pressure', 'density']

        available = [p for p in params if p in df.columns]

        stats = df[available].describe().T
        stats['missing_pct'] = (df[available].isnull().sum() / len(df)) * 100
        stats['valid_pct'] = 100 - stats['missing_pct']

        return stats


def main():
    """Test the data loader."""
    loader = ScrippsDataLoader(data_dir="../../data/real")

    # Load all 3 years
    df = loader.load(years=[2023, 2024, 2025])

    # Show summary stats
    print("\nSummary Statistics:")
    stats = loader.get_summary_stats(df)
    print(stats[['mean', 'std', 'min', 'max', 'valid_pct']].round(2))

    # Resample to hourly
    print("\nResampling to hourly frequency...")
    hourly = loader.resample_to_hourly(df)
    print(f"  Hourly records: {len(hourly):,}")

    # Show sample
    print("\nSample data:")
    print(hourly.head(10))


if __name__ == "__main__":
    main()
