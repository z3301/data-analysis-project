"""
Newport Pier Data Loader
========================
Loads and preprocesses real sensor data from Newport Pier Automated Shore Station.

Data Source: SCCOOS / CeNCOOS Newport Pier station
Location: Newport Beach, California (33.6073°N, 117.9289°W)
Date Range: Jan 1, 2023 - Nov 30, 2025 (3 years)
Sampling: ~4 minute intervals

Parameters:
- Sea Water Temperature (CTD)
- pH (SeaFET external sensor)
- Dissolved Oxygen % saturation (CTD)
- Dissolved Oxygen mg/L (CTD)
- Chlorophyll concentration (CTD)
- Salinity (CTD)
- Conductivity (CTD)
- Pressure/Depth (CTD)
- Density (CTD)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List


class NewportDataLoader:
    """Loads and preprocesses Newport Pier oceanographic data."""

    def __init__(self, data_dir: str = "data/real"):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to directory containing real data CSV files
        """
        self.data_dir = Path(data_dir)

        # Multi-year file mapping
        self.files = {
            2023: self.data_dir / "ism-cencoos-newport-pier-automat_0f4d_0d51_d866.csv",
            2024: self.data_dir / "ism-cencoos-newport-pier-automat_0aa7_4a65_ee36.csv",
            2025: self.data_dir / "newport-pier-automated-shore-sta_50a8_f200_9e59.csv",
        }

    def load(self, years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load Newport Pier dataset for specified years.

        Args:
            years: List of years to load (default: all available)

        Returns:
            Combined DataFrame with standardized column names
        """
        if years is None:
            years = [2023, 2024, 2025]

        print("Loading Newport Pier data...")
        print(f"  Location: 33.6073°N, 117.9289°W (Newport Beach, CA)")
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

        # Remove duplicates (overlap between files)
        df = df.drop_duplicates(subset=['time'], keep='first')

        # Apply cleaning
        df = self._clean_data(df)

        print(f"\n  Total records: {len(df):,}")
        print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

        return df

    def _load_single_file(self, filepath: Path, year: int) -> pd.DataFrame:
        """Load a single CSV file."""

        # Skip row 1 which contains units
        df = pd.read_csv(filepath, low_memory=False, skiprows=[1])

        # Parse time column (ISO 8601 format)
        df['time'] = pd.to_datetime(df['time'], format='ISO8601')

        # Column mapping - handle different naming conventions across files
        # 2023/2024 files use shorter names, 2025 file uses _ctd suffix
        column_mapping = {
            # 2023/2024 format (CeNCOOS)
            'sea_water_temperature': 'water_temp',
            'sea_water_ph_reported_on_total_scale': 'pH',
            'fractional_saturation_of_oxygen_in_sea_water': 'dissolved_oxygen_pct',
            'mass_concentration_of_oxygen_in_sea_water': 'dissolved_oxygen_mgl',
            'mass_concentration_of_chlorophyll_in_sea_water': 'chlorophyll',
            'sea_water_practical_salinity': 'salinity',
            'sea_water_electrical_conductivity': 'conductivity',
            'sea_water_pressure': 'pressure',
            'sea_water_sigma_t': 'density',

            # 2025 format (SCCOOS with _ctd suffix)
            'sea_water_temperature_ctd': 'water_temp',
            'sea_water_ph_reported_on_total_scale_seafet_external': 'pH',
            'fractional_saturation_of_oxygen_in_sea_water_ctd': 'dissolved_oxygen_pct',
            'mass_concentration_of_oxygen_in_sea_water_ctd': 'dissolved_oxygen_mgl',
            'mass_concentration_of_chlorophyll_in_sea_water_ctd': 'chlorophyll',
            'sea_water_practical_salinity_ctd': 'salinity',
            'sea_water_electrical_conductivity_ctd': 'conductivity',
            'sea_water_pressure_ctd': 'pressure',
            'sea_water_density_ctd': 'density',
        }

        # QC flag columns
        qc_columns = {
            'sea_water_temperature_qc_agg': 'water_temp_qc',
            'sea_water_temperature_ctd_qc_agg': 'water_temp_qc',
            'sea_water_ph_reported_on_total_scale_qc_agg': 'pH_qc',
            'sea_water_ph_reported_on_total_scale_seafet_external_qc_agg': 'pH_qc',
            'fractional_saturation_of_oxygen_in_sea_water_qc_agg': 'do_pct_qc',
            'fractional_saturation_of_oxygen_in_sea_water_ctd_qc_agg': 'do_pct_qc',
            'mass_concentration_of_oxygen_in_sea_water_qc_agg': 'do_mgl_qc',
            'mass_concentration_of_oxygen_in_sea_water_ctd_qc_agg': 'do_mgl_qc',
            'mass_concentration_of_chlorophyll_in_sea_water_qc_agg': 'chlorophyll_qc',
            'mass_concentration_of_chlorophyll_in_sea_water_ctd_qc_agg': 'chlorophyll_qc',
            'sea_water_practical_salinity_qc_agg': 'salinity_qc',
            'sea_water_practical_salinity_ctd_qc_agg': 'salinity_qc',
        }

        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df[new_name] = pd.to_numeric(df[old_name], errors='coerce')

        for old_name, new_name in qc_columns.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]

        # Select columns to keep
        keep_cols = ['time', 'water_temp', 'pH', 'dissolved_oxygen_pct',
                     'dissolved_oxygen_mgl', 'chlorophyll', 'salinity',
                     'conductivity', 'pressure', 'density',
                     'water_temp_qc', 'pH_qc', 'do_pct_qc', 'do_mgl_qc',
                     'chlorophyll_qc', 'salinity_qc']

        df = df[[c for c in keep_cols if c in df.columns]]

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning rules based on QC flags and physical limits."""

        # Use QC flags to filter bad data
        qc_mapping = {
            'water_temp': 'water_temp_qc',
            'pH': 'pH_qc',
            'dissolved_oxygen_pct': 'do_pct_qc',
            'dissolved_oxygen_mgl': 'do_mgl_qc',
            'chlorophyll': 'chlorophyll_qc',
            'salinity': 'salinity_qc',
        }

        for param, qc_col in qc_mapping.items():
            if param in df.columns and qc_col in df.columns:
                # Set to NaN where QC flag indicates bad data (4=bad, 9=missing)
                bad_qc = df[qc_col].isin([4, 9])
                df.loc[bad_qc, param] = np.nan

        # Physical limits
        # Temperature: 5-30°C for California coastal waters
        if 'water_temp' in df.columns:
            df.loc[(df['water_temp'] < 5) | (df['water_temp'] > 35), 'water_temp'] = np.nan

        # pH: 7.0-9.0 for seawater
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

        # Chlorophyll: 0-100 µg/L (negative values are sensor noise)
        if 'chlorophyll' in df.columns:
            df.loc[df['chlorophyll'] < -1, 'chlorophyll'] = np.nan
            df.loc[df['chlorophyll'] > 200, 'chlorophyll'] = np.nan

        # Salinity: 0-40 PSU
        if 'salinity' in df.columns:
            df.loc[(df['salinity'] < 0) | (df['salinity'] > 45), 'salinity'] = np.nan

        # Conductivity: 0-60 S/m
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
            'salinity': 'mean',
            'conductivity': 'mean',
            'pressure': 'mean',
            'density': 'mean',
        }

        # Set time as index
        df_indexed = df.set_index('time')

        # Only aggregate columns that exist
        agg_cols = {k: v for k, v in agg_rules.items() if k in df_indexed.columns}

        # Resample using 'h' (not deprecated 'H')
        hourly = df_indexed.resample('h').agg(agg_cols)

        # Reset index
        hourly = hourly.reset_index()

        # Merge tide data if requested
        if include_tides:
            hourly = self._merge_tide_data(hourly)

        return hourly

    def _merge_tide_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge NOAA tide data from newport_tides.csv.

        Args:
            df: DataFrame with 'time' column (hourly)

        Returns:
            DataFrame with tide_height column added
        """
        tide_file = self.data_dir / "newport_tides.csv"

        if not tide_file.exists():
            print(f"  Note: Tide file not found at {tide_file}")
            return df

        print("  Merging NOAA tide data (Station 9410580 - Newport Beach)...")

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
                  'chlorophyll', 'salinity', 'conductivity', 'pressure', 'density']

        available = [p for p in params if p in df.columns]

        stats = df[available].describe().T
        stats['missing_pct'] = (df[available].isnull().sum() / len(df)) * 100
        stats['valid_pct'] = 100 - stats['missing_pct']

        return stats


def main():
    """Test the data loader."""
    loader = NewportDataLoader(data_dir="../../data/real")

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
