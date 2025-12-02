"""
MLML (Moss Landing Marine Laboratories) Data Loader
====================================================
Loads and preprocesses real sensor data from MLML Seawater Intake.

Data Source: CeNCOOS MLML Station
Location: Moss Landing, Monterey Bay, California (36.8025°N, 121.7915°W)
Date Range: Jan 1, 2023 - Oct 27, 2025 (~3 years)
Sampling: ~5 minute intervals

Parameters:
- Sea Water Temperature
- pH (total scale)
- Dissolved Oxygen % saturation
- Dissolved Oxygen (µmol/L)
- Nitrate (µmol/L) - UNIQUE to this site
- Salinity (PSU)
- Conductivity
- Fluorescence (Chlorophyll proxy)
- Transmissivity
- Beam Attenuation (Turbidity proxy)
- Water Level / Tide (m above MLLW) - UNIQUE to this site
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List


class MLMLDataLoader:
    """Loads and preprocesses MLML Monterey Bay oceanographic data."""

    def __init__(self, data_dir: str = "data/real"):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to directory containing CSV files
        """
        self.data_dir = Path(data_dir)

        # Multi-year file mapping
        self.files = {
            2023: self.data_dir / "mlml_mlml_sea_3092_d163_46cd.csv",
            2024: self.data_dir / "mlml_mlml_sea_7fd2_317c_bb54.csv",
            2025: self.data_dir / "mlml_mlml_sea_cf93_1e22_0b66.csv",
        }

    def load(self, years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load MLML dataset for specified years.

        Args:
            years: List of years to load (default: all available)

        Returns:
            Combined DataFrame with standardized column names
        """
        if years is None:
            years = [2023, 2024, 2025]

        print("Loading MLML (Moss Landing) data...")
        print(f"  Location: 36.8025°N, 121.7915°W (Monterey Bay)")
        print(f"  Years: {years}")

        all_data = []

        for year in years:
            if year in self.files and self.files[year].exists():
                print(f"\n  Loading {year}...")
                df_year = self._load_single_file(self.files[year], year)
                all_data.append(df_year)
                print(f"    Records: {len(df_year):,}")
            else:
                print(f"  Warning: File for {year} not found at {self.files.get(year, 'N/A')}")

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

        # Parse time column
        df['time'] = pd.to_datetime(df['time'], format='ISO8601')

        # Column mapping to standardized names
        column_mapping = {
            'sea_water_temperature': 'water_temp',
            'sea_water_ph_reported_on_total_scale': 'pH',
            'fractional_saturation_of_oxygen_in_sea_water': 'dissolved_oxygen_pct',
            'mole_concentration_of_dissolved_molecular_oxygen_in_sea_water': 'dissolved_oxygen_umol',
            'mole_concentration_of_nitrate_in_sea_water': 'nitrate',
            'sea_water_practical_salinity': 'salinity',
            'sea_water_electrical_conductivity': 'conductivity',
            'fluorescence': 'fluorescence',
            'transmissivity': 'transmissivity',
            'volume_beam_attenuation_coefficient_of_radiative_flux_in_sea_water': 'beam_attenuation',
            'water_surface_above_mllw': 'tide_height',
            'pco2_in_sea_water': 'pco2',
        }

        # QC flag columns
        qc_columns = {
            'sea_water_temperature_qc_agg': 'water_temp_qc',
            'sea_water_ph_reported_on_total_scale_qc_agg': 'pH_qc',
            'fractional_saturation_of_oxygen_in_sea_water_qc_agg': 'do_pct_qc',
            'mole_concentration_of_nitrate_in_sea_water_qc_agg': 'nitrate_qc',
            'sea_water_practical_salinity_qc_agg': 'salinity_qc',
        }

        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df[new_name] = pd.to_numeric(df[old_name], errors='coerce')

        for old_name, new_name in qc_columns.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]

        # Select columns to keep
        keep_cols = ['time', 'water_temp', 'pH', 'dissolved_oxygen_pct', 'dissolved_oxygen_umol',
                     'nitrate', 'salinity', 'conductivity', 'fluorescence', 'transmissivity',
                     'beam_attenuation', 'tide_height', 'pco2',
                     'water_temp_qc', 'pH_qc', 'do_pct_qc', 'nitrate_qc', 'salinity_qc']

        df = df[[c for c in keep_cols if c in df.columns]]

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning rules based on physical limits."""

        # Temperature: Monterey Bay ranges from ~8-20°C typically
        if 'water_temp' in df.columns:
            df.loc[(df['water_temp'] < -2) | (df['water_temp'] > 30), 'water_temp'] = np.nan

        # pH: 6.5-9.5 for seawater (allowing for upwelling low pH)
        if 'pH' in df.columns:
            df.loc[(df['pH'] < 6.5) | (df['pH'] > 10), 'pH'] = np.nan

        # DO %: 0-200% saturation
        if 'dissolved_oxygen_pct' in df.columns:
            df.loc[(df['dissolved_oxygen_pct'] < 0) | (df['dissolved_oxygen_pct'] > 200),
                   'dissolved_oxygen_pct'] = np.nan

        # DO µmol/L: 0-600 µmol/L (typical range)
        if 'dissolved_oxygen_umol' in df.columns:
            df.loc[(df['dissolved_oxygen_umol'] < 0) | (df['dissolved_oxygen_umol'] > 600),
                   'dissolved_oxygen_umol'] = np.nan

        # Nitrate: 0-50 µmol/L typical for coastal (can spike during upwelling)
        if 'nitrate' in df.columns:
            df.loc[(df['nitrate'] < -5) | (df['nitrate'] > 100), 'nitrate'] = np.nan

        # Salinity: 25-40 PSU for coastal California
        if 'salinity' in df.columns:
            df.loc[(df['salinity'] < 20) | (df['salinity'] > 40), 'salinity'] = np.nan

        # Fluorescence (chlorophyll proxy): 0-150
        if 'fluorescence' in df.columns:
            df.loc[df['fluorescence'] < -2, 'fluorescence'] = np.nan
            df.loc[df['fluorescence'] > 200, 'fluorescence'] = np.nan

        # Tide height: -1 to 3 meters typical
        if 'tide_height' in df.columns:
            df.loc[(df['tide_height'] < -2) | (df['tide_height'] > 4), 'tide_height'] = np.nan

        # Beam attenuation: positive values
        if 'beam_attenuation' in df.columns:
            df.loc[df['beam_attenuation'] < 0, 'beam_attenuation'] = np.nan

        return df

    def resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to hourly frequency.

        Args:
            df: DataFrame with 'time' column

        Returns:
            Hourly resampled DataFrame
        """
        # Aggregation rules
        agg_rules = {
            'water_temp': 'mean',
            'pH': 'mean',
            'dissolved_oxygen_pct': 'mean',
            'dissolved_oxygen_umol': 'mean',
            'nitrate': 'mean',
            'salinity': 'mean',
            'conductivity': 'mean',
            'fluorescence': 'mean',
            'transmissivity': 'mean',
            'beam_attenuation': 'mean',
            'tide_height': 'mean',
            'pco2': 'mean',
        }

        # Set time as index
        df_indexed = df.set_index('time')

        # Only aggregate columns that exist
        agg_cols = {k: v for k, v in agg_rules.items() if k in df_indexed.columns}

        # Resample using 'h' (not deprecated 'H')
        hourly = df_indexed.resample('h').agg(agg_cols)

        # Reset index
        hourly = hourly.reset_index()

        return hourly

    def get_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for all parameters.

        Args:
            df: DataFrame with oceanographic parameters

        Returns:
            DataFrame with summary statistics
        """
        params = ['water_temp', 'pH', 'dissolved_oxygen_pct', 'dissolved_oxygen_umol',
                  'nitrate', 'salinity', 'conductivity', 'fluorescence',
                  'transmissivity', 'beam_attenuation', 'tide_height']

        available = [p for p in params if p in df.columns]

        stats = df[available].describe().T
        stats['missing_pct'] = (df[available].isnull().sum() / len(df)) * 100
        stats['valid_pct'] = 100 - stats['missing_pct']

        return stats


def main():
    """Test the data loader."""
    loader = MLMLDataLoader(data_dir="../../data/real")

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
