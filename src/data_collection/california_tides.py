"""
California Coastal Tide Data Collector
=======================================
Fetches tide data from NOAA CO-OPS API for California monitoring stations.

Stations:
- La Jolla (Scripps Pier): 9410230
- Newport Beach: 9410580
- Monterey: 9413450

API Documentation: https://api.tidesandcurrents.noaa.gov/api/prod/
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# Import the base NOAA collector
from data_collection.noaa_tides import NOAATidesCollector


# California tide station IDs
CALIFORNIA_STATIONS = {
    'scripps': {
        'id': '9410230',
        'name': 'La Jolla (Scripps Pier)',
        'lat': 32.867,
        'lon': -117.257
    },
    'newport': {
        'id': '9410580',
        'name': 'Newport Beach, Newport Bay Entrance',
        'lat': 33.603,
        'lon': -117.883
    },
    'monterey': {
        'id': '9413450',
        'name': 'Monterey, CA',
        'lat': 36.605,
        'lon': -121.888
    }
}


class CaliforniaTidesCollector:
    """Collector for California coastal tide data."""

    def __init__(self, station_key: str):
        """
        Initialize collector for a specific California station.

        Args:
            station_key: One of 'scripps', 'newport', or 'monterey'
        """
        if station_key not in CALIFORNIA_STATIONS:
            raise ValueError(f"Unknown station: {station_key}. Choose from: {list(CALIFORNIA_STATIONS.keys())}")

        self.station_key = station_key
        self.station_info = CALIFORNIA_STATIONS[station_key]
        self.collector = NOAATidesCollector(station_id=self.station_info['id'])

    def get_tide_data(
        self,
        start_date: str,
        end_date: str,
        product: str = "predictions"
    ) -> pd.DataFrame:
        """
        Fetch tide data for the station.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            product: 'predictions' or 'hourly_height'

        Returns:
            DataFrame with tide_height and time columns
        """
        print(f"Fetching tide data for {self.station_info['name']}...")
        print(f"  Station ID: {self.station_info['id']}")
        print(f"  Date range: {start_date} to {end_date}")

        df = self.collector.get_date_range(start_date, end_date, product=product)

        # Add station metadata
        df['station_name'] = self.station_info['name']
        df['station_lat'] = self.station_info['lat']
        df['station_lon'] = self.station_info['lon']

        print(f"  Retrieved {len(df):,} records")

        return df

    def get_multi_year_data(
        self,
        years: List[int] = [2023, 2024, 2025],
        product: str = "predictions"
    ) -> pd.DataFrame:
        """
        Fetch multiple years of tide data.

        Args:
            years: List of years to fetch
            product: 'predictions' or 'hourly_height'

        Returns:
            DataFrame with all years combined
        """
        all_data = []

        for year in years:
            start_date = f"{year}-01-01"
            # For current year, use today's date
            if year == datetime.now().year:
                end_date = datetime.now().strftime("%Y-%m-%d")
            else:
                end_date = f"{year}-12-31"

            try:
                df = self.get_tide_data(start_date, end_date, product)
                all_data.append(df)
            except Exception as e:
                print(f"  Warning: Failed to fetch {year}: {e}")

        if not all_data:
            raise ValueError("No tide data retrieved")

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)

        return result


def fetch_all_california_tides(
    output_dir: str = "data/real",
    years: List[int] = [2023, 2024, 2025]
) -> dict:
    """
    Fetch tide data for all California monitoring stations.

    Args:
        output_dir: Directory to save CSV files
        years: Years to fetch

    Returns:
        Dictionary with DataFrames for each station
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for station_key in ['scripps', 'newport']:  # MLML already has tide data
        print(f"\n{'='*60}")
        print(f"Fetching tides for {station_key.upper()}")
        print('='*60)

        try:
            collector = CaliforniaTidesCollector(station_key)
            df = collector.get_multi_year_data(years=years)

            # Save to CSV
            output_file = output_path / f"{station_key}_tides.csv"
            df.to_csv(output_file, index=False)
            print(f"\nSaved to: {output_file}")

            results[station_key] = df

        except Exception as e:
            print(f"Error fetching {station_key}: {e}")

    return results


def main():
    """Fetch tide data for Scripps and Newport."""
    print("California Coastal Tide Data Collection")
    print("="*60)

    results = fetch_all_california_tides(
        output_dir="../../data/real",
        years=[2023, 2024, 2025]
    )

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for station, df in results.items():
        print(f"\n{station.upper()}:")
        print(f"  Records: {len(df):,}")
        print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
        print(f"  Tide range: {df['tide_height'].min():.2f}m to {df['tide_height'].max():.2f}m")


if __name__ == "__main__":
    main()
