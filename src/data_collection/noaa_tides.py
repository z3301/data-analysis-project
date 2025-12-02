"""
NOAA Tides Data Collector
==========================
Collects tide data from NOAA CO-OPS API for Hillsboro Inlet, FL (Station 8722862)

Station Info:
- Station ID: 8722862
- Name: Hillsboro Inlet Ocean, FL
- Coordinates: 26.25670°N, 80.08000°W
- Location: Pompano Beach area

API Documentation: https://api.tidesandcurrents.noaa.gov/api/prod/
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Literal
import time


class NOAATidesCollector:
    """Collector for NOAA tide and water level data."""

    BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    STATION_ID = "8722862"  # Hillsboro Inlet Ocean, FL

    def __init__(self, station_id: str = STATION_ID):
        """
        Initialize the NOAA Tides collector.

        Args:
            station_id: NOAA station identifier (default: Hillsboro Inlet)
        """
        self.station_id = station_id

    def get_data(
        self,
        start_date: str,
        end_date: str,
        product: Literal["water_level", "predictions", "hourly_height"] = "predictions",
        datum: str = "MLLW",
        time_zone: str = "GMT",
        units: str = "metric",
        interval: str = "h"  # hourly
    ) -> pd.DataFrame:
        """
        Fetch tide data from NOAA CO-OPS API.

        Args:
            start_date: Start date in YYYYMMDD format (e.g., "20241120")
            end_date: End date in YYYYMMDD format
            product: Type of data product
                - "water_level": Observed water levels
                - "predictions": Tide predictions
                - "hourly_height": Hourly water level heights
            datum: Vertical datum (MLLW, MSL, etc.)
            time_zone: Time zone for data (GMT or LST)
            units: Metric or English
            interval: h (hourly) or hilo (high/low only)

        Returns:
            DataFrame with columns: time, value, quality (for observations)
        """
        params = {
            "begin_date": start_date,
            "end_date": end_date,
            "station": self.station_id,
            "product": product,
            "datum": datum,
            "time_zone": time_zone,
            "units": units,
            "format": "json",
            "application": "WahooBay_Research"
        }

        if product in ["water_level", "hourly_height"]:
            params["interval"] = interval

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Handle different response formats
            if "data" in data:
                records = data["data"]
            elif "predictions" in data:
                records = data["predictions"]
            else:
                if "error" in data:
                    raise ValueError(f"API Error: {data['error'].get('message', 'Unknown error')}")
                raise ValueError(f"Unexpected response format: {list(data.keys())}")

            df = pd.DataFrame(records)

            # Rename columns for consistency
            if "t" in df.columns:
                df.rename(columns={"t": "time"}, inplace=True)
            if "v" in df.columns:
                df.rename(columns={"v": "tide_height"}, inplace=True)
                df["tide_height"] = pd.to_numeric(df["tide_height"], errors="coerce")

            # Convert time to datetime
            df["time"] = pd.to_datetime(df["time"])

            # Add metadata
            df["station_id"] = self.station_id
            df["product"] = product

            return df

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch NOAA data: {e}")

    def get_yearly_data(
        self,
        year: int = 2024,
        product: str = "predictions"
    ) -> pd.DataFrame:
        """
        Fetch a full year of tide data.

        NOAA API has a 31-day limit per request, so this method
        chunks the requests into monthly batches.

        Args:
            year: Year to fetch data for
            product: Type of data product

        Returns:
            DataFrame with full year of hourly tide data
        """
        all_data = []

        for month in range(1, 13):
            # Calculate start and end dates for this month
            start_date = datetime(year, month, 1)

            if month == 12:
                end_date = datetime(year, 12, 31)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(days=1)

            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")

            print(f"Fetching {product} data for {start_date.strftime('%B %Y')}...")

            try:
                df = self.get_data(start_str, end_str, product=product)
                all_data.append(df)

                # Be nice to the API
                time.sleep(0.5)

            except Exception as e:
                print(f"Warning: Failed to fetch data for {start_date.strftime('%B %Y')}: {e}")
                continue

        if not all_data:
            raise ValueError("No data was successfully fetched")

        # Combine all months
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values("time").reset_index(drop=True)

        return result

    def get_date_range(
        self,
        start_date: str,
        end_date: str,
        product: str = "predictions"
    ) -> pd.DataFrame:
        """
        Fetch tide data for an arbitrary date range.

        Automatically chunks requests to respect NOAA's 31-day limit.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            product: Type of data product

        Returns:
            DataFrame with hourly tide data for the specified range
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        all_data = []
        current_start = start

        while current_start < end:
            # NOAA API limit is 31 days per request
            current_end = min(current_start + timedelta(days=30), end)

            start_str = current_start.strftime("%Y%m%d")
            end_str = current_end.strftime("%Y%m%d")

            print(f"Fetching {product} data: {start_str} to {end_str}...")

            try:
                df = self.get_data(start_str, end_str, product=product)
                all_data.append(df)
                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"Warning: Failed to fetch data for {start_str} to {end_str}: {e}")

            current_start = current_end + timedelta(days=1)

        if not all_data:
            raise ValueError("No data was successfully fetched")

        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values("time").reset_index(drop=True)

        return result


def main():
    """Example usage and testing."""
    collector = NOAATidesCollector()

    # Example 1: Get last 7 days of tide predictions
    print("Example 1: Fetching last 7 days of tide predictions...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    df = collector.get_date_range(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        product="predictions"
    )

    print(f"\nFetched {len(df)} records")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nBasic statistics:")
    print(df[["tide_height"]].describe())

    # Save to CSV
    output_path = "../../data/raw/noaa_tides_sample.csv"
    df.to_csv(output_path, index=False)
    print(f"\nData saved to {output_path}")


if __name__ == "__main__":
    main()
