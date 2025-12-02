"""
Weather Data Generator
======================
Generates realistic weather station data matching Wahoo Bay parameters.

Simulates both on-site weather station and external weather API data
for Pompano Beach, FL area.

Weather Station Parameters:
- Barometric pressure (hPa)
- Wind direction & speed (degrees, m/s)
- Air temperature (°C)
- Relative humidity (%)
- Rain accumulation, duration, intensity (mm, hours, mm/h)
- Water level (mm)
- Solar radiation (W/m²)

External Weather Parameters:
- Cloud cover (%)
- Dew point (°C)
- Precipitation intensity & probability
- UV index
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional


class WeatherGenerator:
    """Generates realistic weather data for Southeast Florida coastal areas."""

    def __init__(self, seed: int = 42):
        """
        Initialize weather generator.

        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)

        # Realistic parameter ranges for Pompano Beach, FL
        self.params = {
            "air_temp": {
                "winter_mean": 20.0,  # °C
                "summer_mean": 28.0,
                "std": 2.0,
                "daily_variation": 4.0
            },
            "barometric_pressure": {
                "mean": 1015.0,  # hPa
                "std": 5.0,
                "storm_drop": 20.0
            },
            "humidity": {
                "mean": 75.0,  # %
                "std": 10.0,
                "inverse_temp_correlation": True
            },
            "wind_speed": {
                "mean": 4.5,  # m/s
                "std": 2.0,
                "max": 15.0
            },
            "rain": {
                "prob_rainy_day": 0.15,  # 15% of days
                "wet_season_multiplier": 2.5,  # May-Oct
                "mean_accumulation": 10.0,  # mm when it rains
                "std_accumulation": 8.0
            },
            "solar_radiation": {
                "max": 1000.0,  # W/m²
            },
            "cloud_cover": {
                "clear_mean": 20.0,  # %
                "rainy_mean": 80.0,
                "cloud_reduction": 0.7  # Solar reduction factor
            }
        }

    def generate(
        self,
        start_date: str,
        end_date: str,
        frequency: str = "H"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate weather data matching Wahoo Bay structure.

        Returns two DataFrames:
        1. Weather station data (weatherLatest1)
        2. External weather data (weatherLatest2)

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Data frequency ('H' for hourly)

        Returns:
            Tuple of (station_data, external_data)
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        n = len(date_range)

        # Create base DataFrame
        df = pd.DataFrame({"time": date_range})
        df["day_of_year"] = df["time"].dt.dayofyear
        df["hour"] = df["time"].dt.hour
        df["month"] = df["time"].dt.month

        # Generate weather parameters
        df = self._generate_temperature(df, n)
        df = self._generate_pressure(df, n)
        df = self._generate_humidity(df, n)
        df = self._generate_wind(df, n)
        df = self._generate_rain(df, n)
        df = self._generate_solar(df, n)
        df = self._generate_cloud_cover(df, n)

        # Split into two datasets matching Wahoo Bay structure
        station_data = self._create_station_data(df)
        external_data = self._create_external_data(df)

        return station_data, external_data

    def _generate_temperature(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate air temperature with seasonal and diurnal patterns."""
        params = self.params["air_temp"]

        # Seasonal pattern
        seasonal = params["winter_mean"] + \
                   (params["summer_mean"] - params["winter_mean"]) * \
                   (0.5 - 0.5 * np.cos(2 * np.pi * df["day_of_year"] / 365))

        # Diurnal pattern (peak at 3pm)
        diurnal = params["daily_variation"] * np.sin(2 * np.pi * (df["hour"] - 6) / 24)

        # Random noise
        noise = np.random.normal(0, params["std"], n)

        df["air_temp"] = seasonal + diurnal + noise
        df["air_temp"] = df["air_temp"].clip(10, 35)

        return df

    def _generate_pressure(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate barometric pressure with weather system variations."""
        params = self.params["barometric_pressure"]

        # Base pressure with slow variations (weather systems)
        base = params["mean"] + np.random.normal(0, params["std"], n)

        # Add slow-moving pressure systems
        for i in range(1, n):
            base[i] = 0.95 * base[i-1] + 0.05 * base[i]

        df["barometric_pressure"] = base
        df["barometric_pressure"] = df["barometric_pressure"].clip(980, 1040)

        return df

    def _generate_humidity(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate relative humidity."""
        params = self.params["humidity"]

        # Inverse correlation with temperature
        temp_effect = -0.5 * (df["air_temp"] - 25)

        # Higher at night
        diurnal = 10 * np.sin(2 * np.pi * (df["hour"] + 6) / 24)

        # Random noise
        noise = np.random.normal(0, params["std"], n)

        df["humidity"] = params["mean"] + temp_effect + diurnal + noise
        df["humidity"] = df["humidity"].clip(30, 100)

        return df

    def _generate_wind(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate wind speed and direction."""
        params = self.params["wind_speed"]

        # Wind speed (higher during day)
        diurnal_factor = 1 + 0.3 * np.sin(2 * np.pi * (df["hour"] - 6) / 24)
        base_speed = np.random.exponential(params["mean"], n) * diurnal_factor

        df["wind_speed_avg"] = base_speed.clip(0, params["max"])
        df["wind_speed_min"] = (df["wind_speed_avg"] * 0.7).clip(0, None)
        df["wind_speed_max"] = (df["wind_speed_avg"] * 1.4).clip(0, params["max"])

        # Wind direction (prevailing easterlies in Florida)
        df["wind_dir_avg"] = np.random.normal(90, 45, n) % 360  # East with variation
        df["wind_dir_min"] = (df["wind_dir_avg"] - 20) % 360
        df["wind_dir_max"] = (df["wind_dir_avg"] + 20) % 360

        return df

    def _generate_rain(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate rainfall with wet season patterns."""
        params = self.params["rain"]

        # Wet season (May-October)
        wet_season = df["month"].isin([5, 6, 7, 8, 9, 10])
        rain_prob = np.where(
            wet_season,
            params["prob_rainy_day"] * params["wet_season_multiplier"],
            params["prob_rainy_day"]
        )

        # Rain events
        is_raining = np.random.random(n) < rain_prob

        # Rain accumulation (mm)
        rain_amount = np.random.exponential(params["mean_accumulation"], n)
        df["rain_accumulation"] = np.where(is_raining, rain_amount, 0)

        # Rain duration (hours)
        rain_duration = np.random.exponential(2, n)  # Average 2 hours
        df["rain_duration"] = np.where(is_raining, rain_duration, 0)

        # Rain intensity (mm/hr)
        df["rain_intensity"] = np.where(
            df["rain_duration"] > 0,
            df["rain_accumulation"] / df["rain_duration"],
            0
        )

        # Peak intensity (1.5x average)
        df["rain_peak_intensity"] = df["rain_intensity"] * 1.5

        # Smooth rain over adjacent hours
        df["rain_accumulation"] = df["rain_accumulation"].rolling(window=3, min_periods=1, center=True).mean()

        return df

    def _generate_solar(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate solar radiation."""
        params = self.params["solar_radiation"]

        # Base solar radiation (cosine curve during day)
        hour_angle = 2 * np.pi * (df["hour"] - 12) / 24
        base_solar = params["max"] * np.maximum(0, np.cos(hour_angle))

        # Seasonal variation
        seasonal_factor = 0.8 + 0.2 * np.sin(2 * np.pi * (df["day_of_year"] - 80) / 365)

        # Cloud reduction (will add cloud cover next)
        df["solar_radiation"] = base_solar * seasonal_factor

        return df

    def _generate_cloud_cover(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate cloud cover and adjust solar radiation."""
        params = self.params["cloud_cover"]

        # Cloud cover higher when raining
        is_rainy = df["rain_accumulation"] > 0
        cloud_cover = np.where(
            is_rainy,
            np.random.normal(params["rainy_mean"], 15, n),
            np.random.normal(params["clear_mean"], 20, n)
        )

        df["cloud_cover"] = cloud_cover.clip(0, 100)

        # Reduce solar radiation by cloud cover
        cloud_factor = 1 - (df["cloud_cover"] / 100) * params["cloud_reduction"]
        df["solar_radiation"] = df["solar_radiation"] * cloud_factor

        return df

    def _create_station_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather station dataset (weatherLatest1 format)."""
        station = pd.DataFrame({
            "time": df["time"],
            "barometric_pressure": df["barometric_pressure"],
            "wind_dir_min": df["wind_dir_min"],
            "wind_dir_avg": df["wind_dir_avg"],
            "wind_dir_max": df["wind_dir_max"],
            "wind_speed_min": df["wind_speed_min"],
            "wind_speed_avg": df["wind_speed_avg"],
            "wind_speed_max": df["wind_speed_max"],
            "air_temp": df["air_temp"],
            "humidity": df["humidity"],
            "rain_accumulation": df["rain_accumulation"],
            "rain_duration": df["rain_duration"],
            "rain_intensity": df["rain_intensity"],
            "rain_peak_intensity": df["rain_peak_intensity"],
            "solar_radiation": df["solar_radiation"],
        })

        # Add water level (tide-like variation + noise)
        tidal_pattern = 500 + 200 * np.sin(2 * np.pi * df["hour"] / 12.4)  # Semi-diurnal tide
        station["water_level"] = tidal_pattern + np.random.normal(0, 50, len(df))

        return station

    def _create_external_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create external weather dataset (weatherLatest2 format)."""
        # Calculate dew point from temperature and humidity
        a, b = 17.27, 237.7
        alpha = ((a * df["air_temp"]) / (b + df["air_temp"])) + np.log(df["humidity"] / 100)
        dew_point = (b * alpha) / (a - alpha)

        external = pd.DataFrame({
            "time": df["time"],
            "cloud_cover": df["cloud_cover"],
            "dew_point": dew_point,
            "humidity": df["humidity"],
            "precip_intensity": df["rain_intensity"],
            "precip_probability": np.where(df["rain_accumulation"] > 0, 80, 10),
            "pressure_sea_level": df["barometric_pressure"],
            "pressure_surface": df["barometric_pressure"] - 1,  # Slight difference
            "temperature": df["air_temp"],
            "temperature_apparent": df["air_temp"] - 0.5,  # Simplified "feels like"
            "uv_index": (df["solar_radiation"] / 100).clip(0, 11).round(),
            "wind_direction": df["wind_dir_avg"],
            "wind_speed": df["wind_speed_avg"],
        })

        return external


def main():
    """Example usage and testing."""
    generator = WeatherGenerator(seed=42)

    print("Generating 1 year of synthetic weather data...")
    station_data, external_data = generator.generate(
        start_date="2024-01-01",
        end_date="2024-12-31",
        frequency="H"
    )

    print(f"\n=== Weather Station Data ===")
    print(f"Generated {len(station_data)} hourly records")
    print(f"\nFirst few rows:")
    print(station_data.head())
    print(f"\nStatistics:")
    print(station_data.describe())

    print(f"\n=== External Weather Data ===")
    print(f"Generated {len(external_data)} hourly records")
    print(f"\nFirst few rows:")
    print(external_data.head())

    # Check rain statistics
    rainy_hours = (station_data["rain_accumulation"] > 0).sum()
    total_rain = station_data["rain_accumulation"].sum()
    print(f"\n=== Rain Statistics ===")
    print(f"Rainy hours: {rainy_hours} ({rainy_hours/len(station_data)*100:.1f}%)")
    print(f"Total annual rainfall: {total_rain:.1f} mm ({total_rain/25.4:.1f} inches)")

    # Save to CSV
    station_data.to_csv("../../data/raw/weather_station_synthetic.csv", index=False)
    external_data.to_csv("../../data/raw/weather_external_synthetic.csv", index=False)
    print(f"\nData saved to data/raw/")


if __name__ == "__main__":
    main()
