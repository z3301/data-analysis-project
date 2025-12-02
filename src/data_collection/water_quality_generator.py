"""
Water Quality Data Generator
=============================
Generates realistic proxy water quality data matching Wahoo Bay sensor parameters.

This module creates synthetic data that mimics real coastal water quality patterns
for Pompano Beach, FL area until actual Wahoo Bay API access is available.

Parameters simulated match the Wahoo Bay sensor array:
- Water temperature (°C)
- pH
- Dissolved oxygen (mg/L)
- Chlorophyll (µg/L)
- Phycocyanin (µg/L) - blue-green algae indicator
- Turbidity (NTU)
- Nitrate (mg/L)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


class WaterQualityGenerator:
    """Generates realistic water quality data for coastal environments."""

    def __init__(self, seed: int = 42):
        """
        Initialize the generator with realistic parameter ranges.

        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)

        # Realistic parameter ranges for Southeast Florida coastal waters
        self.params = {
            "water_temp": {
                "winter_mean": 20.0,  # °C (68°F)
                "summer_mean": 29.0,  # °C (84°F)
                "std": 1.5,
                "daily_variation": 2.0
            },
            "pH": {
                "mean": 8.1,
                "std": 0.2,
                "daily_variation": 0.15
            },
            "dissolved_oxygen": {
                "mean": 6.5,  # mg/L
                "std": 1.0,
                "daily_variation": 0.8,
                "temp_correlation": -0.3  # DO decreases with temp
            },
            "chlorophyll": {
                "baseline": 3.0,  # µg/L
                "std": 2.0,
                "bloom_prob": 0.05,  # 5% chance of algal bloom
                "bloom_magnitude": 25.0
            },
            "phycocyanin": {
                "baseline": 2.0,  # µg/L
                "std": 1.5,
                "bloom_prob": 0.02,  # 2% chance of cyanobacteria bloom
                "bloom_magnitude": 30.0
            },
            "turbidity": {
                "baseline": 5.0,  # NTU
                "std": 3.0,
                "rain_spike_factor": 15.0,  # Multiplier during rain
                "decay_rate": 0.85  # How fast turbidity clears
            },
            "nitrate": {
                "baseline": 0.5,  # mg/L
                "std": 0.3,
                "runoff_factor": 8.0  # Increase during rain/runoff
            }
        }

    def generate(
        self,
        start_date: str,
        end_date: str,
        frequency: str = "H"  # Hourly
    ) -> pd.DataFrame:
        """
        Generate synthetic water quality data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Data frequency ('H' for hourly, 'D' for daily)

        Returns:
            DataFrame with realistic water quality measurements
        """
        # Create datetime index
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        n = len(date_range)

        # Initialize DataFrame
        df = pd.DataFrame({"time": date_range})

        # Add day of year for seasonal patterns
        df["day_of_year"] = df["time"].dt.dayofyear
        df["hour"] = df["time"].dt.hour

        # Generate base parameters
        df = self._generate_temperature(df, n)
        df = self._generate_pH(df, n)
        df = self._generate_dissolved_oxygen(df, n)
        df = self._generate_chlorophyll(df, n)
        df = self._generate_phycocyanin(df, n)
        df = self._generate_turbidity(df, n)
        df = self._generate_nitrate(df, n)

        # Clean up temporary columns
        df = df.drop(columns=["day_of_year", "hour"])

        return df

    def _generate_temperature(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate water temperature with seasonal and diurnal patterns."""
        params = self.params["water_temp"]

        # Seasonal pattern (sinusoidal)
        seasonal = params["winter_mean"] + \
                   (params["summer_mean"] - params["winter_mean"]) * \
                   (0.5 - 0.5 * np.cos(2 * np.pi * df["day_of_year"] / 365))

        # Diurnal pattern (warmer in afternoon)
        diurnal = params["daily_variation"] * np.sin(2 * np.pi * (df["hour"] - 6) / 24)

        # Random variation
        noise = np.random.normal(0, params["std"], n)

        df["water_temp"] = seasonal + diurnal + noise
        df["water_temp"] = df["water_temp"].clip(15, 32)  # Physical limits

        return df

    def _generate_pH(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate pH with small random variations."""
        params = self.params["pH"]

        # Slight diurnal variation (photosynthesis increases pH during day)
        diurnal = params["daily_variation"] * np.sin(2 * np.pi * (df["hour"] - 6) / 24)

        # Random variation
        noise = np.random.normal(0, params["std"], n)

        df["pH"] = params["mean"] + diurnal + noise
        df["pH"] = df["pH"].clip(6.0, 9.0)  # Physical limits

        return df

    def _generate_dissolved_oxygen(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate dissolved oxygen with temperature correlation."""
        params = self.params["dissolved_oxygen"]

        # Temperature correlation (inverse)
        temp_effect = params["temp_correlation"] * (df["water_temp"] - 25)

        # Diurnal pattern (higher during day due to photosynthesis)
        diurnal = params["daily_variation"] * np.sin(2 * np.pi * (df["hour"] - 6) / 24)

        # Random variation
        noise = np.random.normal(0, params["std"], n)

        df["dissolved_oxygen"] = params["mean"] + temp_effect + diurnal + noise
        df["dissolved_oxygen"] = df["dissolved_oxygen"].clip(0, 12)  # Physical limits

        return df

    def _generate_chlorophyll(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate chlorophyll with occasional algal bloom events."""
        params = self.params["chlorophyll"]

        # Baseline with noise
        baseline = np.random.lognormal(
            np.log(params["baseline"]),
            params["std"] / params["baseline"],
            n
        )

        # Random bloom events
        blooms = np.random.random(n) < params["bloom_prob"]
        bloom_magnitude = np.random.exponential(params["bloom_magnitude"], n)

        df["chlorophyll"] = np.where(blooms, bloom_magnitude, baseline)
        df["chlorophyll"] = df["chlorophyll"].clip(0, 100)

        return df

    def _generate_phycocyanin(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate phycocyanin (blue-green algae) with rare harmful bloom events."""
        params = self.params["phycocyanin"]

        # Baseline with noise
        baseline = np.random.lognormal(
            np.log(params["baseline"]),
            params["std"] / params["baseline"],
            n
        )

        # Random harmful algal bloom events (rarer than chlorophyll blooms)
        blooms = np.random.random(n) < params["bloom_prob"]
        bloom_magnitude = np.random.exponential(params["bloom_magnitude"], n)

        df["phycocyanin"] = np.where(blooms, bloom_magnitude, baseline)
        df["phycocyanin"] = df["phycocyanin"].clip(0, 80)

        # Smooth blooms over time (blooms persist for days)
        df["phycocyanin"] = df["phycocyanin"].rolling(window=24, min_periods=1).mean()

        return df

    def _generate_turbidity(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate turbidity with rain-driven spikes."""
        params = self.params["turbidity"]

        # Baseline turbidity
        baseline = np.random.lognormal(
            np.log(params["baseline"]),
            params["std"] / params["baseline"],
            n
        )

        # Simulate rain events (will be correlated with actual rain data later)
        rain_events = np.random.random(n) < 0.1  # 10% chance of rain spike
        rain_spike = params["rain_spike_factor"] * np.random.exponential(2, n)

        turbidity = np.where(rain_events, baseline + rain_spike, baseline)

        # Decay effect (turbidity settles over time)
        for i in range(1, n):
            if not rain_events[i]:
                turbidity[i] = max(
                    turbidity[i],
                    turbidity[i-1] * params["decay_rate"]
                )

        df["turbidity"] = turbidity
        df["turbidity"] = df["turbidity"].clip(0, 200)

        return df

    def _generate_nitrate(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Generate nitrate with runoff-driven spikes."""
        params = self.params["nitrate"]

        # Baseline
        baseline = np.random.lognormal(
            np.log(params["baseline"]),
            params["std"] / params["baseline"],
            n
        )

        # Runoff events (correlated with rain)
        runoff_events = np.random.random(n) < 0.08  # 8% chance
        runoff_spike = params["runoff_factor"] * np.random.exponential(1, n)

        df["nitrate"] = np.where(runoff_events, baseline + runoff_spike, baseline)
        df["nitrate"] = df["nitrate"].clip(0, 20)

        return df

    def correlate_with_rain(
        self,
        df: pd.DataFrame,
        rain_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adjust turbidity and nitrate based on actual rainfall data.

        Args:
            df: Water quality DataFrame
            rain_data: DataFrame with 'time' and 'rain_accumulation' columns

        Returns:
            Updated water quality DataFrame
        """
        # Merge rain data
        merged = df.merge(rain_data[["time", "rain_accumulation"]], on="time", how="left")
        merged["rain_accumulation"] = merged["rain_accumulation"].fillna(0)

        # Increase turbidity after rain
        rain_factor = 1 + (merged["rain_accumulation"] / 10)  # Scale factor
        merged["turbidity"] = merged["turbidity"] * rain_factor
        merged["turbidity"] = merged["turbidity"].clip(0, 200)

        # Increase nitrate after rain (with lag)
        merged["nitrate"] = merged["nitrate"] * (1 + merged["rain_accumulation"] / 25)
        merged["nitrate"] = merged["nitrate"].clip(0, 20)

        return merged.drop(columns=["rain_accumulation"])


def main():
    """Example usage and testing."""
    generator = WaterQualityGenerator(seed=42)

    # Generate 1 year of hourly data
    print("Generating 1 year of synthetic water quality data...")
    df = generator.generate(
        start_date="2024-01-01",
        end_date="2024-12-31",
        frequency="H"
    )

    print(f"\nGenerated {len(df)} hourly records")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nBasic statistics:")
    print(df.describe())

    # Check for unsafe conditions
    print("\n\nChecking for unsafe conditions...")
    unsafe_turb = (df["turbidity"] > 100).sum()
    unsafe_pH_low = (df["pH"] < 6.5).sum()
    unsafe_pH_high = (df["pH"] > 8.5).sum()
    unsafe_DO = (df["dissolved_oxygen"] < 2).sum()
    unsafe_phyco = (df["phycocyanin"] > 20).sum()

    print(f"Records with high turbidity (>100 NTU): {unsafe_turb}")
    print(f"Records with low pH (<6.5): {unsafe_pH_low}")
    print(f"Records with high pH (>8.5): {unsafe_pH_high}")
    print(f"Records with low DO (<2 mg/L): {unsafe_DO}")
    print(f"Records with harmful algae (>20 µg/L): {unsafe_phyco}")

    # Save to CSV
    output_path = "../../data/raw/water_quality_synthetic.csv"
    df.to_csv(output_path, index=False)
    print(f"\nData saved to {output_path}")


if __name__ == "__main__":
    main()
