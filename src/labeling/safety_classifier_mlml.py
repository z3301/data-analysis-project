"""
Safety Classification System for MLML (Moss Landing) Data
==========================================================
Generates SAFE/CAUTION/DANGER labels for MLML Monterey Bay oceanographic data.

Location: Moss Landing, Monterey Bay, California (36.8025°N, 121.7915°W)
Data Source: CeNCOOS MLML Seawater Intake

Available Parameters:
- Water Temperature (°C)
- pH (total scale)
- Dissolved Oxygen % saturation
- Dissolved Oxygen (µmol/L)
- Nitrate (µmol/L) - UNIQUE parameter
- Salinity (PSU)
- Fluorescence (Chlorophyll proxy)
- Beam Attenuation (Turbidity proxy)
- Tide Height (m above MLLW)

Monterey Bay Context:
- Upwelling zone - naturally experiences low pH, high nitrate events
- Cold, nutrient-rich waters
- Different baseline than Southern California sites

Classification based on:
- California Ocean Plan water quality objectives
- Monterey Bay ecosystem characteristics
- EPA aquatic life criteria
"""

import pandas as pd
import numpy as np
from typing import Dict


class SafetyClassifierMLML:
    """Classifies water conditions into SAFE/CAUTION/DANGER for MLML data."""

    # Classification constants
    SAFE = 0
    CAUTION = 1
    DANGER = 2

    def __init__(self):
        """Initialize safety classifier with Monterey Bay thresholds."""

        # DANGER thresholds (Priority 1 - Immediate hazards)
        self.danger_thresholds = {
            # Dissolved Oxygen - Severe hypoxia
            # Monterey Bay can naturally have lower DO during upwelling
            "dissolved_oxygen_pct": 25.0,  # Below this is DANGER
            "dissolved_oxygen_umol": 60.0,  # ~2 mg/L equivalent

            # pH extremes - Monterey upwelling can cause low pH naturally
            "pH_low": 7.0,  # More tolerant than SoCal due to upwelling
            "pH_high": 9.0,

            # Temperature extremes
            "water_temp_low": 6.0,   # Monterey is colder
            "water_temp_high": 22.0,

            # Very low salinity (major freshwater event)
            "salinity_low": 25.0,

            # Very high nitrate (extreme upwelling or pollution)
            "nitrate_high": 40.0,  # µmol/L

            # High fluorescence (severe bloom)
            "fluorescence_high": 80.0,
        }

        # CAUTION thresholds (Priority 2 - Elevated risk)
        self.caution_thresholds = {
            # Dissolved Oxygen - Moderate hypoxia
            "dissolved_oxygen_pct": 50.0,
            "dissolved_oxygen_umol": 150.0,  # ~4.8 mg/L equivalent

            # pH outside optimal range
            "pH_low": 7.6,
            "pH_high": 8.3,

            # Temperature outside comfort zone
            "water_temp_low": 8.0,
            "water_temp_high": 18.0,

            # High fluorescence (algal bloom indicator)
            "fluorescence": 30.0,
            "fluorescence_high": 50.0,

            # Elevated nitrate (upwelling or nutrient loading)
            "nitrate": 15.0,  # µmol/L
            "nitrate_high": 25.0,

            # Moderate salinity drop
            "salinity_low": 30.0,

            # High beam attenuation (turbidity proxy)
            "beam_attenuation": 40.0,

            # Extreme tides (may affect mixing)
            "tide_low": -0.3,
            "tide_high": 1.8,
        }

        self.label_stats = {
            "SAFE": 0,
            "CAUTION": 0,
            "DANGER": 0
        }

        self.danger_triggers = {}
        self.caution_triggers = {}

    def classify(
        self,
        df: pd.DataFrame,
        label_col: str = "safety_label"
    ) -> pd.DataFrame:
        """
        Classify each record as SAFE/CAUTION/DANGER.

        Args:
            df: DataFrame with oceanographic parameters
            label_col: Name for the label column

        Returns:
            DataFrame with safety_label column added
        """
        df = df.copy()

        # Initialize all as SAFE
        df[label_col] = self.SAFE

        # Reset trigger counts
        self.danger_triggers = {}
        self.caution_triggers = {}

        # Apply CAUTION conditions first
        caution_mask = self._get_caution_mask(df)
        df.loc[caution_mask, label_col] = self.CAUTION

        # Apply DANGER conditions (overrides CAUTION)
        danger_mask = self._get_danger_mask(df)
        df.loc[danger_mask, label_col] = self.DANGER

        # Update statistics
        self.label_stats["SAFE"] = (df[label_col] == self.SAFE).sum()
        self.label_stats["CAUTION"] = (df[label_col] == self.CAUTION).sum()
        self.label_stats["DANGER"] = (df[label_col] == self.DANGER).sum()

        return df

    def _get_danger_mask(self, df: pd.DataFrame) -> pd.Series:
        """Create boolean mask for DANGER conditions."""
        danger_conditions = []

        # 1. Severe Hypoxia (low DO %)
        if "dissolved_oxygen_pct" in df.columns:
            mask = (df["dissolved_oxygen_pct"] < self.danger_thresholds["dissolved_oxygen_pct"]) & df["dissolved_oxygen_pct"].notna()
            danger_conditions.append(mask)
            self.danger_triggers["Severe Hypoxia (DO < 25%)"] = mask.sum()

        # 2. Severe Hypoxia (low DO µmol/L)
        if "dissolved_oxygen_umol" in df.columns:
            mask = (df["dissolved_oxygen_umol"] < self.danger_thresholds["dissolved_oxygen_umol"]) & df["dissolved_oxygen_umol"].notna()
            danger_conditions.append(mask)
            self.danger_triggers["Severe Hypoxia (DO < 60 µmol/L)"] = mask.sum()

        # 3. Extreme pH
        if "pH" in df.columns:
            low_mask = (df["pH"] < self.danger_thresholds["pH_low"]) & df["pH"].notna()
            high_mask = (df["pH"] > self.danger_thresholds["pH_high"]) & df["pH"].notna()
            mask = low_mask | high_mask
            danger_conditions.append(mask)
            self.danger_triggers["Extreme pH (< 7.0 or > 9.0)"] = mask.sum()

        # 4. Temperature extremes
        if "water_temp" in df.columns:
            low_mask = (df["water_temp"] < self.danger_thresholds["water_temp_low"]) & df["water_temp"].notna()
            high_mask = (df["water_temp"] > self.danger_thresholds["water_temp_high"]) & df["water_temp"].notna()
            mask = low_mask | high_mask
            danger_conditions.append(mask)
            self.danger_triggers["Extreme Temperature"] = mask.sum()

        # 5. Severe freshwater intrusion
        if "salinity" in df.columns:
            mask = (df["salinity"] < self.danger_thresholds["salinity_low"]) & df["salinity"].notna()
            danger_conditions.append(mask)
            self.danger_triggers["Severe Freshwater Event (Salinity < 25)"] = mask.sum()

        # 6. Extreme nitrate (pollution or extreme upwelling)
        if "nitrate" in df.columns:
            mask = (df["nitrate"] > self.danger_thresholds["nitrate_high"]) & df["nitrate"].notna()
            danger_conditions.append(mask)
            self.danger_triggers["Extreme Nitrate (> 40 µmol/L)"] = mask.sum()

        # 7. Severe algal bloom
        if "fluorescence" in df.columns:
            mask = (df["fluorescence"] > self.danger_thresholds["fluorescence_high"]) & df["fluorescence"].notna()
            danger_conditions.append(mask)
            self.danger_triggers["Severe Algal Bloom (Fluorescence > 80)"] = mask.sum()

        # Combine all danger conditions (OR logic)
        if danger_conditions:
            danger_mask = pd.concat(danger_conditions, axis=1).any(axis=1)
        else:
            danger_mask = pd.Series([False] * len(df), index=df.index)

        return danger_mask

    def _get_caution_mask(self, df: pd.DataFrame) -> pd.Series:
        """Create boolean mask for CAUTION conditions."""
        caution_conditions = []

        # 1. Moderate Hypoxia (DO %)
        if "dissolved_oxygen_pct" in df.columns:
            mask = (
                (df["dissolved_oxygen_pct"] >= self.danger_thresholds["dissolved_oxygen_pct"]) &
                (df["dissolved_oxygen_pct"] < self.caution_thresholds["dissolved_oxygen_pct"]) &
                df["dissolved_oxygen_pct"].notna()
            )
            caution_conditions.append(mask)
            self.caution_triggers["Moderate Hypoxia (DO 25-50%)"] = mask.sum()

        # 2. Moderate Hypoxia (DO µmol/L)
        if "dissolved_oxygen_umol" in df.columns:
            mask = (
                (df["dissolved_oxygen_umol"] >= self.danger_thresholds["dissolved_oxygen_umol"]) &
                (df["dissolved_oxygen_umol"] < self.caution_thresholds["dissolved_oxygen_umol"]) &
                df["dissolved_oxygen_umol"].notna()
            )
            caution_conditions.append(mask)
            self.caution_triggers["Moderate Hypoxia (DO 60-150 µmol/L)"] = mask.sum()

        # 3. pH outside optimal range
        if "pH" in df.columns:
            low_mask = (
                (df["pH"] >= self.danger_thresholds["pH_low"]) &
                (df["pH"] < self.caution_thresholds["pH_low"]) &
                df["pH"].notna()
            )
            high_mask = (
                (df["pH"] > self.caution_thresholds["pH_high"]) &
                (df["pH"] <= self.danger_thresholds["pH_high"]) &
                df["pH"].notna()
            )
            mask = low_mask | high_mask
            caution_conditions.append(mask)
            self.caution_triggers["pH Outside Optimal (7.0-7.6 or 8.3-9.0)"] = mask.sum()

        # 4. Temperature outside comfort zone
        if "water_temp" in df.columns:
            low_mask = (
                (df["water_temp"] >= self.danger_thresholds["water_temp_low"]) &
                (df["water_temp"] < self.caution_thresholds["water_temp_low"]) &
                df["water_temp"].notna()
            )
            high_mask = (
                (df["water_temp"] > self.caution_thresholds["water_temp_high"]) &
                (df["water_temp"] <= self.danger_thresholds["water_temp_high"]) &
                df["water_temp"].notna()
            )
            mask = low_mask | high_mask
            caution_conditions.append(mask)
            self.caution_triggers["Temperature Stress"] = mask.sum()

        # 5. Elevated fluorescence (algal bloom)
        if "fluorescence" in df.columns:
            mask = (
                (df["fluorescence"] > self.caution_thresholds["fluorescence"]) &
                (df["fluorescence"] <= self.danger_thresholds["fluorescence_high"]) &
                df["fluorescence"].notna()
            )
            caution_conditions.append(mask)
            self.caution_triggers["Elevated Fluorescence (30-80)"] = mask.sum()

        # 6. Elevated nitrate
        if "nitrate" in df.columns:
            mask = (
                (df["nitrate"] > self.caution_thresholds["nitrate"]) &
                (df["nitrate"] <= self.danger_thresholds["nitrate_high"]) &
                df["nitrate"].notna()
            )
            caution_conditions.append(mask)
            self.caution_triggers["Elevated Nitrate (15-40 µmol/L)"] = mask.sum()

        # 7. Moderate freshwater influence
        if "salinity" in df.columns:
            mask = (
                (df["salinity"] >= self.danger_thresholds["salinity_low"]) &
                (df["salinity"] < self.caution_thresholds["salinity_low"]) &
                df["salinity"].notna()
            )
            caution_conditions.append(mask)
            self.caution_triggers["Freshwater Influence (Salinity 25-30)"] = mask.sum()

        # 8. High beam attenuation (turbidity)
        if "beam_attenuation" in df.columns:
            mask = (df["beam_attenuation"] > self.caution_thresholds["beam_attenuation"]) & df["beam_attenuation"].notna()
            caution_conditions.append(mask)
            self.caution_triggers["High Turbidity (Beam Attenuation > 40)"] = mask.sum()

        # Combine all caution conditions (OR logic)
        if caution_conditions:
            caution_mask = pd.concat(caution_conditions, axis=1).any(axis=1)
        else:
            caution_mask = pd.Series([False] * len(df), index=df.index)

        return caution_mask

    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of safety labels."""
        return self.label_stats.copy()

    def get_label_distribution_pct(self) -> Dict[str, float]:
        """Get distribution of safety labels as percentages."""
        total = sum(self.label_stats.values())
        if total == 0:
            return {k: 0.0 for k in self.label_stats.keys()}

        return {
            k: (v / total) * 100
            for k, v in self.label_stats.items()
        }

    def get_trigger_summary(self) -> str:
        """Get summary of what triggered each classification."""
        lines = []

        if self.danger_triggers:
            lines.append("\nDANGER Triggers:")
            for trigger, count in sorted(self.danger_triggers.items(), key=lambda x: -x[1]):
                if count > 0:
                    lines.append(f"  {trigger}: {count:,}")

        if self.caution_triggers:
            lines.append("\nCAUTION Triggers:")
            for trigger, count in sorted(self.caution_triggers.items(), key=lambda x: -x[1]):
                if count > 0:
                    lines.append(f"  {trigger}: {count:,}")

        return "\n".join(lines)


def main():
    """Example usage."""
    import sys
    sys.path.insert(0, '..')

    from data_collection.mlml_data_loader import MLMLDataLoader

    # Load MLML data
    loader = MLMLDataLoader(data_dir="../../data/real")
    df = loader.load()

    # Resample to hourly
    hourly = loader.resample_to_hourly(df)

    print(f"\nClassifying {len(hourly)} hourly records...")

    # Classify
    classifier = SafetyClassifierMLML()
    labeled_df = classifier.classify(hourly)

    print("\n" + "=" * 60)
    print("Label Distribution:")
    print(f"  SAFE:    {classifier.label_stats['SAFE']:,} ({classifier.get_label_distribution_pct()['SAFE']:.1f}%)")
    print(f"  CAUTION: {classifier.label_stats['CAUTION']:,} ({classifier.get_label_distribution_pct()['CAUTION']:.1f}%)")
    print(f"  DANGER:  {classifier.label_stats['DANGER']:,} ({classifier.get_label_distribution_pct()['DANGER']:.1f}%)")

    print(classifier.get_trigger_summary())


if __name__ == "__main__":
    main()
