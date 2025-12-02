"""
Safety Classification System for Scripps Pier Data
===================================================
Generates SAFE/CAUTION/DANGER labels for Scripps Pier oceanographic data.

Location: La Jolla, California (32.867°N, 117.257°W)
Data Source: SCCOOS Scripps Pier Automated Shore Station

Available Parameters:
- Water Temperature (°C) - CTD
- Salinity (PSU) - CTD
- Chlorophyll (µg/L) - CTD and ECO
- pH (total scale) - SeapHOx
- Dissolved Oxygen % saturation - SeapHOx
- Dissolved Oxygen mg/L - SeapHOx
- Turbidity (NTU) - ECO (2025 mainly)
- fDOM (QSU) - ECO (2025 mainly)

Classification based on:
- California Ocean Plan water quality objectives
- EPA aquatic life criteria
- NOAA coastal water quality standards
"""

import pandas as pd
import numpy as np
from typing import Dict


class SafetyClassifierScripps:
    """Classifies water conditions into SAFE/CAUTION/DANGER for Scripps Pier data."""

    # Classification constants
    SAFE = 0
    CAUTION = 1
    DANGER = 2

    def __init__(self):
        """Initialize safety classifier with California coastal water thresholds."""

        # DANGER thresholds (Priority 1 - Immediate hazards)
        self.danger_thresholds = {
            # Dissolved Oxygen - Severe hypoxia
            "dissolved_oxygen_mgl": 2.0,  # Below this is DANGER
            "dissolved_oxygen_pct": 30.0,  # Below this is DANGER

            # pH extremes
            "pH_low": 6.5,
            "pH_high": 9.0,

            # Temperature extremes for California coastal waters
            "water_temp_low": 8.0,
            "water_temp_high": 28.0,

            # Very low salinity (major freshwater intrusion)
            "salinity_low": 15.0,

            # High turbidity (visibility hazard, indicates runoff)
            "turbidity": 50.0,  # NTU
        }

        # CAUTION thresholds (Priority 2 - Elevated risk)
        self.caution_thresholds = {
            # Dissolved Oxygen - Moderate hypoxia
            "dissolved_oxygen_mgl": 4.0,
            "dissolved_oxygen_pct": 50.0,

            # pH outside optimal range
            "pH_low": 7.5,
            "pH_high": 8.3,

            # Temperature outside comfort zone
            "water_temp_low": 10.0,
            "water_temp_high": 24.0,

            # High chlorophyll (potential bloom)
            "chlorophyll": 20.0,  # µg/L
            "chlorophyll_high": 40.0,  # Probable bloom

            # Moderate turbidity
            "turbidity": 20.0,  # NTU

            # High fDOM (indicates organic matter, potential pollution)
            "fdom": 50.0,  # QSU

            # Moderate salinity drop
            "salinity_low": 28.0,
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

        # 1. Severe Hypoxia (low DO mg/L)
        if "dissolved_oxygen_mgl" in df.columns:
            mask = df["dissolved_oxygen_mgl"] < self.danger_thresholds["dissolved_oxygen_mgl"]
            mask = mask & df["dissolved_oxygen_mgl"].notna()
            danger_conditions.append(mask)
            self.danger_triggers["Severe Hypoxia (DO < 2 mg/L)"] = mask.sum()

        # 2. Severe Hypoxia (low DO %)
        if "dissolved_oxygen_pct" in df.columns:
            mask = df["dissolved_oxygen_pct"] < self.danger_thresholds["dissolved_oxygen_pct"]
            mask = mask & df["dissolved_oxygen_pct"].notna()
            danger_conditions.append(mask)
            self.danger_triggers["Severe Hypoxia (DO < 30%)"] = mask.sum()

        # 3. Extreme pH
        if "pH" in df.columns:
            low_mask = (df["pH"] < self.danger_thresholds["pH_low"]) & df["pH"].notna()
            high_mask = (df["pH"] > self.danger_thresholds["pH_high"]) & df["pH"].notna()
            mask = low_mask | high_mask
            danger_conditions.append(mask)
            self.danger_triggers["Extreme pH (< 6.5 or > 9.0)"] = mask.sum()

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
            self.danger_triggers["Severe Freshwater Event (Salinity < 15)"] = mask.sum()

        # 6. High turbidity
        if "turbidity" in df.columns:
            mask = (df["turbidity"] > self.danger_thresholds["turbidity"]) & df["turbidity"].notna()
            danger_conditions.append(mask)
            self.danger_triggers["High Turbidity (> 50 NTU)"] = mask.sum()

        # Combine all danger conditions (OR logic)
        if danger_conditions:
            danger_mask = pd.concat(danger_conditions, axis=1).any(axis=1)
        else:
            danger_mask = pd.Series([False] * len(df), index=df.index)

        return danger_mask

    def _get_caution_mask(self, df: pd.DataFrame) -> pd.Series:
        """Create boolean mask for CAUTION conditions."""
        caution_conditions = []

        # 1. Moderate Hypoxia (DO mg/L)
        if "dissolved_oxygen_mgl" in df.columns:
            mask = (
                (df["dissolved_oxygen_mgl"] >= self.danger_thresholds["dissolved_oxygen_mgl"]) &
                (df["dissolved_oxygen_mgl"] < self.caution_thresholds["dissolved_oxygen_mgl"]) &
                df["dissolved_oxygen_mgl"].notna()
            )
            caution_conditions.append(mask)
            self.caution_triggers["Moderate Hypoxia (DO 2-4 mg/L)"] = mask.sum()

        # 2. Moderate Hypoxia (DO %)
        if "dissolved_oxygen_pct" in df.columns:
            mask = (
                (df["dissolved_oxygen_pct"] >= self.danger_thresholds["dissolved_oxygen_pct"]) &
                (df["dissolved_oxygen_pct"] < self.caution_thresholds["dissolved_oxygen_pct"]) &
                df["dissolved_oxygen_pct"].notna()
            )
            caution_conditions.append(mask)
            self.caution_triggers["Moderate Hypoxia (DO 30-50%)"] = mask.sum()

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
            self.caution_triggers["pH Outside Optimal (6.5-7.5 or 8.3-9.0)"] = mask.sum()

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

        # 5. Elevated chlorophyll (use both CTD and ECO)
        for chlor_col in ['chlorophyll', 'chlorophyll_eco']:
            if chlor_col in df.columns:
                mask = (df[chlor_col] > self.caution_thresholds["chlorophyll"]) & df[chlor_col].notna()
                caution_conditions.append(mask)
                self.caution_triggers[f"Elevated Chlorophyll ({chlor_col} > 20 µg/L)"] = mask.sum()

                # High chlorophyll (probable bloom)
                mask_high = (df[chlor_col] > self.caution_thresholds["chlorophyll_high"]) & df[chlor_col].notna()
                caution_conditions.append(mask_high)
                self.caution_triggers[f"Algal Bloom Likely ({chlor_col} > 40 µg/L)"] = mask_high.sum()

        # 6. Moderate turbidity
        if "turbidity" in df.columns:
            mask = (
                (df["turbidity"] > self.caution_thresholds["turbidity"]) &
                (df["turbidity"] <= self.danger_thresholds["turbidity"]) &
                df["turbidity"].notna()
            )
            caution_conditions.append(mask)
            self.caution_triggers["Moderate Turbidity (20-50 NTU)"] = mask.sum()

        # 7. High fDOM (organic matter)
        if "fdom" in df.columns:
            mask = (df["fdom"] > self.caution_thresholds["fdom"]) & df["fdom"].notna()
            caution_conditions.append(mask)
            self.caution_triggers["High fDOM (> 50 QSU)"] = mask.sum()

        # 8. Moderate freshwater influence
        if "salinity" in df.columns:
            mask = (
                (df["salinity"] >= self.danger_thresholds["salinity_low"]) &
                (df["salinity"] < self.caution_thresholds["salinity_low"]) &
                df["salinity"].notna()
            )
            caution_conditions.append(mask)
            self.caution_triggers["Freshwater Influence (Salinity 15-28)"] = mask.sum()

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

    from data_collection.scripps_data_loader import ScrippsDataLoader

    # Load Scripps Pier data
    loader = ScrippsDataLoader(data_dir="../../data/real")
    df = loader.load(years=[2023, 2024, 2025])

    # Resample to hourly
    hourly = loader.resample_to_hourly(df)

    print(f"\nClassifying {len(hourly)} hourly records...")

    # Classify
    classifier = SafetyClassifierScripps()
    labeled_df = classifier.classify(hourly)

    print("\n" + "=" * 60)
    print("Label Distribution:")
    print(f"  SAFE:    {classifier.label_stats['SAFE']:,} ({classifier.get_label_distribution_pct()['SAFE']:.1f}%)")
    print(f"  CAUTION: {classifier.label_stats['CAUTION']:,} ({classifier.get_label_distribution_pct()['CAUTION']:.1f}%)")
    print(f"  DANGER:  {classifier.label_stats['DANGER']:,} ({classifier.get_label_distribution_pct()['DANGER']:.1f}%)")

    print(classifier.get_trigger_summary())


if __name__ == "__main__":
    main()
