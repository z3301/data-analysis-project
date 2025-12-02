"""
Safety Classification System for Newport Pier Data
===================================================
Generates SAFE/CAUTION/DANGER labels for Newport Pier oceanographic data.

Location: Newport Beach, California (33.6°N, 117.9°W)
Data Source: NOAA/SCCOOS Newport Pier Automated Shore Station

Available Parameters:
- Water Temperature (°C)
- pH (total scale)
- Dissolved Oxygen % saturation
- Dissolved Oxygen mg/L
- Chlorophyll (µg/L)
- Salinity (PSU)
- Conductivity (S/m)

NOT Available (compared to Wahoo Bay):
- Turbidity
- Nitrate
- Phycoerythrin/Phycocyanin (specific algae indicators)

Classification based on:
- California Ocean Plan water quality objectives
- EPA aquatic life criteria
- NOAA coastal water quality standards

Priority System:
1. DANGER (2) - Immediate health/ecological hazards
2. CAUTION (1) - Elevated risk conditions
3. SAFE (0) - Normal conditions
"""

import pandas as pd
import numpy as np
from typing import Dict


class SafetyClassifierNewport:
    """Classifies water conditions into SAFE/CAUTION/DANGER for Newport Pier data."""

    # Classification constants
    SAFE = 0
    CAUTION = 1
    DANGER = 2

    def __init__(self):
        """Initialize safety classifier with California coastal water thresholds."""

        # DANGER thresholds (Priority 1 - Immediate hazards)
        self.danger_thresholds = {
            # Dissolved Oxygen - Severe hypoxia
            # < 2 mg/L is severe hypoxia (fish kills, dead zones)
            # < 30% saturation equivalent
            "dissolved_oxygen_mgl": 2.0,  # Below this is DANGER
            "dissolved_oxygen_pct": 30.0,  # Below this is DANGER

            # pH extremes - California Ocean Plan
            "pH_low": 6.5,   # Marine life stressed below this
            "pH_high": 9.0,  # Above this indicates algal bloom/eutrophication

            # Temperature extremes for California coastal waters
            # Based on local species tolerance
            "water_temp_low": 8.0,   # Hypothermic stress
            "water_temp_high": 28.0,  # Hyperthermic stress

            # Very low salinity (major freshwater intrusion)
            "salinity_low": 15.0,  # PSU - severe freshwater event
        }

        # CAUTION thresholds (Priority 2 - Elevated risk)
        self.caution_thresholds = {
            # Dissolved Oxygen - Moderate hypoxia
            # 2-4 mg/L or 30-50% is stressful for marine life
            "dissolved_oxygen_mgl": 4.0,  # Below this is CAUTION
            "dissolved_oxygen_pct": 50.0,  # Below this is CAUTION

            # pH outside optimal range
            "pH_low": 7.5,
            "pH_high": 8.3,

            # Temperature outside comfort zone
            "water_temp_low": 10.0,
            "water_temp_high": 24.0,

            # High chlorophyll (potential bloom indicator)
            # California coastal waters typically < 10 µg/L
            "chlorophyll": 20.0,  # µg/L - elevated algae

            # Very high chlorophyll (probable bloom)
            "chlorophyll_high": 40.0,  # µg/L - algal bloom likely

            # Moderate salinity drop (rain event/runoff)
            "salinity_low": 28.0,  # PSU - notable freshwater influence
        }

        self.label_stats = {
            "SAFE": 0,
            "CAUTION": 0,
            "DANGER": 0
        }

        # Track trigger counts
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
            danger_conditions.append(mask)
            self.danger_triggers["Severe Hypoxia (DO < 2 mg/L)"] = mask.sum()

        # 2. Severe Hypoxia (low DO %)
        if "dissolved_oxygen_pct" in df.columns:
            mask = df["dissolved_oxygen_pct"] < self.danger_thresholds["dissolved_oxygen_pct"]
            danger_conditions.append(mask)
            self.danger_triggers["Severe Hypoxia (DO < 30%)"] = mask.sum()

        # 3. Extreme pH
        if "pH" in df.columns:
            low_mask = df["pH"] < self.danger_thresholds["pH_low"]
            high_mask = df["pH"] > self.danger_thresholds["pH_high"]
            mask = low_mask | high_mask
            danger_conditions.append(mask)
            self.danger_triggers["Extreme pH (< 6.5 or > 9.0)"] = mask.sum()

        # 4. Temperature extremes
        if "water_temp" in df.columns:
            low_mask = df["water_temp"] < self.danger_thresholds["water_temp_low"]
            high_mask = df["water_temp"] > self.danger_thresholds["water_temp_high"]
            mask = low_mask | high_mask
            danger_conditions.append(mask)
            self.danger_triggers["Extreme Temperature"] = mask.sum()

        # 5. Severe freshwater intrusion
        if "salinity" in df.columns:
            mask = df["salinity"] < self.danger_thresholds["salinity_low"]
            danger_conditions.append(mask)
            self.danger_triggers["Severe Freshwater Event (Salinity < 15)"] = mask.sum()

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
                (df["dissolved_oxygen_mgl"] < self.caution_thresholds["dissolved_oxygen_mgl"])
            )
            caution_conditions.append(mask)
            self.caution_triggers["Moderate Hypoxia (DO 2-4 mg/L)"] = mask.sum()

        # 2. Moderate Hypoxia (DO %)
        if "dissolved_oxygen_pct" in df.columns:
            mask = (
                (df["dissolved_oxygen_pct"] >= self.danger_thresholds["dissolved_oxygen_pct"]) &
                (df["dissolved_oxygen_pct"] < self.caution_thresholds["dissolved_oxygen_pct"])
            )
            caution_conditions.append(mask)
            self.caution_triggers["Moderate Hypoxia (DO 30-50%)"] = mask.sum()

        # 3. pH outside optimal range
        if "pH" in df.columns:
            low_mask = (
                (df["pH"] >= self.danger_thresholds["pH_low"]) &
                (df["pH"] < self.caution_thresholds["pH_low"])
            )
            high_mask = (
                (df["pH"] > self.caution_thresholds["pH_high"]) &
                (df["pH"] <= self.danger_thresholds["pH_high"])
            )
            mask = low_mask | high_mask
            caution_conditions.append(mask)
            self.caution_triggers["pH Outside Optimal (6.5-7.5 or 8.3-9.0)"] = mask.sum()

        # 4. Temperature outside comfort zone
        if "water_temp" in df.columns:
            low_mask = (
                (df["water_temp"] >= self.danger_thresholds["water_temp_low"]) &
                (df["water_temp"] < self.caution_thresholds["water_temp_low"])
            )
            high_mask = (
                (df["water_temp"] > self.caution_thresholds["water_temp_high"]) &
                (df["water_temp"] <= self.danger_thresholds["water_temp_high"])
            )
            mask = low_mask | high_mask
            caution_conditions.append(mask)
            self.caution_triggers["Temperature Stress"] = mask.sum()

        # 5. Elevated chlorophyll (algae)
        if "chlorophyll" in df.columns:
            mask = df["chlorophyll"] > self.caution_thresholds["chlorophyll"]
            caution_conditions.append(mask)
            self.caution_triggers["Elevated Chlorophyll (> 20 µg/L)"] = mask.sum()

        # 6. High chlorophyll (probable bloom)
        if "chlorophyll" in df.columns:
            mask = df["chlorophyll"] > self.caution_thresholds["chlorophyll_high"]
            caution_conditions.append(mask)
            self.caution_triggers["Algal Bloom Likely (Chlorophyll > 40 µg/L)"] = mask.sum()

        # 7. Moderate freshwater influence
        if "salinity" in df.columns:
            mask = (
                (df["salinity"] >= self.danger_thresholds["salinity_low"]) &
                (df["salinity"] < self.caution_thresholds["salinity_low"])
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

    def explain_classification(
        self,
        row: pd.Series,
        label: int
    ) -> str:
        """Explain why a specific row was classified."""
        reasons = []

        if label == self.DANGER:
            if "dissolved_oxygen_mgl" in row.index and row["dissolved_oxygen_mgl"] < self.danger_thresholds["dissolved_oxygen_mgl"]:
                reasons.append(f"Severe hypoxia (DO: {row['dissolved_oxygen_mgl']:.1f} mg/L)")

            if "dissolved_oxygen_pct" in row.index and row["dissolved_oxygen_pct"] < self.danger_thresholds["dissolved_oxygen_pct"]:
                reasons.append(f"Severe hypoxia (DO: {row['dissolved_oxygen_pct']:.1f}%)")

            if "pH" in row.index:
                if row["pH"] < self.danger_thresholds["pH_low"]:
                    reasons.append(f"Extreme acidic pH ({row['pH']:.2f})")
                elif row["pH"] > self.danger_thresholds["pH_high"]:
                    reasons.append(f"Extreme alkaline pH ({row['pH']:.2f})")

            if "water_temp" in row.index:
                if row["water_temp"] < self.danger_thresholds["water_temp_low"]:
                    reasons.append(f"Hypothermic temperature ({row['water_temp']:.1f}°C)")
                elif row["water_temp"] > self.danger_thresholds["water_temp_high"]:
                    reasons.append(f"Hyperthermic temperature ({row['water_temp']:.1f}°C)")

            if "salinity" in row.index and row["salinity"] < self.danger_thresholds["salinity_low"]:
                reasons.append(f"Severe freshwater intrusion (Salinity: {row['salinity']:.1f} PSU)")

        elif label == self.CAUTION:
            if "dissolved_oxygen_mgl" in row.index:
                if self.danger_thresholds["dissolved_oxygen_mgl"] <= row["dissolved_oxygen_mgl"] < self.caution_thresholds["dissolved_oxygen_mgl"]:
                    reasons.append(f"Moderate hypoxia (DO: {row['dissolved_oxygen_mgl']:.1f} mg/L)")

            if "chlorophyll" in row.index and row["chlorophyll"] > self.caution_thresholds["chlorophyll"]:
                reasons.append(f"Elevated chlorophyll ({row['chlorophyll']:.1f} µg/L)")

            if "pH" in row.index:
                if self.danger_thresholds["pH_low"] <= row["pH"] < self.caution_thresholds["pH_low"]:
                    reasons.append(f"Low pH ({row['pH']:.2f})")
                elif self.caution_thresholds["pH_high"] < row["pH"] <= self.danger_thresholds["pH_high"]:
                    reasons.append(f"Elevated pH ({row['pH']:.2f})")

            if "salinity" in row.index:
                if self.danger_thresholds["salinity_low"] <= row["salinity"] < self.caution_thresholds["salinity_low"]:
                    reasons.append(f"Freshwater influence (Salinity: {row['salinity']:.1f} PSU)")

        else:  # SAFE
            reasons.append("All parameters within safe ranges")

        return "; ".join(reasons) if reasons else "No specific concerns"


def main():
    """Example usage."""
    import sys
    sys.path.insert(0, '..')

    from data_collection.newport_data_loader import NewportDataLoader

    # Load Newport Pier data
    loader = NewportDataLoader(data_dir="../../data/real")
    df = loader.load()

    # Resample to hourly
    hourly = loader.resample_to_hourly(df)

    print(f"\nClassifying {len(hourly)} hourly records...")

    # Classify
    classifier = SafetyClassifierNewport()
    labeled_df = classifier.classify(hourly)

    print("\n" + "=" * 60)
    print("Label Distribution:")
    print(f"  SAFE:    {classifier.label_stats['SAFE']:5,} ({classifier.get_label_distribution_pct()['SAFE']:.1f}%)")
    print(f"  CAUTION: {classifier.label_stats['CAUTION']:5,} ({classifier.get_label_distribution_pct()['CAUTION']:.1f}%)")
    print(f"  DANGER:  {classifier.label_stats['DANGER']:5,} ({classifier.get_label_distribution_pct()['DANGER']:.1f}%)")

    print(classifier.get_trigger_summary())

    # Show example classifications
    if classifier.label_stats['DANGER'] > 0:
        print("\nExample DANGER case:")
        danger_sample = labeled_df[labeled_df["safety_label"] == 2].iloc[0]
        print(f"  {classifier.explain_classification(danger_sample, 2)}")

    if classifier.label_stats['CAUTION'] > 0:
        print("\nExample CAUTION case:")
        caution_sample = labeled_df[labeled_df["safety_label"] == 1].iloc[0]
        print(f"  {classifier.explain_classification(caution_sample, 1)}")


if __name__ == "__main__":
    main()
