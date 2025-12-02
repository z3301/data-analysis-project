"""
Safety Classification System for Real Data
==========================================
Generates SAFE/CAUTION/DANGER labels for real Wahoo Bay water quality data.

Key differences from synthetic data version:
- Uses Phycoerythrin (RFU) instead of Phycocyanin
- Uses Dissolved Oxygen % saturation instead of mg/L
- Uses Chlorophyll (RFU) instead of µg/L
- Thresholds adjusted based on actual data distributions

Classification based on EPA, Florida DEP standards adapted for sensor units.

Priority System:
1. DANGER (2) - Immediate health hazards
2. CAUTION (1) - Elevated risk conditions
3. SAFE (0) - Normal conditions
"""

import pandas as pd
import numpy as np
from typing import Dict


class SafetyClassifierReal:
    """Classifies water conditions into SAFE/CAUTION/DANGER for real sensor data."""

    # Classification constants
    SAFE = 0
    CAUTION = 1
    DANGER = 2

    def __init__(self):
        """Initialize safety classifier with threshold values for real data units."""

        # DANGER thresholds (Priority 1 - Immediate hazards)
        # Adjusted for real sensor units
        self.danger_thresholds = {
            # Phycoerythrin (RFU) - high values indicate harmful algal bloom
            # Based on observed data range (0-150 RFU typically)
            "phycoerythrin_rfu": 100.0,  # Very high algae fluorescence

            # Dissolved Oxygen % saturation
            # <30% is severe hypoxia (equivalent to ~2 mg/L)
            "dissolved_oxygen_pct": 30.0,  # Below this is DANGER

            # pH - same as before
            "pH_low": 6.0,
            "pH_high": 9.0,

            # Turbidity (FNU) - same scale as NTU
            "turbidity": 150.0,
        }

        # CAUTION thresholds (Priority 2 - Elevated risk)
        self.caution_thresholds = {
            # Phycoerythrin moderate levels
            "phycoerythrin_rfu": 50.0,

            # Chlorophyll (RFU) - high fluorescence
            "chlorophyll_rfu": 20.0,

            # Dissolved Oxygen % - moderate hypoxia
            # 30-70% is concerning (equivalent to 2-5 mg/L)
            "dissolved_oxygen_pct": 70.0,

            # pH outside normal range
            "pH_low": 6.5,
            "pH_high": 8.5,

            # Turbidity moderate
            "turbidity": 50.0,

            # Nitrate (mg/L) - same as before
            "nitrate": 10.0,

            # Rain in last 24 hours threshold
            "rain_24h": 25.0,

            # Turbidity after rain
            "turbidity_after_rain": 30.0,

            # Specific conductance (salinity proxy) - sudden drops indicate freshwater influx
            "conductance_low": 30.0,  # mS/cm - low salinity concern
        }

        self.label_stats = {
            "SAFE": 0,
            "CAUTION": 0,
            "DANGER": 0
        }

    def classify(
        self,
        df: pd.DataFrame,
        label_col: str = "safety_label"
    ) -> pd.DataFrame:
        """
        Classify each record as SAFE/CAUTION/DANGER.

        Args:
            df: DataFrame with water quality parameters
            label_col: Name for the label column

        Returns:
            DataFrame with safety_label column added
        """
        df = df.copy()

        # Initialize all as SAFE
        df[label_col] = self.SAFE

        # Calculate 24h cumulative rain if rain_accumulation exists
        if "rain_accumulation" in df.columns:
            df["rain_cumulative_24h"] = df["rain_accumulation"].rolling(
                window=24,
                min_periods=1
            ).sum()
        elif "rain_cumulative_24h" not in df.columns:
            df["rain_cumulative_24h"] = 0

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

        # 1. High Phycoerythrin (Algal Bloom proxy)
        if "phycoerythrin_rfu" in df.columns:
            danger_conditions.append(
                df["phycoerythrin_rfu"] > self.danger_thresholds["phycoerythrin_rfu"]
            )

        # 2. Severe Hypoxia (low DO %)
        if "dissolved_oxygen_pct" in df.columns:
            danger_conditions.append(
                df["dissolved_oxygen_pct"] < self.danger_thresholds["dissolved_oxygen_pct"]
            )

        # 3. Extreme pH
        if "pH" in df.columns:
            danger_conditions.append(
                (df["pH"] < self.danger_thresholds["pH_low"]) |
                (df["pH"] > self.danger_thresholds["pH_high"])
            )

        # 4. Severe Turbidity
        if "turbidity" in df.columns:
            danger_conditions.append(
                df["turbidity"] > self.danger_thresholds["turbidity"]
            )

        # Combine all danger conditions (OR logic)
        if danger_conditions:
            danger_mask = pd.concat(danger_conditions, axis=1).any(axis=1)
        else:
            danger_mask = pd.Series([False] * len(df), index=df.index)

        return danger_mask

    def _get_caution_mask(self, df: pd.DataFrame) -> pd.Series:
        """Create boolean mask for CAUTION conditions."""
        caution_conditions = []

        # 1. Moderate Phycoerythrin
        if "phycoerythrin_rfu" in df.columns:
            caution_conditions.append(
                (df["phycoerythrin_rfu"] > self.caution_thresholds["phycoerythrin_rfu"]) &
                (df["phycoerythrin_rfu"] <= self.danger_thresholds["phycoerythrin_rfu"])
            )

        # 2. High Chlorophyll
        if "chlorophyll_rfu" in df.columns:
            caution_conditions.append(
                df["chlorophyll_rfu"] > self.caution_thresholds["chlorophyll_rfu"]
            )

        # 3. Moderate Hypoxia
        if "dissolved_oxygen_pct" in df.columns:
            caution_conditions.append(
                (df["dissolved_oxygen_pct"] >= self.danger_thresholds["dissolved_oxygen_pct"]) &
                (df["dissolved_oxygen_pct"] < self.caution_thresholds["dissolved_oxygen_pct"])
            )

        # 4. pH Outside Normal Range
        if "pH" in df.columns:
            caution_conditions.append(
                ((df["pH"] < self.caution_thresholds["pH_low"]) &
                 (df["pH"] >= self.danger_thresholds["pH_low"])) |
                ((df["pH"] > self.caution_thresholds["pH_high"]) &
                 (df["pH"] <= self.danger_thresholds["pH_high"]))
            )

        # 5. Moderate Turbidity
        if "turbidity" in df.columns:
            caution_conditions.append(
                (df["turbidity"] > self.caution_thresholds["turbidity"]) &
                (df["turbidity"] <= self.danger_thresholds["turbidity"])
            )

        # 6. High Nitrate
        if "nitrate" in df.columns:
            caution_conditions.append(
                df["nitrate"] > self.caution_thresholds["nitrate"]
            )

        # 7. Post-Rain Turbidity Spike
        if "rain_cumulative_24h" in df.columns and "turbidity" in df.columns:
            caution_conditions.append(
                (df["rain_cumulative_24h"] > self.caution_thresholds["rain_24h"]) &
                (df["turbidity"] > self.caution_thresholds["turbidity_after_rain"])
            )

        # 8. Low Salinity (freshwater influx)
        if "specific_conductance" in df.columns:
            caution_conditions.append(
                df["specific_conductance"] < self.caution_thresholds["conductance_low"]
            )

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

    def explain_classification(
        self,
        row: pd.Series,
        label: int
    ) -> str:
        """Explain why a specific row was classified."""
        reasons = []

        if label == self.DANGER:
            if "phycoerythrin_rfu" in row.index and row["phycoerythrin_rfu"] > self.danger_thresholds["phycoerythrin_rfu"]:
                reasons.append(f"High algal fluorescence (Phycoerythrin: {row['phycoerythrin_rfu']:.1f} RFU)")

            if "dissolved_oxygen_pct" in row.index and row["dissolved_oxygen_pct"] < self.danger_thresholds["dissolved_oxygen_pct"]:
                reasons.append(f"Severe hypoxia (DO: {row['dissolved_oxygen_pct']:.1f}%)")

            if "pH" in row.index:
                if row["pH"] < self.danger_thresholds["pH_low"]:
                    reasons.append(f"Extreme acidic pH ({row['pH']:.2f})")
                elif row["pH"] > self.danger_thresholds["pH_high"]:
                    reasons.append(f"Extreme alkaline pH ({row['pH']:.2f})")

            if "turbidity" in row.index and row["turbidity"] > self.danger_thresholds["turbidity"]:
                reasons.append(f"Severe turbidity ({row['turbidity']:.1f} FNU)")

        elif label == self.CAUTION:
            if "phycoerythrin_rfu" in row.index:
                if self.caution_thresholds["phycoerythrin_rfu"] < row["phycoerythrin_rfu"] <= self.danger_thresholds["phycoerythrin_rfu"]:
                    reasons.append(f"Moderate algal fluorescence ({row['phycoerythrin_rfu']:.1f} RFU)")

            if "chlorophyll_rfu" in row.index and row["chlorophyll_rfu"] > self.caution_thresholds["chlorophyll_rfu"]:
                reasons.append(f"High chlorophyll ({row['chlorophyll_rfu']:.1f} RFU)")

            if "dissolved_oxygen_pct" in row.index:
                if self.danger_thresholds["dissolved_oxygen_pct"] <= row["dissolved_oxygen_pct"] < self.caution_thresholds["dissolved_oxygen_pct"]:
                    reasons.append(f"Moderate hypoxia (DO: {row['dissolved_oxygen_pct']:.1f}%)")

            if "turbidity" in row.index:
                if self.caution_thresholds["turbidity"] < row["turbidity"] <= self.danger_thresholds["turbidity"]:
                    reasons.append(f"Elevated turbidity ({row['turbidity']:.1f} FNU)")

            if "nitrate" in row.index and row["nitrate"] > self.caution_thresholds["nitrate"]:
                reasons.append(f"High nitrate ({row['nitrate']:.1f} mg/L)")

        else:  # SAFE
            reasons.append("All parameters within safe ranges")

        return "; ".join(reasons) if reasons else "No specific concerns"


def main():
    """Example usage."""
    import sys
    sys.path.insert(0, '..')

    from data_collection.real_data_loader import RealDataLoader

    # Load real data
    loader = RealDataLoader(data_dir="../../data/real")
    water_quality, _, _ = loader.load_all()

    # Resample to hourly
    water_hourly = loader.resample_to_hourly(water_quality)

    print(f"\nClassifying {len(water_hourly)} hourly records...")

    # Classify
    classifier = SafetyClassifierReal()
    labeled_df = classifier.classify(water_hourly)

    print("\n" + "=" * 60)
    print("Label Distribution:")
    print(f"  SAFE:    {classifier.label_stats['SAFE']:5,} ({classifier.get_label_distribution_pct()['SAFE']:.1f}%)")
    print(f"  CAUTION: {classifier.label_stats['CAUTION']:5,} ({classifier.get_label_distribution_pct()['CAUTION']:.1f}%)")
    print(f"  DANGER:  {classifier.label_stats['DANGER']:5,} ({classifier.get_label_distribution_pct()['DANGER']:.1f}%)")

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
