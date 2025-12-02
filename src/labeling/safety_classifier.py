"""
Safety Classification System
============================
Generates SAFE/CAUTION/DANGER labels for water quality conditions.

Classification based on EPA, Florida DEP, and WHO standards for
recreational water safety.

Priority System:
1. DANGER (2) - Immediate health hazards
   - Harmful algal blooms (HABs)
   - Severe hypoxia
   - Extreme pH
   - Extreme turbidity

2. CAUTION (1) - Elevated risk conditions
   - Moderate algae levels
   - Moderate hypoxia
   - pH outside normal range
   - Elevated turbidity
   - High nitrate
   - Post-rain turbidity spike

3. SAFE (0) - Normal conditions
   - All parameters within acceptable ranges
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class SafetyClassifier:
    """Classifies water conditions into SAFE/CAUTION/DANGER categories."""

    # Classification constants
    SAFE = 0
    CAUTION = 1
    DANGER = 2

    def __init__(self):
        """Initialize safety classifier with threshold values."""

        # DANGER thresholds (Priority 1 - Immediate hazards)
        self.danger_thresholds = {
            "phycocyanin": 20.0,          # µg/L - Harmful algal bloom
            "dissolved_oxygen": 2.0,       # mg/L - Severe hypoxia (below)
            "pH_low": 6.0,                 # Extreme acidic
            "pH_high": 9.0,                # Extreme alkaline
            "turbidity": 150.0,            # NTU - Severe turbidity
        }

        # CAUTION thresholds (Priority 2 - Elevated risk)
        self.caution_thresholds = {
            "phycocyanin": 10.0,           # µg/L - Moderate algae
            "chlorophyll": 20.0,           # µg/L - High algae (non-toxic)
            "dissolved_oxygen": 5.0,       # mg/L - Moderate hypoxia (below)
            "pH_low": 6.5,                 # Outside normal range
            "pH_high": 8.5,                # Outside normal range
            "turbidity": 50.0,             # NTU - Moderate turbidity
            "nitrate": 10.0,               # mg/L - Elevated nitrate
            "rain_24h": 25.0,              # mm - Significant rain event
            "turbidity_after_rain": 50.0,  # NTU - Post-rain spike
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

        # Calculate 24h cumulative rain if not present
        if "rain_cumulative_24h" not in df.columns and "rain_accumulation" in df.columns:
            df["rain_cumulative_24h"] = df["rain_accumulation"].rolling(
                window=24,
                min_periods=1
            ).sum()

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
        """
        Create boolean mask for DANGER conditions.

        Any ONE of these conditions triggers DANGER.
        """
        danger_conditions = []

        # 1. Harmful Algal Bloom (HAB)
        if "phycocyanin" in df.columns:
            danger_conditions.append(
                df["phycocyanin"] > self.danger_thresholds["phycocyanin"]
            )

        # 2. Severe Hypoxia
        if "dissolved_oxygen" in df.columns:
            danger_conditions.append(
                df["dissolved_oxygen"] < self.danger_thresholds["dissolved_oxygen"]
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
        """
        Create boolean mask for CAUTION conditions.

        Any ONE of these conditions triggers CAUTION.
        """
        caution_conditions = []

        # 1. Moderate Algae (Phycocyanin)
        if "phycocyanin" in df.columns:
            caution_conditions.append(
                (df["phycocyanin"] > self.caution_thresholds["phycocyanin"]) &
                (df["phycocyanin"] <= self.danger_thresholds["phycocyanin"])
            )

        # 2. High Chlorophyll
        if "chlorophyll" in df.columns:
            caution_conditions.append(
                df["chlorophyll"] > self.caution_thresholds["chlorophyll"]
            )

        # 3. Moderate Hypoxia
        if "dissolved_oxygen" in df.columns:
            caution_conditions.append(
                (df["dissolved_oxygen"] >= self.danger_thresholds["dissolved_oxygen"]) &
                (df["dissolved_oxygen"] < self.caution_thresholds["dissolved_oxygen"])
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

        # Combine all caution conditions (OR logic)
        if caution_conditions:
            caution_mask = pd.concat(caution_conditions, axis=1).any(axis=1)
        else:
            caution_mask = pd.Series([False] * len(df), index=df.index)

        return caution_mask

    def get_label_distribution(self) -> Dict[str, int]:
        """
        Get distribution of safety labels.

        Returns:
            Dictionary with counts of each label
        """
        return self.label_stats.copy()

    def get_label_distribution_pct(self) -> Dict[str, float]:
        """
        Get distribution of safety labels as percentages.

        Returns:
            Dictionary with percentages of each label
        """
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
        """
        Explain why a specific row was classified as it was.

        Args:
            row: Single row from DataFrame
            label: The assigned label (0, 1, or 2)

        Returns:
            Human-readable explanation
        """
        reasons = []

        if label == self.DANGER:
            if "phycocyanin" in row and row["phycocyanin"] > self.danger_thresholds["phycocyanin"]:
                reasons.append(f"Harmful algal bloom detected (phycocyanin: {row['phycocyanin']:.1f} µg/L)")

            if "dissolved_oxygen" in row and row["dissolved_oxygen"] < self.danger_thresholds["dissolved_oxygen"]:
                reasons.append(f"Severe hypoxia (DO: {row['dissolved_oxygen']:.1f} mg/L)")

            if "pH" in row:
                if row["pH"] < self.danger_thresholds["pH_low"]:
                    reasons.append(f"Extreme acidic pH ({row['pH']:.2f})")
                elif row["pH"] > self.danger_thresholds["pH_high"]:
                    reasons.append(f"Extreme alkaline pH ({row['pH']:.2f})")

            if "turbidity" in row and row["turbidity"] > self.danger_thresholds["turbidity"]:
                reasons.append(f"Severe turbidity ({row['turbidity']:.1f} NTU)")

        elif label == self.CAUTION:
            if "phycocyanin" in row and self.caution_thresholds["phycocyanin"] < row["phycocyanin"] <= self.danger_thresholds["phycocyanin"]:
                reasons.append(f"Moderate algae levels (phycocyanin: {row['phycocyanin']:.1f} µg/L)")

            if "chlorophyll" in row and row["chlorophyll"] > self.caution_thresholds["chlorophyll"]:
                reasons.append(f"High chlorophyll ({row['chlorophyll']:.1f} µg/L)")

            if "dissolved_oxygen" in row and self.danger_thresholds["dissolved_oxygen"] <= row["dissolved_oxygen"] < self.caution_thresholds["dissolved_oxygen"]:
                reasons.append(f"Moderate hypoxia (DO: {row['dissolved_oxygen']:.1f} mg/L)")

            if "pH" in row:
                if self.danger_thresholds["pH_low"] <= row["pH"] < self.caution_thresholds["pH_low"]:
                    reasons.append(f"Low pH ({row['pH']:.2f})")
                elif self.caution_thresholds["pH_high"] < row["pH"] <= self.danger_thresholds["pH_high"]:
                    reasons.append(f"High pH ({row['pH']:.2f})")

            if "turbidity" in row and self.caution_thresholds["turbidity"] < row["turbidity"] <= self.danger_thresholds["turbidity"]:
                reasons.append(f"Elevated turbidity ({row['turbidity']:.1f} NTU)")

            if "nitrate" in row and row["nitrate"] > self.caution_thresholds["nitrate"]:
                reasons.append(f"High nitrate ({row['nitrate']:.1f} mg/L)")

        else:  # SAFE
            reasons.append("All parameters within safe ranges")

        return "; ".join(reasons) if reasons else "No specific concerns"


def main():
    """Example usage."""
    print("Example: Safety Classification")
    print("=" * 50)

    # Load featured data
    df = pd.read_csv("../../data/processed/featured_dataset.csv")
    df["time"] = pd.to_datetime(df["time"])

    print(f"Input shape: {df.shape}")

    # Classify
    classifier = SafetyClassifier()
    labeled_df = classifier.classify(df)

    print("\n" + "=" * 50)
    print("Label Distribution:")
    print(f"  SAFE:    {classifier.label_stats['SAFE']:5d} ({classifier.get_label_distribution_pct()['SAFE']:.1f}%)")
    print(f"  CAUTION: {classifier.label_stats['CAUTION']:5d} ({classifier.get_label_distribution_pct()['CAUTION']:.1f}%)")
    print(f"  DANGER:  {classifier.label_stats['DANGER']:5d} ({classifier.get_label_distribution_pct()['DANGER']:.1f}%)")

    # Show examples of each category
    print("\n" + "=" * 50)
    print("Example DANGER case:")
    danger_sample = labeled_df[labeled_df["safety_label"] == SafetyClassifier.DANGER].iloc[0]
    print(classifier.explain_classification(danger_sample, SafetyClassifier.DANGER))

    if (labeled_df["safety_label"] == SafetyClassifier.CAUTION).any():
        print("\nExample CAUTION case:")
        caution_sample = labeled_df[labeled_df["safety_label"] == SafetyClassifier.CAUTION].iloc[0]
        print(classifier.explain_classification(caution_sample, SafetyClassifier.CAUTION))

    print("\nExample SAFE case:")
    safe_sample = labeled_df[labeled_df["safety_label"] == SafetyClassifier.SAFE].iloc[0]
    print(classifier.explain_classification(safe_sample, SafetyClassifier.SAFE))

    # Save labeled data
    output_path = "../../data/labels/safety_labels.csv"
    labeled_df[["time", "safety_label"]].to_csv(output_path, index=False)
    print(f"\nSafety labels saved to {output_path}")

    # Save full labeled dataset
    full_output_path = "../../data/processed/labeled_dataset.csv"
    labeled_df.to_csv(full_output_path, index=False)
    print(f"Full labeled dataset saved to {full_output_path}")


if __name__ == "__main__":
    main()
