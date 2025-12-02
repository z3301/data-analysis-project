"""
Experiment 1: Environmental Drivers of Unsafe Water Conditions
===============================================================

Analytical Question:
Does variation in environmental and water-quality variables (e.g., turbidity,
rainfall, tides, salinity) significantly influence the likelihood of unsafe
water conditions at Wahoo Bay?

Techniques:
- Exploratory Data Analysis (EDA)
- Pearson correlation analysis
- Random Forest feature importance
- Time-series visualizations

Expected Outputs:
- Correlation heatmap
- Feature importance rankings
- Time-series plots showing parameter trends before unsafe events
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except (ImportError, ValueError):
    HAS_SEABORN = False
    print("Warning: seaborn not available, using matplotlib fallback")
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class Experiment1:
    """Correlation analysis and feature importance for unsafe water prediction."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize experiment with labeled dataset.

        Args:
            data: DataFrame with features and safety_label column
        """
        self.data = data.copy()
        self.correlation_matrix = None
        self.feature_importance = None
        self.results = {}

    def run_full_analysis(self) -> dict:
        """
        Run complete Experiment 1 analysis.

        Returns:
            Dictionary containing all results
        """
        print("=" * 70)
        print("EXPERIMENT 1: IDENTIFYING ENVIRONMENTAL DRIVERS")
        print("=" * 70)

        # 1. Descriptive Statistics
        print("\n[1/5] Computing descriptive statistics...")
        self.results["descriptive_stats"] = self._descriptive_statistics()

        # 2. Correlation Analysis
        print("\n[2/5] Running correlation analysis...")
        self.results["correlations"] = self._correlation_analysis()

        # 3. Feature Importance
        print("\n[3/5] Computing Random Forest feature importance...")
        self.results["feature_importance"] = self._feature_importance_analysis()

        # 4. Group Comparisons
        print("\n[4/5] Comparing SAFE vs UNSAFE conditions...")
        self.results["group_comparisons"] = self._compare_safe_unsafe()

        # 5. Time-Series Analysis
        print("\n[5/5] Analyzing temporal patterns...")
        self.results["temporal_patterns"] = self._temporal_analysis()

        print("\n" + "=" * 70)
        print("EXPERIMENT 1 COMPLETE")
        print("=" * 70)

        return self.results

    def _descriptive_statistics(self) -> pd.DataFrame:
        """Compute descriptive statistics for key parameters."""

        # Support both synthetic and real data parameter names
        key_params = [
            "turbidity", "pH", "dissolved_oxygen", "dissolved_oxygen_pct",
            "chlorophyll", "chlorophyll_rfu", "phycocyanin", "phycoerythrin_rfu",
            "nitrate", "water_temp", "rain_accumulation",
            "tide_height", "wind_speed_avg", "barometric_pressure",
            "specific_conductance"
        ]

        # Only include parameters that exist
        available_params = [p for p in key_params if p in self.data.columns]

        stats = self.data[available_params].describe().T
        stats["missing_pct"] = (self.data[available_params].isnull().sum() / len(self.data)) * 100

        print("\nDescriptive Statistics:")
        print(stats.round(2))

        return stats

    def _correlation_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute correlations between features and safety label.

        Returns:
            Tuple of (full_correlation_matrix, target_correlations)
        """
        # Select numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

        # Remove time-based columns
        exclude = ["hour", "day_of_week", "month", "day_of_year", "is_weekend", "is_wet_season"]
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        # Compute correlation matrix
        correlation_matrix = self.data[numeric_cols].corr()
        self.correlation_matrix = correlation_matrix

        # Extract correlations with safety_label
        if "safety_label" in correlation_matrix.columns:
            target_corr = correlation_matrix["safety_label"].sort_values(ascending=False)
            target_corr = target_corr[target_corr.index != "safety_label"]

            print("\nTop 10 Positive Correlations with Unsafe Conditions:")
            print(target_corr.head(10).round(3))

            print("\nTop 10 Negative Correlations with Unsafe Conditions:")
            print(target_corr.tail(10).round(3))

            return correlation_matrix, target_corr
        else:
            print("Warning: safety_label not found in data")
            return correlation_matrix, None

    def _feature_importance_analysis(self) -> pd.DataFrame:
        """
        Compute feature importance using Random Forest.

        Returns:
            DataFrame with feature importance scores
        """
        # Prepare data
        feature_cols = [c for c in self.data.columns
                       if c not in ["time", "safety_label"]
                       and self.data[c].dtype in [np.float64, np.int64]]

        X = self.data[feature_cols].fillna(0)  # Simple imputation for RF
        y = self.data["safety_label"]

        # Remove any rows with NaN in target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X, y)

        # Get feature importance
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)

        self.feature_importance = importance_df

        print("\nTop 15 Most Important Features (Random Forest):")
        print(importance_df.head(15).to_string(index=False))

        return importance_df

    def _compare_safe_unsafe(self) -> pd.DataFrame:
        """
        Compare mean values of parameters between SAFE and UNSAFE conditions.

        Returns:
            DataFrame with group comparisons
        """
        # Support both synthetic and real data parameter names
        key_params = [
            "turbidity", "pH", "dissolved_oxygen", "dissolved_oxygen_pct",
            "chlorophyll", "chlorophyll_rfu", "phycocyanin", "phycoerythrin_rfu",
            "nitrate", "water_temp", "rain_accumulation", "specific_conductance"
        ]

        available_params = [p for p in key_params if p in self.data.columns]

        # Group by safety label
        safe_data = self.data[self.data["safety_label"] == 0][available_params]
        caution_data = self.data[self.data["safety_label"] == 1][available_params]
        danger_data = self.data[self.data["safety_label"] == 2][available_params]

        comparison = pd.DataFrame({
            "SAFE_mean": safe_data.mean(),
            "CAUTION_mean": caution_data.mean(),
            "DANGER_mean": danger_data.mean(),
        })

        comparison["DANGER_vs_SAFE_ratio"] = comparison["DANGER_mean"] / comparison["SAFE_mean"]

        print("\nParameter Comparison Across Safety Levels:")
        print(comparison.round(2))

        return comparison

    def _temporal_analysis(self) -> dict:
        """
        Analyze temporal patterns in unsafe conditions.

        Returns:
            Dictionary with temporal statistics
        """
        temporal_stats = {}

        # Unsafe events by hour of day
        if "hour" in self.data.columns:
            hourly = self.data.groupby("hour")["safety_label"].apply(
                lambda x: (x > 0).sum() / len(x) * 100
            )
            temporal_stats["unsafe_by_hour"] = hourly

            print("\nUnsafe Condition Frequency by Hour of Day (%):")
            print(hourly.round(1))

        # Unsafe events by month
        if "month" in self.data.columns:
            monthly = self.data.groupby("month")["safety_label"].apply(
                lambda x: (x > 0).sum() / len(x) * 100
            )
            temporal_stats["unsafe_by_month"] = monthly

            print("\nUnsafe Condition Frequency by Month (%):")
            print(monthly.round(1))

        # Days after rain events
        if "rain_cumulative_24h" in self.data.columns:
            self.data["high_rain_24h"] = self.data["rain_cumulative_24h"] > 25

            rain_impact = self.data.groupby("high_rain_24h")["safety_label"].apply(
                lambda x: (x > 0).sum() / len(x) * 100
            )
            temporal_stats["rain_impact"] = rain_impact

            print("\nUnsafe Frequency After Significant Rain (>25mm in 24h):")
            print(rain_impact.round(1))

        return temporal_stats

    def plot_correlation_heatmap(
        self,
        figsize: Tuple[int, int] = (14, 12),
        top_n: int = 25
    ):
        """
        Create correlation heatmap visualization.

        Args:
            figsize: Figure size
            top_n: Number of top features to include
        """
        if self.correlation_matrix is None:
            print("Run correlation_analysis first")
            return

        # Get top N features by correlation with safety_label
        if "safety_label" in self.correlation_matrix.columns:
            top_features = self.correlation_matrix["safety_label"].abs().sort_values(ascending=False).head(top_n).index
        else:
            # Just use first N features
            top_features = self.correlation_matrix.columns[:top_n]

        # Subset correlation matrix
        subset_corr = self.correlation_matrix.loc[top_features, top_features]

        # Create heatmap
        plt.figure(figsize=figsize)
        if HAS_SEABORN:
            sns.heatmap(
                subset_corr,
                annot=False,
                cmap="RdBu_r",
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8}
            )
        else:
            # Matplotlib fallback
            im = plt.imshow(subset_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(im, shrink=0.8)
            plt.xticks(range(len(subset_corr.columns)), subset_corr.columns, rotation=90, fontsize=8)
            plt.yticks(range(len(subset_corr.index)), subset_corr.index, fontsize=8)
        plt.title("Correlation Heatmap - Top Environmental Variables", fontsize=16, pad=20)
        plt.tight_layout()

        return plt.gcf()

    def plot_feature_importance(
        self,
        figsize: Tuple[int, int] = (10, 8),
        top_n: int = 20
    ):
        """
        Create feature importance bar plot.

        Args:
            figsize: Figure size
            top_n: Number of top features to plot
        """
        if self.feature_importance is None:
            print("Run feature_importance_analysis first")
            return

        top_features = self.feature_importance.head(top_n)

        plt.figure(figsize=figsize)
        plt.barh(range(len(top_features)), top_features["importance"], color="steelblue")
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Importance Score", fontsize=12)
        plt.title(f"Top {top_n} Features for Predicting Unsafe Conditions\n(Random Forest)", fontsize=14, pad=15)
        plt.gca().invert_yaxis()
        plt.tight_layout()

        return plt.gcf()

    def plot_parameter_distributions(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create distribution plots for key parameters by safety label.

        Args:
            figsize: Figure size
        """
        # Support both synthetic and real data parameter names
        key_params = [
            "turbidity", "pH", "dissolved_oxygen", "dissolved_oxygen_pct",
            "phycocyanin", "phycoerythrin_rfu"
        ]
        available_params = [p for p in key_params if p in self.data.columns]

        if len(available_params) == 0:
            print("No key parameters available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        for i, param in enumerate(available_params[:4]):
            for label in [0, 1, 2]:
                label_name = ["SAFE", "CAUTION", "DANGER"][label]
                color = ["green", "orange", "red"][label]

                data_subset = self.data[self.data["safety_label"] == label][param]

                if len(data_subset) > 0:
                    axes[i].hist(
                        data_subset,
                        bins=30,
                        alpha=0.5,
                        label=label_name,
                        color=color
                    )

            axes[i].set_xlabel(param, fontsize=11)
            axes[i].set_ylabel("Frequency", fontsize=11)
            axes[i].set_title(f"Distribution of {param}", fontsize=12)
            axes[i].legend()

        plt.suptitle("Parameter Distributions by Safety Classification", fontsize=14, y=1.02)
        plt.tight_layout()

        return fig


def main():
    """Example usage."""
    print("Loading labeled dataset...")
    df = pd.read_csv("../../data/processed/labeled_dataset.csv")
    df["time"] = pd.to_datetime(df["time"])

    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")

    # Run experiment
    exp1 = Experiment1(df)
    results = exp1.run_full_analysis()

    # Generate plots
    print("\nGenerating visualizations...")

    exp1.plot_correlation_heatmap()
    plt.savefig("../../notebooks/exp1_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    print("Saved: exp1_correlation_heatmap.png")

    exp1.plot_feature_importance()
    plt.savefig("../../notebooks/exp1_feature_importance.png", dpi=150, bbox_inches="tight")
    print("Saved: exp1_feature_importance.png")

    exp1.plot_parameter_distributions()
    plt.savefig("../../notebooks/exp1_parameter_distributions.png", dpi=150, bbox_inches="tight")
    print("Saved: exp1_parameter_distributions.png")

    print("\nExperiment 1 complete!")


if __name__ == "__main__":
    main()
