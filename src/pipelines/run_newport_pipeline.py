#!/usr/bin/env python3
"""
Newport Pier Water Quality Analysis Pipeline
=============================================
Complete pipeline for analyzing Newport Pier oceanographic data.

Location: Newport Beach, California (33.6°N, 117.9°W)
Data Source: NOAA/SCCOOS Newport Pier Automated Shore Station
Date Range: Dec 1, 2024 - Nov 30, 2025 (364 days)

Steps:
1. Load and clean Newport Pier data
2. Resample to hourly frequency
3. Apply safety classification (SAFE/CAUTION/DANGER)
4. Engineer predictive features
5. Run correlation analysis (Experiment 1)
6. Generate visualizations and results

Usage:
    python run_newport_pipeline.py
"""

import sys
import os
from pathlib import Path

# Add src to path (parent of pipelines directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
from data_collection.newport_data_loader import NewportDataLoader
from data_processing.feature_engineering_newport import FeatureEngineerNewport
from labeling.safety_classifier_newport import SafetyClassifierNewport

# Try to import experiment module
try:
    from experiments.experiment_1_correlation import Experiment1
    HAS_EXPERIMENT = True
except ImportError as e:
    print(f"Warning: Could not import Experiment1: {e}")
    HAS_EXPERIMENT = False


def run_pipeline():
    """Run the complete Newport Pier analysis pipeline."""

    print("=" * 70)
    print("NEWPORT PIER WATER QUALITY ANALYSIS PIPELINE")
    print("=" * 70)
    print("Location: Newport Beach, California")
    print("Data Source: NOAA/SCCOOS Automated Shore Station")
    print("=" * 70)

    # Create output directories
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    labels_dir = data_dir / "labels"
    notebooks_dir = Path("notebooks")

    for d in [processed_dir, labels_dir, notebooks_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: LOADING NEWPORT PIER DATA")
    print("=" * 70)

    loader = NewportDataLoader(data_dir="data/real")
    raw_data = loader.load()

    print(f"\nRaw data loaded: {len(raw_data):,} records")
    print(f"Date range: {raw_data['time'].min()} to {raw_data['time'].max()}")

    # Show summary stats
    print("\nParameter Summary:")
    stats = loader.get_summary_stats(raw_data)
    print(stats[['mean', 'std', 'min', 'max', 'missing_pct']].round(2))

    # =========================================================================
    # STEP 2: Resample to Hourly
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: RESAMPLING TO HOURLY FREQUENCY")
    print("=" * 70)

    hourly_data = loader.resample_to_hourly(raw_data)
    print(f"Hourly records: {len(hourly_data):,}")

    # Save merged/cleaned dataset
    hourly_data.to_csv(processed_dir / "newport_hourly_dataset.csv", index=False)
    print(f"Saved: {processed_dir / 'newport_hourly_dataset.csv'}")

    # =========================================================================
    # STEP 3: Safety Classification
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: SAFETY CLASSIFICATION")
    print("=" * 70)

    classifier = SafetyClassifierNewport()
    labeled_data = classifier.classify(hourly_data)

    # Print label distribution
    print("\nLabel Distribution:")
    dist = classifier.get_label_distribution()
    dist_pct = classifier.get_label_distribution_pct()

    print(f"  SAFE:    {dist['SAFE']:,} ({dist_pct['SAFE']:.1f}%)")
    print(f"  CAUTION: {dist['CAUTION']:,} ({dist_pct['CAUTION']:.1f}%)")
    print(f"  DANGER:  {dist['DANGER']:,} ({dist_pct['DANGER']:.1f}%)")

    # Print trigger summary
    print(classifier.get_trigger_summary())

    # Save labels
    labels_df = labeled_data[['time', 'safety_label']].copy()
    labels_df.to_csv(labels_dir / "newport_safety_labels.csv", index=False)
    print(f"\nSaved: {labels_dir / 'newport_safety_labels.csv'}")

    # =========================================================================
    # STEP 4: Feature Engineering
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 70)

    engineer = FeatureEngineerNewport()
    featured_data = engineer.create_all_features(labeled_data, target_col="safety_label")

    print(f"\nTotal features: {featured_data.shape[1]}")
    print(f"New engineered features: {len(engineer.get_feature_names())}")

    # Save featured dataset
    featured_data.to_csv(processed_dir / "newport_featured_dataset.csv", index=False)
    print(f"Saved: {processed_dir / 'newport_featured_dataset.csv'}")

    # =========================================================================
    # STEP 5: Correlation Analysis (Experiment 1)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: CORRELATION ANALYSIS (EXPERIMENT 1)")
    print("=" * 70)

    if HAS_EXPERIMENT:
        exp1 = Experiment1(featured_data)
        results = exp1.run_full_analysis()

        # Generate visualizations
        print("\nGenerating visualizations...")

        try:
            fig = exp1.plot_correlation_heatmap()
            if fig:
                fig.savefig(notebooks_dir / "newport_exp1_correlation_heatmap.png",
                           dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved: {notebooks_dir / 'newport_exp1_correlation_heatmap.png'}")
        except Exception as e:
            print(f"Warning: Could not create correlation heatmap: {e}")

        try:
            fig = exp1.plot_feature_importance()
            if fig:
                fig.savefig(notebooks_dir / "newport_exp1_feature_importance.png",
                           dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved: {notebooks_dir / 'newport_exp1_feature_importance.png'}")
        except Exception as e:
            print(f"Warning: Could not create feature importance plot: {e}")

        try:
            fig = exp1.plot_parameter_distributions()
            if fig:
                fig.savefig(notebooks_dir / "newport_exp1_parameter_distributions.png",
                           dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved: {notebooks_dir / 'newport_exp1_parameter_distributions.png'}")
        except Exception as e:
            print(f"Warning: Could not create distribution plots: {e}")
    else:
        print("Skipping correlation analysis (experiment module not available)")
        results = None

    # =========================================================================
    # STEP 6: Generate Time Series Visualization
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: TIME SERIES VISUALIZATION")
    print("=" * 70)

    try:
        fig = create_timeseries_plot(featured_data)
        fig.savefig(notebooks_dir / "newport_timeseries.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {notebooks_dir / 'newport_timeseries.png'}")
    except Exception as e:
        print(f"Warning: Could not create time series plot: {e}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print("\nDataset Summary:")
    print(f"  Location: Newport Beach, California")
    print(f"  Records: {len(featured_data):,} hourly observations")
    print(f"  Date range: {featured_data['time'].min()} to {featured_data['time'].max()}")
    print(f"  Total features: {featured_data.shape[1]}")

    print("\nSafety Classification:")
    print(f"  SAFE:    {dist['SAFE']:,} ({dist_pct['SAFE']:.1f}%)")
    print(f"  CAUTION: {dist['CAUTION']:,} ({dist_pct['CAUTION']:.1f}%)")
    print(f"  DANGER:  {dist['DANGER']:,} ({dist_pct['DANGER']:.1f}%)")

    print("\nOutput Files:")
    print(f"  {processed_dir / 'newport_hourly_dataset.csv'}")
    print(f"  {processed_dir / 'newport_featured_dataset.csv'}")
    print(f"  {labels_dir / 'newport_safety_labels.csv'}")
    print(f"  {notebooks_dir / 'newport_exp1_*.png'}")

    return featured_data, results


def create_timeseries_plot(df: pd.DataFrame) -> plt.Figure:
    """Create time series visualization of key parameters."""

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Parameters to plot
    params = [
        ('water_temp', 'Water Temperature (°C)', 'blue'),
        ('pH', 'pH', 'green'),
        ('dissolved_oxygen_pct', 'Dissolved Oxygen (%)', 'purple'),
        ('chlorophyll', 'Chlorophyll (µg/L)', 'orange'),
    ]

    for i, (param, label, color) in enumerate(params):
        if param in df.columns:
            ax = axes[i]

            # Plot the parameter
            ax.plot(df['time'], df[param], color=color, alpha=0.7, linewidth=0.5)

            # Add rolling mean
            if len(df) > 24:
                rolling = df[param].rolling(window=24, min_periods=1).mean()
                ax.plot(df['time'], rolling, color=color, linewidth=2, label='24h avg')

            ax.set_ylabel(label, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            # Color background by safety label
            if 'safety_label' in df.columns:
                for label_val, label_color in [(2, 'red'), (1, 'yellow')]:
                    mask = df['safety_label'] == label_val
                    if mask.any():
                        for idx in df[mask].index:
                            ax.axvspan(df.loc[idx, 'time'],
                                      df.loc[idx, 'time'] + pd.Timedelta(hours=1),
                                      alpha=0.2, color=label_color, linewidth=0)

    axes[-1].set_xlabel('Date', fontsize=12)
    fig.suptitle('Newport Pier Water Quality Time Series\n(Red = DANGER, Yellow = CAUTION)',
                fontsize=14, y=1.02)
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    featured_data, results = run_pipeline()
