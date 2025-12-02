#!/usr/bin/env python3
"""
Scripps Pier Water Quality Analysis Pipeline
=============================================
Complete pipeline for analyzing Scripps Pier oceanographic data (3 years).

Location: La Jolla, California (32.867°N, 117.257°W)
Data Source: SCCOOS Scripps Pier Automated Shore Station
Date Range: Jan 1, 2023 - Nov 30, 2025 (~3 years)

Steps:
1. Load and clean Scripps Pier data from NetCDF files
2. Resample to hourly frequency
3. Apply safety classification (SAFE/CAUTION/DANGER)
4. Engineer predictive features
5. Run correlation analysis
6. Generate visualizations and results

Usage:
    python run_scripps_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
from data_collection.scripps_data_loader import ScrippsDataLoader
from data_processing.feature_engineering_newport import FeatureEngineerNewport  # Reuse - same parameters
from labeling.safety_classifier_scripps import SafetyClassifierScripps


def run_pipeline():
    """Run the complete Scripps Pier analysis pipeline."""

    print("=" * 70)
    print("SCRIPPS PIER WATER QUALITY ANALYSIS PIPELINE")
    print("=" * 70)
    print("Location: La Jolla, California (32.867°N, 117.257°W)")
    print("Data Source: SCCOOS Automated Shore Station")
    print("Date Range: 2023-2025 (3 years)")
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
    print("STEP 1: LOADING SCRIPPS PIER DATA (3 YEARS)")
    print("=" * 70)

    loader = ScrippsDataLoader(data_dir="data/real")
    raw_data = loader.load(years=[2023, 2024, 2025])

    print(f"\nRaw data loaded: {len(raw_data):,} records")
    print(f"Date range: {raw_data['time'].min()} to {raw_data['time'].max()}")

    # Show summary stats
    print("\nParameter Summary:")
    stats = loader.get_summary_stats(raw_data)
    print(stats[['mean', 'std', 'min', 'max', 'valid_pct']].round(2))

    # =========================================================================
    # STEP 2: Resample to Hourly
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: RESAMPLING TO HOURLY FREQUENCY")
    print("=" * 70)

    hourly_data = loader.resample_to_hourly(raw_data)
    print(f"Hourly records: {len(hourly_data):,}")

    # Save merged/cleaned dataset
    hourly_data.to_csv(processed_dir / "scripps_hourly_dataset.csv", index=False)
    print(f"Saved: {processed_dir / 'scripps_hourly_dataset.csv'}")

    # =========================================================================
    # STEP 3: Safety Classification
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: SAFETY CLASSIFICATION")
    print("=" * 70)

    classifier = SafetyClassifierScripps()
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
    labels_df.to_csv(labels_dir / "scripps_safety_labels.csv", index=False)
    print(f"\nSaved: {labels_dir / 'scripps_safety_labels.csv'}")

    # =========================================================================
    # STEP 4: Feature Engineering
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 70)

    # Reuse Newport feature engineer (same parameter types)
    engineer = FeatureEngineerNewport()
    featured_data = engineer.create_all_features(labeled_data, target_col="safety_label")

    print(f"\nTotal features: {featured_data.shape[1]}")
    print(f"New engineered features: {len(engineer.get_feature_names())}")

    # Save featured dataset
    featured_data.to_csv(processed_dir / "scripps_featured_dataset.csv", index=False)
    print(f"Saved: {processed_dir / 'scripps_featured_dataset.csv'}")

    # =========================================================================
    # STEP 5: Correlation Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: CORRELATION ANALYSIS")
    print("=" * 70)

    corr_results = run_correlation_analysis(featured_data, notebooks_dir)

    # =========================================================================
    # STEP 6: Generate Time Series Visualization
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: TIME SERIES VISUALIZATION")
    print("=" * 70)

    try:
        fig = create_timeseries_plot(featured_data)
        fig.savefig(notebooks_dir / "scripps_timeseries.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {notebooks_dir / 'scripps_timeseries.png'}")
    except Exception as e:
        print(f"Warning: Could not create time series plot: {e}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print("\nDataset Summary:")
    print(f"  Location: La Jolla, California (Scripps Pier)")
    print(f"  Records: {len(featured_data):,} hourly observations")
    print(f"  Date range: {featured_data['time'].min()} to {featured_data['time'].max()}")
    print(f"  Total features: {featured_data.shape[1]}")

    print("\nSafety Classification:")
    print(f"  SAFE:    {dist['SAFE']:,} ({dist_pct['SAFE']:.1f}%)")
    print(f"  CAUTION: {dist['CAUTION']:,} ({dist_pct['CAUTION']:.1f}%)")
    print(f"  DANGER:  {dist['DANGER']:,} ({dist_pct['DANGER']:.1f}%)")

    print("\nOutput Files:")
    print(f"  {processed_dir / 'scripps_hourly_dataset.csv'}")
    print(f"  {processed_dir / 'scripps_featured_dataset.csv'}")
    print(f"  {labels_dir / 'scripps_safety_labels.csv'}")
    print(f"  {notebooks_dir / 'scripps_*.png'}")

    return featured_data, corr_results


def run_correlation_analysis(df: pd.DataFrame, output_dir: Path) -> dict:
    """Run correlation analysis on the dataset."""

    print("\nComputing correlations...")

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove time-based columns
    exclude = ['hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend']
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr()

    # Get correlations with safety_label
    if 'safety_label' in corr_matrix.columns:
        target_corr = corr_matrix['safety_label'].sort_values(ascending=False)
        target_corr = target_corr[target_corr.index != 'safety_label']

        print("\nTop 10 Positive Correlations with Unsafe Conditions:")
        for i, (feature, corr) in enumerate(target_corr.head(10).items()):
            print(f"  {i+1:2}. {feature:40} {corr:+.3f}")

        print("\nTop 10 Negative Correlations with Unsafe Conditions:")
        for i, (feature, corr) in enumerate(target_corr.tail(10).items()):
            print(f"  {i+1:2}. {feature:40} {corr:+.3f}")

        # Generate visualizations
        print("\nGenerating visualizations...")

        # Correlation heatmap
        create_correlation_heatmap(corr_matrix, target_corr, output_dir)

        # Feature importance chart
        create_feature_importance_chart(target_corr, output_dir)

        # Parameter distributions
        create_distribution_plots(df, output_dir)

        return {'corr_matrix': corr_matrix, 'target_corr': target_corr}

    return {}


def create_correlation_heatmap(corr_matrix, target_corr, output_dir):
    """Create correlation heatmap."""
    top_features = target_corr.abs().sort_values(ascending=False).head(20).index.tolist()
    top_features = ['safety_label'] + top_features

    subset_corr = corr_matrix.loc[top_features, top_features]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(subset_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=12)

    ax.set_xticks(range(len(subset_corr.columns)))
    ax.set_yticks(range(len(subset_corr.index)))
    ax.set_xticklabels(subset_corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(subset_corr.index, fontsize=8)
    ax.set_title('Correlation Heatmap - Top 20 Features\nScripps Pier Water Quality', fontsize=14, pad=20)

    plt.tight_layout()
    fig.savefig(output_dir / 'scripps_exp1_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'scripps_exp1_correlation_heatmap.png'}")


def create_feature_importance_chart(target_corr, output_dir):
    """Create feature importance bar chart."""
    top_features = target_corr.abs().sort_values(ascending=False).head(20)
    feature_corrs = target_corr[top_features.index]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['red' if c > 0 else 'blue' for c in feature_corrs]
    y_pos = range(len(feature_corrs))
    ax.barh(y_pos, feature_corrs, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_corrs.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Correlation with Unsafe Conditions', fontsize=12)
    ax.set_title('Top 20 Features for Predicting Unsafe Conditions\nScripps Pier Water Quality', fontsize=14, pad=15)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlim(-1, 1)

    plt.tight_layout()
    fig.savefig(output_dir / 'scripps_exp1_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'scripps_exp1_feature_importance.png'}")


def create_distribution_plots(df, output_dir):
    """Create distribution plots."""
    params = ['chlorophyll', 'dissolved_oxygen_pct', 'pH', 'water_temp', 'turbidity', 'salinity']
    available = [p for p in params if p in df.columns][:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = {0: 'green', 1: 'orange', 2: 'red'}
    labels = {0: 'SAFE', 1: 'CAUTION', 2: 'DANGER'}

    for i, param in enumerate(available):
        ax = axes[i]
        for label_val in [0, 1, 2]:
            data = df[df['safety_label'] == label_val][param].dropna()
            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.5, color=colors[label_val],
                       label=f"{labels[label_val]} (n={len(data)})")
        ax.set_xlabel(param, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Distribution of {param}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Parameter Distributions by Safety Classification\nScripps Pier', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'scripps_exp1_parameter_distributions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'scripps_exp1_parameter_distributions.png'}")


def create_timeseries_plot(df: pd.DataFrame) -> plt.Figure:
    """Create time series visualization."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    params = [
        ('water_temp', 'Water Temperature (°C)', 'blue'),
        ('pH', 'pH', 'green'),
        ('dissolved_oxygen_pct', 'Dissolved Oxygen (%)', 'purple'),
        ('chlorophyll', 'Chlorophyll (µg/L)', 'orange'),
    ]

    for i, (param, label, color) in enumerate(params):
        if param in df.columns:
            ax = axes[i]
            valid_data = df[['time', param, 'safety_label']].dropna(subset=[param])

            if len(valid_data) > 0:
                ax.plot(valid_data['time'], valid_data[param], color=color, alpha=0.5, linewidth=0.3)

                # Add rolling mean if enough data
                if len(valid_data) > 168:  # 1 week of hourly data
                    rolling = valid_data[param].rolling(window=168, min_periods=1).mean()
                    ax.plot(valid_data['time'], rolling, color=color, linewidth=2, label='7-day avg')

                ax.set_ylabel(label, fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')

    axes[-1].set_xlabel('Date', fontsize=12)
    fig.suptitle('Scripps Pier Water Quality Time Series (2023-2025)', fontsize=14, y=1.02)
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    featured_data, results = run_pipeline()
