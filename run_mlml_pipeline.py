"""
MLML (Moss Landing) Water Quality Analysis Pipeline
====================================================
Complete analysis pipeline for MLML Monterey Bay oceanographic data.

Location: Moss Landing, Monterey Bay, California (36.8025°N, 121.7915°W)
Data Source: CeNCOOS MLML Seawater Intake
Date Range: Jan 1, 2024 - Dec 30, 2024

Unique Parameters:
- Nitrate (µmol/L) - nutrient indicator for upwelling
- Tide Height (m above MLLW) - tidal influence
- Beam Attenuation - turbidity proxy
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_collection.mlml_data_loader import MLMLDataLoader
from labeling.safety_classifier_mlml import SafetyClassifierMLML
from data_processing.feature_engineering_mlml import FeatureEngineerMLML


def run_correlation_analysis(df: pd.DataFrame, output_dir: str = "notebooks"):
    """Run correlation analysis between features and safety labels."""

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    # Binary label (0 = SAFE, 1 = UNSAFE)
    df['is_unsafe'] = (df['safety_label'] > 0).astype(int)

    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['safety_label', 'is_unsafe', 'hour', 'day_of_week', 'month',
                    'day_of_year', 'season', 'is_day', 'is_flooding', 'is_ebbing',
                    'is_low_tide', 'is_high_tide', 'tide_above_mean']
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    # Calculate correlations with is_unsafe
    correlations = {}
    for col in feature_cols:
        valid_mask = df[col].notna() & df['is_unsafe'].notna()
        if valid_mask.sum() > 100:
            corr = df.loc[valid_mask, col].corr(df.loc[valid_mask, 'is_unsafe'])
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 20 Positive Correlations (higher value → more unsafe):")
    print("-" * 50)
    positive = [(k, v) for k, v in sorted_corr if v > 0][:20]
    for feat, corr in positive:
        print(f"  {feat:45s} {corr:+.4f}")

    print("\nTop 10 Negative Correlations (lower value → more unsafe):")
    print("-" * 50)
    negative = [(k, v) for k, v in sorted_corr if v < 0][:10]
    for feat, corr in negative:
        print(f"  {feat:45s} {corr:+.4f}")

    # Create correlation heatmap for core parameters
    core_params = ['water_temp', 'pH', 'dissolved_oxygen_pct', 'dissolved_oxygen_umol',
                   'nitrate', 'salinity', 'fluorescence', 'beam_attenuation',
                   'tide_height', 'is_unsafe']
    available_core = [p for p in core_params if p in df.columns]

    if len(available_core) >= 3:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = df[available_core].corr()

        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

        # Add labels
        ax.set_xticks(range(len(available_core)))
        ax.set_yticks(range(len(available_core)))
        ax.set_xticklabels(available_core, rotation=45, ha='right')
        ax.set_yticklabels(available_core)

        # Add correlation values
        for i in range(len(available_core)):
            for j in range(len(available_core)):
                val = corr_matrix.iloc[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

        plt.colorbar(im, ax=ax, label='Correlation')
        ax.set_title('MLML Monterey Bay - Parameter Correlation Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/mlml_exp1_correlation_heatmap.png', dpi=150)
        plt.close()
        print(f"\nSaved: {output_dir}/mlml_exp1_correlation_heatmap.png")

    # Create feature importance bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    top_n = 25
    top_features = sorted_corr[:top_n]
    features = [x[0] for x in top_features]
    values = [x[1] for x in top_features]
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in values]

    bars = ax.barh(range(len(features)), values, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Correlation with Unsafe Conditions')
    ax.set_title('MLML Monterey Bay - Top Feature Correlations\n(Red = positive, Blue = negative)')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlim(-1, 1)

    # Add value labels
    for i, (feat, val) in enumerate(top_features):
        ax.text(val + 0.02 if val >= 0 else val - 0.02, i, f'{val:.3f}',
                va='center', ha='left' if val >= 0 else 'right', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/mlml_exp1_feature_importance.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/mlml_exp1_feature_importance.png")

    return dict(sorted_corr)


def create_visualizations(df: pd.DataFrame, output_dir: str = "notebooks"):
    """Create time series and distribution visualizations."""

    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    # Ensure time column is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    # 1. Time series plot
    fig, axes = plt.subplots(5, 2, figsize=(16, 16), sharex=True)

    params_to_plot = [
        ('water_temp', 'Water Temperature (°C)', 'tab:red'),
        ('pH', 'pH', 'tab:purple'),
        ('dissolved_oxygen_pct', 'DO Saturation (%)', 'tab:blue'),
        ('dissolved_oxygen_umol', 'DO (µmol/L)', 'tab:cyan'),
        ('nitrate', 'Nitrate (µmol/L)', 'tab:green'),
        ('salinity', 'Salinity (PSU)', 'tab:orange'),
        ('fluorescence', 'Fluorescence', 'tab:olive'),
        ('beam_attenuation', 'Beam Attenuation', 'tab:brown'),
        ('tide_height', 'Tide Height (m)', 'tab:gray'),
        ('safety_label', 'Safety Label', 'tab:pink'),
    ]

    for idx, (param, label, color) in enumerate(params_to_plot):
        ax = axes[idx // 2, idx % 2]
        if param in df.columns:
            ax.plot(df['time'], df[param], color=color, alpha=0.7, linewidth=0.5)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)

            if param == 'safety_label':
                ax.set_yticks([0, 1, 2])
                ax.set_yticklabels(['SAFE', 'CAUTION', 'DANGER'])
        else:
            ax.text(0.5, 0.5, f'{param}\nNot Available', transform=ax.transAxes,
                    ha='center', va='center')

    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[-1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    fig.suptitle('MLML Monterey Bay - Water Quality Parameters (2024)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mlml_timeseries.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/mlml_timeseries.png")

    # 2. Parameter distributions by safety label
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    dist_params = ['water_temp', 'pH', 'dissolved_oxygen_pct', 'dissolved_oxygen_umol',
                   'nitrate', 'salinity', 'fluorescence', 'beam_attenuation', 'tide_height']

    labels_map = {0: 'SAFE', 1: 'CAUTION', 2: 'DANGER'}
    colors_map = {0: 'green', 1: 'orange', 2: 'red'}

    for idx, param in enumerate(dist_params):
        ax = axes[idx // 3, idx % 3]
        if param in df.columns:
            for label_val in [0, 1, 2]:
                data = df[df['safety_label'] == label_val][param].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=50, alpha=0.5, label=labels_map[label_val],
                            color=colors_map[label_val], density=True)
            ax.set_xlabel(param)
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle('MLML Monterey Bay - Parameter Distributions by Safety Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mlml_exp1_parameter_distributions.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/mlml_exp1_parameter_distributions.png")


def save_results(hourly_df: pd.DataFrame, featured_df: pd.DataFrame,
                 classifier: SafetyClassifierMLML):
    """Save processed data and labels."""

    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Create directories if needed
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/labels', exist_ok=True)

    # Save hourly dataset
    hourly_path = 'data/processed/mlml_hourly_dataset.csv'
    hourly_df.to_csv(hourly_path, index=False)
    print(f"Saved: {hourly_path} ({len(hourly_df):,} × {len(hourly_df.columns)})")

    # Save featured dataset
    featured_path = 'data/processed/mlml_featured_dataset.csv'
    featured_df.to_csv(featured_path, index=False)
    print(f"Saved: {featured_path} ({len(featured_df):,} × {len(featured_df.columns)})")

    # Save labels
    labels_df = featured_df[['time', 'safety_label']].copy()
    labels_path = 'data/labels/mlml_safety_labels.csv'
    labels_df.to_csv(labels_path, index=False)
    print(f"Saved: {labels_path}")


def main():
    """Run complete MLML analysis pipeline."""

    print("=" * 70)
    print("MLML MONTEREY BAY WATER QUALITY ANALYSIS PIPELINE")
    print("=" * 70)
    print("Location: Moss Landing, Monterey Bay, CA (36.8°N, 121.8°W)")
    print("Data Source: CeNCOOS MLML Seawater Intake")
    print("=" * 70)

    # Step 1: Load Data
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)

    loader = MLMLDataLoader(data_dir="data/real")
    df = loader.load()

    # Show summary stats
    print("\nSummary Statistics:")
    stats = loader.get_summary_stats(df)
    print(stats[['mean', 'std', 'min', 'max', 'valid_pct']].round(2))

    # Step 2: Resample to Hourly
    print("\n" + "=" * 60)
    print("STEP 2: RESAMPLING TO HOURLY")
    print("=" * 60)

    hourly = loader.resample_to_hourly(df)
    print(f"Hourly records: {len(hourly):,}")

    # Step 3: Safety Classification
    print("\n" + "=" * 60)
    print("STEP 3: SAFETY CLASSIFICATION")
    print("=" * 60)

    classifier = SafetyClassifierMLML()
    labeled = classifier.classify(hourly)

    print("\nLabel Distribution:")
    dist = classifier.get_label_distribution()
    dist_pct = classifier.get_label_distribution_pct()
    print(f"  SAFE:    {dist['SAFE']:,} ({dist_pct['SAFE']:.1f}%)")
    print(f"  CAUTION: {dist['CAUTION']:,} ({dist_pct['CAUTION']:.1f}%)")
    print(f"  DANGER:  {dist['DANGER']:,} ({dist_pct['DANGER']:.1f}%)")

    print(classifier.get_trigger_summary())

    # Step 4: Feature Engineering
    print("\n" + "=" * 60)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 60)

    engineer = FeatureEngineerMLML()
    featured = engineer.engineer_features(labeled)

    # Step 5: Correlation Analysis
    correlations = run_correlation_analysis(featured)

    # Step 6: Visualizations
    create_visualizations(featured)

    # Step 7: Save Results
    save_results(labeled, featured, classifier)

    # Final Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nDataset: MLML Monterey Bay (2024)")
    print(f"Records: {len(featured):,} hourly observations")
    print(f"Features: {len(featured.columns)} total")
    print(f"\nSafety Distribution:")
    print(f"  SAFE:    {dist_pct['SAFE']:.1f}%")
    print(f"  CAUTION: {dist_pct['CAUTION']:.1f}%")
    print(f"  DANGER:  {dist_pct['DANGER']:.1f}%")

    # Top correlations summary
    print(f"\nTop Predictors of Unsafe Conditions:")
    top_3 = list(correlations.items())[:3]
    for feat, corr in top_3:
        print(f"  {feat}: r = {corr:+.3f}")

    print("\n" + "=" * 70)

    return featured, classifier, correlations


if __name__ == "__main__":
    featured, classifier, correlations = main()
