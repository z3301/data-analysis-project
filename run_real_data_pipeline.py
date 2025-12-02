"""
Real Data Pipeline
==================
Complete analysis pipeline using ONLY real data from Wahoo Bay and Pompano Beach.

Data Sources:
- Wahoo Bay Water Quality (wb_water_sensor.csv)
- Wahoo Bay Weather Station (wb_weather_sensor.csv)
- Pompano Beach Weather Station (pb_weather_sensor.csv)

All data from: https://www.sensestream.org/measurements
Date Range: Nov 30, 2024 - Nov 30, 2025 (1 year)

Usage:
    python run_real_data_pipeline.py
"""

import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np

print("=" * 80)
print("WAHOO BAY WATER QUALITY PREDICTION - REAL DATA PIPELINE")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nData Source: https://www.sensestream.org/measurements")
print("=" * 80)

# =============================================================================
# STEP 1: Load Real Data
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING REAL SENSOR DATA")
print("=" * 80)

from data_collection.real_data_loader import RealDataLoader

loader = RealDataLoader(data_dir="data/real")
water_quality_raw, wb_weather_raw, pb_weather_raw = loader.load_all()

# Get overlapping period
start_time, end_time = loader.get_overlapping_period(
    water_quality_raw, wb_weather_raw, pb_weather_raw
)
print(f"\n📅 Overlapping data period: {start_time} to {end_time}")

# =============================================================================
# STEP 2: Resample to Hourly and Align
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: RESAMPLING TO HOURLY FREQUENCY")
print("=" * 80)

# Resample each dataset to hourly
print("\nResampling to hourly frequency...")
water_quality = loader.resample_to_hourly(water_quality_raw)
wb_weather = loader.resample_to_hourly(wb_weather_raw)
pb_weather = loader.resample_to_hourly(pb_weather_raw)

print(f"  Water Quality: {len(water_quality):,} hourly records")
print(f"  WB Weather:    {len(wb_weather):,} hourly records")
print(f"  PB Weather:    {len(pb_weather):,} hourly records")

# Filter to overlapping period
water_quality = water_quality[
    (water_quality['time'] >= start_time) &
    (water_quality['time'] <= end_time)
].reset_index(drop=True)

wb_weather = wb_weather[
    (wb_weather['time'] >= start_time) &
    (wb_weather['time'] <= end_time)
].reset_index(drop=True)

pb_weather = pb_weather[
    (pb_weather['time'] >= start_time) &
    (pb_weather['time'] <= end_time)
].reset_index(drop=True)

print(f"\nAfter filtering to overlapping period:")
print(f"  Water Quality: {len(water_quality):,} records")
print(f"  WB Weather:    {len(wb_weather):,} records")
print(f"  PB Weather:    {len(pb_weather):,} records")

# =============================================================================
# STEP 3: Merge Data Sources
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: MERGING DATA SOURCES")
print("=" * 80)

# Merge water quality with WB weather
print("\nMerging water quality with Wahoo Bay weather...")
merged = water_quality.merge(
    wb_weather,
    on='time',
    how='outer',
    suffixes=('', '_wb')
)
print(f"  After WB weather merge: {len(merged):,} records")

# Merge with PB weather (as backup/supplement)
print("Merging with Pompano Beach weather...")
merged = merged.merge(
    pb_weather,
    on='time',
    how='left',
    suffixes=('', '_pb')
)
print(f"  After PB weather merge: {len(merged):,} records")

# Sort by time
merged = merged.sort_values('time').reset_index(drop=True)

# Fill missing values where possible (use WB values if available, else PB)
for col in ['air_temp', 'humidity', 'barometric_pressure', 'wind_speed_avg', 'rain_accumulation']:
    if f'{col}_pb' in merged.columns and col in merged.columns:
        merged[col] = merged[col].fillna(merged[f'{col}_pb'])

# Drop PB backup columns
pb_cols = [c for c in merged.columns if c.endswith('_pb')]
merged = merged.drop(columns=pb_cols)

print(f"\n✓ Merge complete!")
print(f"  Final shape: {merged.shape}")
print(f"  Columns: {list(merged.columns)}")

# =============================================================================
# STEP 4: Data Cleaning
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: CLEANING DATA")
print("=" * 80)

print("\nChecking for missing values...")
missing = merged.isnull().sum()
missing_pct = (missing / len(merged)) * 100
print(missing[missing > 0].to_string() if missing.any() else "  No missing values")

print("\nInterpolating missing values...")
numeric_cols = merged.select_dtypes(include=[np.number]).columns
merged[numeric_cols] = merged[numeric_cols].interpolate(method='linear', limit_direction='both')

# Fill any remaining NaN with column median
for col in numeric_cols:
    if merged[col].isnull().any():
        merged[col] = merged[col].fillna(merged[col].median())

print(f"  ✓ Missing values handled")
print(f"  Final records: {len(merged):,}")

# Save cleaned merged data
merged.to_csv("data/processed/real_merged_dataset.csv", index=False)
print(f"\n  Saved: data/processed/real_merged_dataset.csv")

# =============================================================================
# STEP 5: Feature Engineering
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: FEATURE ENGINEERING")
print("=" * 80)

from data_processing.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()

print("\nCreating engineered features...")
featured_data = engineer.create_all_features(merged)

print(f"\n✓ Feature engineering complete!")
print(f"  Input features: {len(merged.columns)}")
print(f"  Output features: {len(featured_data.columns)}")
print(f"  New features created: {len(engineer.get_feature_names())}")

# Save featured data
featured_data.to_csv("data/processed/real_featured_dataset.csv", index=False)
print(f"\n  Saved: data/processed/real_featured_dataset.csv")

# =============================================================================
# STEP 6: Generate Safety Labels
# =============================================================================
print("\n" + "=" * 80)
print("STEP 6: GENERATING SAFETY LABELS")
print("=" * 80)

from labeling.safety_classifier_real import SafetyClassifierReal

classifier = SafetyClassifierReal()

print("\nClassifying water conditions...")
labeled_data = classifier.classify(featured_data, label_col="safety_label")

print(f"\n✓ Safety classification complete!")
print(f"\n  Label Distribution:")
dist = classifier.get_label_distribution()
dist_pct = classifier.get_label_distribution_pct()
print(f"    🟢 SAFE:    {dist['SAFE']:5,} records ({dist_pct['SAFE']:.1f}%)")
print(f"    🟡 CAUTION: {dist['CAUTION']:5,} records ({dist_pct['CAUTION']:.1f}%)")
print(f"    🔴 DANGER:  {dist['DANGER']:5,} records ({dist_pct['DANGER']:.1f}%)")

# Show example classifications
if dist['DANGER'] > 0:
    print(f"\n  Example DANGER condition:")
    danger_example = labeled_data[labeled_data["safety_label"] == 2].iloc[0]
    print(f"    Time: {danger_example['time']}")
    print(f"    Reason: {classifier.explain_classification(danger_example, 2)}")

if dist['CAUTION'] > 0:
    print(f"\n  Example CAUTION condition:")
    caution_sample = labeled_data[labeled_data["safety_label"] == 1].iloc[0]
    print(f"    Time: {caution_sample['time']}")
    print(f"    Reason: {classifier.explain_classification(caution_sample, 1)}")

# Save labeled data
labeled_data.to_csv("data/processed/real_labeled_dataset.csv", index=False)
labeled_data[["time", "safety_label"]].to_csv("data/labels/real_safety_labels.csv", index=False)
print(f"\n  Saved: data/processed/real_labeled_dataset.csv")
print(f"  Saved: data/labels/real_safety_labels.csv")

# =============================================================================
# STEP 7: Run Experiment 1 (Correlation Analysis)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 7: RUNNING EXPERIMENT 1 (CORRELATION ANALYSIS)")
print("=" * 80)

try:
    from experiments.experiment_1_correlation import Experiment1

    exp1 = Experiment1(labeled_data)

    print("\nRunning correlation analysis on REAL data...")
    results = exp1.run_full_analysis()
except Exception as e:
    print(f"\nWarning: Could not run Experiment 1 due to library compatibility issue: {e}")
    print("Skipping to basic correlation analysis...")
    exp1 = None
    results = {}

    # Basic correlation analysis without sklearn
    print("\nBasic Correlation Analysis:")
    numeric_cols = labeled_data.select_dtypes(include=[np.number]).columns
    if 'safety_label' in numeric_cols:
        corr_with_label = labeled_data[numeric_cols].corr()['safety_label'].sort_values(ascending=False)
        print("\nTop 10 correlations with safety_label:")
        print(corr_with_label.head(11).drop('safety_label').round(3))
        print("\nBottom 10 correlations with safety_label:")
        print(corr_with_label.tail(10).round(3))

# =============================================================================
# STEP 8: Generate Visualizations
# =============================================================================
print("\n" + "=" * 80)
print("STEP 8: GENERATING VISUALIZATIONS")
print("=" * 80)

import matplotlib.pyplot as plt

print("\nCreating visualizations from REAL data...")

# Correlation heatmap
print("\n[1/4] Correlation heatmap...")
if exp1 is not None:
    fig1 = exp1.plot_correlation_heatmap(figsize=(14, 12), top_n=25)
    plt.savefig("notebooks/real_exp1_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: notebooks/real_exp1_correlation_heatmap.png")
else:
    # Basic correlation heatmap without exp1
    numeric_cols = labeled_data.select_dtypes(include=[np.number]).columns
    corr_matrix = labeled_data[numeric_cols].corr()
    top_features = corr_matrix['safety_label'].abs().sort_values(ascending=False).head(25).index
    subset_corr = corr_matrix.loc[top_features, top_features]

    plt.figure(figsize=(14, 12))
    im = plt.imshow(subset_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, shrink=0.8)
    plt.xticks(range(len(subset_corr.columns)), subset_corr.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(subset_corr.index)), subset_corr.index, fontsize=8)
    plt.title("Correlation Heatmap - Top Environmental Variables (Real Data)", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("notebooks/real_exp1_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: notebooks/real_exp1_correlation_heatmap.png")

# Feature importance
print("\n[2/4] Feature importance chart...")
if exp1 is not None:
    fig2 = exp1.plot_feature_importance(figsize=(10, 8), top_n=20)
    plt.savefig("notebooks/real_exp1_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: notebooks/real_exp1_feature_importance.png")
else:
    # Use correlation as proxy for importance
    top_corr = corr_with_label.abs().sort_values(ascending=False).head(21).drop('safety_label')
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_corr)), top_corr.values, color='steelblue')
    plt.yticks(range(len(top_corr)), top_corr.index)
    plt.xlabel('Absolute Correlation with Safety Label', fontsize=12)
    plt.title('Top 20 Features by Correlation with Unsafe Conditions\n(Real Data)', fontsize=14, pad=15)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("notebooks/real_exp1_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: notebooks/real_exp1_feature_importance.png")

# Parameter distributions
print("\n[3/4] Parameter distribution plots...")
if exp1 is not None:
    fig3 = exp1.plot_parameter_distributions(figsize=(15, 10))
    plt.savefig("notebooks/real_exp1_parameter_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: notebooks/real_exp1_parameter_distributions.png")
else:
    # Basic distribution plots
    key_params = ['turbidity', 'pH', 'dissolved_oxygen_pct', 'phycoerythrin_rfu']
    available_params = [p for p in key_params if p in labeled_data.columns]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, param in enumerate(available_params[:4]):
        for label in [0, 1, 2]:
            label_name = ["SAFE", "CAUTION", "DANGER"][label]
            color = ["green", "orange", "red"][label]
            data_subset = labeled_data[labeled_data["safety_label"] == label][param]
            if len(data_subset) > 0:
                axes[i].hist(data_subset, bins=30, alpha=0.5, label=label_name, color=color)
        axes[i].set_xlabel(param, fontsize=11)
        axes[i].set_ylabel("Frequency", fontsize=11)
        axes[i].set_title(f"Distribution of {param}", fontsize=12)
        axes[i].legend()

    plt.suptitle("Parameter Distributions by Safety Classification (Real Data)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("notebooks/real_exp1_parameter_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: notebooks/real_exp1_parameter_distributions.png")

# Time series plot
print("\n[4/4] Time series visualization...")
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

# Plot key parameters over time
params = ['turbidity', 'dissolved_oxygen_pct', 'phycoerythrin_rfu', 'rain_accumulation']
titles = ['Turbidity (FNU)', 'Dissolved Oxygen (%)', 'Phycoerythrin (RFU)', 'Rain Accumulation (mm)']
colors = ['brown', 'blue', 'green', 'purple']

for ax, param, title, color in zip(axes, params, titles, colors):
    if param in labeled_data.columns:
        ax.plot(labeled_data['time'], labeled_data[param], color=color, alpha=0.7, linewidth=0.5)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Highlight CAUTION and DANGER periods
        for label, lbl_color in [(1, 'orange'), (2, 'red')]:
            mask = labeled_data['safety_label'] == label
            if mask.any():
                ax.scatter(labeled_data.loc[mask, 'time'], labeled_data.loc[mask, param],
                          color=lbl_color, s=2, alpha=0.7, label=f"{'CAUTION' if label==1 else 'DANGER'}")

axes[0].set_title('Real Data: Key Water Quality Parameters Over Time (1 Year)', fontsize=14, fontweight='bold')
axes[-1].set_xlabel('Date', fontsize=12)
plt.tight_layout()
plt.savefig("notebooks/real_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: notebooks/real_timeseries.png")

# =============================================================================
# STEP 9: Summary Statistics
# =============================================================================
print("\n" + "=" * 80)
print("STEP 9: GENERATING SUMMARY STATISTICS")
print("=" * 80)

# Key statistics
print("\n📊 REAL DATA SUMMARY STATISTICS:")
print(f"\n  Date Range: {labeled_data['time'].min()} to {labeled_data['time'].max()}")
print(f"  Total Hours: {len(labeled_data):,}")
print(f"  Total Days: {len(labeled_data) // 24}")

# Water quality stats
print("\n  Water Quality Parameters (Mean ± Std):")
wq_params = ['water_temp', 'pH', 'dissolved_oxygen_pct', 'turbidity', 'phycoerythrin_rfu', 'nitrate']
for param in wq_params:
    if param in labeled_data.columns:
        mean = labeled_data[param].mean()
        std = labeled_data[param].std()
        print(f"    {param:25s}: {mean:8.2f} ± {std:.2f}")

# Weather stats
print("\n  Weather Parameters (Mean ± Std):")
weather_params = ['air_temp', 'humidity', 'rain_accumulation', 'wind_speed_avg']
for param in weather_params:
    if param in labeled_data.columns:
        mean = labeled_data[param].mean()
        std = labeled_data[param].std()
        print(f"    {param:25s}: {mean:8.2f} ± {std:.2f}")

# Safety label stats
print(f"\n  Safety Classification:")
print(f"    SAFE:    {dist['SAFE']:5,} hours ({dist_pct['SAFE']:.1f}%)")
print(f"    CAUTION: {dist['CAUTION']:5,} hours ({dist_pct['CAUTION']:.1f}%)")
print(f"    DANGER:  {dist['DANGER']:5,} hours ({dist_pct['DANGER']:.1f}%)")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)

print("\n📊 SUMMARY:")
print(f"  Data Source: Real sensors from SenseStream.org")
print(f"  Data Period: {labeled_data['time'].min().date()} to {labeled_data['time'].max().date()}")
print(f"  Total Records: {len(labeled_data):,} hourly observations")
print(f"  Features: {len(labeled_data.columns)}")
print(f"  Safety Labels: {dist['SAFE']:,} SAFE, {dist['CAUTION']:,} CAUTION, {dist['DANGER']:,} DANGER")

print("\n📁 OUTPUT FILES:")
print("  Data:")
print("    - data/processed/real_merged_dataset.csv")
print("    - data/processed/real_featured_dataset.csv")
print("    - data/processed/real_labeled_dataset.csv")
print("    - data/labels/real_safety_labels.csv")
print("  Visualizations:")
print("    - notebooks/real_exp1_correlation_heatmap.png")
print("    - notebooks/real_exp1_feature_importance.png")
print("    - notebooks/real_exp1_parameter_distributions.png")
print("    - notebooks/real_timeseries.png")

print("\n🎯 NEXT STEPS:")
print("  1. Review visualizations in notebooks/")
print("  2. Run Experiments 2 & 3 (classification and clustering)")
print("  3. Update paper with real data results")

print(f"\n✓ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
