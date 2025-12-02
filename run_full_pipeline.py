"""
Master Pipeline Script
======================
Runs the complete Wahoo Bay analysis pipeline from data generation to experiments.

This script demonstrates the full workflow:
1. Generate synthetic data (water quality, weather, tides)
2. Merge and clean data
3. Engineer features
4. Generate safety labels
5. Run Experiment 1 (correlation analysis)
6. Generate visualizations

Usage:
    python run_full_pipeline.py
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 80)
print("WAHOO BAY WATER QUALITY PREDICTION - FULL PIPELINE")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# STEP 1: Generate Synthetic Data
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: GENERATING SYNTHETIC DATA")
print("=" * 80)

from data_collection.water_quality_generator import WaterQualityGenerator
from data_collection.weather_generator import WeatherGenerator
from data_collection.noaa_tides import NOAATidesCollector

# Date range: 1 year
end_date = datetime(2024, 11, 20)
start_date = end_date - timedelta(days=365)

print(f"\nDate range: {start_date.date()} to {end_date.date()}")
print("Frequency: Hourly (8,760 observations expected)")

# Generate water quality data
print("\n[1/3] Generating water quality data...")
wq_generator = WaterQualityGenerator(seed=42)
water_quality = wq_generator.generate(
    start_date=start_date.strftime("%Y-%m-%d"),
    end_date=end_date.strftime("%Y-%m-%d"),
    frequency="H"
)
print(f"  ✓ Generated {len(water_quality)} water quality records")
water_quality.to_csv("data/raw/water_quality_synthetic.csv", index=False)

# Generate weather data
print("\n[2/3] Generating weather data...")
weather_gen = WeatherGenerator(seed=42)
weather_station, weather_external = weather_gen.generate(
    start_date=start_date.strftime("%Y-%m-%d"),
    end_date=end_date.strftime("%Y-%m-%d"),
    frequency="H"
)
print(f"  ✓ Generated {len(weather_station)} weather station records")
print(f"  ✓ Generated {len(weather_external)} external weather records")
weather_station.to_csv("data/raw/weather_station_synthetic.csv", index=False)
weather_external.to_csv("data/raw/weather_external_synthetic.csv", index=False)

# Generate simple tide data (using sine wave for demonstration)
print("\n[3/3] Generating tide data...")
import pandas as pd
import numpy as np

tide_data = pd.DataFrame({
    "time": water_quality["time"],
    "tide_height": 0.5 + 0.4 * np.sin(2 * np.pi * np.arange(len(water_quality)) / 24.8)
})
print(f"  ✓ Generated {len(tide_data)} tide records")
tide_data.to_csv("data/raw/tide_data_synthetic.csv", index=False)

print(f"\n✓ Data generation complete!")
print(f"  Total records: {len(water_quality)}")
print(f"  Files saved to data/raw/")

# =============================================================================
# STEP 2: Merge and Clean Data
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: MERGING AND CLEANING DATA")
print("=" * 80)

from data_processing.merger import DataMerger

merger = DataMerger()

print("\nMerging all data sources...")
merged_data = merger.merge_all(
    water_quality=water_quality,
    weather_station=weather_station,
    weather_external=weather_external,
    tide_data=tide_data
)

print("\nCleaning merged data...")
cleaned_data = merger.clean_data(
    merged_data,
    handle_missing="interpolate",
    remove_outliers=True
)

print(f"\n✓ Merge and clean complete!")
print(f"  Final shape: {cleaned_data.shape}")
print(f"  Columns: {len(cleaned_data.columns)}")

# Save
cleaned_data.to_csv("data/processed/merged_dataset.csv", index=False)
print(f"  Saved to data/processed/merged_dataset.csv")

# =============================================================================
# STEP 3: Feature Engineering
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 80)

from data_processing.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()

print("\nCreating engineered features...")
featured_data = engineer.create_all_features(cleaned_data)

print(f"\n✓ Feature engineering complete!")
print(f"  Input features: {len(cleaned_data.columns)}")
print(f"  Output features: {len(featured_data.columns)}")
print(f"  New features created: {len(engineer.get_feature_names())}")
print(f"\n  Sample features:")
for feat in engineer.get_feature_names()[:10]:
    print(f"    - {feat}")

# Save
featured_data.to_csv("data/processed/featured_dataset.csv", index=False)
print(f"\n  Saved to data/processed/featured_dataset.csv")

# =============================================================================
# STEP 4: Generate Safety Labels
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: GENERATING SAFETY LABELS")
print("=" * 80)

from labeling.safety_classifier import SafetyClassifier

classifier = SafetyClassifier()

print("\nClassifying water conditions...")
labeled_data = classifier.classify(featured_data, label_col="safety_label")

print(f"\n✓ Safety classification complete!")
print(f"\n  Label Distribution:")
dist = classifier.get_label_distribution()
dist_pct = classifier.get_label_distribution_pct()
print(f"    SAFE:    {dist['SAFE']:5,} records ({dist_pct['SAFE']:.1f}%)")
print(f"    CAUTION: {dist['CAUTION']:5,} records ({dist_pct['CAUTION']:.1f}%)")
print(f"    DANGER:  {dist['DANGER']:5,} records ({dist_pct['DANGER']:.1f}%)")

# Show example classifications
print(f"\n  Example DANGER condition:")
if dist['DANGER'] > 0:
    danger_example = labeled_data[labeled_data["safety_label"] == 2].iloc[0]
    explanation = classifier.explain_classification(danger_example, 2)
    print(f"    {explanation}")

# Save
labeled_data.to_csv("data/processed/labeled_dataset.csv", index=False)
labeled_data[["time", "safety_label"]].to_csv("data/labels/safety_labels.csv", index=False)
print(f"\n  Saved to data/processed/labeled_dataset.csv")

# =============================================================================
# STEP 5: Run Experiment 1
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: RUNNING EXPERIMENT 1 (CORRELATION ANALYSIS)")
print("=" * 80)

from experiments.experiment_1_correlation import Experiment1

exp1 = Experiment1(labeled_data)

print("\nRunning full correlation analysis...")
results = exp1.run_full_analysis()

print(f"\n✓ Experiment 1 complete!")
print(f"  Results generated:")
print(f"    - Descriptive statistics")
print(f"    - Correlation matrix")
print(f"    - Feature importance rankings")
print(f"    - Group comparisons")
print(f"    - Temporal patterns")

# =============================================================================
# STEP 6: Generate Visualizations
# =============================================================================
print("\n" + "=" * 80)
print("STEP 6: GENERATING VISUALIZATIONS")
print("=" * 80)

import matplotlib.pyplot as plt

print("\nCreating visualizations...")

# Correlation heatmap
print("\n[1/3] Correlation heatmap...")
fig1 = exp1.plot_correlation_heatmap(figsize=(14, 12), top_n=25)
plt.savefig("notebooks/exp1_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: notebooks/exp1_correlation_heatmap.png")

# Feature importance
print("\n[2/3] Feature importance chart...")
fig2 = exp1.plot_feature_importance(figsize=(10, 8), top_n=20)
plt.savefig("notebooks/exp1_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: notebooks/exp1_feature_importance.png")

# Parameter distributions
print("\n[3/3] Parameter distribution plots...")
fig3 = exp1.plot_parameter_distributions(figsize=(15, 10))
plt.savefig("notebooks/exp1_parameter_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: notebooks/exp1_parameter_distributions.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)

print("\n📊 SUMMARY:")
print(f"  Data records: {len(labeled_data):,}")
print(f"  Date range: {labeled_data['time'].min()} to {labeled_data['time'].max()}")
print(f"  Features: {len(labeled_data.columns)}")
print(f"  Safety labels: {dist['SAFE']:,} SAFE, {dist['CAUTION']:,} CAUTION, {dist['DANGER']:,} DANGER")

print("\n📁 OUTPUT FILES:")
print("  Data:")
print("    - data/raw/ (4 CSV files)")
print("    - data/processed/ (3 CSV files)")
print("    - data/labels/ (1 CSV file)")
print("  Visualizations:")
print("    - notebooks/ (3 PNG files)")

print("\n🎯 NEXT STEPS:")
print("  1. Review visualizations in notebooks/")
print("  2. Run Experiments 2 & 3 (classification and clustering)")
print("  3. Use labeled_dataset.csv for modeling")
print("  4. Include visualizations in your paper")

print(f"\n✓ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
