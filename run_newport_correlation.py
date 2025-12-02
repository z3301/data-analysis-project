#!/usr/bin/env python3
"""
Newport Pier Correlation Analysis
==================================
Standalone correlation analysis without sklearn dependency.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def run_correlation_analysis():
    """Run correlation analysis on Newport Pier data."""

    print("=" * 70)
    print("NEWPORT PIER CORRELATION ANALYSIS")
    print("=" * 70)

    # Load the featured dataset
    data_path = Path("data/processed/newport_featured_dataset.csv")
    print(f"\nLoading: {data_path}")

    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])

    print(f"Records: {len(df):,}")
    print(f"Features: {df.shape[1]}")

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove time-based columns from correlation
    exclude = ['hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend']
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    print(f"Numeric columns for correlation: {len(numeric_cols)}")

    # Compute correlation matrix
    print("\nComputing correlation matrix...")
    corr_matrix = df[numeric_cols].corr()

    # Get correlations with safety_label
    if 'safety_label' in corr_matrix.columns:
        target_corr = corr_matrix['safety_label'].sort_values(ascending=False)
        target_corr = target_corr[target_corr.index != 'safety_label']

        print("\n" + "=" * 70)
        print("TOP 15 POSITIVE CORRELATIONS WITH UNSAFE CONDITIONS")
        print("(Higher values → More likely unsafe)")
        print("=" * 70)
        for i, (feature, corr) in enumerate(target_corr.head(15).items()):
            print(f"{i+1:2}. {feature:45} {corr:+.3f}")

        print("\n" + "=" * 70)
        print("TOP 15 NEGATIVE CORRELATIONS WITH UNSAFE CONDITIONS")
        print("(Lower values → More likely unsafe)")
        print("=" * 70)
        for i, (feature, corr) in enumerate(target_corr.tail(15).items()):
            print(f"{i+1:2}. {feature:45} {corr:+.3f}")

    # Create visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    notebooks_dir = Path("notebooks")

    # 1. Correlation heatmap for top features
    print("\nCreating correlation heatmap...")
    create_correlation_heatmap(corr_matrix, target_corr, notebooks_dir)

    # 2. Feature importance bar chart
    print("Creating feature importance chart...")
    create_feature_importance_chart(target_corr, notebooks_dir)

    # 3. Parameter distributions by safety class
    print("Creating parameter distributions...")
    create_distribution_plots(df, notebooks_dir)

    # Generate summary
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS COMPLETE")
    print("=" * 70)

    # Key findings
    print("\nKEY FINDINGS:")
    print("-" * 50)

    # Top predictors
    top_positive = target_corr.head(5)
    top_negative = target_corr.tail(5)

    print("\nStrongest Positive Correlations (higher = more unsafe):")
    for feature, corr in top_positive.items():
        print(f"  • {feature}: r = {corr:+.3f}")

    print("\nStrongest Negative Correlations (lower = more unsafe):")
    for feature, corr in top_negative.items():
        print(f"  • {feature}: r = {corr:+.3f}")

    return corr_matrix, target_corr


def create_correlation_heatmap(corr_matrix, target_corr, output_dir):
    """Create correlation heatmap for top features."""

    # Get top 20 features by absolute correlation with target
    top_features = target_corr.abs().sort_values(ascending=False).head(20).index.tolist()
    top_features = ['safety_label'] + top_features

    # Subset correlation matrix
    subset_corr = corr_matrix.loc[top_features, top_features]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(subset_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=12)

    # Set ticks
    ax.set_xticks(range(len(subset_corr.columns)))
    ax.set_yticks(range(len(subset_corr.index)))
    ax.set_xticklabels(subset_corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(subset_corr.index, fontsize=8)

    ax.set_title('Correlation Heatmap - Top 20 Features\nNewport Pier Water Quality',
                fontsize=14, pad=20)

    plt.tight_layout()

    output_path = output_dir / 'newport_exp1_correlation_heatmap.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_feature_importance_chart(target_corr, output_dir):
    """Create feature importance bar chart."""

    # Get top 20 features by absolute correlation
    top_features = target_corr.abs().sort_values(ascending=False).head(20)

    # Get actual correlation values (with sign)
    feature_corrs = target_corr[top_features.index]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['red' if c > 0 else 'blue' for c in feature_corrs]

    y_pos = range(len(feature_corrs))
    ax.barh(y_pos, feature_corrs, color=colors, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_corrs.index, fontsize=9)
    ax.invert_yaxis()

    ax.set_xlabel('Correlation with Unsafe Conditions', fontsize=12)
    ax.set_title('Top 20 Features for Predicting Unsafe Conditions\nNewport Pier Water Quality',
                fontsize=14, pad=15)

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlim(-1, 1)

    # Add legend
    ax.text(0.95, 0.02, 'Red = Higher → More Unsafe\nBlue = Lower → More Unsafe',
           transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
           horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    output_path = output_dir / 'newport_exp1_feature_importance.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_distribution_plots(df, output_dir):
    """Create distribution plots for key parameters by safety class."""

    # Key parameters
    params = ['chlorophyll', 'dissolved_oxygen_pct', 'pH', 'water_temp']
    available = [p for p in params if p in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = {0: 'green', 1: 'orange', 2: 'red'}
    labels = {0: 'SAFE', 1: 'CAUTION', 2: 'DANGER'}

    for i, param in enumerate(available[:4]):
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

    fig.suptitle('Parameter Distributions by Safety Classification\nNewport Pier',
                fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'newport_exp1_parameter_distributions.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    corr_matrix, target_corr = run_correlation_analysis()
