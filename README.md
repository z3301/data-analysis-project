# Forecasting Unsafe Water Conditions Along the California Coast

**Author:** Dan Zimmerman

**Email:** <dzimmerman2021@fau.edu>

**Course:** CAP5768 Introduction to Data Analytics, Fall 2025

**Instructor:** Dr. Fernando Koch

**Institution:** Florida Atlantic University

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/z3301/data-analysis-project/blob/main/notebooks/California_Coastal_Water_Quality_Analysis.ipynb)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/danzimmerman/california-coastal-water-quality-analysis)

---

## Project Overview

This research investigates whether unsafe water conditions at California coastal sites can be predicted in advance using oceanographic sensor data. Lab-based water-quality advisories typically lag real conditions by 24-48 hours, limiting the ability to issue proactive warnings. This project develops predictive models using nearly three years of real sensor data from three California coastal monitoring stations spanning a latitude gradient from Monterey Bay to San Diego.

### Research Question

**Can unsafe water conditions at California coastal sites be predicted in advance using oceanographic sensor data and tidal information?**

---

## Key Findings

Analysis of 75,000+ hourly observations revealed a striking **latitude gradient** in water quality drivers:

| Site | Latitude | Primary Driver | Correlation | DANGER Events |
|------|----------|----------------|-------------|---------------|
| MLML Monterey Bay | 36.8 N | Nitrate (upwelling) | r = +0.76 | 9.8% |
| Scripps Pier | 32.9 N | Chlorophyll (algal blooms) | r = +0.67 | 3.7% |
| Newport Pier | 33.6 N | Chlorophyll (algal blooms) | r = +0.86 | 0.0% |

- **Northern California (Monterey Bay):** Driven by coastal upwelling events
- **Southern California (La Jolla, Newport):** Driven by algal bloom dynamics
- **Random Forest classification:** 87-99% accuracy across all sites
- **24-hour forecasting:** Feasible at all sites using lagged features

---

## Data Sources

### Primary Oceanographic Data

Data was obtained from the Central and Northern California Ocean Observing System (CeNCOOS) and Southern California Coastal Ocean Observing System (SCCOOS) via their ERDDAP data servers.

| Station | Provider | Parameters | Sampling | Records |
|---------|----------|------------|----------|---------|
| MLML Monterey Bay | CeNCOOS | 11 parameters | ~5 min | 501,467 |
| Scripps Pier | SCCOOS | 13 parameters | ~3 min | 501,467 |
| Newport Pier | SCCOOS | 9 parameters | ~4 min | 501,467 |

**Date Range:** January 1, 2023 - November 30, 2025

### NOAA Tide Data

Tide predictions were obtained from NOAA Center for Operational Oceanographic Products and Services (CO-OPS) via their public API.

| Station ID | Station Name | Location |
|------------|--------------|----------|
| 9413450 | Monterey, CA | 36.605 N, 121.888 W |
| 9410230 | La Jolla (Scripps Pier), CA | 32.867 N, 117.257 W |
| 9410580 | Newport Beach, CA | 33.603 N, 117.883 W |

---

## References

[1] Central and Northern California Ocean Observing System, "MLML Seawater Intake Mooring," CeNCOOS ERDDAP Server, 2023-2025. [Online]. Available: [erddap.cencoos.org](https://erddap.cencoos.org/erddap/tabledap/mlml-mlml-sea.html)

[2] Southern California Coastal Ocean Observing System, "Scripps Pier Automated Shore Station," SCCOOS ERDDAP Server, 2023-2025. [Online]. Available: [erddap.sccoos.org](https://erddap.sccoos.org/erddap/tabledap/autoss_scripps_pier.html)

[3] Southern California Coastal Ocean Observing System, "Newport Pier Automated Shore Station," SCCOOS/CeNCOOS ERDDAP Server, 2023-2025. [Online]. Available: [erddap.cencoos.org](https://erddap.cencoos.org/erddap/tabledap/ism-cencoos-newport-pier-automated.html)

[4] National Oceanic and Atmospheric Administration, "CO-OPS API for Data Retrieval," NOAA Tides and Currents, 2023-2025. [Online]. Available: [tidesandcurrents.noaa.gov](https://api.tidesandcurrents.noaa.gov/api/prod/)

[5] California State Water Resources Control Board, "California Ocean Plan: Water Quality Control Plan for Ocean Waters of California," 2019. [Online]. Available: [waterboards.ca.gov](https://www.waterboards.ca.gov/water_issues/programs/ocean/docs/cop2019.pdf)

[6] U.S. Environmental Protection Agency, "National Recommended Water Quality Criteria - Aquatic Life Criteria Table," 2023. [Online]. Available: [epa.gov](https://www.epa.gov/wqc/national-recommended-water-quality-criteria-aquatic-life-criteria-table)

---

## Project Structure

```text
data-analysis-project/
├── README.md
├── PAPER_SECTIONS_CALIFORNIA.md      # Paper sections for submission
├── requirements.txt
├── notebooks/
│   └── California_Coastal_Water_Quality_Analysis.ipynb
├── data/
│   ├── raw/                          # Original NetCDF/CSV files
│   ├── real/                         # Downloaded sensor data
│   └── processed/                    # Featured datasets (.csv.gz)
└── src/
    ├── data_collection/
    │   ├── mlml_data_loader.py       # MLML Monterey Bay
    │   ├── scripps_data_loader.py    # Scripps Pier
    │   ├── newport_data_loader.py    # Newport Pier
    │   └── california_tides.py       # NOAA tide fetcher
    ├── data_processing/
    │   ├── feature_engineering_mlml.py
    │   └── feature_engineering_newport.py
    ├── labeling/
    │   ├── safety_classifier_mlml.py
    │   ├── safety_classifier_scripps.py
    │   └── safety_classifier_newport.py
    └── experiments/
        └── experiment_1_correlation.py
```

---

## Safety Classification System

Water conditions are classified into three categories based on California Ocean Plan [5] and EPA [6] thresholds:

### DANGER (Label = 2)

Immediate health hazards:

- Dissolved oxygen < 5.0 mg/L (severe hypoxia)
- pH < 7.0 or > 8.5 (extreme values)
- Chlorophyll > 50 ug/L (harmful algal bloom)
- Nitrate > 35 umol/L (MLML only, upwelling stress)

### CAUTION (Label = 1)

Elevated risk conditions:

- Dissolved oxygen 5.0-6.5 mg/L
- pH 7.0-7.5 or 8.3-8.5
- Chlorophyll 20-50 ug/L
- Nitrate 25-35 umol/L (MLML only)

### SAFE (Label = 0)

All parameters within acceptable ranges for recreational water contact.

---

## Quick Start

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badge at the top of this README. The notebook will:

1. Automatically clone this repository
2. Load the compressed datasets
3. Run all analyses

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/z3301/data-analysis-project.git
cd data-analysis-project

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook
jupyter notebook notebooks/California_Coastal_Water_Quality_Analysis.ipynb
```

### Option 3: Regenerate Data from Source

```bash
# Run site-specific pipelines
python run_mlml_pipeline.py
python run_scripps_pipeline.py
python run_newport_pipeline.py
```

---

## Methodology

### Experiment 1: Environmental Drivers

Identifies which variables most influence unsafe water conditions at each site using correlation analysis and feature importance ranking.

### Experiment 2: Predictive Classification

Trains Logistic Regression, Naive Bayes, and Random Forest models to predict safety labels using lagged features (1h, 6h, 12h, 24h).

### Experiment 3: Clustering Analysis

Applies K-means clustering to discover natural water quality regimes and their alignment with safety classifications.

---

## Results Summary

### Model Performance (Random Forest)

| Site | Accuracy | F1-Score | DANGER Recall |
|------|----------|----------|---------------|
| MLML Monterey | 87.1% | 0.868 | 71.6% |
| Scripps Pier | 95.9% | 0.955 | N/A (rare) |
| Newport Pier | 98.8% | 0.988 | N/A (none) |

### Key Insights

1. **Site-specific models are essential** - different oceanographic drivers at different latitudes
2. **Lagged features retain predictive power** - 6-24 hour forecasting is feasible
3. **Upwelling vs. bloom dynamics** - fundamentally different prediction approaches needed for Northern vs. Southern California

---

## Technologies

- Python 3.10+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- h5py (for NetCDF files)
- NOAA CO-OPS API

---

## Reproducibility

- **Random seed:** 42
- **Train/test split:** 80/20 temporal split (no shuffle)
- **All timestamps:** UTC

---

## License

This project is submitted as coursework for CAP4773/CAP5768 at Florida Atlantic University.

---

## Acknowledgments

- Dr. Fernando Koch, Course Instructor
- CeNCOOS and SCCOOS for open oceanographic data access
- NOAA CO-OPS for tide prediction data

---

## Contact

Dan Zimmerman
Florida Atlantic University
Email: <dzimmerman2021@fau.edu>

---

Last Updated: November 2025
