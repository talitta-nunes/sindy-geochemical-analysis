# SINDy Geochemical Analysis

Enhanced SINDy analysis for Ordovician geochemical proxy data.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains a reproducible Python workflow for discovering reduced-order differential models from Ordovician sedimentary geochemical proxy time series using Sparse Identification of Nonlinear Dynamics (SINDy).

The workflow was developed for the manuscript:

**Data-Driven Biogeochemical Dynamics: Extracting Differential Models from Complex Geological Datasets**

The main script includes data preprocessing, outlier treatment, interval-specific filtering, SINDy model fitting, cross-validation, coefficient export, model-performance metrics, and figure generation.

## Description

The code implements a data-driven approach to infer differential equations from geochemical proxy records, focusing on three variables:

* Total Organic Carbon (TOC)
* Pyrite iron ratio / highly reactive iron proxy (FePy/FeHR)
* Phosphorus concentration (P)

The workflow includes:

* Age-windowed SINDy modeling
* Butterworth filtering with interval-specific parameters
* Outlier removal before filtering
* Kalman-smoothed derivative estimation
* Sparse regression using Sequential Thresholded Least Squares (STLSQ)
* Polynomial feature libraries
* Ensemble SINDy fitting
* Forward simulation using `solve_ivp`
* Time-series cross-validation with adaptive folds
* Separate tracking of seen and unseen R²
* RMSE calculation
* Coefficient export
* Heatmaps and publication-ready figures

## Installation

### Prerequisites

* Python 3.8 or higher
* pip package manager

### Setup

Clone the repository:

```bash
git clone https://github.com/talitta-nunes/sindy-geochemical-analysis.git
cd sindy-geochemical-analysis
```

Create a virtual environment:

```bash
python -m venv env
```

Activate the environment:

```bash
# Linux / macOS
source env/bin/activate

# Windows
env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The input data file should be placed in the repository root directory.

The expected input file is:

```text
DATA.csv
```

Run the main analysis script:

run:

```bash
python sindy_geochemical_analysis_enhanced.py
```

## Input Data

Expected input file:

```text
DATA.csv
```

The dataset should contain the following variables:

* `age`: geological age in Ma
* `toc`: total organic carbon
* `pyrite`: FePy/FeHR proxy
* `p`: phosphorus concentration
* `ironspec`: iron speciation proxy, if present in the raw file

The modeled variables are:

* TOC
* FePy/FeHR
* P

## Methodology

The workflow follows these main steps.

### 1. Data preprocessing

* Sorting by geological age
* Interpolation of missing values
* Selection of Ordovician age intervals
* Grouping by rounded age
* Phosphorus normalization

### 2. Outlier treatment

Outliers are removed before filtering.

Available methods include:

* Standard IQR-based filtering
* Strict IQR-based filtering
* Percentile-based filtering
* Z-score filtering
* Aggressive multi-pass filtering

### 3. Signal smoothing

* Butterworth low-pass filtering
* Interval-specific cutoff and filter order
* Padding correction for short time series

### 4. SINDy model fitting

* Kalman-smoothed derivative estimation
* STLSQ sparse regression
* Polynomial feature library
* Ensemble fitting
* Selection of best-performing models

### 5. Validation

* Forward simulation
* In-sample performance metrics
* Time-series cross-validation for unseen-data evaluation
* R² calculation
* RMSE calculation
* Separate tracking of seen and unseen model performance

### 6. Output generation

* Differential equations
* Model coefficients
* Performance metrics
* Observed vs. simulated time-series plots
* Heatmaps and summary figures

## Output

The script creates the following directories:

```text
outputs/
outputs/TOC/
outputs/FePy_FeHr/
outputs/P/
results/
```

Generated files may include:

* Time-series plots comparing observed and simulated data
* Outlier visualization plots
* CSV files with model coefficients
* CSV files with R² and RMSE metrics
* Summary figures for model comparison
* Heatmaps of model coefficients

## Repository Structure

```text
sindy-geochemical-analysis/
│
├── sindy_geochemical_analysis_enhanced.py      # Main SINDy analysis script
├── DATA.csv                           # Input dataset
├── requirements.txt                   # Python dependencies
├── README.md                          # Repository documentation
├── LICENSE                            # License information
│
├── outputs/                           # Generated figures
│   ├── TOC/
│   ├── FePy_FeHr/
│   └── P/
│
└── results/                           # Generated metrics and coefficients
```

If the current script is named `sindy_geochemical_analysis_enhanced.py`, it should either be renamed to `sindy_geochemical_analysis_enhanced.py` or the usage command should be updated accordingly.

## Dependencies

Main libraries:

* `pysindy==1.7.5`
* `numpy==1.26.4`
* `scipy==1.11.4`
* `pandas==2.1.4`
* `scikit-learn==1.3.2`
* `matplotlib==3.8.0`
* `seaborn`
* `statsmodels`

See `requirements.txt` for the complete list.

## Configuration

Important configuration parameters are defined inside the main script:

```python
GLOBAL_THRESHOLD = 1e-6
TOP_N_RESULTS = 2
POLY_DEGREE = 1
MAX_COEFFICIENT = 100000.0

MIN_FOLDS = 2
MAX_FOLDS = 4
MIN_TRAIN_PTS = 2
PTS_PER_FOLD = 5

OUTLIER_METHOD = "standard"
```

The default value used to reproduce the final selected models in the manuscript is:

```python
POLY_DEGREE = 1
```

To reproduce the second-degree polynomial-library comparison reported in the manuscript appendix, set:

```python
POLY_DEGREE = 2
```

When `POLY_DEGREE = 2`, the model includes nonlinear and interaction terms such as:

```text
C², CX, CP, X², XP, P²
```

The age intervals and Butterworth filter parameters are defined in:

```python
INTERVAL_CONFIGS = {
    (440, 445): {"cutoff": 0.10, "order": 2},
    (445, 448): {"cutoff": 0.10, "order": 2},
    (448, 452): {"cutoff": 0.10, "order": 1},
    (452, 458): {"cutoff": 0.05, "order": 1},
    (458, 462): {"cutoff": 0.10, "order": 2},
    (462, 467): {"cutoff": 0.10, "order": 1},
    (467, 473): {"cutoff": 0.05, "order": 1},
    (473, 480): {"cutoff": 0.05, "order": 2},
    (480, 483): {"cutoff": 0.10, "order": 1},
    (483, 488): {"cutoff": 0.10, "order": 2},
}
```

## Notes on Reproducibility

This repository is organized around a single main analysis script:

```text
sindy_geochemical_analysis_enhanced.py
```

The script contains the full enhanced SINDy workflow, including preprocessing, filtering, outlier removal, model fitting, validation, coefficient export, and figure generation.

By default, the repository should be configured to reproduce the first-order SINDy models reported as the final models in the manuscript.

Because the workflow includes configurable preprocessing and validation steps, the final equations and performance metrics depend on the selected configuration.

For the final manuscript configuration, use:

```python
POLY_DEGREE = 1
```

For the polynomial-library comparison discussed in the Appendix, use:

```python
POLY_DEGREE = 2
```

## Important Notes

* The input data file must be placed in the same directory as the script.
* The script expects the input file to be named `DATA.csv`.
* If your data file has another name, either rename it to `DATA.csv` or update the file path inside the script.
* Dependencies should be installed through `requirements.txt`.
* Generated folders such as `outputs/` and `results/` can be ignored by Git if they are large or automatically generated.
* For clarity and reproducibility, the main script should be named `sindy_geochemical_analysis_enhanced.py`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

Talitta Nunes Manoel
Federal University of Minas Gerais
Email: [nunestalitta@gmail.com](mailto:nunestalitta@gmail.com)

## References

1. Brunton, S. L., Proctor, J. L., and Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *Proceedings of the National Academy of Sciences of the United States of America*, 113(15), 3932–3937. https://doi.org/10.1073/pnas.1517384113

2. Kaptanoglu, A. A., de Silva, B. M., Fasel, U., Kaheman, K., Goldschmidt, A. J., Callaham, J. L., Delahunt, C. B., Nicolaou, Z. G., Champion, K., Loiseau, J.-C., Kutz, J. N., and Brunton, S. L. (2022). PySINDy: A comprehensive Python package for robust sparse system identification. *Journal of Open Source Software*, 7(69), 3994. https://doi.org/10.21105/joss.03994

3. de Silva, B. M., Champion, K., Quade, M., Loiseau, J. C., Kutz, J. N., and Brunton, S. L. (2020). PySINDy: A Python package for the sparse identification of nonlinear dynamical systems from data. *Journal of Open Source Software*, 5(49), 2104. https://doi.org/10.21105/joss.02104
