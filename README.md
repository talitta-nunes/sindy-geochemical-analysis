# sindy-geochemical-analysis
"SINDy analysis for Ordovician geochemical proxy data"
# SINDy-Geochemical-Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code and data for discovering governing equations from Ordovician geochemical proxy data using Sparse Identification of Nonlinear Dynamics (SINDy).

## Citation

If you use this code or data in your research, please cite:

```
[Manoel et al. (2025). Data-Driven Biogeochemical Dynamics: Extracting Governing Equations from Complex Geological Datasets. Computers & Geosciences]
```

## Description

The code implements a data-driven approach to identify nonlinear dynamical systems from geochemical time series data, including:

- **Total Organic Carbon (TOC)**
- **Highly Reactive Iron ratio (FePy/FeHR)**
- **Phosphorus concentrations**

Key features:
- Butterworth filtering for noise reduction with systematic parameter testing
- SINDy implementation using Kalman-smoothed derivatives
- Sparse regression (STLSQ) for parsimonious model selection
- Forward simulation and validation against observations

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/[seu-usuario]/sindy-geochemical-analysis.git
cd sindy-geochemical-analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running in Google Colab (Recommended)

1. Open the notebook in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[seu-usuario]/sindy-geochemical-analysis/blob/main/sindy_analysis.ipynb)

2. Upload the `DATA.csv` file when prompted

3. Run all cells sequentially

### Running Locally

```bash
jupyter notebook sindy_analysis.ipynb
```

Ensure the `DATA.csv` file is in the same directory as the notebook.

## Input Data

**File**: `DATA.csv`

**Columns**:
- `age`: Geological age (Ma)
- `toc`: Total Organic Carbon (%)
- `ironspec`: Iron speciation data
- `pyrite`: Pyrite content (FePy/FeHR ratio)
- `p`: Phosphorus concentration (normalized)

**Time range**: 440-488 Ma (Late Ordovician)

## Methodology

1. **Data Preprocessing**
   - Temporal sorting and interpolation
   - Age filtering (440-488 Ma)
   - Binning to 0.1 Ma resolution
   - Phosphorus normalization

2. **Signal Processing**
   - Butterworth low-pass filtering
   - Parameter sweep: cutoff ∈ {0.1, 0.2, 0.3, 0.4}, order ∈ {1, 2, 3, 4, 5}

3. **SINDy Model**
   - Differentiation: Kalman smoother (α=0.3)
   - Optimizer: STLSQ (threshold=1e-6)
   - Integration: solve_ivp

4. **Validation**
   - Visual comparison of data vs. simulation
   - Systematic evaluation across filter parameters

## Output

The code generates:
- Discovered differential equations for each variable
- Time series plots comparing observed vs. simulated data
- Results for all filter parameter combinations

## Repository Structure

```
sindy-geochemical-analysis/
│
├── sindy_analysis.ipynb          # Main Jupyter notebook
├── DATA.csv                      # Input dataset
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── LICENSE                       # License information
└── figures/                      # Output plots (generated)
```

## Dependencies

Main libraries:
- `pysindy>=1.7.0` - SINDy implementation
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `matplotlib>=3.4.0` - Plotting
- `scipy>=1.7.0` - Signal processing
- `scikit-learn>=0.24.0` - ML utilities
- `statsmodels>=0.13.0` - Statistical models
- `dask>=2021.10.0` - Parallel computing

See `requirements.txt` for complete list with version specifications.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Talitta Nunes Manoel  
Federal University of Minas Gerais  
Email: nunestalitta@gmail.com

## References

1. Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *Proceedings of the National Academy of Sciences*, 113(15), 3932-3937.

2. de Silva, B. M., et al. (2020). PySINDy: A Python package for the sparse identification of nonlinear dynamical systems from data. *Journal of Open Source Software*, 5(49), 2104.
