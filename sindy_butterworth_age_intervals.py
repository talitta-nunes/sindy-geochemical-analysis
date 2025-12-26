"""
SINDy Model Reproduction Script for Ordovician Biogeochemistry
Paper: "Data-Driven Biogeochemical Dynamics: Extracting data-driven
 differential model from Complex Geological Datasets"
Submitted to Computers & Geosciences

Description:
This script reproduces the governing equations and time-series simulations 
presented in the manuscript using the Sparse Identification of Nonlinear 
Dynamics (SINDy) algorithm. It applies specific filtering parameters 
optimized for each geological time interval.

Usage:
    Ensure 'DATA.csv' is in the same directory.
    Run: python sindy-geochemical-analysis.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pysindy as ps
from scipy.signal import butter, filtfilt

# --- 1. GENERAL CONFIGURATION ---
# Global Threshold: Since StandardScaler is NOT used to preserve interpretability,
# coefficients can be small. We use 1e-6 to avoid cutting off subtle dynamics.
GLOBAL_THRESHOLD = 1e-6 

# --- 2. INTERVAL CONFIGURATION DICTIONARY ---
# Exact pairs of cutoff/order optimized for each time interval based on data density.
# Format: (Start_Ma, End_Ma): {'cutoff': value, 'order': value}

INTERVAL_CONFIGS = {
    INTERVAL_CONFIGS = {
    (440, 445): {'cutoff': 0.10, 'order': 2},
    (445, 448): {'cutoff': 0.10, 'order': 4},
    (448, 452): {'cutoff': 0.10, 'order': 2},
    (452, 458): {'cutoff': 0.10, 'order': 1},
    (458, 465): {'cutoff': 0.15, 'order': 2},
    (465, 467): {'cutoff': 0.10, 'order': 1},
    (467, 470): {'cutoff': 0.30, 'order': 1},
    (470, 473): {'cutoff': 0.10, 'order': 1},
    (473, 480): {'cutoff': 0.10, 'order': 4},
    (480, 483): {'cutoff': 0.10, 'order': 1},
    (483, 488): {'cutoff': 0.10, 'order': 1},
    
}

def apply_butterworth_and_sindy(cutoff, order, start_age, end_age):
    # Load and prepare the data
    df = pd.read_csv("DATA_IRON_ONLY.csv", dtype=float)
    df.columns = ['toc', 'age', 'ironspec', 'pyrite', 'p']
    df.sort_values('age', inplace=True)
    df.interpolate(limit_direction="both", inplace=True)

    # Filter the data based on the age range
    filtered_df = df[(df['age'] >= start_age) & (df['age'] <= end_age)]
    filtered_df['rounded_age'] = filtered_df['age'].round(1)
    grouped_data = filtered_df.groupby('rounded_age').mean().reset_index()

    # Normalize phosphorus
    grouped_data['p'] = grouped_data['p'] / 10000

    # Butterworth filter
    def apply_butterworth_filter(data, cutoff, order):
        b, a = butter(order, cutoff, btype='low', analog=False)
        filtered = filtfilt(b, a, data)
        return np.clip(filtered, 0, None)

    for col in ['toc', 'pyrite', 'p', 'ironspec']:
        grouped_data[f'{col}_smooth'] = apply_butterworth_filter(
            grouped_data[col].values, cutoff, order
        )

    # Prepare data
    t = grouped_data["age"].values
    X = np.stack((
        grouped_data["toc_smooth"].values,
        grouped_data["pyrite_smooth"].values,
        grouped_data["p_smooth"].values
    ), axis=-1)

    # SINDy model
    dif = ps.SINDyDerivative(kind='kalman', alpha=0.3)
    optimizer = ps.STLSQ(threshold=1e-6)
    feature_names = ["toc", "pyrite", "p"]

    model = ps.SINDy(
        differentiation_method=dif,
        optimizer=optimizer,
        feature_names=feature_names
    )

    model.fit(X, t, ensemble=True)
    print(f"\nDerivatives for SINDy | Interval {start_age}-{end_age} Ma")
    model.print()

    simulated = model.simulate(X[0], t, integrator='solve_ivp')

    plot_simulation(
        t,
        grouped_data["toc_smooth"].values,
        grouped_data["pyrite_smooth"].values,
        grouped_data["p_smooth"].values,
        simulated
    )


def plot_simulation(t, data_toc, data_pyrite, data_p, simulated):
    labels = ['TOC', 'FePy/FeHR', 'Phosphorus']
    data_list = [data_toc, data_pyrite, data_p]

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for i, (data, sim, label) in enumerate(zip(data_list, simulated.T, labels)):
        axs[i].plot(t, data, label=f'DATA {label}')
        axs[i].plot(t, sim, '--', label=f'SINDY {label}')
        axs[i].set_ylabel(label)
        axs[i].legend()
        axs[i].grid(True, linestyle=':', alpha=0.6)

    axs[-1].set_xlabel('Age (Ma)')
    plt.tight_layout()
    plt.show()


# ==========================================================
# EXECUCTION
# ==========================================================

for (start_age, end_age), params in INTERVAL_CONFIGS.items():
    cutoff = params['cutoff']
    order = params['order']
    print(f"\nTesting interval {start_age}-{end_age} with cutoff={cutoff} and order={order}")
    apply_butterworth_and_sindy(cutoff, order, start_age, end_age)





