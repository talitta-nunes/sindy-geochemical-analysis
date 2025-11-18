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
    (440, 445): {'cutoff': 0.10, 'order': 3},
    (445, 448): {'cutoff': 0.10, 'order': 4},
    (448, 452): {'cutoff': 0.10, 'order': 2},
    (452, 458): {'cutoff': 0.10, 'order': 1},
    (458, 465): {'cutoff': 0.10, 'order': 5},
    (465, 467): {'cutoff': 0.10, 'order': 1},
    (467, 470): {'cutoff': 0.08, 'order': 1},
    (470, 473): {'cutoff': 0.10, 'order': 1},
    (473, 480): {'cutoff': 0.10, 'order': 4},
    (480, 483): {'cutoff': 0.10, 'order': 1},
    (483, 488): {'cutoff': 0.10, 'order': 1},
}

def apply_butterworth_filter(data, cutoff, order):
    """Applies a low-pass Butterworth filter to smooth geological data."""
    b, a = butter(order, cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    filtered_data[filtered_data < 0] = 0  # Ensures physical non-negativity
    return filtered_data

def run_analysis():
    print("--- Starting SINDy Reproduction Script (Raw Data Mode) ---")
    
    # 1. Load and Prepare Data
    try:
        df = pd.read_csv("DATA.csv")
        # Ensure consistent column naming
        df.columns = ['toc', 'age', 'pyrite', 'p']
    except Exception as e:
        print(f"Error reading file: {e}")
        print("Please ensure 'DATA_IRON_ONLY.csv' is in the script directory.")
        return

    # Interpolation to handle missing geological steps
    df = df.sort_values('age')
    df.interpolate(limit_direction="both", inplace=True)

    # 2. Iterate through defined time intervals
    for (start_time, end_time), params in INTERVAL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f" PROCESSING INTERVAL: {start_time} - {end_time} Ma")
        print(f" Filter Parameters: Cutoff={params['cutoff']}, Order={params['order']}")
        print(f"{'='*60}")

        # Filter data for the specific interval
        mask = (df['age'] >= start_time) & (df['age'] <= end_time)
        filtered_df = df[mask].copy()

        if len(filtered_df) < 5:
            print(" [!] Insufficient data points for this interval. Skipping.")
            continue

        # Grouping by rounded age to handle duplicates/noise
        filtered_df['rounded_age'] = filtered_df['age'].round(1)
        grouped_data = filtered_df.groupby('rounded_age').mean().reset_index()
        
        # --- CRITICAL: Manual Normalization ---
        # Since StandardScaler is disabled, we manually scale Phosphorus
        # to bring it to a comparable numerical range.
        grouped_data['p'] = grouped_data['p'] / 10000 

        # Apply Smoothing Filter
        cutoff = params['cutoff']
        order = params['order']
        
        grouped_data['toc_smooth'] = apply_butterworth_filter(grouped_data['toc'].values, cutoff, order)
        grouped_data['pyrite_smooth'] = apply_butterworth_filter(grouped_data['pyrite'].values, cutoff, order)
        grouped_data['p_smooth'] = apply_butterworth_filter(grouped_data['p'].values, cutoff, order)

        # Prepare Vectors for SINDy
        t = grouped_data["age"].values
        data_toc = grouped_data["toc_smooth"].values
        data_py = grouped_data['pyrite_smooth'].values
        data_p = grouped_data['p_smooth'].values
        
        # Stack Data (X Matrix) - No extra normalization applied here
        X = np.stack((data_toc, data_py, data_p), axis=-1)

        # Configure SINDy Model
        dif = ps.SINDyDerivative(kind='kalman', alpha=0.3)
        feature_names = ["toc", "pyrite", "p"]
        
        # Using global threshold for raw data
        custom_optimizer = ps.STLSQ(threshold=GLOBAL_THRESHOLD)
        
        model = ps.SINDy(
            differentiation_method=dif, 
            feature_names=feature_names, 
            optimizer=custom_optimizer,
            feature_library=ps.PolynomialLibrary(degree=2, include_bias=True)
        )

        try:
            # Fit Model
            model.fit(X, t, ensemble=True, quiet=True)
            
            print("\n>>> Discovered Governing Equations (Real Scale):")
            model.print()

            # Simulate Model
            x0 = X[0]
            simulated_data = model.simulate(x0, t, integrator='solve_ivp')

            # Safety Check (Overflow/NaN)
            if np.any(np.isinf(simulated_data)) or np.any(np.isnan(simulated_data)):
                raise ValueError("Simulation resulted in infinity (numerical instability).")

            # Plot Results
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            
            # TOC Plot
            axs[0].plot(t, data_toc, 'k-', label='Observed Data (Smoothed)')
            axs[0].plot(t, simulated_data[:, 0], 'r--', label='SINDy Simulation')
            axs[0].set(ylabel='TOC (wt. %)')
            axs[0].legend()
            axs[0].grid(True, linestyle=':', alpha=0.6)

            # FePy/FeHR Plot
            axs[1].plot(t, data_py, 'k-', label='Observed Data')
            axs[1].plot(t, simulated_data[:, 1], 'r--', label='SINDy Simulation')
            axs[1].set(ylabel='FePy/FeHR')
            axs[1].legend()
            axs[1].grid(True, linestyle=':', alpha=0.6)

            # Phosphorus Plot
            axs[2].plot(t, data_p, 'k-', label='Observed Data')
            axs[2].plot(t, simulated_data[:, 2], 'r--', label='SINDy Simulation')
            axs[2].set(ylabel='Phosphorus (Scaled)')
            axs[2].legend()
            axs[2].grid(True, linestyle=':', alpha=0.6)

            axs[-1].set(xlabel='Age (Ma)')
            plt.suptitle(f'Interval {start_time}-{end_time} Ma (Cutoff={cutoff})')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f" [!] Error in this interval: {e}")
            print("     Suggestion: Try adjusting the cutoff for this specific interval.")

if __name__ == "__main__":
    run_analysis()


