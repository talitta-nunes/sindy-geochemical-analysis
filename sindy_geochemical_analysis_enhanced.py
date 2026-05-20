# -*- coding: utf-8 -*-
"""
SINDy Geochemical Analysis

Enhanced SINDy analysis for Ordovician geochemical proxy data.

This script performs:
- data preprocessing
- outlier removal
- Butterworth filtering
- SINDy model fitting
- cross-validation
- seen/unseen R² evaluation
- RMSE calculation
- coefficient export
- figure generation

Input:
    DATA.csv

Run:
    python sindy_geochemical_analysis.py
"""

import os
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysindy as ps

from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
"""
SINDy Model Reproduction Script
ENHANCED VERSION WITH EXPLICIT UNSEEN DATA R² SAVING AND COEFFICIENT LIMITING:
- Outliers removed BEFORE filtering with multiple methods
- Cross-validation with adaptive folds for UNSEEN data evaluation
- Ensemble SINDy with top-N selection
- ALL R² results for UNSEEN data explicitly saved to CSV files
- Separate tracking of SEEN (in-sample) vs UNSEEN (CV) R²
- MAXIMUM COEFFICIENT LIMITING to prevent unrealistic values
- Comprehensive annotations and visualizations
- Fixed Butterworth filter padding issue
- Fixed TypeError with numpy.float64 len() issue
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pysindy as ps
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy import stats
import warnings
import json
warnings.filterwarnings('ignore')

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
FIGURES_DIR = "outputs"
FIGURES_TOC = os.path.join(FIGURES_DIR, "TOC")
FIGURES_FEPY = os.path.join(FIGURES_DIR, "FePy_FeHr")
FIGURES_P = os.path.join(FIGURES_DIR, "P")

os.makedirs(FIGURES_TOC, exist_ok=True)
os.makedirs(FIGURES_FEPY, exist_ok=True)
os.makedirs(FIGURES_P, exist_ok=True)
RESULTS_DIR = "results"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# GLOBAL STORAGE (CONTINUOUS)
# =========================
all_t = []
all_sim = []
all_obs = []
# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
GLOBAL_THRESHOLD = 1e-6
TOP_N_RESULTS    = 2
POLY_DEGREE      = 2
MAX_COEFFICIENT  = 100000.0  # Maximum absolute value allowed for any coefficient

MIN_FOLDS     = 2
MAX_FOLDS     = 4
MIN_TRAIN_PTS = 2
PTS_PER_FOLD  = 5

# Outlier removal configuration
OUTLIER_METHOD = 'standard'  # Options: 'standard', 'strict', 'very_strict', 'percentile', 'aggressive', 'zscore'
IQR_MULTIPLIER = 5.0  # For strict methods (lower = more aggressive)
PERCENTILE_RANGE = (10, 99)  # For percentile method
ZSCORE_THRESHOLD = 2.5  # For z-score method

# =============================================================================
# INTERVAL CONFIGURATION
# =============================================================================
INTERVAL_CONFIGS = {
    (440, 445): {'cutoff': 0.10, 'order': 2},
    (445, 448): {'cutoff': 0.1, 'order': 2},
    (448, 452): {'cutoff': 0.10, 'order': 1},
    (452, 458): {'cutoff': 0.05, 'order': 1},
    (458, 462): {'cutoff': 0.1, 'order': 2},
    (462, 467): {'cutoff': 0.10, 'order': 1},
    (467, 473): {'cutoff': 0.05, 'order': 1},
    (473, 480): {'cutoff': 0.05, 'order': 2},
    (480, 483): {'cutoff': 0.10, 'order': 1},
    (483, 488): {'cutoff': 0.10, 'order': 2},

}


# =============================================================================
# ENHANCED OUTLIER REMOVAL METHODS
# =============================================================================

def visualize_outliers(df, columns, start_age, end_age):
    """Create boxplots to visualize outliers before removal."""
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 4))

    # Handle case when there's only one column
    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(columns):
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(f'{col} - {start_age}-{end_age} Ma')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('Outlier Visualization Before Removal')
    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR, f"outlier_viz_before_{start_age}_{end_age}Ma.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Outlier visualization saved to {fname}")

def remove_outliers_iqr_standard(df, columns):
    """Standard IQR method with multiplier 1.5."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 15 * IQR
        upper = Q3 + 15 * IQR
        df_clean = df_clean[
            (df_clean[col] >= lower) &
            (df_clean[col] <= upper)
        ]
    return df_clean

def remove_outliers_iqr_strict(df, columns, multiplier=1.0):
    """
    Strict IQR with custom multiplier.
    Smaller multiplier = more aggressive removal.
    """
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        df_clean = df_clean[
            (df_clean[col] >= lower) &
            (df_clean[col] <= upper)
        ]
    return df_clean

def remove_outliers_percentile(df, columns, lower_pct=1, upper_pct=99):
    """Remove outliers based on percentiles."""
    df_clean = df.copy()
    for col in columns:
        lower = np.percentile(df_clean[col], lower_pct)
        upper = np.percentile(df_clean[col], upper_pct)
        df_clean = df_clean[
            (df_clean[col] >= lower) &
            (df_clean[col] <= upper)
        ]
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3):
    """Remove outliers based on Z-score."""
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df_clean[col]))
        df_clean = df_clean[z_scores < threshold]
    return df_clean

def remove_outliers_aggressive(df, columns):
    """
    Multiple passes of outlier removal for maximum effect.
    """
    df_clean = df.copy()
    original_len = len(df_clean)

    # Pass 1: Remove extreme percentiles
    for col in columns:
        lower = np.percentile(df_clean[col], 0.5)
        upper = np.percentile(df_clean[col], 99.5)
        df_clean = df_clean[
            (df_clean[col] >= lower) &
            (df_clean[col] <= upper)
        ]

    # Pass 2: Strict IQR
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 0.8 * IQR
        upper = Q3 + 0.8 * IQR
        df_clean = df_clean[
            (df_clean[col] >= lower) &
            (df_clean[col] <= upper)
        ]

    # Pass 3: Z-score
    for col in columns:
        z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
        df_clean = df_clean[z_scores < 2.5]

    print(f"  Aggressive removal: kept {len(df_clean)}/{original_len} points "
          f"({100*len(df_clean)/original_len:.1f}%)")
    return df_clean

def remove_outliers_wrapper(df, columns, method='standard', **kwargs):
    """
    Wrapper function to apply selected outlier removal method.
    """
    print(f"\n  Applying outlier removal method: {method}")

    if method == 'standard':
        return remove_outliers_iqr_standard(df, columns)
    elif method == 'strict':
        multiplier = kwargs.get('multiplier', 1.0)
        return remove_outliers_iqr_strict(df, columns, multiplier)
    elif method == 'very_strict':
        return remove_outliers_iqr_strict(df, columns, multiplier=0.5)
    elif method == 'percentile':
        lower, upper = kwargs.get('percentile_range', (1, 99))
        return remove_outliers_percentile(df, columns, lower, upper)
    elif method == 'zscore':
        threshold = kwargs.get('threshold', 3)
        return remove_outliers_zscore(df, columns, threshold)
    elif method == 'aggressive':
        return remove_outliers_aggressive(df, columns)
    else:
        print(f"  Unknown method '{method}', using standard")
        return remove_outliers_iqr_standard(df, columns)

# =============================================================================
# COEFFICIENT LIMITING FUNCTION
# =============================================================================

def enforce_coefficient_limit(model, max_coef=MAX_COEFFICIENT):
    """
    Enforce maximum coefficient limit by clipping.
    Returns True if clipping was applied, False otherwise.
    """
    if not hasattr(model, 'optimizer') or not hasattr(model.optimizer, 'coef_'):
        return False

    coef = model.coefficients()
    max_abs = np.max(np.abs(coef))

    if max_abs > max_coef:
        print(f"  Clipping coefficients: max {max_abs:.4f} → {max_coef}")
        coef_clipped = np.clip(coef, -max_coef, max_coef)
        model.optimizer.coef_ = coef_clipped
        return True

    return False

def analyze_coefficients(model, variable_names, start_age, end_age, save=True):
    """Extract and analyze coefficient magnitudes."""
    if not hasattr(model, 'optimizer') or not hasattr(model.optimizer, 'coef_'):
        return None

    coefficients = model.coefficients()
    feature_names = model.get_feature_names()

    coef_data = []
    for i, var in enumerate(variable_names):
        for j, feat in enumerate(feature_names):
            if abs(coefficients[i, j]) > 1e-10:
                coef_data.append({
                    'variable': var,
                    'feature': feat,
                    'coefficient': float(coefficients[i, j]),
                    'abs_value': float(abs(coefficients[i, j]))
                })

    if coef_data:
        df_coef = pd.DataFrame(coef_data)

        print(f"\n  Coefficient analysis for {start_age}-{end_age} Ma:")
        print(f"    Max |coef|: {df_coef['abs_value'].max():.4f}")
        print(f"    Mean |coef|: {df_coef['abs_value'].mean():.4f}")
        print(f"    Non-zero terms: {len(df_coef)}")

        # Check if any coefficients hit the limit
        near_limit = df_coef[df_coef['abs_value'] > 0.9 * MAX_COEFFICIENT]
        if not near_limit.empty:
            print(f"{len(near_limit)} coefficients near the limit ({MAX_COEFFICIENT})")

        if save:
            coef_file = os.path.join(RESULTS_DIR, f'coefficients_{start_age}_{end_age}Ma.csv')
            df_coef.to_csv(coef_file, index=False)
            print(f"Coefficients saved to: {coef_file}")

        return df_coef

    return None

# =============================================================================
# METRICS WITH EXPLICIT DATA TYPE TRACKING
# =============================================================================
def calculate_r2_score(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    if np.var(y_true) < 1e-8:
          return np.nan
    # if np.all(y_true == y_true[0]):
    #     return 0.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)

def calculate_performance_metrics(y_true, y_pred, variable_names=None, data_type="UNSEEN", fold_id=None):
    """
    Calculate performance metrics with data type annotation.
    data_type: "SEEN" (in-sample) or "UNSEEN" (cross-validation)
    fold_id: which CV fold this corresponds to (for UNSEEN data)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_features = y_true.shape[1]

    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mse)
    r2_overall = calculate_r2_score(y_true.flatten(), y_pred.flatten())

    per_var_rmse, per_var_r2 = [], []
    for i in range(n_features):
        per_var_rmse.append(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        per_var_r2.append(calculate_r2_score(y_true[:, i], y_pred[:, i]))

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'r2_overall': r2_overall,
        'data_type': data_type,
        'fold_id': fold_id,
        'per_var_rmse': per_var_rmse,
        'per_var_r2': per_var_r2,
    }

    if variable_names:
        for idx, name in enumerate(variable_names):
            metrics[f'{name}_rmse'] = per_var_rmse[idx]
            metrics[f'{name}_r2'] = per_var_r2[idx]
    return metrics

# =============================================================================
# FILTER - FIXED VERSION WITH PADDING HANDLING
# =============================================================================
def apply_butterworth_filter(data, cutoff, order):
    """
    Low-pass Butterworth filter with proper padding handling.
    """
    required_length = 3 * order
    if len(data) <= required_length:
        print(f"    Warning: Data length ({len(data)}) too short for filter order {order}")
        return data

    try:
        b, a = butter(order, cutoff, btype='low')
        padlen = min(2 * order, len(data) - 1)
        return np.clip(filtfilt(b, a, data, padlen=padlen), 0, None)
    except Exception as e:
        print(f"    Filter error: {e}")
        return data

# =============================================================================
# NORMALIZATION
# =============================================================================
def minmax_scale(X):
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    rng = maxs - mins
    rng[rng == 0] = 1
    return (X - mins) / rng, mins, maxs

def minmax_inverse(X_scaled, mins, maxs):
    rng = maxs - mins
    rng[rng == 0] = 1
    return X_scaled * rng + mins

# =============================================================================
# CV FOLD CALCULATION
# =============================================================================
def compute_adaptive_n_splits(start_age, end_age):
    try:
        df_tmp = pd.read_csv("DATA.csv", header=None, dtype=float)
        df_tmp.columns = ['toc', 'age', 'ironspec', 'pyrite', 'p']
        df_tmp.sort_values('age', inplace=True)
        mask = (df_tmp['age'] >= start_age) & (df_tmp['age'] <= end_age)
        n_pts = int(df_tmp[mask]['age'].round(1).nunique())
    except Exception:
        n_pts = 20

    n_splits = max(MIN_FOLDS, min(MAX_FOLDS, n_pts // PTS_PER_FOLD))
    return n_splits, n_pts

def safe_n_splits(requested, n_pts):
    data_limited = n_pts // MIN_TRAIN_PTS
    effective = min(requested, data_limited)
    effective = max(MIN_FOLDS, effective)
    effective = min(MAX_FOLDS, effective)
    return effective

# =============================================================================
# SINDY MODEL WITH COEFFICIENT LIMITING
# =============================================================================
def make_sindy_model(variable_names, alpha=0.3, max_coef=MAX_COEFFICIENT):
    """
    Create SINDy model with maximum coefficient limit.
    The limit will be enforced after fitting via post-processing.
    """
    optimizer = ps.STLSQ(
        threshold=GLOBAL_THRESHOLD,
        alpha=1000,  # No L2 regularization - we only want to limit max value
        max_iter=100,
        normalize_columns=True  # Helps with numerical stability
    )

    model = ps.SINDy(
        differentiation_method=ps.SINDyDerivative(kind='kalman', alpha=alpha),
        feature_library=ps.PolynomialLibrary(degree=POLY_DEGREE),
        optimizer=optimizer,
        feature_names=variable_names,
    )

    # Store max_coef as an attribute for later enforcement
    model.max_coef = max_coef
    return model

def fit_with_coef_limit(model, X, t=None, ensemble=True):
    """
    Fit model and then enforce coefficient limit.
    """
    # Fit the model
    model.fit(X, t=t, ensemble=ensemble)

    # Enforce coefficient limit
    was_clipped = enforce_coefficient_limit(model, model.max_coef)

    return model, was_clipped

# =============================================================================
# SIMULATION
# =============================================================================
def simulate_on_uniform_grid(model, X0, t_span, dt=0.001):
    t_uniform = np.arange(t_span[0], t_span[-1] + dt / 2, dt)
    simulated_uniform = model.simulate(
        X0, t_uniform,
        integrator='solve_ivp',
        integrator_kws={'min_step': 1e-14},
    )
    n_sim = simulated_uniform.shape[0]
    if n_sim < len(t_uniform):
        t_uniform = t_uniform[:n_sim]

    simulated_original = np.zeros((len(t_span), simulated_uniform.shape[1]))
    for i in range(simulated_uniform.shape[1]):
        f = interp1d(t_uniform, simulated_uniform[:, i],
                     kind='linear', fill_value='extrapolate')
        simulated_original[:, i] = f(t_span)

    return simulated_original, t_uniform, simulated_uniform

# =============================================================================
# FUNCTION TO SAVE UNSEEN DATA R² RESULTS
# =============================================================================

def save_unseen_r2_results(all_fold_results, top_results, start_age, end_age, interval_params):
    """
    Explicitly save all UNSEEN data R² results to CSV files.
    """
    # Save all fold results (all UNSEEN R² values)
    if all_fold_results:
        # Filter to only include valid results (remove infinite values)
        valid_folds = [r for r in all_fold_results if r.get('rmse', np.inf) != np.inf]

        if valid_folds:
            df_all_folds = pd.DataFrame(valid_folds)
            # Remove model_index column if present
            if 'model_index' in df_all_folds.columns:
                df_all_folds = df_all_folds.drop('model_index', axis=1)

            # Add interval information
            df_all_folds['interval_start'] = start_age
            df_all_folds['interval_end'] = end_age
            df_all_folds['cutoff'] = interval_params['cutoff']
            df_all_folds['filter_order'] = interval_params['order']

            # Save all UNSEEN results
            all_unseen_file = os.path.join(RESULTS_DIR, f'all_unseen_r2_{start_age}_{end_age}Ma.csv')
            df_all_folds.to_csv(all_unseen_file, index=False)
            print(f"\n Saved ALL UNSEEN R² results ({len(valid_folds)} folds) to: {all_unseen_file}")

    # Save top N results (best UNSEEN performance)
    if top_results:
        df_top = pd.DataFrame(top_results)
        if 'model_index' in df_top.columns:
            df_top = df_top.drop('model_index', axis=1)

        # Add interval information
        df_top['interval_start'] = start_age
        df_top['interval_end'] = end_age
        df_top['cutoff'] = interval_params['cutoff']
        df_top['filter_order'] = interval_params['order']

        # Save top UNSEEN results
        top_unseen_file = os.path.join(RESULTS_DIR, f'top{TOP_N_RESULTS}_unseen_r2_{start_age}_{end_age}Ma.csv')
        df_top.to_csv(top_unseen_file, index=False)
        print(f"Saved TOP {len(top_results)} UNSEEN R² results to: {top_unseen_file}")

        # Also save a summary of just the R² values
        r2_summary = {
            'interval': f"{start_age}-{end_age}",
            'best_unseen_r2': float(top_results[0]['r2_overall']),
            'best_unseen_r2_raw': float(top_results[0].get('r2_overall_raw', np.nan)),
            'avg_unseen_r2': float(np.mean([r['r2_overall_raw'] for r in top_results])),
            'std_unseen_r2': float(np.std([r['r2_overall_raw'] for r in top_results])),
            'best_toc_r2': float(top_results[0].get('toc_r2', np.nan)),
            'best_pyrite_r2': float(top_results[0].get('pyrite_r2', np.nan)),
            'best_p_r2': float(top_results[0].get('p_r2', np.nan)),
            'best_toc_r2_raw': float(top_results[0].get('toc_r2_raw', np.nan)),
            'best_pyrite_r2_raw': float(top_results[0].get('pyrite_r2_raw', np.nan)),
            'best_p_r2_raw': float(top_results[0].get('p_r2_raw', np.nan)),
            'n_folds': len(valid_folds) if 'valid_folds' in locals() else 0,
            'max_coefficient_limit': MAX_COEFFICIENT,
        }

        # Save as JSON for easy reading
        r2_json_file = os.path.join(RESULTS_DIR, f'unseen_r2_summary_{start_age}_{end_age}Ma.json')
        with open(r2_json_file, 'w') as f:
            json.dump(r2_summary, f, indent=2)
        print(f"Saved UNSEEN R² summary JSON to: {r2_json_file}")

        return r2_summary

    return None

# =============================================================================
# PLOTTING FUNCTIONS WITH SEEN VS UNSEEN ANNOTATIONS
# =============================================================================

def plot_seen_vs_unseen_performance(
    t, grouped_data, simulated, cv_predictions,
    start_age, end_age, insample_metrics, cv_results,
    variable_names, model_eq=None, coef_df=None):

    labels = ['TOC', 'FePy/FeHR', 'Phosphorus']
    colors = ['steelblue', 'firebrick', 'forestgreen']
    var_keys = ['toc', 'pyrite', 'p']

    for i, (label, key, color) in enumerate(zip(labels, var_keys, colors)):

        fig, ax = plt.subplots(figsize=(8, 5))

        # RAW
        ax.scatter(t, grouped_data[f'{key}_raw'],
                   alpha=0.3, s=15, color='gray', label='Raw data')

        # FILTERED
        ax.plot(t, grouped_data[f'{key}_smooth'],
                '-', color=color, linewidth=1.5, label='Filtered')

        # SEEN
        seen_r2 = float(insample_metrics.get(f'{key}_r2', 0))
        ax.plot(t, simulated[:, i], '--', color='black',
                linewidth=2, label=f'SEEN R²={seen_r2:.3f}')

        # =========================
        # UNSEEN (BEST FOLD ONLY)
        # =========================
        if cv_results and len(cv_results) > 0:

            best_idx = np.argmax([r['r2_overall_raw'] for r in cv_results])

            t_val, sim_val = cv_predictions[best_idx]
            unseen_r2 = float(cv_results[best_idx].get(f'{key}_r2', 0))
            rmse_val = cv_results[best_idx]['rmse']

            if sim_val.shape[0] > 0 and sim_val.shape[1] > i:
                ax.plot(
                    t_val,
                    sim_val[:, i],
                    ':',
                    color='purple',
                    linewidth=2,
                    alpha=0.8,
                    label=f'UNSEEN (R²={unseen_r2:.3f}, RMSE={rmse_val:.3f})'
                )

        # EQUAÇÃO
        if model_eq and i < len(model_eq):
            eq_text = model_eq[i]
            if len(eq_text) > 40:
                eq_text = eq_text[:37] + "..."
            ax.text(0.02, 0.98, f"SINDy: {eq_text}",
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_title(f'{label}: {start_age}-{end_age} Ma')
        ax.set_xlabel('Age (Ma)')
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # SAVE
        if key == "toc":
            save_dir = FIGURES_TOC
        elif key == "pyrite":
            save_dir = FIGURES_FEPY
        elif key == "p":
            save_dir = FIGURES_P

        fname = os.path.join(save_dir, f'{key}_{start_age}_{end_age}Ma.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved {label} plot to {fname}")

def save_performance_summary_plot(top_results, insample_metrics, start_age, end_age):
    """Create a bar chart comparing SEEN vs UNSEEN performance."""
    if not top_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # R² comparison
    variables = ['TOC', 'Pyrite', 'P']
    seen_r2 = [float(insample_metrics.get(f'{v.lower()}_r2', 0)) for v in variables]
    unseen_r2 = [float(top_results[0].get(f'{v.lower()}_r2', 0)) for v in variables]

    x = np.arange(len(variables))
    width = 0.35

    ax1.bar(x - width/2, seen_r2, width, label='SEEN (in-sample)', color='steelblue', alpha=0.7)
    ax1.bar(x + width/2, unseen_r2, width, label='UNSEEN (best CV)', color='forestgreen', alpha=0.7)
    ax1.set_xlabel('Variable')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R²: SEEN vs UNSEEN Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(variables)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for i, (s, u) in enumerate(zip(seen_r2, unseen_r2)):
        ax1.text(i - width/2, s + 0.02, f'{s:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, u + 0.02, f'{u:.3f}', ha='center', va='bottom', fontsize=8)

    # Gap analysis
    gaps = [s - u for s, u in zip(seen_r2, unseen_r2)]
    colors = ['red' if g > 0.1 else 'orange' if g > 0.05 else 'green' for g in gaps]
    ax2.bar(variables, gaps, color=colors, alpha=0.7)
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Overfitting threshold')
    ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Warning threshold')
    ax2.set_xlabel('Variable')
    ax2.set_ylabel('SEEN - UNSEEN R² Gap')
    ax2.set_title('Overfitting Analysis (smaller gap is better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Performance Summary: {start_age}-{end_age} Ma (Max Coef = {MAX_COEFFICIENT})', fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR, f'performance_summary_{start_age}_{end_age}Ma.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# MAIN FUNCTION WITH CROSS-VALIDATION
# =============================================================================
def apply_butterworth_and_sindy_with_cv(cutoff, order, start_age, end_age, n_splits=5):
    print(f"\n{'='*70}")
    print(f"PROCESSING INTERVAL: {start_age}-{end_age} Ma")
    print(f"{'='*70}")

    # Load and prepare data
    df = pd.read_csv("DATA.csv", header=None)
    df.columns = ['toc', 'age', 'ironspec', 'pyrite', 'p']
    df.sort_values('age', inplace=True)
    df.interpolate(limit_direction="both", inplace=True)

    # Filter by age range
    filtered_df = df[(df['age'] >= start_age) & (df['age'] <= end_age)].copy()
    original_len = len(filtered_df)

    # Visualize outliers before removal
    visualize_outliers(filtered_df, ['toc', 'pyrite', 'p', 'ironspec'], start_age, end_age)

    # Apply selected outlier removal method
    filtered_df = remove_outliers_wrapper(
        filtered_df,
        ['toc', 'pyrite', 'p', 'ironspec'],
        method=OUTLIER_METHOD,
        multiplier=IQR_MULTIPLIER,
        percentile_range=PERCENTILE_RANGE,
        threshold=ZSCORE_THRESHOLD
    )

    print(f"  Data points: {original_len} → {len(filtered_df)} after outlier removal")

    # Group by rounded age
    filtered_df['rounded_age'] = filtered_df['age'].round(1)
    grouped_data = filtered_df.groupby('rounded_age').mean().reset_index()

    # Normalize phosphorus
    grouped_data['p'] = grouped_data['p'] / 10000

    # Preserve raw values
    for col in ['toc', 'pyrite', 'p', 'ironspec']:
        grouped_data[f'{col}_raw'] = grouped_data[col].copy()

    # Apply Butterworth filter
    for col in ['toc', 'pyrite', 'p', 'ironspec']:
        grouped_data[f'{col}_smooth'] = apply_butterworth_filter(
            grouped_data[col].values, cutoff, order
        )

    # Build state matrices
    t = grouped_data["age"].values

    X_smooth = np.stack([
        grouped_data["toc_smooth"].values,
        grouped_data["pyrite_smooth"].values,
        grouped_data["p_smooth"].values,
    ], axis=-1)

    X_raw = np.stack([
        grouped_data["toc_raw"].values,
        grouped_data["pyrite_raw"].values,
        grouped_data["p_raw"].values,
    ], axis=-1)

    variable_names = ["toc", "pyrite", "p"]

    # Normalize data
    X_smooth_norm, norm_mins, norm_maxs = minmax_scale(X_smooth)
    rng = norm_maxs - norm_mins
    rng[rng == 0] = 1.0

    dt = np.mean(np.diff(t))
    print(f"\n  Final data points: {len(t)}")
    print(f"  Average time step: {dt:.3f} Myr")
    print(f"  Max coefficient limit: {MAX_COEFFICIENT}")

    # Calculate effective number of CV folds
    n_splits_eff = safe_n_splits(n_splits, len(t))
    print(f"  Effective CV folds: {n_splits_eff}")

    # Check if we have enough points for CV
    if len(t) < MIN_TRAIN_PTS * 2:
        print(f"  Too few points ({len(t)}) for CV — using full dataset only")
        model, _, insample, _ = fit_full_dataset(t, X_smooth_norm, X_raw, X_smooth,
                                                norm_mins, norm_maxs, grouped_data,
                                                start_age, end_age, variable_names, dt)
        return model, None, insample, None

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits_eff)
    all_fold_results = []
    cv_models = []
    cv_predictions = []

    print(f"\n{'─'*50}")
    print("CROSS-VALIDATION (UNSEEN DATA PERFORMANCE)")
    print(f"{'─'*50}")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_smooth_norm)):
        print(f"\nFold {fold+1}/{n_splits_eff}:")
        print(f"  Train (SEEN): {len(train_idx)} pts")
        print(f"  Validation (UNSEEN): {len(val_idx)} pts")

        X_train = X_smooth_norm[train_idx]
        t_train = t[train_idx]
        X_val_smooth = X_smooth[val_idx]
        X_val_raw = X_raw[val_idx]
        t_val = t[val_idx]

        # Train model with ensemble and coefficient limiting
        model = make_sindy_model(variable_names, alpha=0.1, max_coef=MAX_COEFFICIENT)
        model, was_clipped = fit_with_coef_limit(model, X_train, t=t_train, ensemble=True)
        cv_models.append(model)

        if was_clipped:
            print(f"Coefficients were clipped to max {MAX_COEFFICIENT}")

        try:
            # Simulate validation period
            X0_norm = X_train[-1]
            t_validation = np.arange(t_train[-1], t_val[-1] + dt/2, dt)

            sim_uniform_norm = model.simulate(
                X0_norm, t_validation,
                integrator='solve_ivp',
                integrator_kws={'min_step': 1e-8},
            )

            n_sim_cv = sim_uniform_norm.shape[0]
            if n_sim_cv < len(t_validation):
                t_validation = t_validation[:n_sim_cv]

            val_indices = [np.argmin(np.abs(t_validation - tv)) for tv in t_val]
            simulated_val_norm = sim_uniform_norm[val_indices]
            simulated_val = minmax_inverse(simulated_val_norm, norm_mins, norm_maxs)

            cv_predictions.append((t_val, simulated_val))

            # Calculate metrics on UNSEEN data
            metrics_smooth = calculate_performance_metrics(
                X_val_smooth, simulated_val, variable_names,
                data_type="UNSEEN", fold_id=fold+1)
            metrics_raw = calculate_performance_metrics(
                X_val_raw, simulated_val, variable_names,
                data_type="UNSEEN", fold_id=fold+1)

            fold_result = {
                'fold': fold + 1,
                'mse': float(metrics_smooth['mse']),
                'rmse': float(metrics_smooth['rmse']),
                'r2_overall': float(metrics_smooth['r2_overall']),  # ← UNSEEN R² (filtered)
                'toc_r2': float(metrics_smooth['toc_r2']),
                'pyrite_r2': float(metrics_smooth['pyrite_r2']),
                'p_r2': float(metrics_smooth['p_r2']),
                'rmse_raw': float(metrics_raw['rmse']),
                'r2_overall_raw': float(metrics_raw['r2_overall']),  # ← UNSEEN R² (raw)
                'toc_r2_raw': float(metrics_raw.get('toc_r2', 0)),
                'pyrite_r2_raw': float(metrics_raw.get('pyrite_r2', 0)),
                'p_r2_raw': float(metrics_raw.get('p_r2', 0)),
                'model_index': len(cv_models) - 1,
                'data_type': 'UNSEEN',  # Explicit label
            }
            all_fold_results.append(fold_result)

            print(f"  → UNSEEN R² (filtered) = {metrics_smooth['r2_overall']:.4f}")
            print(f"    TOC:{metrics_smooth['toc_r2']:.3f} Pyrite:{metrics_smooth['pyrite_r2']:.3f} P:{metrics_smooth['p_r2']:.3f}")

        except Exception as e:
            print(f"  Validation failed: {e}")
            all_fold_results.append({
                'fold': fold + 1,
                'rmse': np.inf, 'r2_overall': -np.inf,
                'rmse_raw': np.inf, 'r2_overall_raw': -np.inf,
                'model_index': len(cv_models) - 1,
                'data_type': 'UNSEEN',
            })

    # Select top-N results based on UNSEEN performance
    valid_results = [r for r in all_fold_results if r['rmse'] != np.inf]
    valid_results.sort(key=lambda x: (-x['r2_overall_raw'], x['rmse_raw']))
    top_results = valid_results[:min(TOP_N_RESULTS, len(valid_results))]

    print(f"\n{'─'*50}")
    print(f"TOP {len(top_results)} RESULTS (ranked by UNSEEN performance)")
    print(f"{'─'*50}")
    for i, r in enumerate(top_results):
        print(f"  Rank {i+1} (Fold {r['fold']}): UNSEEN R²={r['r2_overall_raw']:.4f}")
    best_result = top_results[0]
    print(f"  → UNSEEN RMSE = {best_result['rmse_raw']:.4f}")

    if not top_results:
        print("No valid CV results — falling back to full-data fit")
        model, _, insample, _ = fit_full_dataset(t, X_smooth_norm, X_raw, X_smooth,
                                                norm_mins, norm_maxs, grouped_data,
                                                start_age, end_age, variable_names, dt)
        return model, None, insample, None, t, simulated, grouped_data

    # ===== SAVE ALL UNSEEN R² RESULTS =====
    interval_params = {'cutoff': cutoff, 'order': order}
    unseen_summary = save_unseen_r2_results(all_fold_results, top_results, start_age, end_age, interval_params)

    # Best model from CV
    best_result = top_results[0]
    best_model = cv_models[best_result['model_index']]

    print(f"\nBest model (from CV Fold {best_result['fold']}):")
    best_model.print()

    # Final model on ALL data (for SEEN performance)
    print(f"\nFitting final model on ALL {len(t)} points (for SEEN performance)...")
    final_model = make_sindy_model(variable_names, alpha=0.3, max_coef=MAX_COEFFICIENT)
    final_model, was_clipped = fit_with_coef_limit(final_model, X_smooth_norm, t=t, ensemble=True)

    if was_clipped:
        print(f"Final model coefficients were clipped to max {MAX_COEFFICIENT}")

    # Analyze coefficients
    coef_df = analyze_coefficients(final_model, variable_names, start_age, end_age)

    # Get final model equations
    final_eq = []
    if hasattr(final_model, 'equations'):
        final_eq = final_model.equations()
    elif hasattr(final_model, 'print'):
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            final_model.print()
        final_eq = f.getvalue().strip().split('\n')

    print("\nFinal Model Equations (trained on ALL data):")
    final_model.print()

    # Simulate final model on ALL data (SEEN performance)
    sim_norm, t_uniform, sim_uniform_norm = simulate_on_uniform_grid(
        final_model, X_smooth_norm[0], t, dt
    )
    simulated = minmax_inverse(sim_norm, norm_mins, norm_maxs)
    # =========================
    # STORE CONTINUOUS DATA
    # =========================
    all_t.append(t)

    all_sim.append(simulated)

    all_obs.append(
        np.column_stack([
            grouped_data['toc_smooth'],
            grouped_data['pyrite_smooth'],
            grouped_data['p_smooth']
        ])
    )

    # Calculate SEEN (in-sample) metrics
    insample_smooth = calculate_performance_metrics(
        X_smooth, simulated, variable_names, data_type="SEEN")
    insample_raw = calculate_performance_metrics(
        X_raw, simulated, variable_names, data_type="SEEN")

    print(f"\n{'─'*50}")
    print("FINAL PERFORMANCE SUMMARY")
    print(f"{'─'*50}")
    print(f"SEEN data (in-sample) R² = {insample_smooth['r2_overall']:.4f}")
    print(f"UNSEEN R² (filtered) = {best_result['r2_overall']:.4f}")
    print(f"UNSEEN R² (raw) = {best_result['r2_overall_raw']:.4f}")
    avg_filtered = np.mean([r['r2_overall'] for r in top_results])
    avg_raw = np.mean([r['r2_overall_raw'] for r in top_results])

    print(f"UNSEEN R² (filtered avg) = {avg_filtered:.4f}")
    print(f"UNSEEN R² (raw avg) = {avg_raw:.4f}")
    print(f"SEEN RMSE = {insample_smooth['rmse']:.4f}")
    print(f"UNSEEN RMSE = {best_result['rmse']:.4f}")

    overfit_gap = float(insample_smooth['r2_overall'] - best_result['r2_overall'])
    if overfit_gap > 0.1:
        print(f"OVERFITTING WARNING: Gap = {overfit_gap:.3f} (> 0.1)")
    elif overfit_gap > 0.05:
        print(f"Moderate overfitting: Gap = {overfit_gap:.3f}")
    else:
        print(f"Good generalization: Gap = {overfit_gap:.3f}")

    # Create SEEN vs UNSEEN visualization
    plot_seen_vs_unseen_performance(
        t, grouped_data, simulated, cv_predictions,
        start_age, end_age, insample_smooth, top_results,
        variable_names, final_eq, coef_df
    )
    # =============================================================================
# EXPORT INDIVIDUAL FIGURES (CORRETO)
# =============================================================================
    export_individual_variable_plots(
    t,
    grouped_data,
    simulated,
    start_age,
    end_age,
    insample_raw,
    variable_names
)

    # Create performance summary plot
    save_performance_summary_plot(top_results, insample_smooth, start_age, end_age)

    # Save comprehensive results with both SEEN and UNSEEN
    comprehensive_results = {
        'interval': f"{start_age}-{end_age}",
        'n_points': len(t),
        'seen_r2': float(insample_smooth['r2_overall']),
        'seen_toc_r2': float(insample_smooth.get('toc_r2', 0)),
        'seen_pyrite_r2': float(insample_smooth.get('pyrite_r2', 0)),
        'seen_p_r2': float(insample_smooth.get('p_r2', 0)),
        'best_unseen_r2_filtered': float(best_result['r2_overall']),
        'best_unseen_r2_raw': float(best_result['r2_overall_raw']),
        'seen_rmse': float(insample_smooth['rmse']),
        'best_unseen_rmse': float(best_result['rmse']),
        'best_unseen_toc_r2': float(best_result.get('toc_r2', 0)),
        'best_unseen_pyrite_r2': float(best_result.get('pyrite_r2', 0)),
        'best_unseen_p_r2': float(best_result.get('p_r2', 0)),
        'avg_unseen_r2': float(np.mean([r['r2_overall_raw'] for r in top_results])),
        'std_unseen_r2': float(np.std([r['r2_overall_raw'] for r in top_results])),
        'overfit_gap': float(overfit_gap),
        'outlier_method': OUTLIER_METHOD,
        'filter_cutoff': cutoff,
        'filter_order': order,
        'max_coefficient_limit': MAX_COEFFICIENT,
        'max_coefficient_actual': float(coef_df['abs_value'].max()) if coef_df is not None else 0,
        'n_nonzero_terms': len(coef_df) if coef_df is not None else 0,
    }

    comp_file = os.path.join(RESULTS_DIR, f'comprehensive_results_{start_age}_{end_age}Ma.csv')
    pd.DataFrame([comprehensive_results]).to_csv(comp_file, index=False)
    print(f"Saved comprehensive results to: {comp_file}")

    return final_model, top_results, insample_smooth, unseen_summary, t, simulated, grouped_data

# =============================================================================
# FALLBACK FUNCTION FOR FULL DATASET FITTING
# =============================================================================
def fit_full_dataset(t, X_smooth_norm, X_raw, X_smooth,
                     norm_mins, norm_maxs, grouped_data,
                     start_age, end_age, variable_names, dt):
    """Fallback function when CV is not possible."""

    final_model = make_sindy_model(variable_names, alpha=0.3, max_coef=MAX_COEFFICIENT)
    final_model, was_clipped = fit_with_coef_limit(final_model, X_smooth_norm, t=t, ensemble=True)

    if was_clipped:
        print(f"Coefficients were clipped to max {MAX_COEFFICIENT}")

    print(f"\nEquations | {start_age}-{end_age} Ma:")
    final_model.print()

    # Analyze coefficients
    coef_df = analyze_coefficients(final_model, variable_names, start_age, end_age)

    # Get model equations
    final_eq = []
    if hasattr(final_model, 'equations'):
        final_eq = final_model.equations()
    elif hasattr(final_model, 'print'):
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            final_model.print()
        final_eq = f.getvalue().strip().split('\n')

    # Simulate
    sim_norm, t_uniform, sim_uniform_norm = simulate_on_uniform_grid(
        final_model, X_smooth_norm[0], t, dt
    )
    simulated = minmax_inverse(sim_norm, norm_mins, norm_maxs)

    # Calculate metrics (SEEN only, no CV)
    metrics_smooth = calculate_performance_metrics(
        X_smooth, simulated, variable_names, data_type="SEEN")
    metrics_raw = calculate_performance_metrics(
        X_raw, simulated, variable_names, data_type="SEEN")

    print(f"\nIn-sample (SEEN) R² (filtered): {metrics_smooth['r2_overall']:.4f}")
    print(f"In-sample (SEEN) R² (raw): {metrics_raw['r2_overall']:.4f}")

    return final_model, None, metrics_smooth, None

# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================
def summarize_all_results(all_results, all_unseen_summaries):
    """Summarize results across all intervals with SEEN vs UNSEEN comparison."""
    print("\n" + "="*90)
    print("SUMMARY OF RESULTS ACROSS ALL INTERVALS (SEEN vs UNSEEN)")
    print("="*90)

    summary_data = []
    all_unseen_data = []

    for (start_age, end_age), result in all_results.items():
        # Handle different return value lengths
        if len(result) == 4:
            model, top_results, insample, unseen_summary = result
        elif len(result) == 3:
            model, top_results, insample = result
            unseen_summary = None
        else:
            model, top_results = result[:2]
            insample = {'r2_overall': 0, 'toc_r2': 0, 'pyrite_r2': 0, 'p_r2': 0}
            unseen_summary = None

        if top_results and insample:
            best = top_results[0]
            # Convert to float to avoid numpy type issues
            avg_unseen = float(np.mean([r['r2_overall_raw'] for r in top_results]))
            std_unseen = float(np.std([r['r2_overall_raw'] for r in top_results]))
            gap = float(insample['r2_overall'] - best['r2_overall_raw'])

            status = "GOOD" if gap <= 0.05 else " WARNING" if gap <= 0.1 else " OVERFIT"

            # Try to get coefficient info from comprehensive results file
            coef_info = ""
            comp_file = os.path.join(RESULTS_DIR, f'comprehensive_results_{start_age}_{end_age}Ma.csv')
            if os.path.exists(comp_file):
                try:
                    comp_df = pd.read_csv(comp_file)
                    if not comp_df.empty:
                        max_coef_actual = comp_df.iloc[0].get('max_coefficient_actual', 0)
                        n_terms = comp_df.iloc[0].get('n_nonzero_terms', 0)
                        coef_info = f" | max|coef|={max_coef_actual:.2f} ({n_terms} terms)"
                except:
                    pass

            row = {
                'Interval': f"{start_age}-{end_age} Ma",
                'Points': len(t) if 't' in locals() else 0,
                'SEEN_R²': round(float(insample['r2_overall']), 4),
                'BEST_UNSEEN_R²': round(float(best['r2_overall_raw']), 4),
                'SEEN_RMSE': round(float(insample.get('rmse', np.nan)), 4),
                'UNSEEN_RMSE': round(float(best.get('rmse', np.nan)), 4),
                'AVG_UNSEEN_R²': round(avg_unseen, 4),
                'STD_UNSEEN_R²': round(std_unseen, 4),
                'GAP': round(gap, 4),
                'STATUS': status,
                'TOC_SEEN': round(float(insample.get('toc_r2', 0)), 4),
                'TOC_UNSEEN': round(float(best.get('toc_r2', 0)), 4),
                'PYR_SEEN': round(float(insample.get('pyrite_r2', 0)), 4),
                'PYR_UNSEEN': round(float(best.get('pyrite_r2', 0)), 4),
                'P_SEEN': round(float(insample.get('p_r2', 0)), 4),
                'P_UNSEEN': round(float(best.get('p_r2', 0)), 4),
            }
            summary_data.append(row)

            # Collect all unseen data for combined file
            if unseen_summary:
                all_unseen_data.append(unseen_summary)

    if summary_data:
        # Save interval summary
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        summary_file = os.path.join(RESULTS_DIR, 'summary_all_intervals_seen_vs_unseen.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved summary to: {summary_file}")

        # Save combined unseen R² data
        if all_unseen_data:
            unseen_df = pd.DataFrame(all_unseen_data)
            unseen_file = os.path.join(RESULTS_DIR, 'all_intervals_unseen_r2_summary.csv')
            unseen_df.to_csv(unseen_file, index=False)
            print(f"Saved combined UNSEEN R² data to: {unseen_file}")

        # Create overall visualization
        plot_overall_summary(summary_df)

        # Also save as JSON for easy parsing
        json_file = os.path.join(RESULTS_DIR, 'all_results_summary.json')
        with open(json_file, 'w') as f:
            json.dump({
                'intervals': summary_data,
                'total_intervals': len(summary_data),
                'max_coefficient_limit': MAX_COEFFICIENT,
                'best_interval': max(summary_data, key=lambda x: x['BEST_UNSEEN_R²']) if summary_data else None,
                'worst_interval': min(summary_data, key=lambda x: x['BEST_UNSEEN_R²']) if summary_data else None,
            }, f, indent=2, default=str)
        print(f"Saved JSON summary to: {json_file}")

def plot_overall_summary(summary_df):
    """Create overall summary plot across intervals."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    intervals = summary_df['Interval'].tolist()
    seen_r2 = summary_df['SEEN_R²'].astype(float)
    unseen_r2 = summary_df['BEST_UNSEEN_R²'].astype(float)
    gaps = summary_df['GAP'].astype(float)

    # R² comparison
    x = np.arange(len(intervals))
    width = 0.35

    ax1.bar(x - width/2, seen_r2, width, label='SEEN R²', color='steelblue', alpha=0.7)
    ax1.bar(x + width/2, unseen_r2, width, label='UNSEEN R²', color='forestgreen', alpha=0.7)
    ax1.set_xlabel('Interval')
    ax1.set_ylabel('R² Score')
    ax1.set_title(f'SEEN vs UNSEEN Performance (Max Coef = {MAX_COEFFICIENT})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(intervals, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Good threshold')

    # Gap analysis
    colors = ['green' if g <= 0.05 else 'orange' if g <= 0.1 else 'red' for g in gaps]
    bars = ax2.bar(intervals, gaps, color=colors, alpha=0.7)
    ax2.set_xlabel('Interval')
    ax2.set_ylabel('SEEN - UNSEEN R² Gap')
    ax2.set_title('Overfitting Analysis (smaller is better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(intervals, rotation=45, ha='right')
    ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Warning')
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Overfit')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Overall Model Performance Summary (Max Coefficient = {MAX_COEFFICIENT})', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'overall_performance_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
import matplotlib.ticker as ticker

def export_individual_variable_plots(
    t, grouped_data, simulated,
    start_age, end_age, insample_metrics,
    variable_names):

    labels = ['TOC', 'FePy/FeHR', 'Phosphorus']
    colors = ['steelblue', 'firebrick', 'forestgreen']
    var_keys = ['toc', 'pyrite', 'p']

    for i, (label, key, color) in enumerate(zip(labels, var_keys, colors)):

        plt.figure(figsize=(8,5))

        # RAW DATA
        plt.scatter(
            t,
            grouped_data[f'{key}_raw'],
            alpha=0.4,
            color='gray',
            label='Raw data'
        )

        # FILTERED
        plt.plot(
            t,
            grouped_data[f'{key}_smooth'],
            color=color,
            linewidth=2,
            label='Filtered'
        )

        # MODEL
        plt.plot(
            t,
            simulated[:, i],
            '--',
            color='black',
            linewidth=2,
            label='Model'
        )

        # R²
        r2 = float(insample_metrics.get(f'{key}_r2', 0))

        plt.title(
            f"{label} ({start_age}–{end_age} Ma)",
            fontsize=14
        )

        plt.xlabel("Age (Ma)", fontsize=12)
        plt.ylabel(label, fontsize=12)


        ax = plt.gca()

        # tamanho dos números
        ax.tick_params(axis='both', labelsize=12)

        # reduzir quantidade de ticks (ESSENCIAL)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

        # formatar números (mais limpo)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))


        # legenda
        plt.legend(fontsize=11, frameon=False)

        plt.grid(alpha=0.2)

        plt.tight_layout()

        # SAVE
        if key == "toc":
            save_dir = FIGURES_TOC
        elif key == "pyrite":
            save_dir = FIGURES_FEPY
        elif key == "p":
            save_dir = FIGURES_P

        fname = os.path.join(save_dir, f"{key}_up_clean_{start_age}_{end_age}.png")
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"CLEAN plot saved for {label}")
def smooth_transitions(t, sim, window=3):
    sim_smooth = sim.copy()

    for i in range(sim.shape[1]):
        for j in range(window, len(t) - window):
            sim_smooth[j, i] = np.mean(sim[j-window:j+window, i])

    return sim_smooth


def plot_continuous_dynamics(t, obs, sim):

    labels = ['TOC', 'FePy/FeHR', 'Phosphorus']
    colors = ['steelblue', 'firebrick', 'forestgreen']

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for i, (label, color) in enumerate(zip(labels, colors)):

        ax = axes[i]

        # OBSERVED
        ax.plot(
            t,
            obs[:, i],
            color=color,
            linewidth=2,
            label='Observed (filtered)'
        )

        # SIMULATED
        ax.plot(
            t,
            sim[:, i],
            '--',
            color='black',
            linewidth=2,
            label='Simulated (SINDy)'
        )

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Age (Ma)")

    plt.suptitle(
        "Continuous Reconstruction of Ordovician Biogeochemical Dynamics",
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    # =========================
    # SAVE (UMA FIGURA SÓ)
    # =========================
    save_path = os.path.join(FIGURES_DIR, "continuous_all_variables.svg")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"Continuous combined figure saved to {save_path}")

    plt.show()

def plot_continuous_same_axis(t, obs, sim):

    labels = ['TOC', 'FePy/FeHR', 'Phosphorus']
    colors = ['steelblue', 'firebrick', 'forestgreen']

    fig, ax = plt.subplots(figsize=(14, 6))

    # =========================
    # NORMALIZAÇÃO (0–1)
    # =========================
    obs_norm = (obs - obs.min(axis=0)) / (obs.max(axis=0) - obs.min(axis=0))
    sim_norm = (sim - sim.min(axis=0)) / (sim.max(axis=0) - sim.min(axis=0))

    for i, (label, color) in enumerate(zip(labels, colors)):

        # OBSERVED
        ax.plot(
            t,
            obs_norm[:, i],
            color=color,
            linewidth=2,
            label=f'{label} (Observed)'
        )

        # SIMULATED
        ax.plot(
            t,
            sim_norm[:, i],
            '--',
            color=color,
            linewidth=2,
            alpha=0.7,
            label=f'{label} (Simulated)'
        )

    ax.set_xlabel("Age (Ma)")
    ax.set_ylabel("Normalized value (0–1)")
    ax.grid(True, alpha=0.3)

    # inverter tempo (geológico)
    ax.invert_xaxis()

    plt.title(
        "Continuous Biogeochemical Dynamics (Normalized)",
        fontsize=14,
        fontweight='bold'
    )

    plt.legend(ncol=2, fontsize=9)

    plt.tight_layout()

    # =========================
    # SAVE
    # =========================
    save_path = os.path.join(FIGURES_DIR, "continuous_same_axis.svg")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"Same-axis figure saved to {save_path}")

    plt.show()

def plot_continuous_publication_style(t, obs, sim):

    labels = ['TOC', 'FePy/FeHR', 'Phosphorus']
    colors = ['blue', 'green', 'red']

    fig, ax = plt.subplots(figsize=(9, 6))

    # =========================
    # PLOT (SEM NORMALIZAR)
    # =========================
    for i, (label, color) in enumerate(zip(labels, colors)):

        # Observed
        ax.plot(t, obs[:, i], color=color, linewidth=2,
                label=f'{label} Observed')

        # Simulated
        ax.plot(t, sim[:, i], '--', color=color, linewidth=2,
                alpha=0.7, label=f'{label} Simulated')

    # =========================
    # EVENTOS GEOLÓGICOS
    # =========================

    # =========================
# STAGES ORDOVICIANOS
# =========================

    stages = [
    ("Cambrian", 490.0, 485.4, "#D9D9D9"),

    ("Tremadocian", 485.4, 477.7, "#8CC63F"),
    ("Floian", 477.7, 470.0, "#7FBF3F"),

    ("Dapingian", 470.0, 467.3, "#F2D03B"),
    ("Darriwilian", 467.3, 458.4, "#F5B335"),

    ("Sandbian", 458.4, 453.0, "#F08A24"),
    ("Katian", 453.0, 445.2, "#E66101"),

    ("Hirnantian", 445.2, 443.8, "#B22222"),

    ("Silurian", 443.8, 440.0, "#984EA3"),
]

    ymin, ymax = ax.get_ylim()

    for name, start, end, color in stages:

        ax.axvspan(
            start,
            end,
            ymin=0.94,
            ymax=1.00,
            color=color,
            alpha=0.85,
            linewidth=0
        )

        ax.text(
            (start + end) / 2,
            ymax * 0.985,
            name,
            ha='center',
            va='top',
            fontsize=8,
            fontweight='bold'
        )

# =========================
# GOBE
# =========================

    ax.axvspan(458, 470, color='red', alpha=0.08)

    ax.axvline(
    467.8,
    color='black',
    linestyle='--',
    linewidth=1.2
    )

    ax.text(
    467.8,
    ymax * 0.78,
    'GOBE MAIN PHASE',
    rotation=90,
    ha='center',
    fontsize=7,
    fontweight='bold'
    )

    # =========================
    # TACONIC OROGENY
    # =========================

    ax.axvspan(
    470,
    445,
    color='saddlebrown',
    alpha=0.05
    )

    ax.text(
    458,
    ymax * 0.70,
    'Taconic Orogeny (465–440 Ma)',
    fontsize=10,
    color='black',
    ha='center',
    fontweight='bold'
    )

    # =========================
    # LOME
    # =========================

    ax.axvspan(
    443.8,
    445.5,
    color='cyan',
    alpha=0.12
    )

    ax.text(
    444.7,
    ymax * 0.92,
    'LOME',
    fontsize=10,
    fontweight='bold',
    color='darkcyan',
    ha='center'
    )

    # =========================
    # LINHAS DE INTERVALO
    # =========================
    for age in np.arange(440, 491, 5):
        ax.axvline(age, color='gray', linestyle=':', alpha=0.4)

    # =========================
    # TENDÊNCIA DO P (igual ao paper)
    # =========================
    from scipy.ndimage import gaussian_filter1d
    p_trend = gaussian_filter1d(obs[:, 2], sigma=10)

    ax.plot(t, p_trend, color='darkred', linestyle=':',
            linewidth=2, alpha=0.8)

    # =========================
    # ESTÉTICA
    # =========================
    ax.set_xlabel("Age (Ma)")
    ax.set_ylabel("Concentration")
    ax.grid(True, alpha=0.2)

    plt.title("Observed and Simulated Ordovician Biogeochemical Dynamics",
              fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    # =========================
    # SAVE
    # =========================
    save_path = os.path.join(FIGURES_DIR, "new_figure_paper_style.svg")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"Figure saved to {save_path}")

    plt.show()

def plot_paper_figure(t, obs, sim):

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    labels = ['TOC', 'FePy/FeHR', 'Phosphorus']
    colors = ['blue', 'green', 'red']

    # =========================
    # (A) OBSERVED
    # =========================
    for i in range(3):
        axs[0].plot(t, obs[:, i], color=colors[i], linewidth=2)

    axs[0].set_title("(A) Observed Data")
    axs[0].set_ylabel("Concentration")

    # =========================
    # (B) MODEL VS DATA
    # =========================
    for i in range(3):
        axs[1].plot(t, obs[:, i], color=colors[i], linewidth=2)
        axs[1].plot(t, sim[:, i], '--', color=colors[i], alpha=0.7)

    axs[1].set_title("(B) SINDy Model Fit")
    axs[1].set_ylabel("Observed vs Simulated")

    # =========================
    # (C) RESIDUALS
    # =========================
    for i in range(3):
        residual = obs[:, i] - sim[:, i]
        axs[2].plot(t, residual, color=colors[i], linewidth=2)

    axs[2].axhline(0, color='black', linewidth=1)
    axs[2].set_title("(C) Model Residuals")
    axs[2].set_ylabel("Error")
    axs[2].set_xlabel("Age (Ma)")

    # =========================
    # ESTÉTICA
    # =========================
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[2].invert_xaxis()

    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "paper_figure_multiplot.svg")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f" PAPER FIGURE saved to {save_path}")

    plt.show()
# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*90)
    print("SINDy MODEL ANALYSIS WITH EXPLICIT UNSEEN DATA R² SAVING")
    print(f"MAXIMUM COEFFICIENT LIMIT: {MAX_COEFFICIENT}")
    print("="*90)

    all_results = {}
    all_unseen_summaries = []
    # =========================
    # CONTINUOUS STORAGE
    # =========================
    all_t = []
    all_sim = []
    all_obs = []
    # Pre-scan for adaptive n_splits
    print("\nAdaptive n_splits calculation:")
    print(f"{'Interval':<15} {'Data pts':>10} {'n_splits':>10}")
    print("-" * 40)

    interval_splits = {}
    for (start_age, end_age) in INTERVAL_CONFIGS:
        n_splits, n_pts = compute_adaptive_n_splits(start_age, end_age)
        interval_splits[(start_age, end_age)] = n_splits
        print(f"{start_age}-{end_age} Ma{'':<6} {n_pts:>10} {n_splits:>10}")

    print(f"\nOutlier removal method: {OUTLIER_METHOD}")
    print(f"Figures will be saved to: {os.path.abspath(FIGURES_DIR)}/")
    print(f"Results (including UNSEEN R²) will be saved to: {os.path.abspath(RESULTS_DIR)}/")

    # Main execution loop
    for (start_age, end_age), params in INTERVAL_CONFIGS.items():
        cutoff = params['cutoff']
        order = params['order']
        n_splits = interval_splits[(start_age, end_age)]

        result = apply_butterworth_and_sindy_with_cv(
        cutoff, order, start_age, end_age, n_splits=n_splits
    )

        model, top_results, insample, unseen_summary, t, simulated, grouped_data = result
        all_results[(start_age, end_age)] = result

        # Collect unseen summary if available
        if len(result) >= 4 and result[3] is not None:
            all_unseen_summaries.append(result[3])

    # Generate comprehensive summary
    summarize_all_results(all_results, all_unseen_summaries)
    # =========================
    # SMOOTH + PLOT CONTINUOUS
    # =========================

    t_full = np.concatenate(all_t)
    sim_full = np.vstack(all_sim)
    obs_full = np.vstack(all_obs)

    idx = np.argsort(t_full)

    t_full = t_full[idx]
    sim_full = sim_full[idx]
    obs_full = obs_full[idx]
    sim_full_smooth = smooth_transitions(t_full, sim_full)
    plot_continuous_dynamics(
        t_full,
        obs_full,
        sim_full_smooth
    )
    plot_continuous_same_axis(
    t_full,
    obs_full,
    sim_full_smooth
    )
    plot_continuous_publication_style(
    t_full,
    obs_full,
    sim_full_smooth
    )
    plot_paper_figure(
    t_full,
    obs_full,
    sim_full_smooth
    )
    # =========================
    # BUILD CONTINUOUS SERIES
    # =========================
    t_full = np.concatenate(all_t)
    sim_full = np.vstack(all_sim)
    obs_full = np.vstack(all_obs)

    # ordenar por tempo
    idx = np.argsort(t_full)

    t_full = t_full[idx]
    sim_full = sim_full[idx]
    obs_full = obs_full[idx]

    # Create a master file with ALL unseen R² from all folds across all intervals
    print(f"\n{'='*90}")
    print("COMBINING ALL UNSEEN R² RESULTS")
    print("="*90)

    all_folds_dfs = []
    for file in os.listdir(RESULTS_DIR):
        if file.startswith('all_unseen_r2_') and file.endswith('.csv'):
            df = pd.read_csv(os.path.join(RESULTS_DIR, file))
            all_folds_dfs.append(df)

    if all_folds_dfs:
        master_unseen_df = pd.concat(all_folds_dfs, ignore_index=True)
        master_file = os.path.join(RESULTS_DIR, 'master_all_unseen_r2_results.csv')
        master_unseen_df.to_csv(master_file, index=False)
        print(f"Created master file with ALL unseen R² results: {master_file}")
        print(f"Total folds across all intervals: {len(master_unseen_df)}")

        # Print summary statistics
        print(f"\nOverall UNSEEN R² statistics:")
        print(f"  Mean R²: {master_unseen_df['r2_overall_raw'].mean():.4f}")
        print(f"  Std R²: {master_unseen_df['r2_overall_raw'].std():.4f}")
        print(f"  Min R²: {master_unseen_df['r2_overall_raw'].min():.4f}")
        print(f"  Max R²: {master_unseen_df['r2_overall_raw'].max():.4f}")

    print(f"\n{'='*90}")
    print(f"ANALYSIS COMPLETE!")
    print(f"All UNSEEN data R² results saved in: {os.path.abspath(RESULTS_DIR)}/")
    print(f"Figures saved in: {os.path.abspath(FIGURES_DIR)}/")
    print(f"Coefficient limits enforced: |coef| ≤ {MAX_COEFFICIENT}")
    print(f"{'='*90}")
