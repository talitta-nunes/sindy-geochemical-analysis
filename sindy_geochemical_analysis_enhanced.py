"""
SINDy geochemical analysis for Ordovician proxy data.

Outputs created by this script:
- outputs/TOC/toc_up_clean_<start>_<end>.png
- outputs/FePy_FeHr/pyrite_up_clean_<start>_<end>.png
- outputs/P/p_up_clean_<start>_<end>.png
- results/unseen_r2_summary_<start>_<end>Ma.json
- results/all_results_summary.json
- results/model_evaluation_table.csv

The numerical workflow follows the original analysis:
- permissive IQR outlier removal (15 x IQR);
- interval-level interpolation, grouping, Butterworth smoothing, and min-max scaling;
- time-series cross-validation with top-N fold ranking;
- final first-order SINDy fit using all points in each interval.

"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pysindy as ps
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS AND OUTPUTS
# =============================================================================

DATA_FILE = "DATA.csv"

FIGURES_DIR = Path("outputs")
FIGURES_TOC = FIGURES_DIR / "TOC"
FIGURES_FEPY = FIGURES_DIR / "FePy_FeHr"
FIGURES_P = FIGURES_DIR / "P"
RESULTS_DIR = Path("results")


def prepare_output_directories() -> None:
    """Create output folders without deleting previous files."""
    for directory in (FIGURES_DIR, FIGURES_TOC, FIGURES_FEPY, FIGURES_P, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_THRESHOLD = 1e-6
TOP_N_RESULTS = 2
POLY_DEGREE = 1
MAX_COEFFICIENT = 100000.0

MIN_FOLDS = 2
MAX_FOLDS = 4
MIN_TRAIN_PTS = 2
PTS_PER_FOLD = 5

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
    (483, 488): {"cutoff": 0.10, "order": 1},
}

VARIABLE_NAMES = ["toc", "pyrite", "p"]
PLOT_LABELS = ["TOC", "FePy/FeHR", "Phosphorus"]
PLOT_COLORS = ["steelblue", "firebrick", "forestgreen"]


# =============================================================================
# PREPROCESSING
# =============================================================================

def load_input_data() -> pd.DataFrame:
    """Load the expected five-column SGP-derived input file."""
    data_path = Path(DATA_FILE)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {data_path.resolve()}. "
            "Place DATA.csv in the working directory."
        )

    df = pd.read_csv(data_path, header=None)
    if df.shape[1] < 5:
        raise ValueError("DATA.csv must contain at least five columns.")

    df = df.iloc[:, :5].copy()
    df.columns = ["toc", "age", "ironspec", "pyrite", "p"]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.sort_values("age", inplace=True)
    df.interpolate(limit_direction="both", inplace=True)
    return df


def remove_outliers_iqr_permissive(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply the original permissive outlier rule: Q1 - 15 IQR to Q3 + 15 IQR."""
    cleaned = df.copy()

    for column in columns:
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 15.0 * iqr
        upper = q3 + 15.0 * iqr
        cleaned = cleaned[(cleaned[column] >= lower) & (cleaned[column] <= upper)]

    return cleaned


def apply_butterworth_filter(data: np.ndarray, cutoff: float, order: int) -> np.ndarray:
    """Apply the same low-pass Butterworth filtering strategy used in the original code."""
    required_length = 3 * order
    if len(data) <= required_length:
        print(f"    Warning: data length ({len(data)}) is too short for filter order {order}.")
        return data

    try:
        b, a = butter(order, cutoff, btype="low")
        padlen = min(2 * order, len(data) - 1)
        return np.clip(filtfilt(b, a, data, padlen=padlen), 0, None)
    except Exception as exc:
        print(f"    Filter error: {exc}")
        return data


def minmax_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    spans = maxs - mins
    spans[spans == 0] = 1.0
    return (x - mins) / spans, mins, maxs


def minmax_inverse(x_scaled: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    spans = maxs - mins
    spans[spans == 0] = 1.0
    return x_scaled * spans + mins


def prepare_interval_data(
    start_age: float,
    end_age: float,
    cutoff: float,
    order: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare one interval using the original ordering of operations:
    interpolation -> outlier removal -> age grouping -> P rescaling -> filtering -> scaling.
    """
    df = load_input_data()
    interval_df = df[(df["age"] >= start_age) & (df["age"] <= end_age)].copy()
    original_count = len(interval_df)

    if interval_df.empty:
        raise ValueError(f"No observations found for {start_age}-{end_age} Ma.")

    interval_df = remove_outliers_iqr_permissive(
        interval_df,
        columns=["toc", "pyrite", "p", "ironspec"],
    )
    print(f"  Data points: {original_count} -> {len(interval_df)} after outlier removal")

    if interval_df.empty:
        raise ValueError(f"No observations remain after outlier removal for {start_age}-{end_age} Ma.")

    interval_df["rounded_age"] = interval_df["age"].round(1)
    grouped_data = interval_df.groupby("rounded_age").mean(numeric_only=True).reset_index()

    # Original numerical rescaling used before filtering and fitting.
    grouped_data["p"] = grouped_data["p"] / 10000.0

    for column in VARIABLE_NAMES:
        grouped_data[f"{column}_raw"] = grouped_data[column].copy()
        grouped_data[f"{column}_smooth"] = apply_butterworth_filter(
            grouped_data[column].to_numpy(),
            cutoff=cutoff,
            order=order,
        )

    t = grouped_data["age"].to_numpy()

    x_smooth = np.column_stack(
        [grouped_data[f"{column}_smooth"].to_numpy() for column in VARIABLE_NAMES]
    )
    x_raw = np.column_stack(
        [grouped_data[f"{column}_raw"].to_numpy() for column in VARIABLE_NAMES]
    )

    x_smooth_norm, norm_mins, norm_maxs = minmax_scale(x_smooth)
    return grouped_data, t, x_smooth, x_raw, x_smooth_norm, norm_mins, norm_maxs


# =============================================================================
# MODEL AND METRICS
# =============================================================================

def calculate_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R2 implementation retained from the original workflow."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if np.var(y_true) < 1e-8:
        return float("nan")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1.0 - ss_res / ss_tot)


def calculate_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variable_names: list[str],
) -> dict:
    """Calculate overall and variable-level RMSE/R2 values."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Metric arrays must have the same shape; got {y_true.shape} and {y_pred.shape}."
        )

    metrics = {
        "mse": float(mean_squared_error(y_true.flatten(), y_pred.flatten())),
        "rmse": float(np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))),
        "r2_overall": calculate_r2_score(y_true.flatten(), y_pred.flatten()),
    }

    for index, variable in enumerate(variable_names):
        metrics[f"{variable}_rmse"] = float(
            np.sqrt(mean_squared_error(y_true[:, index], y_pred[:, index]))
        )
        metrics[f"{variable}_r2"] = calculate_r2_score(
            y_true[:, index],
            y_pred[:, index],
        )

    return metrics


def make_sindy_model(alpha: float) -> ps.SINDy:
    """Create the first-order SINDy model used in the original analysis."""
    optimizer = ps.STLSQ(
        threshold=GLOBAL_THRESHOLD,
        alpha=1000,
        max_iter=100,
        normalize_columns=True,
    )

    return ps.SINDy(
        differentiation_method=ps.SINDyDerivative(kind="kalman", alpha=alpha),
        feature_library=ps.PolynomialLibrary(degree=POLY_DEGREE),
        optimizer=optimizer,
        feature_names=VARIABLE_NAMES,
    )


def enforce_coefficient_limit(model: ps.SINDy) -> bool:
    """Clip fitted coefficients only if their magnitude exceeds MAX_COEFFICIENT."""
    if not hasattr(model, "optimizer") or not hasattr(model.optimizer, "coef_"):
        return False

    coefficients = model.coefficients()
    max_abs = np.max(np.abs(coefficients))

    if max_abs > MAX_COEFFICIENT:
        print(f"  Clipping coefficients: max {max_abs:.4f} -> {MAX_COEFFICIENT}")
        model.optimizer.coef_ = np.clip(coefficients, -MAX_COEFFICIENT, MAX_COEFFICIENT)
        return True

    return False


def fit_with_coefficient_limit(model: ps.SINDy, x: np.ndarray, t: np.ndarray) -> tuple[ps.SINDy, bool]:
    """Fit with ensemble mode and enforce the original post-fit coefficient bound."""
    model.fit(x, t=t, ensemble=True)
    return model, enforce_coefficient_limit(model)


def coefficient_summary(model: ps.SINDy) -> dict:
    """Return coefficient diagnostics without writing per-interval CSV files."""
    coefficients = model.coefficients()
    active = coefficients[np.abs(coefficients) > 1e-10]

    return {
        "max_abs_coefficient": float(np.max(np.abs(coefficients))),
        "mean_abs_active_coefficient": float(np.mean(np.abs(active))) if active.size else 0.0,
        "n_nonzero_terms": int(active.size),
    }


def simulate_on_uniform_grid(
    model: ps.SINDy,
    x0: np.ndarray,
    t_span: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Simulate on a uniform age grid and interpolate results back to observed ages."""
    if len(t_span) < 2:
        raise ValueError("At least two time points are required for simulation.")

    t_uniform = np.arange(t_span[0], t_span[-1] + dt / 2, dt)

    simulated_uniform = model.simulate(
        x0,
        t_uniform,
        integrator="solve_ivp",
        integrator_kws={"min_step": 1e-14},
    )

    if simulated_uniform.shape[0] < len(t_uniform):
        t_uniform = t_uniform[: simulated_uniform.shape[0]]

    simulated_original = np.zeros((len(t_span), simulated_uniform.shape[1]))
    for index in range(simulated_uniform.shape[1]):
        interpolator = interp1d(
            t_uniform,
            simulated_uniform[:, index],
            kind="linear",
            fill_value="extrapolate",
        )
        simulated_original[:, index] = interpolator(t_span)

    return simulated_original


# =============================================================================
# CROSS-VALIDATION AND JSON REPORTING
# =============================================================================

def compute_adaptive_n_splits(start_age: float, end_age: float) -> tuple[int, int]:
    """Use the original age-count rule for requested TimeSeriesSplit folds."""
    try:
        df = load_input_data()
        mask = (df["age"] >= start_age) & (df["age"] <= end_age)
        n_points = int(df.loc[mask, "age"].round(1).nunique())
    except Exception:
        n_points = 20

    n_splits = max(MIN_FOLDS, min(MAX_FOLDS, n_points // PTS_PER_FOLD))
    return n_splits, n_points


def safe_n_splits(requested: int, n_points: int) -> int:
    """Guarantee a feasible fold count for the final grouped data."""
    data_limited = n_points // MIN_TRAIN_PTS
    effective = min(requested, data_limited)
    effective = max(MIN_FOLDS, effective)
    return min(MAX_FOLDS, effective)


def to_json_safe(value):
    """Convert NumPy scalars and non-finite values to JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]

    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())

    return value


def save_interval_validation_json(
    all_fold_results: list[dict],
    top_results: list[dict],
    start_age: float,
    end_age: float,
    cutoff: float,
    order: int,
    full_interval_r2: float,
    coefficient_info: dict,
) -> dict:
    """Save one complete evaluation JSON per interval; no fold-level CSVs are created."""
    valid_folds = [
        result for result in all_fold_results
        if np.isfinite(result.get("rmse", np.inf))
    ]

    filtered_scores = [result["r2_overall"] for result in top_results]
    raw_scores = [result["r2_overall_raw"] for result in top_results]

    mean_top_filtered_r2 = float(np.mean(filtered_scores)) if filtered_scores else float("nan")
    mean_top_raw_r2 = float(np.mean(raw_scores)) if raw_scores else float("nan")

    report = {
        "interval": f"{start_age}-{end_age} Ma",
        "interval_start_ma": start_age,
        "interval_end_ma": end_age,
        "filter_cutoff": cutoff,
        "filter_order": order,
        "top_n_requested": TOP_N_RESULTS,
        "n_valid_folds": len(valid_folds),
        "full_interval_r2": full_interval_r2,
        "best_filtered_validation_r2": top_results[0]["r2_overall"] if top_results else None,
        "mean_top_ranked_filtered_validation_r2": mean_top_filtered_r2,
        "std_top_ranked_filtered_validation_r2": (
            float(np.std(filtered_scores)) if filtered_scores else float("nan")
        ),
        "best_raw_validation_r2": top_results[0]["r2_overall_raw"] if top_results else None,
        "mean_top_ranked_raw_validation_r2": mean_top_raw_r2,
        "best_validation_rmse": top_results[0]["rmse"] if top_results else None,
        "delta_r2_abs_full_vs_mean_top_ranked": abs(full_interval_r2 - mean_top_filtered_r2),
        "max_coefficient_limit": MAX_COEFFICIENT,
        "coefficient_summary": coefficient_info,
        "all_fold_results": valid_folds,
        "top_ranked_folds": top_results,
    }

    json_path = RESULTS_DIR / f"unseen_r2_summary_{start_age}_{end_age}Ma.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(to_json_safe(report), file, indent=2)

    print(f"  Saved validation JSON: {json_path}")
    return report


# =============================================================================
# ONLY FIGURES WRITTEN BY THIS SCRIPT
# =============================================================================

def export_individual_variable_plots(
    t: np.ndarray,
    grouped_data: pd.DataFrame,
    simulated: np.ndarray,
    start_age: float,
    end_age: float,
    full_metrics: dict,
) -> None:
    """Write only the three up_clean figures required for each interval."""
    output_dirs = [FIGURES_TOC, FIGURES_FEPY, FIGURES_P]

    for index, (label, color, key, output_dir) in enumerate(
        zip(PLOT_LABELS, PLOT_COLORS, VARIABLE_NAMES, output_dirs)
    ):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(
            t,
            grouped_data[f"{key}_raw"],
            alpha=0.4,
            color="gray",
            label="Raw data",
        )
        ax.plot(
            t,
            grouped_data[f"{key}_smooth"],
            color=color,
            linewidth=2,
            label="Filtered",
        )
        ax.plot(
            t,
            simulated[:, index],
            "--",
            color="black",
            linewidth=2,
            label="Model",
        )

        r2 = full_metrics.get(f"{key}_r2", float("nan"))
        ax.set_title(f"{label} ({start_age}--{end_age} Ma)\n$R^2$ = {r2:.3f}", fontsize=14)
        ax.set_xlabel("Age (Ma)", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.tick_params(axis="both", labelsize=12)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=11, frameon=False)
        ax.grid(alpha=0.2)
        fig.tight_layout()

        figure_path = output_dir / f"{key}_up_clean_{start_age}_{end_age}.png"
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved up_clean figure: {figure_path}")


# =============================================================================
# INTERVAL EXECUTION
# =============================================================================

def fit_full_interval_model(
    t: np.ndarray,
    x_smooth: np.ndarray,
    x_raw: np.ndarray,
    x_smooth_norm: np.ndarray,
    norm_mins: np.ndarray,
    norm_maxs: np.ndarray,
    dt: float,
) -> tuple[ps.SINDy, np.ndarray, dict, dict]:
    """Fit the final model on all interval points and calculate full-interval metrics."""
    final_model = make_sindy_model(alpha=0.3)
    final_model, was_clipped = fit_with_coefficient_limit(final_model, x_smooth_norm, t=t)

    if was_clipped:
        print(f"  Coefficients clipped to max {MAX_COEFFICIENT}")

    simulated_norm = simulate_on_uniform_grid(final_model, x_smooth_norm[0], t, dt)
    simulated = minmax_inverse(simulated_norm, norm_mins, norm_maxs)

    filtered_metrics = calculate_performance_metrics(x_smooth, simulated, VARIABLE_NAMES)
    raw_metrics = calculate_performance_metrics(x_raw, simulated, VARIABLE_NAMES)

    return final_model, simulated, filtered_metrics, raw_metrics


def apply_butterworth_and_sindy_with_cv(
    cutoff: float,
    order: int,
    start_age: float,
    end_age: float,
    requested_n_splits: int,
) -> tuple[ps.SINDy, list[dict], dict, dict, np.ndarray, np.ndarray, pd.DataFrame]:
    """Process, validate, fit, plot, and report one geological interval."""
    print(f"\n{'=' * 70}")
    print(f"PROCESSING INTERVAL: {start_age}-{end_age} Ma")
    print(f"{'=' * 70}")

    (
        grouped_data,
        t,
        x_smooth,
        x_raw,
        x_smooth_norm,
        norm_mins,
        norm_maxs,
    ) = prepare_interval_data(start_age, end_age, cutoff, order)

    if len(t) < 2:
        raise ValueError(f"Too few final points for {start_age}-{end_age} Ma.")

    dt = float(np.mean(np.diff(t)))
    print(f"  Final data points: {len(t)}")
    print(f"  Average time step: {dt:.3f} Myr")
    print(f"  Effective coefficient limit: {MAX_COEFFICIENT}")

    effective_splits = safe_n_splits(requested_n_splits, len(t))
    print(f"  Effective CV folds: {effective_splits}")

    all_fold_results: list[dict] = []
    cv_models: list[ps.SINDy] = []

    can_cross_validate = len(t) >= MIN_TRAIN_PTS * 2 and effective_splits >= MIN_FOLDS

    if can_cross_validate:
        print(f"\n{'-' * 50}")
        print("CROSS-VALIDATION (FILTERED VALIDATION PERFORMANCE)")
        print(f"{'-' * 50}")

        splitter = TimeSeriesSplit(n_splits=effective_splits)

        for fold_number, (train_idx, validation_idx) in enumerate(
            splitter.split(x_smooth_norm),
            start=1,
        ):
            print(f"\nFold {fold_number}/{effective_splits}:")
            print(f"  Train: {len(train_idx)} pts")
            print(f"  Validation: {len(validation_idx)} pts")

            x_train = x_smooth_norm[train_idx]
            t_train = t[train_idx]
            x_validation_smooth = x_smooth[validation_idx]
            x_validation_raw = x_raw[validation_idx]
            t_validation_observed = t[validation_idx]

            model = make_sindy_model(alpha=0.1)
            model, was_clipped = fit_with_coefficient_limit(model, x_train, t=t_train)
            cv_models.append(model)

            if was_clipped:
                print(f"  Coefficients clipped to max {MAX_COEFFICIENT}")

            try:
                t_validation = np.arange(
                    t_train[-1],
                    t_validation_observed[-1] + dt / 2,
                    dt,
                )

                simulated_validation_norm = model.simulate(
                    x_train[-1],
                    t_validation,
                    integrator="solve_ivp",
                    integrator_kws={"min_step": 1e-8},
                )

                if simulated_validation_norm.shape[0] < len(t_validation):
                    t_validation = t_validation[: simulated_validation_norm.shape[0]]

                validation_positions = [
                    int(np.argmin(np.abs(t_validation - current_time)))
                    for current_time in t_validation_observed
                ]
                simulated_validation_norm = simulated_validation_norm[validation_positions]
                simulated_validation = minmax_inverse(
                    simulated_validation_norm,
                    norm_mins,
                    norm_maxs,
                )

                filtered_metrics = calculate_performance_metrics(
                    x_validation_smooth,
                    simulated_validation,
                    VARIABLE_NAMES,
                )
                raw_metrics = calculate_performance_metrics(
                    x_validation_raw,
                    simulated_validation,
                    VARIABLE_NAMES,
                )

                fold_result = {
                    "fold": fold_number,
                    "filtered_mse": filtered_metrics["mse"],
                    "rmse": filtered_metrics["rmse"],
                    "r2_overall": filtered_metrics["r2_overall"],
                    "toc_r2": filtered_metrics["toc_r2"],
                    "pyrite_r2": filtered_metrics["pyrite_r2"],
                    "p_r2": filtered_metrics["p_r2"],
                    "raw_rmse": raw_metrics["rmse"],
                    "r2_overall_raw": raw_metrics["r2_overall"],
                    "toc_r2_raw": raw_metrics["toc_r2"],
                    "pyrite_r2_raw": raw_metrics["pyrite_r2"],
                    "p_r2_raw": raw_metrics["p_r2"],
                    "model_index": len(cv_models) - 1,
                    "n_train_points": len(train_idx),
                    "n_validation_points": len(validation_idx),
                }
                all_fold_results.append(fold_result)

                print(f"  Filtered validation R2 = {filtered_metrics['r2_overall']:.4f}")
                print(
                    "  Variable R2: "
                    f"TOC={filtered_metrics['toc_r2']:.3f}, "
                    f"Pyrite={filtered_metrics['pyrite_r2']:.3f}, "
                    f"P={filtered_metrics['p_r2']:.3f}"
                )

            except Exception as exc:
                print(f"  Validation failed: {exc}")
                all_fold_results.append(
                    {
                        "fold": fold_number,
                        "rmse": float("inf"),
                        "r2_overall": float("-inf"),
                        "raw_rmse": float("inf"),
                        "r2_overall_raw": float("-inf"),
                        "model_index": len(cv_models) - 1,
                        "n_train_points": len(train_idx),
                        "n_validation_points": len(validation_idx),
                    }
                )
    else:
        print("  Cross-validation skipped: too few final interval points.")

    valid_results = [
        result for result in all_fold_results
        if np.isfinite(result.get("rmse", np.inf))
    ]
    valid_results.sort(key=lambda result: (-result["r2_overall"], result["rmse"]))
    top_results = valid_results[:TOP_N_RESULTS]

    if top_results:
        print(f"\n{'-' * 50}")
        print(f"TOP {len(top_results)} VALIDATION FOLDS")
        print(f"{'-' * 50}")
        for rank, result in enumerate(top_results, start=1):
            print(
                f"  Rank {rank} (fold {result['fold']}): "
                f"filtered validation R2={result['r2_overall']:.4f}, "
                f"RMSE={result['rmse']:.4f}"
            )

    print(f"\nFitting final model on all {len(t)} points...")
    final_model, simulated, full_metrics, full_raw_metrics = fit_full_interval_model(
        t=t,
        x_smooth=x_smooth,
        x_raw=x_raw,
        x_smooth_norm=x_smooth_norm,
        norm_mins=norm_mins,
        norm_maxs=norm_maxs,
        dt=dt,
    )

    coefficient_info = coefficient_summary(final_model)
    print(f"  Max |coefficient|: {coefficient_info['max_abs_coefficient']:.4f}")
    print(f"  Active terms: {coefficient_info['n_nonzero_terms']}")

    print("\nFinal model equations:")
    final_model.print()

    print(f"\nFull-interval filtered R2 = {full_metrics['r2_overall']:.4f}")
    if top_results:
        mean_top_r2 = float(np.mean([result["r2_overall"] for result in top_results]))
        print(f"Mean top-ranked filtered validation R2 = {mean_top_r2:.4f}")
        print(f"Best validation RMSE = {top_results[0]['rmse']:.4f}")

    validation_report = save_interval_validation_json(
        all_fold_results=all_fold_results,
        top_results=top_results,
        start_age=start_age,
        end_age=end_age,
        cutoff=cutoff,
        order=order,
        full_interval_r2=full_metrics["r2_overall"],
        coefficient_info=coefficient_info,
    )

    export_individual_variable_plots(
        t=t,
        grouped_data=grouped_data,
        simulated=simulated,
        start_age=start_age,
        end_age=end_age,
        full_metrics=full_metrics,
    )

    # The raw full-interval metrics are retained in memory only. The selected
    # raw validation statistics are saved in the interval JSON report.
    _ = full_raw_metrics

    return (
        final_model,
        top_results,
        full_metrics,
        validation_report,
        t,
        simulated,
        grouped_data,
    )


# =============================================================================
# FINAL TABLE AND COMBINED JSON
# =============================================================================

def summarize_all_results(
    all_results: dict[
        tuple[int, int],
        tuple[ps.SINDy, list[dict], dict, dict, np.ndarray, np.ndarray, pd.DataFrame],
    ],
) -> None:
    """
    Write the single CSV evaluation table and the global JSON summary.

    The in-sample RMSE column is intentionally omitted. The table reports full-interval R2,
    validation R2, validation RMSE, and the absolute R2 difference.
    """
    print(f"\n{'=' * 90}")
    print("SUMMARY OF RESULTS ACROSS ALL INTERVALS")
    print(f"{'=' * 90}")

    table_rows: list[dict] = []
    interval_reports: list[dict] = []

    for (start_age, end_age), result in all_results.items():
        (
            _model,
            top_results,
            full_metrics,
            validation_report,
            t,
            _simulated,
            _grouped_data,
        ) = result

        if top_results:
            mean_top_r2 = float(np.mean([item["r2_overall"] for item in top_results]))
            std_top_r2 = float(np.std([item["r2_overall"] for item in top_results]))
            best_result = top_results[0]
            best_validation_r2 = float(best_result["r2_overall"])
            best_validation_rmse = float(best_result["rmse"])
            delta_r2 = abs(float(full_metrics["r2_overall"]) - mean_top_r2)
        else:
            mean_top_r2 = float("nan")
            std_top_r2 = float("nan")
            best_validation_r2 = float("nan")
            best_validation_rmse = float("nan")
            delta_r2 = float("nan")

        table_rows.append(
            {
                "Interval": f"{start_age}-{end_age} Ma",
                "Points": int(len(t)),
                "Full_interval_R2": round(float(full_metrics["r2_overall"]), 4),
                "Best_validation_R2": round(best_validation_r2, 4),
                "Mean_top_ranked_validation_R2": round(mean_top_r2, 4),
                "Std_top_ranked_validation_R2": round(std_top_r2, 4),
                "Best_validation_RMSE": round(best_validation_rmse, 4),
                "Delta_R2_abs_full_vs_mean_top_ranked": round(delta_r2, 4),
            }
        )

        if validation_report:
            interval_reports.append(validation_report)

    summary_df = pd.DataFrame(table_rows)
    print(summary_df.to_string(index=False))

    table_path = RESULTS_DIR / "model_evaluation_table.csv"
    summary_df.to_csv(table_path, index=False)
    print(f"\nSaved evaluation table: {table_path}")

    global_json = {
        "configuration": {
            "polynomial_degree": POLY_DEGREE,
            "top_n_ranked_folds": TOP_N_RESULTS,
            "stlsq_threshold": GLOBAL_THRESHOLD,
            "stlsq_alpha": 1000,
            "stlsq_max_iterations": 100,
            "max_coefficient_limit": MAX_COEFFICIENT,
            "outlier_rule": "Q1 - 15*IQR to Q3 + 15*IQR",
        },
        "total_intervals": len(table_rows),
        "evaluation_table": table_rows,
        "interval_reports": interval_reports,
    }

    json_path = RESULTS_DIR / "all_results_summary.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(to_json_safe(global_json), file, indent=2)

    print(f"Saved global evaluation JSON: {json_path}")


# =============================================================================
# EXECUTION
# =============================================================================

def main() -> None:
    prepare_output_directories()

    print("\n" + "=" * 90)
    print("SINDY MODEL ANALYSIS")
    print("=" * 90)
    print(f"Input data: {Path(DATA_FILE).resolve()}")
    print(f"Figures: {FIGURES_DIR.resolve()}")
    print(f"Results: {RESULTS_DIR.resolve()}")

    print("\nAdaptive fold calculation:")
    print(f"{'Interval':<15} {'Raw age points':>15} {'Requested folds':>17}")
    print("-" * 52)

    requested_splits: dict[tuple[int, int], int] = {}
    for start_age, end_age in INTERVAL_CONFIGS:
        n_splits, n_points = compute_adaptive_n_splits(start_age, end_age)
        requested_splits[(start_age, end_age)] = n_splits
        print(f"{start_age}-{end_age} Ma{'':<5} {n_points:>15} {n_splits:>17}")

    all_results = {}
    for (start_age, end_age), params in INTERVAL_CONFIGS.items():
        all_results[(start_age, end_age)] = apply_butterworth_and_sindy_with_cv(
            cutoff=params["cutoff"],
            order=params["order"],
            start_age=start_age,
            end_age=end_age,
            requested_n_splits=requested_splits[(start_age, end_age)],
        )

    summarize_all_results(all_results)

    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print("Created only up_clean figures, interval JSON reports, one global JSON,")
    print("and model_evaluation_table.csv. Figures are saved without interactive display.")
    print("=" * 90)


if __name__ == "__main__":
    main()
