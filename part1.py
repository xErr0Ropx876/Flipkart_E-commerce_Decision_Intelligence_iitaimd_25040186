"""Part 1 — Estimation, inference and a regression baseline.

Produces: outputs/part1_results.txt
"""

import math
import numpy as np
from common import load_records, sample_mean, sample_sd, write_results_txt


def closed_form_simple_regression(x, y):
    """Fit y = intercept + slope * x manually using sample covariance and variance."""
    n = len(x)
    mx = sample_mean(x)
    my = sample_mean(y)
    dx = x - mx
    dy = y - my
    cov_xy = float(np.sum(dx * dy) / (n - 1))
    var_x = float(np.sum(dx * dx) / (n - 1))
    slope = cov_xy / var_x
    intercept = my - slope * mx
    fitted = intercept + slope * x

    tss = float(np.sum((y - my) ** 2))
    ess = float(np.sum((fitted - my) ** 2))
    rss = float(np.sum((y - fitted) ** 2))
    decomp_ok = abs(tss - ess - rss) < 1e-6 * tss
    r2 = 1.0 - rss / tss
    rmse = math.sqrt(rss / n)
    mae = float(np.mean(np.abs(y - fitted)))

    return {
        "slope": slope,
        "intercept": intercept,
        "fitted": fitted,
        "tss": tss,
        "ess": ess,
        "rss": rss,
        "decomp_ok": decomp_ok,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }


def gradient_descent_simple_regression(x, y, lr=0.1, updates=500):
    """Fit y on standardised x via full-batch gradient descent, then convert back."""
    n = len(x)
    mean_x = sample_mean(x)
    sd_x = sample_sd(x)
    z = (x - mean_x) / sd_x

    m = 0.0
    c = 0.0
    snapshot_steps = {0, 1, 10, 100, 500}
    costs = {}

    pred0 = m * z + c
    costs[0] = float(np.mean((y - pred0) ** 2))

    for step in range(1, updates + 1):
        pred = m * z + c
        err = y - pred
        dm = -(2.0 / n) * float(np.sum(z * err))
        dc = -(2.0 / n) * float(np.sum(err))
        m = m - lr * dm
        c = c - lr * dc

        if step in snapshot_steps:
            pred_after = m * z + c
            costs[step] = float(np.mean((y - pred_after) ** 2))

    gd_slope = m / sd_x
    gd_intercept = c - m * mean_x / sd_x
    return costs, gd_slope, gd_intercept


def residual_funnel_stats(fitted, residuals):
    """Sort rows stably by fitted values ascending and compare upper/lower half residual SDs."""
    order = np.argsort(fitted, kind="stable")
    half = len(fitted) // 2
    lower_resid = residuals[order[:half]]
    upper_resid = residuals[order[half:]]
    sd_lower = sample_sd(lower_resid)
    sd_upper = sample_sd(upper_resid)
    ratio = sd_upper / sd_lower
    return sd_lower, sd_upper, ratio


def main():
    _, X_raw, _, y_value = load_records("data/records.csv")
    n = len(y_value)
    x4 = X_raw[:, 3]  # avg_order_value_inr_hundreds

    # Task 1.1: Estimate with an interval
    mean_value = sample_mean(y_value)
    sd_value = sample_sd(y_value)
    se_value = sd_value / math.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    # Task 1.2: Hypothesis test (H0: mu = 8.0)
    z_stat = (mean_value - 8.0) / se_value
    reject_h0 = abs(z_stat) > 1.96

    # Task 1.3: Required sample size for margin of error E = 0.5
    n_required = math.ceil((1.96 * sd_value / 0.5) ** 2)

    # Task 1.4: Simple regression by closed form (value ~ X4)
    raw_reg = closed_form_simple_regression(x4, y_value)

    # Task 1.5: Gradient descent for the same regression
    gd_costs, gd_slope, gd_intercept = gradient_descent_simple_regression(
        x4, y_value, lr=0.1, updates=500
    )
    gd_slope_gap = abs(gd_slope - raw_reg["slope"])

    # Task 1.6: Residual funnel check and log-log remedy
    resid_raw = y_value - raw_reg["fitted"]
    resid_sd_lower_raw, resid_sd_upper_raw, resid_sd_ratio_raw = residual_funnel_stats(
        raw_reg["fitted"], resid_raw
    )

    ln_x4 = np.log(x4)
    ln_y = np.log(y_value)
    log_reg = closed_form_simple_regression(ln_x4, ln_y)
    resid_log = ln_y - log_reg["fitted"]
    resid_sd_lower_log, resid_sd_upper_log, resid_sd_ratio_log = residual_funnel_stats(
        log_reg["fitted"], resid_log
    )

    # Task 1.7: Multiple regression by normal equation on ln(value)
    # Design matrix order: [1, ln(X4), X1, X2, X3, X5, X6, X7]
    ones = np.ones(n, dtype=np.float64)
    X_mr = np.column_stack(
        [
            ones,
            ln_x4,
            X_raw[:, 0],  # X1
            X_raw[:, 1],  # X2
            X_raw[:, 2],  # X3
            X_raw[:, 4],  # X5
            X_raw[:, 5],  # X6
            X_raw[:, 6],  # X7
        ]
    )
    beta = np.linalg.solve(X_mr.T @ X_mr, X_mr.T @ ln_y)
    fitted_mr = X_mr @ beta
    tss_mr = float(np.sum((ln_y - sample_mean(ln_y)) ** 2))
    rss_mr = float(np.sum((ln_y - fitted_mr) ** 2))
    mr_r2 = 1.0 - rss_mr / tss_mr
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - 7 - 1)
    loglog_adj_r2 = 1.0 - (1.0 - log_reg["r2"]) * (n - 1) / (n - 1 - 1)

    results = [
        ("n", n),
        ("mean_value", mean_value),
        ("sd_value", sd_value),
        ("se_value", se_value),
        ("ci95_low", ci95_low),
        ("ci95_high", ci95_high),
        ("z_stat", z_stat),
        ("reject_h0", reject_h0),
        ("n_required", n_required),
        ("reg_slope", raw_reg["slope"]),
        ("reg_intercept", raw_reg["intercept"]),
        ("tss", raw_reg["tss"]),
        ("ess", raw_reg["ess"]),
        ("rss", raw_reg["rss"]),
        ("tss_equals_ess_plus_rss", raw_reg["decomp_ok"]),
        ("r2", raw_reg["r2"]),
        ("rmse", raw_reg["rmse"]),
        ("mae", raw_reg["mae"]),
        ("gd_J0", gd_costs[0]),
        ("gd_J1", gd_costs[1]),
        ("gd_J10", gd_costs[10]),
        ("gd_J100", gd_costs[100]),
        ("gd_J500", gd_costs[500]),
        ("gd_slope", gd_slope),
        ("gd_intercept", gd_intercept),
        ("gd_slope_gap", gd_slope_gap),
        ("resid_sd_lower_raw", resid_sd_lower_raw),
        ("resid_sd_upper_raw", resid_sd_upper_raw),
        ("resid_sd_ratio_raw", resid_sd_ratio_raw),
        ("loglog_intercept", log_reg["intercept"]),
        ("loglog_slope", log_reg["slope"]),
        ("loglog_r2", log_reg["r2"]),
        ("loglog_rmse", log_reg["rmse"]),
        ("resid_sd_lower_log", resid_sd_lower_log),
        ("resid_sd_upper_log", resid_sd_upper_log),
        ("resid_sd_ratio_log", resid_sd_ratio_log),
        ("mr_beta_0", float(beta[0])),
        ("mr_beta_1", float(beta[1])),
        ("mr_beta_2", float(beta[2])),
        ("mr_beta_3", float(beta[3])),
        ("mr_beta_4", float(beta[4])),
        ("mr_beta_5", float(beta[5])),
        ("mr_beta_6", float(beta[6])),
        ("mr_beta_7", float(beta[7])),
        ("mr_r2", mr_r2),
        ("mr_adj_r2", mr_adj_r2),
        ("loglog_adj_r2", loglog_adj_r2),
    ]

    write_results_txt("outputs/part1_results.txt", results)


if __name__ == "__main__":
    main()
