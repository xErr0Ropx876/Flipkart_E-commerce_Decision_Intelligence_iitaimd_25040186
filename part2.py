"""Part 2 — Classification and evaluation.

Produces:
    outputs/part2_results.txt
    outputs/part2_thresholds.csv
    outputs/part2_knn_cv.csv
"""

import math
import numpy as np
from common import (
    binary_cross_entropy,
    classification_metrics,
    confusion_matrix,
    knn_cross_validate,
    knn_predict,
    load_records,
    predict_logistic_proba,
    standardise_train_test,
    stratified_split,
    train_logistic_regression,
    write_csv,
    write_results_txt,
)


def evaluate_at_threshold(y_true, y_prob, tau):
    """Threshold probabilities at tau and return (tp, fp, tn, fn, metrics_dict)."""
    y_pred = (y_prob >= tau).astype(np.int64)
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    metrics = classification_metrics(tp, fp, tn, fn)
    return tp, fp, tn, fn, metrics


def main():
    _, X_raw, y_outcome, _ = load_records("data/records.csv")

    # Task 2.1: Deterministic stratified 70/30 split and training-only standardisation
    X_train_raw, X_test_raw, y_train, y_test, _, _ = stratified_split(X_raw, y_outcome)
    X_train_std, X_test_std, _, _ = standardise_train_test(X_train_raw, X_test_raw)

    train_rows = len(y_train)
    train_positives = int(np.sum(y_train))
    test_rows = len(y_test)
    test_positives = int(np.sum(y_test))

    # Task 2.2: Logistic regression from scratch (lr=0.1, epochs=1000)
    weights, bias, snapshots = train_logistic_regression(
        X_train_std,
        y_train,
        lr=0.1,
        epochs=1000,
        snapshot_epochs={0, 1, 10, 100, 1000},
    )
    train_probs = predict_logistic_proba(X_train_std, weights, bias)
    test_probs = predict_logistic_proba(X_test_std, weights, bias)
    test_bce = binary_cross_entropy(y_test, test_probs)

    # Task 2.3: Learning-rate effect (200 updates from zero)
    lr_losses = {}
    for eta in (0.01, 0.1, 1.0):
        _, _, sn = train_logistic_regression(
            X_train_std, y_train, lr=eta, epochs=200, snapshot_epochs={200}
        )
        lr_losses[eta] = sn[200]

    # Task 2.4: Odds ratios = exp(weight)
    odds_ratios = [math.exp(float(w)) for w in weights]

    # Task 2.5: Evaluate on test rows at tau = 0.5 and tau = 0.3
    eval_05 = evaluate_at_threshold(y_test, test_probs, 0.5)
    eval_03 = evaluate_at_threshold(y_test, test_probs, 0.3)
    blind_accuracy = (test_rows - test_positives) / test_rows

    # Task 2.6: Threshold sweep for tau in 0.1, ..., 0.9
    threshold_rows = []
    for k in range(1, 10):
        tau = k / 10.0
        tp, fp, tn, fn, metrics = evaluate_at_threshold(y_test, test_probs, tau)
        threshold_rows.append([tau, tp, fp, tn, fn, metrics["recall"], metrics["fpr"]])
    write_csv(
        "outputs/part2_thresholds.csv",
        ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"],
        threshold_rows,
    )

    # Task 2.7: K-nearest neighbours with 5-fold CV on training rows
    cv_results, knn_best_k = knn_cross_validate(X_train_std, y_train)
    write_csv("outputs/part2_knn_cv.csv", ["K", "mean_cv_f1"], cv_results)

    knn_test_preds, _ = knn_predict(X_train_std, y_train, X_test_std, knn_best_k)
    knn_tp, knn_fp, knn_tn, knn_fn = confusion_matrix(y_test, knn_test_preds)
    knn_metrics = classification_metrics(knn_tp, knn_fp, knn_tn, knn_fn)

    # Task 2.8: Economically optimal alert threshold on training rows
    # Cost = 50 * fp + 400 * fn; ties go to smaller tau
    best_cost_tau = None
    best_train_cost = None
    for k in range(1, 20):
        tau = k / 20.0
        _, fp, _, fn, _ = evaluate_at_threshold(y_train, train_probs, tau)
        cost = 50 * fp + 400 * fn
        if best_train_cost is None or cost < best_train_cost:
            best_train_cost = cost
            best_cost_tau = tau

    _, test_fp_tau, _, test_fn_tau, _ = evaluate_at_threshold(
        y_test, test_probs, best_cost_tau
    )
    cost_test_at_tau = 50 * test_fp_tau + 400 * test_fn_tau
    cost_test_at_05 = 50 * eval_05[1] + 400 * eval_05[3]
    cost_test_flag_nobody = 400 * test_positives

    results = [
        ("train_rows", train_rows),
        ("train_positives", train_positives),
        ("test_rows", test_rows),
        ("test_positives", test_positives),
        ("loss_epoch_0", snapshots[0]),
        ("loss_epoch_1", snapshots[1]),
        ("loss_epoch_10", snapshots[10]),
        ("loss_epoch_100", snapshots[100]),
        ("loss_epoch_1000", snapshots[1000]),
    ]
    for j in range(7):
        results.append((f"weight_X{j + 1}", float(weights[j])))
    results.extend(
        [
            ("bias", float(bias)),
            ("test_bce", test_bce),
            ("loss_epoch_200_eta_0.01", lr_losses[0.01]),
            ("loss_epoch_200_eta_0.1", lr_losses[0.1]),
            ("loss_epoch_200_eta_1.0", lr_losses[1.0]),
        ]
    )
    for j in range(7):
        results.append((f"odds_ratio_X{j + 1}", odds_ratios[j]))

    for prefix, ev in (("tau_0.5", eval_05), ("tau_0.3", eval_03)):
        tp, fp, tn, fn, m = ev
        results.extend(
            [
                (f"{prefix}_tp", tp),
                (f"{prefix}_fp", fp),
                (f"{prefix}_tn", tn),
                (f"{prefix}_fn", fn),
                (f"{prefix}_accuracy", m["accuracy"]),
                (f"{prefix}_precision", m["precision"]),
                (f"{prefix}_recall", m["recall"]),
                (f"{prefix}_f1", m["f1"]),
                (f"{prefix}_specificity", m["specificity"]),
            ]
        )

    results.extend(
        [
            ("blind_accuracy", blind_accuracy),
            ("knn_best_k", knn_best_k),
            ("knn_tp", knn_tp),
            ("knn_fp", knn_fp),
            ("knn_tn", knn_tn),
            ("knn_fn", knn_fn),
            ("knn_accuracy", knn_metrics["accuracy"]),
            ("knn_precision", knn_metrics["precision"]),
            ("knn_recall", knn_metrics["recall"]),
            ("knn_f1", knn_metrics["f1"]),
            ("cost_tau", best_cost_tau),
            ("cost_train_at_tau", best_train_cost),
            ("cost_test_at_tau", cost_test_at_tau),
            ("cost_test_at_0.5", cost_test_at_05),
            ("cost_test_flag_nobody", cost_test_flag_nobody),
        ]
    )

    write_results_txt("outputs/part2_results.txt", results)


if __name__ == "__main__":
    main()
