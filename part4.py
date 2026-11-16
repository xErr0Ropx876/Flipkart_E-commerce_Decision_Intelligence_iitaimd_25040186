"""Part 4 — Model-choice and architecture memo.

Produces:
    outputs/part4_results.txt
    outputs/part4_splits.csv
    outputs/part4_voting.csv
    outputs/part4_adaboost.csv
    outputs/part4_architecture.csv
"""

import math
import numpy as np
from common import (
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


def node_entropy(positives, total):
    """Binary entropy in bits (-sum p_i log2 p_i), 0 for a pure or empty node."""
    if total == 0 or positives == 0 or positives == total:
        return 0.0
    p1 = positives / total
    p0 = 1.0 - p1
    return -(p1 * math.log2(p1) + p0 * math.log2(p0))


def node_gini(positives, total):
    """Binary Gini impurity (1 - sum p_i^2), 0 for a pure or empty node."""
    if total == 0:
        return 0.0
    p1 = positives / total
    p0 = 1.0 - p1
    return 1.0 - (p1 * p1 + p0 * p0)


def main():
    _, X_raw, y_outcome, _ = load_records("data/records.csv")

    # Reuse the exact Task 2.1 deterministic stratified 70/30 split
    X_train_raw, X_test_raw, y_train, y_test, _, _ = stratified_split(X_raw, y_outcome)
    X_train_std, X_test_std, _, _ = standardise_train_test(X_train_raw, X_test_raw)

    n_train = len(y_train)
    pos_train = int(np.sum(y_train))

    # Task 4.1: Decision-tree split purity on original unstandardised training rows
    root_entropy = node_entropy(pos_train, n_train)
    root_gini = node_gini(pos_train, n_train)

    split_specs = [
        ("S1", lambda X: X[:, 5] == 1.0),   # X6 = 1
        ("S2", lambda X: X[:, 6] == 1.0),   # X7 = 1
        ("S3", lambda X: X[:, 2] > 0.5),    # X3 > 0.5
        ("S4", lambda X: X[:, 0] <= 24.0),  # X1 <= 24
    ]

    split_rows = []
    best_split_info = None
    best_info_gain = -1.0
    best_split_gini = None
    best_gini_gain = -1.0
    split_conds = {}

    for name, cond_fn in split_specs:
        split_conds[name] = cond_fn
        yes_mask = cond_fn(X_train_raw)
        no_mask = ~yes_mask
        n_yes = int(np.sum(yes_mask))
        pos_yes = int(np.sum(y_train[yes_mask]))
        n_no = int(np.sum(no_mask))
        pos_no = int(np.sum(y_train[no_mask]))

        weighted_ent = (
            (n_yes / n_train) * node_entropy(pos_yes, n_yes)
            + (n_no / n_train) * node_entropy(pos_no, n_no)
        )
        weighted_gini = (
            (n_yes / n_train) * node_gini(pos_yes, n_yes)
            + (n_no / n_train) * node_gini(pos_no, n_no)
        )
        info_gain = root_entropy - weighted_ent
        gini_gain = root_gini - weighted_gini

        split_rows.append([name, n_yes, pos_yes, n_no, pos_no, info_gain, gini_gain])
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_split_info = name
        if gini_gain > best_gini_gain:
            best_gini_gain = gini_gain
            best_split_gini = name

    write_csv(
        "outputs/part4_splits.csv",
        ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"],
        split_rows,
    )

    # Task 4.2: Stump, hard voting, and soft voting on the 240 test rows
    best_cond = split_conds[best_split_info]
    train_yes = best_cond(X_train_raw)
    stump_rate_yes = float(np.sum(y_train[train_yes]) / np.sum(train_yes))
    stump_rate_no = float(np.sum(y_train[~train_yes]) / np.sum(~train_yes))

    # 1. Logistic (Task 2.2 model, threshold 0.3)
    weights, bias, _ = train_logistic_regression(X_train_std, y_train, lr=0.1, epochs=1000)
    log_probs = predict_logistic_proba(X_test_std, weights, bias)
    log_preds = (log_probs >= 0.3).astype(np.int64)

    # 2. KNN (Task 2.7 model with best K from 5-fold CV)
    _, knn_best_k = knn_cross_validate(X_train_std, y_train)
    knn_preds, knn_probs = knn_predict(X_train_std, y_train, X_test_std, knn_best_k)

    # 3. Stump (threshold 0.3)
    test_yes = best_cond(X_test_raw)
    stump_probs = np.where(test_yes, stump_rate_yes, stump_rate_no)
    stump_preds = (stump_probs >= 0.3).astype(np.int64)

    # 4. Hard vote (at least 2 of 3 positive)
    hard_preds = ((log_preds + knn_preds + stump_preds) >= 2).astype(np.int64)

    # 5. Soft vote (mean of 3 probabilities >= 0.3)
    soft_probs = (log_probs + knn_probs + stump_probs) / 3.0
    soft_preds = (soft_probs >= 0.3).astype(np.int64)

    voting_rows = []
    for model_name, preds in (
        ("logistic", log_preds),
        ("knn", knn_preds),
        ("stump", stump_preds),
        ("hard_vote", hard_preds),
        ("soft_vote", soft_preds),
    ):
        tp, fp, tn, fn = confusion_matrix(y_test, preds)
        m = classification_metrics(tp, fp, tn, fn)
        voting_rows.append(
            [model_name, tp, fp, tn, fn, m["precision"], m["recall"], m["f1"]]
        )

    write_csv(
        "outputs/part4_voting.csv",
        ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
        voting_rows,
    )

    # Task 4.3: Two rounds of AdaBoost manually
    y_ada = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=np.float64)
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1], dtype=np.float64)
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1], dtype=np.float64)

    w = np.full(8, 1.0 / 8.0, dtype=np.float64)
    adaboost_rows = []
    alphas = []
    errors = []

    for round_idx, h in ((1, h1), (2, h2)):
        err = float(np.sum(w[h != y_ada]))
        alpha = 0.5 * math.log((1.0 - err) / err)
        w_unnorm = w * np.exp(-alpha * y_ada * h)
        w_after = w_unnorm / np.sum(w_unnorm)

        errors.append(err)
        alphas.append(alpha)
        for pt in range(8):
            adaboost_rows.append(
                [round_idx, pt + 1, float(w[pt]), float(w_unnorm[pt]), float(w_after[pt])]
            )
        w = w_after

    ensemble_score = alphas[0] * h1 + alphas[1] * h2
    ensemble_pred = np.where(ensemble_score >= 0.0, 1.0, -1.0)
    ensemble_accuracy = float(np.mean(ensemble_pred == y_ada))

    write_csv(
        "outputs/part4_adaboost.csv",
        ["round", "point", "weight_before", "weight_unnormalised", "weight_after"],
        adaboost_rows,
    )

    # Task 4.4: Out-of-bag fraction
    oob_fraction = (1.0 - 1.0 / 560.0) ** 560

    # Task 4.5: Neural-network architecture parameter calculations
    # Model A Option 1 (MLP: 784 -> 16 -> 16 -> 10)
    mlp_hidden1 = 784 * 16 + 16
    mlp_hidden2 = 16 * 16 + 16
    mlp_output = 16 * 10 + 10
    mlp_total = mlp_hidden1 + mlp_hidden2 + mlp_output

    # Model A Option 2 (CNN on 28x28x1)
    # Conv1: 5x5x1, 6 filters, stride 1, pad 2 -> 28x28x6
    cnn_conv1 = (5 * 5 * 1 + 1) * 6
    # Pool1: 2x2 stride 2 -> 14x14x6
    # Conv2: 5x5x6, 16 filters, stride 1, pad 0 -> 10x10x16
    cnn_conv2 = (5 * 5 * 6 + 1) * 16
    # Pool2: 2x2 stride 2 -> 5x5x16 -> flatten = 400
    flat_dim = 5 * 5 * 16
    cnn_dense1 = (flat_dim + 1) * 32
    cnn_dense2 = (32 + 1) * 10
    cnn_total = cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2

    write_csv(
        "outputs/part4_architecture.csv",
        ["layer", "parameters"],
        [
            ["mlp_hidden1", mlp_hidden1],
            ["mlp_hidden2", mlp_hidden2],
            ["mlp_output", mlp_output],
            ["cnn_conv1", cnn_conv1],
            ["cnn_conv2", cnn_conv2],
            ["cnn_dense1", cnn_dense1],
            ["cnn_dense2", cnn_dense2],
        ],
    )

    # Why convolution scales: 128 x 128 x 3
    fc_layer_params = (128 * 128 * 3 + 1) * 1024
    conv_layer_params = (3 * 3 * 3 + 1) * 64

    # Model B: RNN (input=64, hidden=128, output=16)
    w_xh_params = 128 * 64
    w_hh_params = 128 * 128
    w_hy_params = 16 * 128
    b_h_params = 128
    b_y_params = 16
    rnn_total = w_xh_params + w_hh_params + w_hy_params + b_h_params + b_y_params

    results = [
        ("root_entropy", root_entropy),
        ("root_gini", root_gini),
        ("best_split_by_info_gain", best_split_info),
        ("best_split_by_gini_gain", best_split_gini),
        ("stump_rate_yes", stump_rate_yes),
        ("stump_rate_no", stump_rate_no),
        ("round1_error", errors[0]),
        ("round1_alpha", alphas[0]),
        ("round2_error", errors[1]),
        ("round2_alpha", alphas[1]),
        ("ensemble_accuracy", ensemble_accuracy),
        ("oob_fraction", oob_fraction),
        ("mlp_total", mlp_total),
        ("cnn_total", cnn_total),
        ("fc_layer_params", fc_layer_params),
        ("conv_layer_params", conv_layer_params),
        ("rnn_total", rnn_total),
    ]
    write_results_txt("outputs/part4_results.txt", results)


if __name__ == "__main__":
    main()
