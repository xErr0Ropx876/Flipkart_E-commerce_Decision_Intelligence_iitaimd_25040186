"""Shared data loading, splitting, standardisation, modelling, and I/O helpers for ShopperPulse.

Importing this module never executes any analysis or writes output files.
"""

import csv
import os
import numpy as np

PREDICTOR_NAMES = [
    "X1",
    "X2",
    "X3",
    "X4",
    "X5",
    "X6",
    "X7",
]

PREDICTOR_COLUMNS = [
    "customer_tenure_months",
    "complaint_count_90d",
    "discount_order_share",
    "avg_order_value_inr_hundreds",
    "loyalty_score",
    "cod_flag",
    "plus_member_flag",
]


def load_records(filepath="data/records.csv"):
    """Load data/records.csv preserving original row order.

    Returns:
        record_ids: list[str]
        X_raw: np.ndarray of shape (n, 7), float64 (X1..X7)
        y_outcome: np.ndarray of shape (n,), int64 (churned)
        y_value: np.ndarray of shape (n,), float64 (order_value_inr_hundreds)
    """
    record_ids = []
    features = []
    outcomes = []
    values = []

    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) != 10:
            raise ValueError(f"Expected 10 columns in {filepath}, got {len(header)}")
        for row in reader:
            record_ids.append(row[0])
            features.append([float(v) for v in row[1:8]])
            outcomes.append(int(row[8]))
            values.append(float(row[9]))

    return (
        record_ids,
        np.array(features, dtype=np.float64),
        np.array(outcomes, dtype=np.int64),
        np.array(values, dtype=np.float64),
    )


def sample_mean(arr):
    """Arithmetic mean."""
    return float(np.sum(arr) / len(arr))


def sample_sd(arr):
    """Sample standard deviation using denominator n - 1."""
    n = len(arr)
    mean_val = np.sum(arr) / n
    return float(np.sqrt(np.sum((arr - mean_val) ** 2) / (n - 1)))


def standardise_columns(X):
    """Standardise each column using its mean and sample SD (ddof=1).

    Returns:
        X_std, means, sds
    """
    n = X.shape[0]
    means = np.sum(X, axis=0) / n
    sds = np.sqrt(np.sum((X - means) ** 2, axis=0) / (n - 1))
    X_std = (X - means) / sds
    return X_std, means, sds


def standardise_train_test(X_train, X_test):
    """Standardise X_train and X_test using ONLY training mean and training sample SD."""
    X_train_std, means, sds = standardise_columns(X_train)
    X_test_std = (X_test - means) / sds
    return X_train_std, X_test_std, means, sds


def stratified_split(X, y):
    """Deterministic stratified 70/30 split preserving original file order.

    For each class (0 and 1) in file order, the first floor(7 * count / 10) rows
    go to train and the remainder to test. Both sets retain original file order.

    Returns:
        X_train, X_test, y_train, y_test, train_indices, test_indices
    """
    n = len(y)
    idx_0 = [i for i in range(n) if y[i] == 0]
    idx_1 = [i for i in range(n) if y[i] == 1]

    n0_train = (7 * len(idx_0)) // 10
    n1_train = (7 * len(idx_1)) // 10

    train_set = set(idx_0[:n0_train] + idx_1[:n1_train])
    train_indices = [i for i in range(n) if i in train_set]
    test_indices = [i for i in range(n) if i not in train_set]

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices],
        train_indices,
        test_indices,
    )


def sigmoid(z):
    """Numerically stable sigmoid function."""
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z_arr, dtype=np.float64)
    pos_mask = z_arr >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1.0 / (1.0 + np.exp(-z_arr[pos_mask]))
    exp_z = np.exp(z_arr[neg_mask])
    out[neg_mask] = exp_z / (1.0 + exp_z)
    return out


def binary_cross_entropy(y_true, y_prob):
    """Binary cross-entropy averaged over observations with log underflow protection."""
    eps = 1e-15
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def train_logistic_regression(X, y, lr=0.1, epochs=1000, snapshot_epochs=None):
    """Full-batch gradient-descent binary logistic regression starting from zeros.

    Returns:
        weights: np.ndarray of shape (d,)
        bias: float
        snapshots: dict[int, float] mapping epoch -> training BCE
    """
    if snapshot_epochs is None:
        snapshot_epochs = set()
    else:
        snapshot_epochs = set(snapshot_epochs)

    n, d = X.shape
    weights = np.zeros(d, dtype=np.float64)
    bias = 0.0
    snapshots = {}

    if 0 in snapshot_epochs:
        p0 = sigmoid(X @ weights + bias)
        snapshots[0] = binary_cross_entropy(y, p0)

    for epoch in range(1, epochs + 1):
        z = X @ weights + bias
        p = sigmoid(z)
        error = p - y
        dw = (X.T @ error) / n
        db = float(np.sum(error) / n)
        weights = weights - lr * dw
        bias = bias - lr * db

        if epoch in snapshot_epochs:
            p_after = sigmoid(X @ weights + bias)
            snapshots[epoch] = binary_cross_entropy(y, p_after)

    return weights, bias, snapshots


def predict_logistic_proba(X, weights, bias):
    """Return predicted positive probabilities for logistic regression."""
    return sigmoid(X @ weights + bias)


def confusion_matrix(y_true, y_pred):
    """Compute (tp, fp, tn, fn) manually from 0/1 arrays."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    tp = int(np.sum((y_p == 1) & (y_t == 1)))
    fp = int(np.sum((y_p == 1) & (y_t == 0)))
    tn = int(np.sum((y_p == 0) & (y_t == 0)))
    fn = int(np.sum((y_p == 0) & (y_t == 1)))
    return tp, fp, tn, fn


def classification_metrics(tp, fp, tn, fn):
    """Compute accuracy, precision, recall, specificity, fpr, and f1 manually."""
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "f1": float(f1),
    }


def knn_predict(X_ref, y_ref, X_query, k):
    """Manual K-nearest neighbours using Euclidean distance and stable tie-breaking.

    Reference rows must be in original file order; stable sort preserves file order
    on exact distance ties. Predicts positive when at least (k + 1) // 2 neighbours
    are positive (for odd k).

    Returns:
        preds: np.ndarray of int64 (0/1)
        probs: np.ndarray of float64 (positive neighbour share)
    """
    n_query = X_query.shape[0]
    preds = np.zeros(n_query, dtype=np.int64)
    probs = np.zeros(n_query, dtype=np.float64)
    threshold_count = (k + 1) // 2

    for i in range(n_query):
        diff = X_ref - X_query[i]
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        order = np.argsort(dists, kind="stable")
        top_k = order[:k]
        pos_count = int(np.sum(y_ref[top_k]))
        probs[i] = pos_count / k
        preds[i] = 1 if pos_count >= threshold_count else 0

    return preds, probs


def knn_cross_validate(X_train_std, y_train, k_values=None):
    """Run 5-fold CV on training set using fold = position % 5.

    Returns:
        cv_results: list of (k, mean_cv_f1)
        best_k: int (highest mean CV F1, ties go to smaller k)
    """
    if k_values is None:
        k_values = list(range(1, 22, 2))

    n_train = X_train_std.shape[0]
    folds = np.arange(n_train) % 5
    cv_results = []
    best_k = None
    best_f1 = -1.0

    for k in k_values:
        fold_f1s = []
        for f_idx in range(5):
            val_mask = folds == f_idx
            ref_mask = ~val_mask
            preds, _ = knn_predict(
                X_train_std[ref_mask],
                y_train[ref_mask],
                X_train_std[val_mask],
                k,
            )
            tp, fp, tn, fn = confusion_matrix(y_train[val_mask], preds)
            metrics = classification_metrics(tp, fp, tn, fn)
            fold_f1s.append(metrics["f1"])
        mean_f1 = float(sum(fold_f1s) / 5.0)
        cv_results.append((k, mean_f1))
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_k = k

    return cv_results, best_k


def format_value(val):
    """Format values according to assignment rules (4 decimals for float, int, yes/no)."""
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        return f"{float(val):.4f}"
    return str(val)


def write_results_txt(filepath, items):
    """Write key = value lines to filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="\n") as f:
        for key, val in items:
            f.write(f"{key} = {format_value(val)}\n")


def write_csv(filepath, header, rows):
    """Write CSV file with exact header and formatted rows."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([format_value(v) for v in row])
