"""Part 3 — Segmentation and dimensionality reduction.

Produces:
    outputs/part3_results.txt
    outputs/part3_segments.csv
    outputs/part3_hierarchical.csv
"""

import numpy as np
from common import load_records, sample_mean, standardise_columns, write_csv, write_results_txt


def run_kmeans(X_std, k):
    """Run deterministic K-means with prescribed starting rows and convergence rule."""
    n = X_std.shape[0]
    step = n // k
    init_indices = [j * step for j in range(k)]
    centroids = X_std[init_indices].copy()

    prev_labels = None
    passes = 0
    while True:
        # Squared Euclidean distances from each point to each centroid: shape (n, k)
        diff = X_std[:, None, :] - centroids[None, :, :]
        sq_dists = np.sum(diff * diff, axis=2)
        # np.argmin picks the lowest cluster index on exact ties
        labels = np.argmin(sq_dists, axis=1)
        passes += 1

        if prev_labels is not None and np.array_equal(labels, prev_labels):
            break

        prev_labels = labels
        new_centroids = centroids.copy()
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                new_centroids[c] = np.mean(X_std[mask], axis=0)
        centroids = new_centroids

    # Final WCSS in standardised space
    final_diff = X_std - centroids[labels]
    wcss = float(np.sum(final_diff * final_diff))
    return labels, centroids, wcss, passes


def agglomerative_clustering(dist_matrix, linkage):
    """Perform agglomerative clustering on a dissimilarity matrix for a given linkage rule.

    Returns:
        merges: list of (merge_no, cluster_a_name, cluster_b_name, height)
        clusters_at_cut: int
    """
    n = dist_matrix.shape[0]
    clusters = [[i] for i in range(n)]
    merges = []
    heights = []

    def cluster_name(members):
        return "+".join(f"E{m + 1}" for m in sorted(members))

    def linkage_distance(c1, c2):
        pairs = [dist_matrix[a, b] for a in c1 for b in c2]
        if linkage == "single":
            return min(pairs)
        if linkage == "complete":
            return max(pairs)
        if linkage == "average":
            return sum(pairs) / len(pairs)
        raise ValueError(f"Unknown linkage: {linkage}")

    for merge_no in range(1, n):
        best_i = -1
        best_j = -1
        best_dist = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = linkage_distance(clusters[i], clusters[j])
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_i = i
                    best_j = j

        c_i = clusters[best_i]
        c_j = clusters[best_j]
        if min(c_i) < min(c_j):
            cluster_a, cluster_b = c_i, c_j
        else:
            cluster_a, cluster_b = c_j, c_i

        merges.append(
            (merge_no, cluster_name(cluster_a), cluster_name(cluster_b), float(best_dist))
        )
        heights.append(float(best_dist))

        merged = sorted(c_i + c_j)
        # Replace lower index with merged cluster and remove higher index
        clusters[best_i] = merged
        clusters.pop(best_j)

    gaps = [heights[idx + 1] - heights[idx] for idx in range(len(heights) - 1)]
    max_gap_idx = max(range(len(gaps)), key=lambda idx: gaps[idx])
    merges_below_cut = max_gap_idx + 1
    clusters_at_cut = n - merges_below_cut
    return merges, clusters_at_cut


def run_pca_seven_steps(X_std):
    """Execute PCA in the exact seven steps using numpy.linalg.eigh."""
    n = X_std.shape[0]
    # 1. Feature mean vector
    mean_vec = np.mean(X_std, axis=0)
    # 2. Centre the data
    X_centered = X_std - mean_vec
    # 3. Sample covariance matrix S = (1 / (n - 1)) X^T X
    cov_matrix = (X_centered.T @ X_centered) / (n - 1)
    # 4. Eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(cov_matrix)
    # 5. Sort descending by eigenvalue and enforce sign convention
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    for j in range(evecs.shape[1]):
        max_abs_idx = int(np.argmax(np.abs(evecs[:, j])))
        if evecs[max_abs_idx, j] < 0:
            evecs[:, j] = -evecs[:, j]

    # 6. Top two eigenvectors
    W = evecs[:, :2]
    # 7. Project Z = X W
    Z = X_centered @ W
    return evals, evecs, Z


def main():
    _, X_raw, y_outcome, y_value = load_records("data/records.csv")
    X_cont = X_raw[:, :5]  # Continuous predictors X1..X5
    X_std, _, _ = standardise_columns(X_cont)

    # Task 3.1: K-means for K = 2..6
    wcss_by_k = {}
    labels_k3 = None
    for k in (2, 3, 4, 5, 6):
        labels, _, wcss, _ = run_kmeans(X_std, k)
        wcss_by_k[k] = wcss
        if k == 3:
            labels_k3 = labels

    # Task 3.2: Elbow method
    elbow_k = None
    for k in (3, 4, 5):
        drop_next = wcss_by_k[k] - wcss_by_k[k + 1]
        drop_prev = wcss_by_k[k - 1] - wcss_by_k[k]
        if drop_next < 0.5 * drop_prev:
            elbow_k = k
            break

    # Task 3.3: Profile K=3 segments ordered by ascending mean of X3 (index 2)
    cluster_x3_means = []
    for c in range(3):
        mask = labels_k3 == c
        cluster_x3_means.append((sample_mean(X_cont[mask, 2]), c))
    cluster_x3_means.sort(key=lambda item: item[0])

    segment_names = ["A", "B", "C"]
    segment_masks = {}
    segment_rows = []
    for seg_name, (_, c_idx) in zip(segment_names, cluster_x3_means):
        mask = labels_k3 == c_idx
        segment_masks[seg_name] = mask
        size = int(np.sum(mask))
        means_x = [sample_mean(X_cont[mask, j]) for j in range(5)]
        outcome_rate = sample_mean(y_outcome[mask])
        mean_val = sample_mean(y_value[mask])
        segment_rows.append(
            [seg_name, size] + means_x + [outcome_rate, mean_val]
        )

    write_csv(
        "outputs/part3_segments.csv",
        [
            "segment",
            "size",
            "mean_X1",
            "mean_X2",
            "mean_X3",
            "mean_X4",
            "mean_X5",
            "outcome_rate",
            "mean_value",
        ],
        segment_rows,
    )

    # Task 3.4: Hierarchical clustering on supplied 6x6 dissimilarity table
    dist_table = np.array(
        [
            [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
            [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
            [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
            [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
            [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
            [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
        ],
        dtype=np.float64,
    )
    hier_rows = []
    clusters_at_cut = {}
    for linkage in ("single", "complete", "average"):
        merges, n_clust = agglomerative_clustering(dist_table, linkage)
        clusters_at_cut[linkage] = n_clust
        for merge_no, ca, cb, h in merges:
            hier_rows.append([linkage, merge_no, ca, cb, h])

    write_csv(
        "outputs/part3_hierarchical.csv",
        ["linkage", "merge_no", "cluster_a", "cluster_b", "height"],
        hier_rows,
    )

    # Task 3.5: PCA by the seven steps
    evals, evecs, Z = run_pca_seven_steps(X_std)
    eval_sum = float(np.sum(evals))
    explained_ratios = evals / eval_sum
    cum_ratios = np.cumsum(explained_ratios)
    components_for_80pct = int(np.argmax(cum_ratios >= 0.8) + 1)

    results = [
        ("wcss_k2", wcss_by_k[2]),
        ("wcss_k3", wcss_by_k[3]),
        ("wcss_k4", wcss_by_k[4]),
        ("wcss_k5", wcss_by_k[5]),
        ("wcss_k6", wcss_by_k[6]),
        ("elbow_k", elbow_k),
        ("size_segment_A", int(np.sum(segment_masks["A"]))),
        ("size_segment_B", int(np.sum(segment_masks["B"]))),
        ("size_segment_C", int(np.sum(segment_masks["C"]))),
        ("single_clusters_at_cut", clusters_at_cut["single"]),
        ("complete_clusters_at_cut", clusters_at_cut["complete"]),
        ("average_clusters_at_cut", clusters_at_cut["average"]),
        ("eigenvalue_sum", eval_sum),
    ]
    for j in range(5):
        results.append((f"eigenvalue_{j + 1}", float(evals[j])))
    for j in range(5):
        results.append((f"explained_ratio_{j + 1}", float(explained_ratios[j])))
    for j in range(5):
        results.append((f"cumulative_ratio_{j + 1}", float(cum_ratios[j])))
    results.append(("components_for_80pct", components_for_80pct))
    for j in range(5):
        results.append((f"pc1_loading_X{j + 1}", float(evecs[j, 0])))
    for j in range(5):
        results.append((f"pc2_loading_X{j + 1}", float(evecs[j, 1])))
    for r in range(3):
        results.append((f"row{r + 1}_pc1", float(Z[r, 0])))
        results.append((f"row{r + 1}_pc2", float(Z[r, 1])))
    for seg_name in segment_names:
        mean_pc1 = sample_mean(Z[segment_masks[seg_name], 0])
        results.append((f"mean_pc1_segment_{seg_name}", mean_pc1))

    write_results_txt("outputs/part3_results.txt", results)


if __name__ == "__main__":
    main()
