Domain: E-commerce

This repository is an offline decision-intelligence pipeline built in Python 3.10+ and NumPy (`1.26.4`) for a synthetic Flipkart e-commerce customer analytics and retention scenario. Every observation in `data/records.csv` is produced deterministically by `generate_data.py` and does not represent real Flipkart customer data, products, policies, or business performance. Across four connected modules (`part1.py` to `part4.py`, sharing core utilities in `common.py`), the repository evaluates the INR 800 average-order-value planning assumption, predicts 90-day churn to set a cost-minimising alert threshold, profiles three customer segments with PCA and clustering, and compares structured-data classifiers alongside parameter sizing for future image and text neural networks.

## How to run

Run the following commands in order from a clean clone of the repository:

```bash
git clone https://github.com/xErr0Ropx876/Flipkart_E-commerce_Decision_Intelligence_iitaimd_25040186.git
cd Flipkart_E-commerce_Decision_Intelligence_iitaimd_25040186
python -m venv .venv
```

Activate the virtual environment:

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install the pinned dependency, generate the dataset, and run all four parts in sequence:

```bash
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

`common.py` is imported automatically by the Part scripts and executes no computations on import.

## Part 1 findings

- **R1.1 (Hypothesis test & practical gap):** `H0: mu = 8.0` was rejected (`reject_h0 = yes`) because `z_stat = 2.8930` exceeds the two-tailed 5% critical value of `1.96` (`|2.8930| > 1.96`). In this domain, a Type I error would mean concluding that Flipkart's true average order value differs from the INR 800 planning baseline when the true average actually equals INR 800, prompting unnecessary changes to category budgets. The gap `mean_value - 8.0 = 8.9382 - 8.0 = 0.9382` column units equals INR 93.82 (with 95% confidence interval `8.3026` to `9.5738`, or INR 830.26 to INR 957.38); at roughly 11.7% above INR 800, this difference is practically material for category revenue forecasting as well as statistically significant.
- **R1.2 (Sample-size adequacy):** The required sample size for a margin of error of `0.5` column units (INR 50) at 95% confidence is `n_required = 1293`, compared with `800` records currently available (`n = 800`, yielding `se_value = 0.3243` and a 95% half-width of `0.6356` units / INR 63.56). The planning team needs at least 493 additional observations if it requires INR 50 estimation precision.
- **R1.3 (Gradient descent convergence):** `gd_slope_gap = 0.0000` (well within `0.001`), confirming that gradient descent recovered the closed-form slope (`reg_slope = 0.6701`, `gd_slope = 0.6701`). Because simple linear regression has a strictly convex quadratic loss surface and standardising `X4` keeps the problem well-conditioned, 500 full-batch updates at `eta = 0.1` converge directly to the unique global minimum.
- **R1.4 (Residual funnel & log remedy):** In the raw linear regression, `resid_sd_ratio_raw = 3.7525` (`resid_sd_lower_raw = 2.3155`, `resid_sd_upper_raw = 8.6891`), revealing a pronounced residual funnel where error spread grows sharply with fitted order value. Fitting the log-log model reduces this ratio to `resid_sd_ratio_log = 1.0396` (`resid_sd_lower_log = 0.5165`, `resid_sd_upper_log = 0.5369`), effectively removing the heteroscedastic funnel.
- **R1.5 (Multiple regression coefficients & adjusted R²):** Applying the mechanical `|beta| > 0.05` rule (excluding the constant `mr_beta_0 = -1.8613`) yields four predictors: `ln(X4)` (`mr_beta_1 = 1.1576`), `X2` (`complaint_count_90d`, `mr_beta_3 = 0.0940`), `X3` (`discount_order_share`, `mr_beta_4 = 0.1067`), and `X6` (`cod_flag`, `mr_beta_6 = 0.1518`). Because the predictors have heterogeneous scales, this cutoff is a mechanical filter rather than a ranking of importance. On the `ln(value)` scale, `mr_adj_r2 = 0.7403` compared with `loglog_adj_r2 = 0.6561` (+0.0842), confirming that the extra predictors substantially improve fit after adjusting for degrees of freedom.

## Part 2 findings

- **R2.1 (Learning-rate comparison):** Ordering the 200-update training BCE losses from lowest to highest gives `loss_epoch_200_eta_1.0 = 0.3703`, `loss_epoch_200_eta_0.1 = 0.3776`, and `loss_epoch_200_eta_0.01 = 0.5091`. This shows that too small a learning rate (`0.01`) moves weights too slowly to converge within 200 updates, whereas a larger step (`1.0`) reaches near the convex optimum quickly here without overshooting, though excessively large rates risk oscillation on less smooth surfaces.
- **R2.2 (Largest weights & odds ratio):** The three predictors with the largest absolute logistic-regression weights are `X3` (`discount_order_share`, `weight_X3 = 0.9013`), `X1` (`customer_tenure_months`, `weight_X1 = -0.5554`), and `X2` (`complaint_count_90d`, `weight_X2 = 0.5505`). The largest odds ratio is `odds_ratio_X3 = 2.4629`, meaning that each one-training-SD increase in a customer's share of discounted orders multiplies their odds of 90-day churn by `2.4629` when all other predictors are held fixed.
- **R2.3 (Accuracy trap & threshold trade-off):** The always-zero baseline achieves `blind_accuracy = 0.7875`, nearly matching `tau_0.5_accuracy = 0.8042` despite catching zero churners; accuracy alone is a weak metric here because only 51 of 240 test customers (21.25%) churned. Lowering `tau` from `0.5` to `0.3` increased recall from `0.4706` (`tau_0.5_recall`) to `0.7451` (`tau_0.3_recall`), while precision fell from `0.5455` (`tau_0.5_precision`) to `0.4634` (`tau_0.3_precision`).
- **R2.4 (Logistic vs KNN):** On the 240 test rows, logistic regression at `tau = 0.3` outperforms KNN (`knn_best_k = 5`) by a wide margin, achieving `tau_0.3_f1 = 0.5714` and `tau_0.3_recall = 0.7451` versus `knn_f1 = 0.3696` and `knn_recall = 0.3333`.
- **R2.5 (Deployment & cost optimisation):** Deploy logistic regression at `cost_tau = 0.1500` (`cost_train_at_tau = 12150`), which reduces test cost to `cost_test_at_tau = 6050` INR compared with `cost_test_at_0.5 = 11800` INR at `tau = 0.5` and `cost_test_flag_nobody = 20400` INR when flagging nobody. Choosing `cost_tau` exclusively on training rows prevents test-set leakage so that the reported test cost remains an unbiased evaluation.

## Part 3 findings

- **R3.1 (K-means WCSS drops & elbow):** The four successive WCSS drops are `WCSS(2) - WCSS(3) = 659.8282`, `WCSS(3) - WCSS(4) = 108.0739`, `WCSS(4) - WCSS(5) = 171.0575`, and `WCSS(5) - WCSS(6) = 45.5466`. The elbow rule selects `elbow_k = 3`, since `108.0739 < 0.5 * 659.8282 = 329.9141`. Because the prescribed deterministic starting centroids yield a reproducible local solution rather than necessarily the global WCSS optimum for every K, the successive drops do not shrink monotonically (`171.0575` exceeds `108.0739`).
- **R3.2 (Segment profiles & actions):**
  - **Segment A — Premium Loyalists (`size = 322`):** Longest tenure (`mean_X1 = 47.5028` months), lowest complaints (`mean_X2 = 1.9503`), lowest discount reliance (`mean_X3 = 0.2250`), highest loyalty score (`mean_X5 = 62.1584`), `mean_value = 3.5976` (INR 359.76), and lowest churn (`outcome_rate = 0.0186`). Action: retain through non-discount Plus recognition and early-access benefits to preserve margin.
  - **Segment B — High-Spend Friction Shoppers (`size = 255`):** Highest past order value (`mean_X4 = 28.7192`) and current order value (`mean_value = 17.1248`, INR 1,712.48), highest complaint count (`mean_X2 = 6.0706`), moderate discount share (`mean_X3 = 0.4534`), and `outcome_rate = 0.2353`. Action: prioritise proactive customer-support resolution and post-complaint recovery to protect high-ticket revenue.
  - **Segment C — Bargain-Focused Churn-Prone Shoppers (`size = 223`):** Highest discount share (`mean_X3 = 0.7118`), shortest tenure (`mean_X1 = 14.3892` months), lowest loyalty (`mean_X5 = 38.5215`), `mean_value = 7.2883` (INR 728.83), and the highest churn rate (`outcome_rate = 0.4664`). Action: deploy targeted onboarding nudges and time-limited retention coupons before 90-day inactivity sets in.
- **R3.3 (Dendrogram cuts):** Single linkage (`single_clusters_at_cut = 4`) cuts at the gap between `1.4000` and `2.8000`, yielding four clusters: `{E2, E3, E6}`, `{E1}`, `{E4}`, and `{E5}`. Complete linkage (`complete_clusters_at_cut = 2`) cuts at the gap between `2.9000` and `7.9000`, yielding two clusters: `{E1, E4, E5}` and `{E2, E3, E6}`. Single linkage chains through the closest member-to-member bridge (`3.4` between `E5` and `E6`), keeping the final group merge low so the largest relative height jump occurs early, whereas complete linkage uses the farthest pair (`7.9` between `E1` and `E2`) and creates a large separation once the two coherent three-cohort groups have formed.
- **R3.4 (PCA interpretation):** After standardisation, all five feature means are zero. PC1 (`eigenvalue_1 = 2.6136`, `52.27%` of variance) loads positively on `X1` (`+0.5317`) and `X5` (`+0.4418`) and negatively on `X3` (`-0.5187`), `X2` (`-0.4100`), and `X4` (`-0.2916`), representing an axis from long-tenure, high-loyalty shoppers to discount-reliant, complaint-prone shoppers. At least 80% of variance requires `components_for_80pct = 3` components (`cumulative_ratio_3 = 0.8616`). The segment PC1 means (`mean_pc1_segment_A = 1.7773`, `mean_pc1_segment_B = -0.7831`, `mean_pc1_segment_C = -1.6709`) confirm that PC1 separates loyal Segment A at the positive end from friction- and discount-driven Segments B and C at the negative end.

## Part 4 findings

- **R4.1 (Split purity & overfitting):** `S4` (`customer_tenure_months <= 24`) is the best split by both criteria (`info_gain = 0.1252`, `gini_gain = 0.0606`) because newer shoppers churn at `45.74%` (`86/188`) compared with `8.87%` (`33/372`) for longer-tenured shoppers, achieving the sharpest class separation. Allowing a tree to split until every leaf is pure memorises random noise in individual training rows and degrades test generalisation; enforcing a maximum tree depth (`max_depth`) prevents excessive branching.
- **R4.2 (Structured model comparison & voting):** Among the five Task 4.2 strategies, `logistic` (`tau = 0.3`) is the strongest model to ship with `f1 = 0.5714` and `recall = 0.7451`. Combining models did not help: `hard_vote` (`f1 = 0.5344`, `recall = 0.6863`) and `soft_vote` (`f1 = 0.5507`, `recall = 0.7451`) both trail standalone `logistic` (and `stump` has `f1 = 0.5532`, `recall = 0.7647`) because the weak `knn` classifier (`f1 = 0.3696`, `recall = 0.3333`) pulls the combined decisions down.
- **R4.3 (AdaBoost weights):** After round 1 normalisation, the two misclassified points (points 3 and 7) each carry weight `0.2500`, giving a combined weight of `0.5000`. Round 2 assigns higher weight to `h2` (`round2_alpha = 0.8047`) than round 1 gave `h1` (`round1_alpha = 0.5493`) because `h2` achieves a lower weighted error (`round2_error = 0.1667` vs `round1_error = 0.2500`) on the reweighted sample where points 3 and 7 dominate.
- **R4.4 (Neural-network architecture & sizing):**

  | Stage (Model A, Option 2 CNN) | Output shape (H x W x D) | Parameters |
  |---|---|---:|
  | Input image | 28 x 28 x 1 | 0 |
  | Conv1 (5x5, 6 filters, stride 1, pad 2) | 28 x 28 x 6 | 156 |
  | ReLU 1 | 28 x 28 x 6 | 0 |
  | MaxPool 1 (2x2, stride 2) | 14 x 14 x 6 | 0 |
  | Conv2 (5x5, 16 filters, stride 1, pad 0) | 10 x 10 x 16 | 2416 |
  | ReLU 2 | 10 x 10 x 16 | 0 |
  | MaxPool 2 (2x2, stride 2) | 5 x 5 x 16 | 0 |
  | Flatten | 400 (flattened) | 0 |
  | Dense 1 (32 neurons, ReLU) | 32 | 12832 |
  | Output Dense 2 (10 neurons) | 10 | 330 |
  | Total (`cnn_total`) | — | 15734 |

  For Model B (RNN), the weight-matrix shapes are `W_xh: 128 x 64` (8,192 parameters), `W_hh: 128 x 128` (16,384), `W_hy: 16 x 128` (2,048), plus biases `b_h` (`128`) and `b_y` (`16`), giving `rnn_total = 26768`; this total does not grow with sequence length because the exact same weights are shared across every time step. For 10 mutually exclusive classes, use `ReLU` in hidden layers and `softmax` with categorical cross-entropy loss at the output. In a very deep plain network, repeated multiplication by derivatives smaller than 1 during backpropagation causes gradients to vanish exponentially before reaching early layers, stalling their learning; skip connections add an identity shortcut across layers so gradients can propagate directly back to early weights.

## Model-choice memo

**To:** Head of the Retention Team  
**Subject:** Churn-Model Deployment Recommendation

I recommend deploying **logistic regression** from Task 4.2 as our structured-data churn predictor, paired with the cost-optimal alert threshold **`tau = 0.1500`**. Across the Task 4.2 test comparison (at `tau = 0.3`), logistic regression achieved the highest F1 score (`0.5714`) and strong recall (`0.7451`), decisively beating KNN (`F1 = 0.3696`, `recall = 0.3333`), hard voting (`F1 = 0.5344`), and soft voting (`F1 = 0.5507`), with well-calibrated probabilities (`test_bce = 0.3818`).

Economically, tuning the logistic threshold on training records (`cost_train_at_tau = 12150` INR) lowers test retention cost to **`6050` INR** at `tau = 0.1500` (`cost_test_at_tau`). By comparison, the default `tau = 0.5` threshold costs **`11800` INR** (`cost_test_at_0.5`) and flagging nobody costs **`20400` INR** (`cost_test_flag_nobody`), saving up to 70.3% of churn-plus-coupon cost under the INR 50 coupon / INR 400 churn assumption.

**Key limitation:** These figures come from a single 560/240 split of 800 synthetic records containing only 51 test churners, so synthetic accuracy does not guarantee live production performance and requires shadow-mode validation on real orders before automated coupon dispatch.

Neither Model A (product-image CNN, 15,734 parameters) nor Model B (review-text RNN, 26,768 parameters) should be built before the structured churn model is live in production, because immediate retention savings depend on deploying structured customer-risk scoring first rather than constructing separate merchandising or text pipelines.

## Limitations

1. **Synthetic dataset rather than real customer behaviour:** All 800 observations were generated from fixed parametric distributions in `generate_data.py`; actual Flipkart customers exhibit seasonal spikes, category mix effects, and campaign responses that synthetic equations cannot reproduce, so performance here cannot guarantee real-world deployment effectiveness.
2. **Single deterministic train/test split and small positive test count:** Evaluating on one 560/240 split with only `test_positives = 51` churners introduces substantial sampling variance in precision, recall, and monetary cost estimates; one or two borderline customers can noticeably shift F1 or INR cost.
3. **Static snapshot and non-causal relationships:** The dataset captures a single cross-section with no temporal drift, and regression or logistic coefficients reflect statistical associations under the generator rather than causal proof that changing a feature (such as reducing discount share) will prevent churn.
4. **Simplified uniform economic cost assumptions:** The cost rule (`50 * FP + 400 * FN`) assumes every customer has the same INR 400 future margin and INR 50 coupon cost, whereas Segment B's average order value (`17.1248`) is nearly five times Segment A's (`3.5976`), meaning real retention economics should weight high-value customers differently.
