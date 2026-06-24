# Model Card — TelcoWave Churn Scoring

This Model Card follows the framework proposed by Mitchell et al. (2019). It
documents the churn scoring model served by this repository (V2). The model
itself was trained in the [V1 repository](https://github.com/capristunt/telco-churn-scoring);
this card reflects the artifact `artifacts/finetuned.joblib` as deployed here.

---

## 1. Model details

- **Model type**: Binary classifier — `LogisticRegression(penalty="l2", class_weight="balanced")`, wrapped in `CalibratedClassifierCV(method="isotonic", cv=5)` for probability calibration.
- **Library**: scikit-learn 1.5.2
- **Version**: V1 finetuned (frozen), copied into this repo as `artifacts/finetuned.joblib`
- **Date**: trained and frozen in 2026, no retraining since.
- **Authors**: capristunt (portfolio project, DATAGONG Data Scientist certification)
- **License**: portfolio and educational use only
- **Decision threshold**: `s* = 0.141`, derived from cost/benefit optimization on the validation set (see Section 7).
- **Input contract**: 18 features, validated by Pydantic in `app/schemas.py`. See API reference in the [README](README.md).
- **Output**: predicted churn probability, binary label at threshold `s*`, and a risk segment (Q1 to Q4) based on frozen quartile bounds computed on train+valid.

---

## 2. Intended use

### Primary intended use

Prioritize a retention marketing campaign at TelcoWave. The model assigns each
customer a calibrated churn probability, supporting targeted offers under a
limited marketing budget rather than blanket outreach.

The score is meant to be **one input among others** for the retention team.
A retention manager remains responsible for the decision to contact a given
customer and the offer extended.

### Primary intended users

- Retention marketing team (operational use of the scoring).
- Data team (monitoring, validation, periodic refresh).

### Out-of-scope uses

The model should **not** be used for:

- **Adverse decisions against customers** (refusal of service, price discrimination, account closure). It estimates churn likelihood, not creditworthiness or eligibility.
- **Causal claims**. Correlations identified (e.g. month-to-month contract → higher churn) are predictive, not causal. They do not justify forcing customers into longer contracts.
- **Customers outside the training distribution** (B2B accounts, prepaid plans, regions or product lines not covered by the Telco Kaggle dataset).
- **Real-time triggering of automated communications** without human review.

---

## 3. Factors

### Relevant factors

The model relies on three structural feature groups:

- **Contractual**: `Contract`, `PaymentMethod`, `PaperlessBilling`, `tenure`.
- **Service mix**: `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `MultipleLines`, `nb_services` (derived).
- **Demographic**: `SeniorCitizen`, `Partner`, `Dependents`.

Excluded features (V1 EDA conclusion, Cramér's V ≈ 0): `customerID`, `gender`, `PhoneService`.

### Evaluation factors

No subgroup-disaggregated evaluation was performed at training time. This is a
known gap (see Section 8). The dataset does carry sensitive attributes
(`SeniorCitizen`) which could be used for fairness auditing in a follow-up.

---

## 4. Metrics

### Selected metrics

- **ROC-AUC**: threshold-independent ranking quality.
- **PR-AUC**: more informative than ROC-AUC under class imbalance (churn rate ≈ 26.5%).
- **Recall and precision at `s* = 0.141`**: operational view at the deployed threshold.
- **Recall and precision at top 10%**: portfolio-style metric for capacity-constrained outreach.
- **Economic gain**: business-level metric defined as `gain = 105 · TP − 15 · FP`, where 105 € is the saved customer value (120 €) net of the offer cost (15 €).

### Decision threshold

`s* = 0.141`, optimal on the validation set under the gain function above. Close
to the theoretical Bayes threshold `15 / (15 + 105) = 0.125`, with the small
gap explained by the discrete nature of the optimization grid.

### Variation approaches

Threshold sensitivity was studied at training time. Top-K strategies (10%, 20%,
30%) were also evaluated as operational alternatives when the retention team
faces a fixed contact capacity. See Section 7 for figures.

---

## 5. Evaluation data

- **Source**: Telco Customer Churn dataset, Kaggle (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
- **Test set size**: 1,406 customers (20% of usable rows, stratified on `Churn`).
- **Preprocessing**: identical to training (see Section 6). Specifically: `TotalCharges` cast to numeric, `tenure = 0` customers excluded, `nb_services` derived, `gender` and `PhoneService` removed.
- **Class balance**: 26.5% churn rate, consistent with train and valid splits.
- **Hold-out discipline**: the test set was not touched during model selection, hyperparameter tuning, calibration, or threshold optimization. It was scored exactly once for the final reporting.

---

## 6. Training data

- **Source**: same Telco Kaggle dataset.
- **Train size**: 4,218 customers (60%).
- **Validation size**: 1,408 customers (20%, used for model selection, calibration fit, and threshold tuning).
- **Split**: stratified on `Churn`, fixed `random_state=42`. Customer IDs persisted to disk for reproducibility.
- **Features used**: 18 (5 numeric, 13 categorical).
- **Class imbalance**: 26.5% positive class. Handled at training time by `class_weight="balanced"`, then corrected at inference time by isotonic calibration (see Section 7).

---

## 7. Quantitative analyses

### Performance on the test set (1,406 customers)

| Metric | Value |
|---|---|
| ROC-AUC | 0.832 |
| PR-AUC | 0.615 |
| Recall at `s* = 0.141` | 91.7% (343 / 374 churners captured) |
| Precision at `s* = 0.141` | ≈ 43.3% (343 / 793 contacts) |
| Contacts at `s*` | 793 (56% of the base) |
| Economic gain at `s*` | **29,265 €** |

### Comparison of operational strategies (validation set)

| Strategy | Contacts | Recall | Gain (€) |
|---|---|---|---|
| Top 10% | 140 | 29.1% | 10,980 |
| Top 20% | 281 | 54.0% | 20,025 |
| Top 30% | 422 | 68.7% | 24,510 |
| **`s* = 0.141`** | **821** | **94.1%** | **29,925** |

`s*` outperforms the best top-K strategy by ~22% in economic gain, justifying
the choice of a probability threshold over a fixed-volume policy in the
nominal scenario. Top-K remains documented as a fallback when retention
capacity is hard-capped.

### Calibration

`class_weight="balanced"` produces a systematic upward bias in raw probabilities
(mean predicted 0.42 vs. observed churn rate 0.27 on validation). Isotonic
calibration corrects this:

| Metric | Before calibration | After calibration |
|---|---|---|
| Mean predicted probability (valid) | 0.42 | 0.27 |
| Brier score (valid) | 0.164 | 0.131 (−20%) |
| ROC-AUC, PR-AUC | unchanged (isotonic is monotone) | unchanged |

Within-quartile calibration was also verified on validation: mean predicted
probability and observed churn rate agree within 4 percentage points across all
four quartiles, indicating the calibration holds locally and not only on
average.

### Feature importance (permutation importance on validation)

| Feature | Permutation importance (ΔROC-AUC) |
|---|---|
| `tenure` | 0.177 |
| `Contract` | 0.043 |
| `InternetService` | 0.023 |
| `nb_services` | ≈ −0.001 (negligible) |

`nb_services` was confirmed near-zero, consistent with L1 analysis that zeroed
its coefficient. Its bell-shaped relationship with churn (EDA peak ≈ 45% at 3
services) is interesting descriptively but is not a strong model driver.

---

## 8. Ethical considerations

### Data sensitivity

The training data contains demographic attributes (`SeniorCitizen`, `Partner`,
`Dependents`). These were retained because they carry signal and removing them
would not have eliminated discrimination risk (proxy variables exist via
service mix and billing patterns). Their use must remain restricted to the
intended marketing purpose described in Section 2.

### Potential biases

- **Senior citizens** churn at ~42% in the dataset vs. ~24% overall. The model
  reflects this and is likely to flag senior customers more often. This can
  steer retention offers toward this group, which is not inherently harmful
  but warrants monitoring to avoid stigmatization or over-solicitation.
- **No external benchmark for fairness**: no protected-class disaggregated
  evaluation was performed. This is a documented gap.
- **`PaperlessBilling` and `Electronic check` payment** are strong churn
  predictors. Used carelessly, these factors could lead to retention offers
  conditional on switching back to paper billing or mailed checks, which would
  be a regression on digital adoption goals.

### Recommended human oversight

- A retention manager must validate the offer template **before** automated
  outreach uses the scoring output.
- An audit of contacted vs. ignored customers, disaggregated by senior status,
  is recommended quarterly.

---

## 9. Caveats and recommendations

### Known limitations

1. **Static model**. Trained once on the Telco Kaggle snapshot. No retraining
   pipeline is in place; concept drift will erode performance over time.
2. **No drift monitoring** in V2. Population shift (e.g. new fiber tier
   launched, new payment method) would not be detected automatically.
   Mentioned as a natural next step.
3. **Quartile bounds are frozen** at training time. As the customer mix
   evolves, the proportion of customers in each quartile may diverge from the
   intended 25%. Bounds should be refreshed alongside any model retraining.
4. **No behavioral signal**. Features are limited to contractual and demographic
   attributes. Usage data (call duration, data volume, support tickets) would
   likely improve the model substantially but is out of scope of the Telco
   Kaggle dataset.
5. **Single dataset origin**. Performance on TelcoWave's actual production
   data is unknown until measured. The figures in Section 7 should be treated
   as a ceiling estimate, not a guarantee.
6. **`nb_services` is a request input** rather than computed server-side.
   See README for the V2.1 improvement proposal.

### Recommendations for safe operation

- **Refresh schedule**: retrain at least once per year, or when ROC-AUC on a
  monthly sample drops below 0.78 (≈5% below test).
- **Population monitoring**: track the distribution of `Contract`,
  `InternetService`, `nb_services` and the share of customers in each risk
  quartile. Alert on shifts > 10 points.
- **Outcome tracking**: link scoring output to actual churn outcomes 3 months
  out to measure realized gain vs. the 29,265 € test estimate.
- **Threshold review**: `s* = 0.141` reflects the 15 € / 120 € economics. If
  the offer cost or the saved customer value changes materially, recompute the
  threshold with `find_optimal_threshold` rather than nudging by intuition.

---

## References

- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAT* 2019.
  ([arXiv:1810.03677](https://arxiv.org/abs/1810.03677))
- V1 repository: [`capristunt/telco-churn-scoring`](https://github.com/capristunt/telco-churn-scoring).