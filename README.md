# Patient or Clinician?

**When a clinical model predicts sepsis, is it seeing the patient — or seeing what clinicians chose to measure?**

A research prototype for *GIBC V2, Track 02 (Applied: Medical Technology & Finance)*. It is an empirical audit,
not a sepsis predictor: we split the information available to a model into **physiology** and **measurement
process**, train matched models on each, and test which survives transfer between two hospitals.

> **Status.** The pipeline has been run once on a **subset of the real PhysioNet data: the first 3,000 patients (by file order) of each hospital** — 3,000 of 20,336 in Set A and 3,000 of 20,000 in Set B — due to download/compute limits. Results below are from that run (`results/`, `synthetic: false`). With only 177–270 septic patients per hospital, confidence intervals are wide. A full-data run is the obvious next step (`make run`).

## The question

- **Physiological signal** — what was happening to the patient (vitals, labs).
- **Measurement-process signal** — what clinicians chose to measure: which variables, when, how often, how large a lab panel, when observation stopped.

Clinical records contain both. Process patterns can be predictive (sicker or more-suspected patients get more tests),
and they are set by local workflow, so they may not transfer. We test whether this happens here; we do **not** assume it.

## The experiment

All models receive the **base** features (age, gender, HospAdmTime, ICULOS) plus:

| Model | Adds | Meaning |
|---|---|---|
| `M_BASE` | – | floor |
| `M_V` | physiology | values only, causal LOCF, **no** missingness/timing columns |
| `M_P` | process | observation mask, hours-since, counts, panel size, density — **no values** |
| `M_VP` | both | combined |
| `LR_V` | physiology | logistic-regression baseline (Baseline A); `M_V` is the gradient-boosting baseline (Baseline B) |

All boosted models share identical hyperparameters (`src/config.py`; scikit-learn `HistGradientBoostingClassifier`,
chosen to avoid extra dependencies — LightGBM/XGBoost would be drop-in via `src/models.py`).
Scenarios: `within_A`, `within_B` (patient-grouped CV), `A_to_B`, `B_to_A`.

Feature provenance is enforced by column prefix (`b__`, `v__`, `p__`) and by signatures: `process_features` receives only the boolean mask;
`physiology_locf` only values. See `docs/temporal_rules.md`.

## Dataset

[PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) — hourly,
de-identified ICU records from two hospital systems (Set A: Beth Israel Deaconess; Set B: Emory) with 34 time-varying
measurements, 6 demographic/administrative fields and `SepsisLabel`, one `.psv` file per patient. Obtain it from PhysioNet
(accept its terms), then place:

```
data/training_setA/*.psv
data/training_setB/*.psv
```

(e.g. `wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/` and move the two folders — verify the folder layout on the dataset page.)
The data is never committed (`.gitignore`). Cohort sizes and prevalence are computed by the pipeline into `results/model_metrics.json`, not hard-coded here.

## Methodology

- **Splits.** `StratifiedGroupKFold` grouped by patient (stratified on "patient ever septic"). Row-level splitting is never used; `assert_disjoint` guards every partition.
- **Transfer.** Train on all of one hospital, evaluate on all of the other. Imputer fit on the source hospital only.
- **Temporal correctness.** Every feature at hour *t* uses hours ≤ *t* (`docs/temporal_rules.md`), tested by prefix-invariance and future-edit tests.
- **Physiology imputation.** Causal LOCF; before a variable's first measurement, a per-patient seeded draw from the training distribution (avoids a shared "not yet measured" sentinel). A **missingness-leak audit** (`src/audit.py`) reports how well physiology features alone predict that a lab is being measured (0.5 = nothing recovered). LOCF can still leak weak staleness information; this is reported, not claimed away.
- **Metrics.** AUROC, AUPRC, Brier, expected calibration error (10 bins) and reliability bins, on all hourly rows. 95% CIs by percentile bootstrap resampling **patients** (default 100 resamples). CIs are per-model; paired model-difference CIs are not computed, so differences between models should not be called "significant" from overlap alone.
- **Attribution.** For `M_VP`, replace one family with its training median and measure the log-odds change: `c_P`, `c_V`; `PRI = c_P / (c_P + c_V)` when both are positive, otherwise undefined. This is a **model-based associational diagnostic, not causal inference**; ablated inputs are off-distribution and the reference is a modelling choice. Demo patients are a seeded random sample, not hand-picked.
- **Stress test (optional, implemented).** Randomly delete 0/25/50% of the target hospital's measurements, recompute features, rescore the transferred models.
- **Not implemented:** per-hospital process normalization (optional item).

## Results

Subset run: 3,000 patients per hospital (A: 116,651 hourly rows, 270 septic patients; B: 113,642 rows, 177 septic patients; row-level label prevalence 2.2% / 1.5%). 5-fold patient-grouped CV within hospital; 95% CIs from 100 patient-level bootstrap resamples; seed 42; single run, no tuning. Rows are hourly predictions, so AUPRC is low in absolute terms given the low prevalence.

**AUROC** (95% patient-bootstrap CI)

| model | within_A | within_B | A_to_B | B_to_A |
|---|---|---|---|---|
| M_BASE | 0.661 [0.626, 0.687] | 0.674 [0.629, 0.712] | 0.599 [0.547, 0.656] | 0.598 [0.569, 0.624] |
| M_V | 0.715 [0.689, 0.742] | 0.787 [0.758, 0.818] | 0.677 [0.636, 0.719] | 0.662 [0.622, 0.691] |
| M_P | 0.757 [0.730, 0.782] | 0.764 [0.728, 0.790] | 0.592 [0.549, 0.640] | 0.580 [0.549, 0.609] |
| M_VP | 0.779 [0.751, 0.802] | 0.817 [0.791, 0.847] | 0.634 [0.597, 0.673] | 0.661 [0.627, 0.683] |
| LR_V | 0.705 [0.675, 0.731] | 0.697 [0.657, 0.737] | 0.710 [0.668, 0.746] | 0.653 [0.618, 0.681] |

**AUPRC** (95% patient-bootstrap CI)

| model | within_A | within_B | A_to_B | B_to_A |
|---|---|---|---|---|
| M_BASE | 0.075 [0.065, 0.094] | 0.036 [0.029, 0.047] | 0.036 [0.030, 0.055] | 0.046 [0.038, 0.054] |
| M_V | 0.082 [0.068, 0.102] | 0.058 [0.048, 0.073] | 0.054 [0.043, 0.081] | 0.070 [0.058, 0.082] |
| M_P | 0.088 [0.075, 0.106] | 0.053 [0.046, 0.068] | 0.032 [0.026, 0.054] | 0.037 [0.030, 0.049] |
| M_VP | 0.092 [0.078, 0.111] | 0.061 [0.050, 0.078] | 0.045 [0.034, 0.070] | 0.056 [0.046, 0.068] |
| LR_V | 0.076 [0.065, 0.093] | 0.045 [0.038, 0.065] | 0.043 [0.036, 0.065] | 0.064 [0.053, 0.076] |

**BRIER** (95% patient-bootstrap CI)

| model | within_A | within_B | A_to_B | B_to_A |
|---|---|---|---|---|
| M_BASE | 0.027 [0.024, 0.031] | 0.020 [0.016, 0.023] | 0.021 [0.017, 0.025] | 0.028 [0.024, 0.031] |
| M_V | 0.023 [0.020, 0.026] | 0.016 [0.014, 0.019] | 0.016 [0.014, 0.019] | 0.023 [0.020, 0.025] |
| M_P | 0.024 [0.021, 0.028] | 0.017 [0.014, 0.019] | 0.018 [0.015, 0.021] | 0.022 [0.019, 0.024] |
| M_VP | 0.024 [0.021, 0.027] | 0.017 [0.014, 0.020] | 0.017 [0.014, 0.021] | 0.022 [0.019, 0.024] |
| LR_V | 0.021 [0.019, 0.024] | 0.015 [0.013, 0.017] | 0.018 [0.014, 0.022] | 0.021 [0.019, 0.024] |

**What the numbers show (and do not show).**

- *Within a hospital, process alone is highly predictive.* `M_P` (0.757 / 0.764 AUROC in A / B) clearly beats `M_BASE` (0.661 / 0.674) and is comparable to `M_V` (0.715 / 0.787); CIs overlap.
- *Across hospitals, the process model's advantage over base disappears.* `M_P` transfers at 0.592 (A→B) and 0.580 (B→A), statistically indistinguishable from `M_BASE` (0.599 / 0.598) given the CIs.
- *Physiology retains some gain over base after transfer, but also degrades.* `M_V`: 0.677 (A→B) and 0.662 (B→A). From within-B to A→B it falls by about 0.11 AUROC, so physiology does not transfer "cleanly" either; with these intervals, `M_V` vs `M_BASE` is only marginally separated in A→B.
- *Combining does not help under shift.* `M_VP` is 0.634 (A→B), below `M_V`'s 0.677 (intervals overlap), and equal in B→A (0.661 vs 0.662), despite being best within-hospital (0.779 / 0.817). The logistic baseline `LR_V` is comparable to the boosted `M_V` (0.710 / 0.653).
- *Calibration/Brier* differences between models are small and CIs overlap; see `transfer_metrics.json` for reliability bins.
- *Stress test* (delete 25% / 50% of the target hospital's measurements): all models degrade modestly (e.g. A→B `M_V` 0.677 → 0.647, `M_VP` 0.634 → 0.604, `M_P` 0.592 → 0.564). `M_P` is already near the base-model level, so this does **not** show process models being uniquely fragile. B→A is flat-to-noisy (`M_P` 0.580 → 0.580).
- *Attribution (PRI).* Where defined (16% of sampled rows A→B, 72% B→A), median PRI is 0.54 / 0.56, i.e. process and physiology contribute comparably under this ablation. In A→B mean contributions are negative: replacing a family with its reference tends to *raise* predicted risk, a sign that the reference inputs are off-distribution. Treat PRI as illustrative only.

**Honest summary.** The data support: process features are informative within a hospital but their added value over base features is not detectable after cross-hospital transfer in this subset. They do **not** show that physiology transfers well (it also drops), that the model is "biased", or that process reliance causes the drop. Hospital differences in case mix and labeling are confounded with measurement practice.

**Missingness-leak audit (a limitation of `M_V`).** From physiology features alone, a gradient-boosted classifier predicts "Lactate measured this hour" with AUROC **0.72 (A) and 0.92 (B)** on held-out patients. So the physiology family is *values-only*, not provably process-free: values correlate with when things are measured (partly genuine illness severity, partly ordering behavior), and we cannot separate these here. We checked one candidate artifact — interpolated imputation draws — by switching to draws of actually observed values; the audit was unchanged (0.717 / 0.915), so the retained code keeps the original imputer. The `M_V` vs `M_P` contrast should be read as "inputs restricted to values" vs "inputs restricted to measurement pattern".

## Reproduction

```bash
git clone <this repo> && cd patient-or-clinician
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place the dataset as described above
make test                      # leakage / temporal / label tests
make run-small                 # quick subsample run (2000 patients/hospital, 3 folds)
make run                       # full run: 5 folds x 5 models x 2 hospitals + transfers; expect a long run on a laptop
```
Outputs in `results/`: `transfer_metrics.json` (full, with calibration), `transfer_matrix.json` (compact), `model_metrics.json`
(cohort counts, config, versions), `attribution_cases.json`, `stress_test.json`, `audit.json`, `results_table.md`.

## Demo

```bash
make viewer      # serves the repo root; open http://localhost:8000/viewer/
```
Static HTML/JS, no build step, no backend, no retraining. Views: the question, transfer matrices, patient case
(All / Physiology only / Process only — *model-input* counterfactuals, not clinical ones), stress test. With no results it shows an empty state; with synthetic results it shows a red banner.

## Limitations

Observational, retrospective data; two hospital systems only, so "hospital shift" conflates measurement practice with case mix, population and labeling. Challenge labels are derived from a sepsis definition applied by the organizers, with onset-relative positives. Process and physiology are correlated, so family ablations are model-dependent and not causal. LOCF can retain faint timing information. Hourly-row metrics weight long stays more heavily. No external validity beyond these two systems; no clinical-utility claim; not for clinical use.

## References

1. Reyna MA, Josef CS, et al. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. *Crit Care Med* 2020;48(2):210–217.
2. Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 2000;101(23):e215–e220.
3. Wong A, et al. External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients. *JAMA Intern Med* 2021;181(8):1065–1070. (Prior art on the Epic Sepsis Model; this project is a different study and does not reproduce it.)
4. Agniel D, Kohane IS, Weber GM. Biases in electronic health record data due to processes within the healthcare system. *BMJ* 2018;361:k1479.
5. Che Z, et al. Recurrent Neural Networks for Multivariate Time Series with Missing Values. *Sci Rep* 2018;8:6085.

*Citations were written from memory without network access; verify volume/page details before submission.*

## License

Code: MIT (see `LICENSE`, rationale in `NOTICE`). The dataset is not included and carries its own terms.
