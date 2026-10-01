# Patient or Clinician?

**When a clinical model predicts sepsis, is it seeing the patient — or seeing what clinicians chose to measure?**

A research prototype for *GIBC V2, Track 02 (Applied: Medical Technology & Finance)*. It is an empirical audit,
not a sepsis predictor: we split the information available to a model into **physiology** and **measurement
process**, train matched models on each, and test which survives transfer between two hospitals.

> **Status (read first).** The pipeline, tests and viewer are implemented. **The experiment has not yet been run on the
> real PhysioNet data** — it was built in an environment without network access, so no real results exist and none are
> reported here. Code paths were exercised only on a small synthetic fixture (`tests/synth.py`), whose outputs are
> meaningless and are marked `synthetic: true`. Run the steps under *Reproduction*, then fill in *Results*.

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

*Not yet generated.* After running the pipeline, `results/results_table.md` contains the transfer tables (with CIs) to paste here,
and the viewer reads `results/*.json`. We will report whatever is found, including if process information transfers as well as physiology.

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
