# Temporal feature-generation rules

Row *t* of a patient is one ICU hour. **Every feature at row t may use rows ≤ t of the same patient only.**

| Family | Construction | Why it is causal |
|---|---|---|
| base | Age, Gender, HospAdmTime, ICULOS as recorded at row t | Recorded at/before t; ICULOS is elapsed time |
| physiology | Last observation carried forward per variable (`groupby(patient).ffill()`) | Looks backward only |
| physiology (pre-first-measurement) | Draw from the training distribution, keyed by hash(seed, patient, variable) | Depends on no data from the patient |
| process: obs / cnt | observation mask at t; cumulative count to t | `cumsum` |
| process: since | rows since last observation (ffill of position), capped at 72 | Looks backward only |
| process: 6h window | `cnt_t − cnt_{t−6}` | `shift(6)`, positive shift only |
| process: aggregates | functions of the above and of row position within the patient | Past-only inputs |

Imputer parameters (value quantiles) are fit **on training patients only**, per fold / per source hospital.
Labels are never an input to any feature. Enforced by `tests/test_temporal.py`: (i) truncating a patient's record
after row t leaves features at rows ≤ t unchanged; (ii) rewriting all values, masks and labels after row t leaves features at rows ≤ t unchanged.

**Label caveat.** The challenge's `SepsisLabel` is defined by the organizers relative to clinical sepsis onset
(positives begin several hours *before* onset; see Reyna et al. 2020). The audit therefore measures early-warning
discrimination under that labeling, not detection of an observed event.
