# Demo script (~100 s, matches `patient-or-clinician-demo.mp4`)

All numbers are from `results/` (subset run: first 3,000 patients per hospital; AUROC unless noted). Read them as written; do not round up the claims.

**0:00–0:16 · The question**
"A clinical model can see two kinds of information: the patient's physiology, and what clinicians chose to measure. We separate them, train matched models on each, and then move the models between two hospitals — to ask whether the signal survives when the measurement process changes."

**0:17–0:27 · Transfer, first row (BASE, PHYSIOLOGY)**
"Each grid is one model. The diagonal is within-hospital. The bold off-diagonal cells are trained on one hospital and tested on the other. Physiology-only goes from 0.79 within hospital B to 0.68 when trained on A — it degrades, too."

**0:28–0:37 · Process and combined**
"Process-only is strong inside a hospital: 0.76 in both. But transferred, it falls to about 0.59 and 0.58 — the same as the base model, which sees only age, sex and time in ICU. The combined model is best within hospital, but under transfer it's no better than physiology alone."

**0:38–0:48 · Audit line, honest caveat**
"One caveat: physiology values alone can still predict when a lab is being measured, so our physiology family is values-only, not provably process-free. Intervals are wide — about 3,000 patients per hospital."

**0:52–1:20 · Patient case** *(case chosen from the 9-patient seeded random sample for visible contrast; say so)*
"Same patient timeline. Top: heart rate. Middle: when each variable was measured — a separate signal. Bottom: the model's risk. All information. Now remove process, keep physiology. Now remove physiology, keep process. These are model-input counterfactuals, not clinical ones, and other patients look different — in many, removing either family flattens the prediction."

**1:21–1:33 · Stress test**
"If the target hospital measured less, every model degrades modestly. Process-only isn't uniquely fragile here — it was already near baseline."

**1:34–1:40 · Close**
"So: process information helped within a hospital, but its added value wasn't detectable after transfer in this subset. This is a research prototype and an audit, not a clinical tool, and our attribution is associational, not causal."

*Known cosmetic gaps in the video: the stress-test chart has no y-axis labels and the M_VP / M_BASE labels overlap at the right edge.*
