# Demo script — Clinician's Shadow (~2:11, matches `clinicians-shadow-demo.mp4`)

Timings assume ~150 words/minute (2.5 words/s) plus 1 s buffer per segment; the video holds each view for exactly that long, so read at a relaxed pace. The video is silent. Numbers are from `results/` (subset run: first 3,000 patients per hospital; AUROC). Say them as written; do not strengthen the claims.

> Add intro/outro (team name, GIBC V2 Track 02) before or after if the submission format wants it. The hackathon brief gives no video length requirement; check the submission page and trim segments if there is a limit.

**0:00–0:22 · The question**  
"Clinician's Shadow asks one question. When a clinical model predicts sepsis, is it seeing the patient, or seeing what clinicians chose to measure? A model can see two kinds of information: physiology, and the measurement process. We separate them, train matched models on each, and then move the models between two hospitals."

**0:22–0:41 · Transfer: base and physiology**  
"Each grid is one model. The diagonal is within one hospital. The bold cells are trained on one hospital and tested on the other. Physiology-only falls from point seven nine within hospital B to point six eight when trained on A, so it degrades too."

**0:41–1:01 · Transfer: process and combined**  
"Process-only is strong inside a hospital, point seven six in both. Transferred, it falls to about point five nine and point five eight, the same as the base model. The combined model is best within a hospital, but under transfer it is no better than physiology alone."

**1:01–1:14 · Audit caveat**  
"One caveat. Physiology values alone can still predict when a lab is measured, so this family is values-only, not provably process-free. And intervals are wide, with about three thousand patients per hospital."

**1:14–1:26 · Patient case: all information**  
"Now one patient, picked from a seeded random sample to show contrast. Top: heart rate. Middle: when each variable was measured. Bottom: the model's predicted risk."

**1:26–1:32 · Patient case: physiology only**  
"Remove process, keep physiology."

**1:32–1:38 · Patient case: process only**  
"Remove physiology, keep process."

**1:38–1:46 · Patient case: caveat**  
"These are model-input counterfactuals, not clinical ones. In many other patients, removing either family flattens the prediction."

**1:46–1:55 · Stress test**  
"If the target hospital measured less, every model degrades modestly. Process-only is not uniquely fragile here. It was already near baseline."

**1:55–2:11 · Close**  
"So process information helped within a hospital, but its added value was not detectable after transfer in this subset. This is a research prototype and an audit, not a clinical tool, and the attribution is associational, not causal."

*Note for the patient-case segment: the case (`A_to_B:100632`) was picked from the 9-patient seeded random sample for visible contrast between modes; say it is illustrative, not representative.*