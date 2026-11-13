# ADR-0008: CTG branch and rule-based decision-level fusion

* Status: accepted (13 Nov 2026)
* Module: CTG model and tests (Yashwanth Gowda M N), API and fusion endpoint (Vishwas K), UI (Rakshitha V)

## Context

The project title promises multimodal screening. The second modality available as public data is
cardiotocography (UCI Cardiotocography, 2,126 records, NSP labels). It comes from different patients than the
ultrasound datasets, so no record links one fetus's scan to its CTG.

## Decision

* **CTG model:** XGBoost on the 21 SisPorto features, compared with logistic regression and random forest on the
  same 70/15/15 stratified split (duplicate records kept together). Class imbalance is handled with
  balanced sample weights computed from the train split only. Hyper-parameters are fixed in advance and the
  validation split is used only for early stopping. Explanations come from SHAP TreeExplainer.
* **Deployment:** the model is saved truncated to the early-stopping iteration and loaded as a plain
  `xgboost.Booster`. The scikit-learn wrapper failed to load under the scikit-learn 1.9 that the Docker image
  resolved, and truncation makes the predictions and SHAP use the same trees. The model (1.7 MB) is committed
  so CI and Docker use the real classifier.
* **API:** `/ctg/predict` with Pydantic-validated features (422 on bad input), `/ctg/examples` (held-out test
  records), `/screening/summary`.
* **Fusion:** decision-level and **rule-based** (R1–R6 in `ml/fusion.py`). A pathological CTG always escalates
  to Urgent review; any other abnormal input gives Review recommended. Missing inputs are listed as "not
  assessed", never assumed normal. Every response names the rules that fired and states that the data come
  from different patients.
* **Not done:** learned fusion and a calibrated combined risk, because there is no paired data to train or
  validate them.

## Consequences

* Held-out test: XGBoost macro-F1 0.963 (95 % CI 0.936–0.986), Pathological recall 27/27 (Wilson CI
  0.875–1.000), versus random forest 0.910 and logistic regression 0.778. The validation macro-F1 was 0.918, so
  the test figure may be optimistic, and the pathological test set is small.
* The summary is transparent and testable (every rule has a unit test) but has no measured accuracy. Its
  levels are illustrative until paired clinical data allows learned fusion and calibration (Phase 2).
