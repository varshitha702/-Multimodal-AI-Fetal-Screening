# Upgrade notes

- Reworked `app.py` into a multi-section research dashboard.
- Added a cohesive responsive navy/teal stylesheet.
- Added in-app HC18 dataset audit counts.
- Kept the existing U-Net architecture and HC18 training script unchanged.
- Preserved the originally uploaded application as `app_original.py`.
- Kept the ultrasound+ECG experiment files but isolated them because `MultimodalFetalModel`, its paired data, and checkpoint were not supplied.
- Updated Windows launch and HC18 validation scripts.
- No dataset or trained model checkpoint was included in the upload, so inference will remain unavailable until the checkpoint is trained/generated.
