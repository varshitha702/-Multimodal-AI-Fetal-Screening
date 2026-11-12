# Multimodal AI Fetal Screening (FetalAI)

A Streamlit academic research workspace built around the supplied HC18 fetal-head-region U-Net segmentation pipeline.

## Verified current capability

- Upload a PNG/JPG ultrasound still image.
- Run the existing grayscale 256×256 U-Net, when its trained checkpoint is available.
- View the predicted mask, overlay and pixel-level probability heatmap.
- Optionally calculate a geometric contour perimeter when exact image-specific pixel spacing is supplied.
- Inspect the expected HC18 dataset folder and matched image/annotation counts.
- Export a research summary PDF.

**This is an academic prototype, not a medical device. It is not clinically validated and must not be used for diagnosis, reassurance, triage or pregnancy-care decisions.** The U-Net does not predict gestational age, presentation, fetal weight, heartbeat, anatomy or overall fetal health.

## Project structure

```text
Multimodal AI Fetal Screening/
├── app.py                    # upgraded Streamlit dashboard
├── app_original.py           # original uploaded app preserved
├── model.py                  # original HC18 U-Net
├── train_hc18.py             # original HC18 trainer, preserved
├── check_hc18.py             # dataset audit utility
├── config.py                 # separate multimodal experiment config
├── train.py                  # separate multimodal experiment trainer
├── assets/style.css          # dashboard theme
├── data/raw/HC18/             # place HC18 dataset here
├── outputs/                   # trained checkpoint output
├── requirements.txt
├── requirements-live.txt
└── run_app_windows.bat
```

## Windows setup

Open PowerShell in this project folder:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

## HC18 dataset layout

Place the extracted HC18 dataset as:

```text
data/raw/HC18/
├── training_set/
│   ├── training_set/
│   │   ├── *_HC.png
│   │   └── *_HC_Annotation.png
│   └── training_set_pixel_size_and_HC.csv
```

The image/mask naming convention and CSV columns are checked by `check_hc18.py`. Run:

```powershell
python check_hc18.py
```

Then train the existing segmentation model:

```powershell
python train_hc18.py
```

The trainer saves the best validation checkpoint to:

```text
outputs/hc18_unet_best.pth
```

The trainer currently uses a random image-level train/validation split. This is not equivalent to a patient/study-level split. For research reporting, document and use a leakage-safe split when patient/study identifiers are available.

## Launch

```powershell
streamlit run app.py
```

Or double-click `run_app_windows.bat`.

## Important multimodal experiment status

The supplied `config.py` and `train.py` describe a different ultrasound + ECG classification experiment. `train.py` imports `MultimodalFetalModel`, but that class was not present in the supplied `model.py`. The paired manifest, ECG data and trained multimodal checkpoint were also not supplied. Therefore this experiment is deliberately not connected to the dashboard and is not represented as working functionality.

Before enabling it, provide/implement and validate the model class, data manifest schema, data provenance, preprocessing, checkpoint and evaluation protocol. Do not combine its six ultrasound-view labels with HC18 fetal-head segmentation labels.

## Privacy and safety

The app does not intentionally save uploaded images or report entries to a database. Session state is temporary, but the host, browser, operating system and deployment platform may have independent logs or retention. Avoid identifiable patient data in demonstrations. The contour perimeter is a geometric estimate, not a clinically validated head-circumference measurement.
