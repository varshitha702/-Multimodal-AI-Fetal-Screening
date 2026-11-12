# dashboard/: Streamlit research dashboard (fallback UI)

A research workspace kept alongside the Next.js app. It runs the **same** inference code as the API
(`ml.inference.Segmenter`), so its HC values match the API exactly.

```powershell
# from the repository root, with the trained checkpoint at outputs/hc18_unet_best.pth
streamlit run dashboard\app.py
# or: scripts\run_dashboard.bat
```

Pages: overview, analysis (upload, overlay, mask, probability heatmap, HC by ellipse fit, PDF summary),
dataset audit, model information, about.

The checkpoint is not committed; generate it with `python -m ml.train`. Do not upload identifiable patient
data. Academic research prototype, not a medical device.
