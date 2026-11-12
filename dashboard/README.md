# Deployment notes

For Streamlit Community Cloud, select `app.py` as the entry point and use `requirements-live.txt`.

The trained checkpoint is not included in this project bundle. Provide `outputs/hc18_unet_best.pth` through an appropriate private/model-artifact workflow before expecting segmentation to run. Do not commit identifiable ultrasound images or patient information.

The current project is a research prototype only, not a medical device.
