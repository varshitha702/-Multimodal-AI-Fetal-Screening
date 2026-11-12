from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from model import UNet

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data" / "raw" / "HC18"
IMAGE_DIR = DATA_ROOT / "training_set" / "training_set"
CSV_PATH = DATA_ROOT / "training_set_pixel_size_and_HC.csv"
CHECKPOINT = BASE_DIR / "outputs" / "hc18_unet_best.pth"
METRICS_PATH = BASE_DIR / "outputs" / "hc18_metrics.json"
IMAGE_SIZE = 256
MAX_IMAGE_PIXELS = 20_000_000

st.set_page_config(
    page_title="FetalAI | Research Workspace",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    css = BASE_DIR / "assets" / "style.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

load_css()

@st.cache_resource(show_spinner="Loading the HC18 segmentation model...")
def load_model(checkpoint_path: str):
    path = Path(checkpoint_path)
    if not path.exists():
        return None, None, "Checkpoint not found"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    try:
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state)
        model.eval()
        return model, device, None
    except Exception as exc:
        return None, device, f"Checkpoint could not be loaded: {exc}"

def run_segmentation(model, device, pil_image):
    gray = pil_image.convert("L")
    width, height = gray.size
    resized = gray.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability_small = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    binary_small = (probability_small > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(binary_small)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 0:
            cv2.drawContours(cleaned, [largest], -1, 255, thickness=cv2.FILLED)
    mask = cv2.resize(cleaned, (width, height), interpolation=cv2.INTER_NEAREST)
    probability = cv2.resize(probability_small, (width, height), interpolation=cv2.INTER_LINEAR)
    original = np.asarray(pil_image.convert("RGB")).copy()
    tint = np.zeros_like(original)
    tint[:, :, 1] = mask
    overlay = cv2.addWeighted(original, 1.0, tint, 0.38, 0)
    heatmap = cv2.applyColorMap(np.uint8(np.clip(probability, 0, 1) * 255), cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return mask, overlay, probability, heatmap

def png_bytes(array, mode=None):
    buffer = BytesIO()
    (Image.fromarray(array, mode=mode) if mode else Image.fromarray(array)).save(buffer, format="PNG")
    return buffer.getvalue()

def contour_perimeter(mask, spacing_mm):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(contour) < 5:
        return None
    scaled = contour * float(spacing_mm)
    return float(cv2.arcLength(scaled.reshape(-1, 1, 2), True))

def dataset_audit():
    images = list(IMAGE_DIR.glob("*_HC.png")) if IMAGE_DIR.exists() else []
    masks = list(IMAGE_DIR.glob("*_HC_Annotation.png")) if IMAGE_DIR.exists() else []
    image_ids = {p.stem.removesuffix("_HC") for p in images}
    mask_ids = {p.stem.removesuffix("_HC_Annotation") for p in masks}
    csv_ok = CSV_PATH.exists()
    csv_rows = None
    csv_columns = []
    if csv_ok:
        try:
            frame = pd.read_csv(CSV_PATH)
            csv_rows, csv_columns = len(frame), frame.columns.tolist()
        except Exception:
            csv_ok = False
    return {
        "image_count": len(images), "mask_count": len(masks),
        "matched": len(image_ids & mask_ids),
        "images_without_masks": len(image_ids - mask_ids),
        "masks_without_images": len(mask_ids - image_ids),
        "csv_exists": csv_ok, "csv_rows": csv_rows, "csv_columns": csv_columns,
    }

def build_pdf(details, filename, image=None, overlay=None, perimeter=None, spacing=None, model_status=""):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=17*mm, leftMargin=17*mm,
                            topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Foot", parent=styles["BodyText"], fontSize=8, leading=10,
                              textColor=colors.HexColor("#526174")))
    story = [
        Paragraph("FetalAI | Research Analysis Summary", styles["Title"]),
        Paragraph("Academic prototype output. Not a medical diagnosis or clinical report.", styles["Foot"]),
        Spacer(1, 6*mm),
        Paragraph("Session information", styles["Heading2"]),
        Paragraph(f"Image: {filename}", styles["BodyText"]),
        Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", styles["BodyText"]),
        Paragraph(f"Model status: {model_status}", styles["BodyText"]),
        Spacer(1, 4*mm),
        Paragraph("Information transcribed from the supplied report", styles["Heading2"]),
    ]
    rows = [["Field", "Entered value"]]
    rows.extend([[str(k), str(v) if str(v).strip() else "Not provided"] for k, v in details.items()])
    table = Table(rows, colWidths=[55*mm, 110*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#12304A")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#D9E3EC")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F3F7FA")]),
        ("PADDING", (0,0), (-1,-1), 6), ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story += [table, Spacer(1, 5*mm), Paragraph("AI image-processing output", styles["Heading2"])]
    story.append(Paragraph(
        "The supplied U-Net is trained for candidate fetal-head-region segmentation only. "
        "It does not determine fetal presentation, gestational age, fetal weight, heartbeat, "
        "anatomy, or overall fetal health.", styles["Foot"]))
    if perimeter is not None:
        story.append(Paragraph(
            f"Predicted-mask contour perimeter: {perimeter:.2f} mm, using entered spacing "
            f"of {spacing} mm/pixel. This is not a clinically validated head-circumference measurement.",
            styles["BodyText"]))
    else:
        story.append(Paragraph("No perimeter in millimetres was calculated.", styles["BodyText"]))
    for heading, arr in [("Uploaded image", image), ("Segmentation overlay", overlay)]:
        if arr is not None:
            story += [Spacer(1, 4*mm), Paragraph(heading, styles["Heading3"])]
            imbuf = BytesIO()
            (arr if isinstance(arr, Image.Image) else Image.fromarray(arr)).save(imbuf, format="PNG")
            imbuf.seek(0)
            story.append(RLImage(imbuf, width=150*mm, height=85*mm, kind="proportional"))
    story += [Spacer(1, 5*mm), Paragraph(
        "Use the original signed ultrasound report for clinical interpretation. Do not make care "
        "decisions from this prototype output.", styles["Foot"])]
    doc.build(story)
    return buffer.getvalue()

model, device, model_error = load_model(str(CHECKPOINT))
model_ready = model is not None
audit = dataset_audit()

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">✚</div><div><b>FetalAI</b><small>RESEARCH WORKSPACE</small></div></div>', unsafe_allow_html=True)
    st.caption("Multimodal AI Fetal Screening")
    page = st.radio("WORKSPACE", ["Overview", "Ultrasound analysis", "Dataset & training", "Model information", "About & safety"], label_visibility="collapsed")
    st.divider()
    st.markdown("**MODEL AVAILABILITY**")
    if model_ready:
        st.success("HC18 U-Net loaded")
    elif model_error and "not found" in model_error.lower():
        st.warning("Checkpoint required")
    else:
        st.error("Model load issue")
    st.caption("Academic research prototype · REVA University")

# Session counters are intentionally ephemeral, not patient records.
if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0

if page == "Overview":
    st.markdown('<div class="eyebrow">RESEARCH CONTROL CENTER</div>', unsafe_allow_html=True)
    st.title("FetalAI research dashboard")
    st.write("A workspace for ultrasound image segmentation, dataset inspection and research reporting.")
    st.info("Research prototype only. It is not a medical device and must not be used for diagnosis or pregnancy-care decisions.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analyses this session", st.session_state.analysis_count)
    c2.metric("HC18 image files", audit["image_count"])
    c3.metric("Matched image-mask pairs", audit["matched"])
    c4.metric("Segmentation model", "Ready" if model_ready else "Unavailable")
    st.markdown("### Research workflow")
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="feature-card"><div class="feature-icon">01</div><h3>Inspect dataset</h3><p>Check HC18 image and annotation availability before training.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="feature-card"><div class="feature-icon">02</div><h3>Analyze ultrasound</h3><p>Generate a candidate fetal-head-region mask with the HC18 U-Net.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="feature-card"><div class="feature-icon">03</div><h3>Review outputs</h3><p>Inspect overlays, probability maps and downloadable research summaries.</p></div>', unsafe_allow_html=True)
    st.markdown("### System status")
    s1, s2 = st.columns(2)
    with s1:
        st.write("**Checkpoint**")
        st.code(str(CHECKPOINT.relative_to(BASE_DIR)))
        st.write("**Compute device**")
        st.write(str(device).upper() if device else "Not initialized")
    with s2:
        st.write("**HC18 dataset folder**")
        st.code(str(IMAGE_DIR.relative_to(BASE_DIR)))
        st.write("**Dataset CSV**")
        st.write("Available" if audit["csv_exists"] else "Not found")

elif page == "Ultrasound analysis":
    st.markdown('<div class="eyebrow">IMAGE PROCESSING WORKSPACE</div>', unsafe_allow_html=True)
    st.title("Ultrasound image analysis")
    st.write("Upload a still ultrasound image to inspect the model's candidate fetal-head-region segmentation.")
    st.warning("The model output is an experimental image segmentation, not a diagnosis or a confirmed anatomical measurement.")
    uploaded = st.file_uploader("Choose an ultrasound image", type=["png", "jpg", "jpeg"])
    image = None
    if uploaded:
        raw = uploaded.getvalue()
        fingerprint = sha256(raw).hexdigest()
        if st.session_state.get("upload_fingerprint") != fingerprint:
            st.session_state.pop("segmentation", None)
            st.session_state.upload_fingerprint = fingerprint
        try:
            image = Image.open(BytesIO(raw)).convert("RGB")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                st.error("Image exceeds the supported 20-megapixel limit.")
                image = None
            else:
                st.success(f"Image loaded · {image.width} × {image.height} pixels")
        except Exception:
            st.error("The selected file could not be decoded as an image.")
    if image is not None:
        st.image(image, caption="Uploaded ultrasound image", use_container_width=True)
        spacing = st.number_input(
            "Source image pixel spacing (mm/pixel), if documented",
            min_value=0.0, value=0.0, step=0.001, format="%.6f",
            help="Enter only the exact image-specific spacing from a trusted source. Leave zero if unknown.")
        if not model_ready:
            st.warning(f"Segmentation unavailable: {model_error or 'checkpoint missing'}.")
        if st.button("Run HC18 segmentation", type="primary", disabled=not model_ready, use_container_width=True):
            with st.spinner("Running U-Net inference..."):
                try:
                    mask, overlay, probability, heatmap = run_segmentation(model, device, image)
                    st.session_state.segmentation = {
                        "mask": mask, "overlay": overlay, "probability": probability,
                        "heatmap": heatmap, "width": image.width, "height": image.height,
                        "filename": uploaded.name, "image": image.copy(), "spacing": float(spacing),
                    }
                    st.session_state.analysis_count += 1
                    st.success("Inference completed. Review the outputs below.")
                except Exception as exc:
                    st.error(f"Inference failed: {exc}")
    result = st.session_state.get("segmentation")
    if result:
        st.divider()
        st.subheader("Segmentation results")
        t1, t2, t3 = st.tabs(["Overlay", "Binary mask", "Probability heatmap"])
        with t1:
            st.image(result["overlay"], use_container_width=True)
            st.download_button("Download overlay PNG", png_bytes(result["overlay"]), "fetalai_overlay.png", "image/png")
        with t2:
            st.image(result["mask"], clamp=True, use_container_width=True)
            st.download_button("Download mask PNG", png_bytes(result["mask"], "L"), "fetalai_mask.png", "image/png")
        with t3:
            st.image(result["heatmap"], use_container_width=True)
            st.caption("Colours represent pixel-level sigmoid output, not a calibrated clinical confidence score.")
        perimeter = contour_perimeter(result["mask"], result["spacing"]) if result["spacing"] > 0 else None
        m1, m2, m3 = st.columns(3)
        m1.metric("Image dimensions", f'{result["width"]} × {result["height"]} px')
        m2.metric("Inference device", str(device).upper())
        m3.metric("Contour perimeter", f"{perimeter:.2f} mm" if perimeter is not None else "Not calculated")
        st.caption("A contour perimeter is a geometric estimate and must not be represented as a validated fetal head-circumference measurement.")
        details = st.session_state.get("report_details", {})
        st.markdown("### Report details for the PDF summary")
        st.caption("These values are manually transcribed from an existing report. They are not AI predictions.")
        with st.form("report_details"):
            x1, x2, x3 = st.columns(3)
            with x1:
                ga = st.text_input("Gestational age")
                bpd = st.text_input("BPD")
                hc = st.text_input("HC")
            with x2:
                presentation = st.selectbox("Presentation as reported", ["Not provided", "Cephalic", "Breech", "Transverse", "Oblique", "Other / unclear"])
                ac = st.text_input("AC")
                fl = st.text_input("FL")
            with x3:
                efw = st.text_input("Estimated fetal weight")
                fhr = st.text_input("Fetal heart rate")
                report_date = st.text_input("Report date")
            note = st.text_area("Other report wording (optional)")
            submitted = st.form_submit_button("Update report details")
            if submitted:
                st.session_state.report_details = {
                    "Gestational age": ga, "Fetal presentation": "" if presentation == "Not provided" else presentation,
                    "BPD": bpd, "HC": hc, "AC": ac, "FL": fl,
                    "Estimated fetal weight": efw, "Fetal heart rate": fhr,
                    "Report date": report_date, "Other report wording": note,
                }
                st.success("Report details updated for this session.")
        details = st.session_state.get("report_details", {})
        if st.button("Prepare PDF research summary", type="primary"):
            try:
                pdf = build_pdf(details, result["filename"], result["image"], result["overlay"],
                                perimeter, result["spacing"], "Loaded" if model_ready else "Unavailable")
                st.download_button("Download PDF summary", pdf, "FetalAI_research_summary.pdf", "application/pdf")
            except Exception as exc:
                st.error(f"Could not create PDF: {exc}")

elif page == "Dataset & training":
    st.markdown('<div class="eyebrow">DATA PIPELINE</div>', unsafe_allow_html=True)
    st.title("Dataset inspection")
    st.write("Dataset checks use the expected HC18 folder layout from the supplied training script.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ultrasound images", audit["image_count"])
    c2.metric("Annotation masks", audit["mask_count"])
    c3.metric("Matched pairs", audit["matched"])
    d1, d2, d3 = st.columns(3)
    d1.metric("Images without masks", audit["images_without_masks"])
    d2.metric("Masks without images", audit["masks_without_images"])
    d3.metric("Metadata CSV", "Found" if audit["csv_exists"] else "Missing")
    st.markdown("### Expected directory layout")
    st.code("data/raw/HC18/\n├── training_set/\n│   └── training_set/\n│       ├── *_HC.png\n│       └── *_HC_Annotation.png\n└── training_set_pixel_size_and_HC.csv", language="text")
    if audit["csv_exists"]:
        st.caption(f"CSV rows: {audit['csv_rows']} · Columns: {', '.join(audit['csv_columns'])}")
    if not IMAGE_DIR.exists():
        st.warning("HC18 image directory not found. Add the dataset using the expected folder structure.")
    if st.button("Refresh dataset audit"):
        st.rerun()
    st.markdown("### Training")
    st.write("Run the existing HC18 training script from the project root after verifying the dataset:")
    st.code("python check_hc18.py\npython train_hc18.py", language="powershell")
    st.warning("The supplied trainer uses a random image-level train/validation split. For stronger research claims, use a documented patient/study-level split where dataset metadata supports it. No training is started from this dashboard.")

elif page == "Model information":
    st.markdown('<div class="eyebrow">MODEL CARD</div>', unsafe_allow_html=True)
    st.title("HC18 U-Net segmentation model")
    st.write("The active model is the convolutional U-Net defined in `model.py` and trained by `train_hc18.py`.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Architecture**")
        st.write("Encoder-decoder U-Net with skip connections.")
        st.write("Input: grayscale ultrasound, 256 × 256.")
        st.write("Output: one-channel candidate fetal-head-region mask.")
    with right:
        st.markdown("**Runtime status**")
        st.write("Loaded" if model_ready else (model_error or "Checkpoint unavailable"))
        st.write(f"Expected checkpoint: `{CHECKPOINT.relative_to(BASE_DIR)}`")
        st.write(f"Compute: {str(device).upper() if device else 'Unavailable'}")
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            st.markdown("### Saved evaluation metrics")
            st.json(metrics)
        except Exception:
            st.warning("Metrics file exists but could not be parsed.")
    else:
        st.info("No evaluation metrics JSON is present. The supplied HC18 training script reports validation Dice in the terminal but does not currently save a metrics JSON.")
    st.markdown("### Separate multimodal experiment")
    st.warning("The uploaded `train.py` and `config.py` describe an ultrasound + ECG classification experiment. It is not connected to this application: the required `MultimodalFetalModel` implementation, paired dataset manifest, ECG inputs and checkpoint were not included in the supplied files.")

else:
    st.markdown('<div class="eyebrow">PROJECT DOCUMENTATION</div>', unsafe_allow_html=True)
    st.title("About FetalAI")
    st.write("Multimodal AI Fetal Screening is an academic research project developed as a prototype for ultrasound image analysis.")
    st.markdown("### Current verified capability")
    st.write("The supplied HC18 U-Net performs candidate fetal-head-region segmentation on a still ultrasound image.")
    st.markdown("### Not predicted")
    st.write("Fetal presentation, gestational age, fetal weight, growth percentile, heartbeat, anatomy, placenta location, amniotic-fluid level and overall fetal health are not inferred by this U-Net.")
    st.markdown("### Safety, privacy and intended use")
    st.warning(
        "This is not a medical device and has not been clinically validated. Do not use it to diagnose, "
        "reassure, triage or make pregnancy-care decisions. Report values are manually entered and are "
        "not verified. Avoid identifiable patient data in demonstrations. Uploaded files are processed "
        "in the app session and are not intentionally written to a database by this application; hosting "
        "and operating-system environments may have separate logs or retention."
    )

st.markdown('<div class="footer">FetalAI · REVA University · Academic research prototype · Not for diagnosis</div>', unsafe_allow_html=True)
