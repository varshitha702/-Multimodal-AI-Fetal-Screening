from pathlib import Path
from io import BytesIO
from datetime import datetime
import hashlib
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from model import UNet

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = BASE_DIR / "outputs" / "hc18_unet_best.pth"
CSS_PATH = BASE_DIR / "assets" / "style.css"
IMAGE_SIZE = 256

st.set_page_config(
    page_title="FetalAI | Ultrasound Research Companion",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_model(checkpoint_path: str):
    path = Path(checkpoint_path)
    if not path.exists():
        return None, None, "Trained checkpoint not found."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model, device, None


def run_segmentation(model, device, pil_image):
    gray = pil_image.convert("L")
    width, height = gray.size
    resized = gray.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
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


def contour_perimeter(mask, spacing_mm):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(contour) < 5:
        return None
    scaled = contour * float(spacing_mm)
    return float(cv2.arcLength(scaled.reshape(-1, 1, 2), True))


def png_bytes(array, mode=None):
    image = Image.fromarray(array, mode=mode) if mode else Image.fromarray(array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def pdf_report(details, image=None, overlay=None, perimeter=None, spacing=None, model_status=""):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.utils import ImageReader

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"], fontSize=8, leading=11,
                              textColor=colors.HexColor("#526174")))
    story = [
        Paragraph("FetalAI | Ultrasound Research Summary", styles["Title"]),
        Paragraph("Academic research prototype report. Not a medical diagnosis or clinical report.", styles["SmallNote"]),
        Spacer(1, 8*mm),
        Paragraph("Information transcribed from the supplied ultrasound report", styles["Heading2"]),
    ]
    rows = [["Field", "Value"]]
    for key, value in details.items():
        rows.append([key, str(value) if str(value).strip() else "Not provided"])
    table = Table(rows, colWidths=[58*mm, 105*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#102A43")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#D8E1EA")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F4F8FB")]),
        ("PADDING", (0,0), (-1,-1), 7),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story += [table, Spacer(1, 7*mm), Paragraph("AI image-processing output", styles["Heading2"])]
    story.append(Paragraph(f"Model status: {model_status}", styles["BodyText"]))
    story.append(Paragraph("The supplied U-Net is trained for fetal-head-region segmentation only. It does not determine fetal presentation, gestational age, fetal weight, heartbeat, anatomy, or overall fetal health.", styles["SmallNote"]))
    if perimeter is not None:
        story.append(Paragraph(f"Predicted mask contour perimeter: {perimeter:.2f} mm (geometric estimate using entered spacing of {spacing} mm/pixel; not clinically validated).", styles["BodyText"]))
    else:
        story.append(Paragraph("No perimeter in millimetres was calculated. Exact source-image pixel spacing was not provided or a usable contour was not found.", styles["BodyText"]))
    if image is not None:
        story += [Spacer(1, 5*mm), Paragraph("Uploaded image", styles["Heading3"])]
        imbuf = BytesIO()
        image.save(imbuf, format="PNG")
        imbuf.seek(0)
        story.append(RLImage(imbuf, width=150*mm, height=90*mm, kind="proportional"))
    if overlay is not None:
        story += [Spacer(1, 4*mm), Paragraph("Segmentation overlay", styles["Heading3"])]
        ovbuf = BytesIO()
        Image.fromarray(overlay).save(ovbuf, format="PNG")
        ovbuf.seek(0)
        story.append(RLImage(ovbuf, width=150*mm, height=90*mm, kind="proportional"))
    story += [
        Spacer(1, 7*mm),
        Paragraph("Next steps", styles["Heading2"]),
        Paragraph("Use the original signed ultrasound report for clinical measurements and interpretation. Discuss questions or unexpected findings with a qualified obstetric clinician. Do not make care decisions from this prototype output.", styles["BodyText"]),
        Spacer(1, 5*mm),
        Paragraph(f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')} | No uploaded information is saved by this app to a database.", styles["SmallNote"]),
    ]
    doc.build(story)
    return buffer.getvalue()


model, device, model_error = load_model(str(CHECKPOINT))
model_ready = model is not None

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">✚</div><div><b>FetalAI</b><small>ULTRASOUND RESEARCH COMPANION</small></div></div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Patient workspace", "How it works", "About & safety"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="side-label">MODEL STATUS</div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="pill good">● U-Net ready</span>' if model_ready
        else '<span class="pill warn">● Checkpoint required</span>',
        unsafe_allow_html=True,
    )
    st.caption("Academic prototype · REVA University")


if page == "Patient workspace":
    st.markdown('<div class="eyebrow">ULTRASOUND IMAGE WORKSPACE</div>', unsafe_allow_html=True)
    st.title("Your ultrasound, explained clearly.")
    st.write("Upload an ultrasound image and, if available, enter the measurements exactly as written in the official report.")
    st.info("This is a student research demonstration, not a medical service. It cannot assess fetal health or replace your clinician's report.")

    uploaded = st.file_uploader("Upload ultrasound image", type=["png", "jpg", "jpeg"], help="Use a still ultrasound image in PNG or JPEG format.")
    image = None
    if uploaded is not None:
        try:
            upload_bytes = uploaded.getvalue()
            upload_fingerprint = hashlib.sha256(upload_bytes).hexdigest()
            if st.session_state.get("upload_fingerprint") != upload_fingerprint:
                st.session_state.pop("segmentation", None)
                st.session_state["upload_fingerprint"] = upload_fingerprint
            image = Image.open(BytesIO(upload_bytes)).convert("RGB")
            if image.width * image.height > 20_000_000:
                st.error("This image is larger than the supported 20-megapixel limit.")
                image = None
            else:
                st.success(f"Image loaded: {image.width} × {image.height} pixels")
        except Exception:
            st.error("The selected file could not be opened as an image.")
            image = None

    st.markdown("### Details from the existing ultrasound report")
    st.caption("These are report-entry fields, not AI-detected findings. Leave anything blank if it is not written in your report.")
    with st.form("report_details"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gestational_age = st.text_input("Gestational age", placeholder="e.g.  twenty weeks + 3 days")
            presentation = st.selectbox("Fetal presentation (as reported)", ["Not provided", "Cephalic", "Breech", "Transverse", "Oblique", "Other / unclear"])
        with c2:
            bpd = st.text_input("BPD", placeholder="Value and unit, if reported")
            hc = st.text_input("HC", placeholder="Value and unit, if reported")
        with c3:
            ac = st.text_input("AC", placeholder="Value and unit, if reported")
            fl = st.text_input("FL", placeholder="Value and unit, if reported")
        c4, c5 = st.columns(2)
        with c4:
            efw = st.text_input("Estimated fetal weight", placeholder="Value and unit, if reported")
            fhr = st.text_input("Fetal heart rate", placeholder="Value and unit, if reported")
        with c5:
            report_date = st.text_input("Report date", placeholder="As printed on the report")
            report_note = st.text_area("Other report wording (optional)", placeholder="Transcribe relevant wording exactly. Avoid adding your own interpretation.")
        submitted = st.form_submit_button("Save report details for this session", type="primary", use_container_width=True)
        if submitted:
            st.session_state["report_details"] = {
                "Gestational age": gestational_age,
                "Fetal presentation": presentation if presentation != "Not provided" else "",
                "BPD": bpd, "HC": hc, "AC": ac, "FL": fl,
                "Estimated fetal weight": efw, "Fetal heart rate": fhr,
                "Report date": report_date, "Other report wording": report_note,
            }
            st.success("Details are available for this browser session. They are not stored in a medical record.")

    st.markdown("### AI image analysis")
    if not model_ready:
        st.warning("The trained HC18 checkpoint is not present. Add `outputs/hc18_unet_best.pth` to enable segmentation. The report-entry section can still be demonstrated.")
    if image is not None:
        st.image(image, caption="Uploaded ultrasound image", use_container_width=True)
        spacing = st.number_input(
            "Source image pixel spacing (mm/pixel), if known",
            min_value=0.0, value=0.0, step=0.001, format="%.6f",
            help="Only enter the exact spacing associated with this image. Do not estimate it.",
        )
        if st.button("Run fetal-head segmentation", type="primary", disabled=not model_ready, use_container_width=True):
            with st.spinner("Processing image with the HC18 U-Net..."):
                try:
                    mask, overlay, probability, heatmap = run_segmentation(model, device, image)
                    st.session_state["segmentation"] = {
                        "mask": mask, "overlay": overlay, "probability": probability,
                        "heatmap": heatmap, "width": image.width, "height": image.height,
                        "filename": uploaded.name,
                    }
                    st.session_state["spacing"] = float(spacing)
                    st.success("Segmentation finished. Review the outputs below.")
                except Exception as exc:
                    st.error(f"Segmentation failed: {exc}")

    result = st.session_state.get("segmentation")
    if result is not None:
        st.markdown("---")
        st.markdown("## Image analysis results")
        st.caption("The green region is the model's predicted fetal-head segmentation. It is a model output, not a confirmed anatomical boundary.")
        tab1, tab2, tab3 = st.tabs(["Segmentation overlay", "Binary mask", "Probability map"])
        with tab1:
            st.image(result["overlay"], use_container_width=True)
        with tab2:
            st.image(result["mask"], clamp=True, use_container_width=True)
            st.download_button("Download binary mask (PNG)", png_bytes(result["mask"], "L"), "fetalai_head_mask.png", "image/png")
        with tab3:
            st.image(result["heatmap"], use_container_width=True)
            st.caption("Colour indicates the model's pixel-level sigmoid output. It is not a clinical confidence score.")
        spacing_used = st.session_state.get("spacing", 0.0)
        perimeter = contour_perimeter(result["mask"], spacing_used) if spacing_used > 0 else None
        m1, m2, m3 = st.columns(3)
        m1.metric("Image dimensions", f'{result["width"]} × {result["height"]} px')
        m2.metric("Inference device", str(device).upper())
        m3.metric("Contour perimeter", f"{perimeter:.2f} mm" if perimeter is not None else "Not calculated")
        if perimeter is not None:
            st.warning("This is a geometric estimate from the predicted mask and entered pixel spacing. It is not a validated fetal head-circumference measurement.")
        else:
            st.caption("A perimeter in millimetres is not shown because exact source-image spacing was not supplied or no usable contour was detected.")
        st.download_button("Download segmentation overlay (PNG)", png_bytes(result["overlay"]), "fetalai_overlay.png", "image/png")

    details = st.session_state.get("report_details", {})
    if details:
        st.markdown("---")
        st.markdown("## Your entered report details")
        st.caption("Values below are copied from the information entered above. FetalAI has not independently measured or verified them.")
        shown = [(k, v) for k, v in details.items() if str(v).strip()]
        if shown:
            st.dataframe([{"Report field": k, "Entered value": v} for k, v in shown], hide_index=True, use_container_width=True)
        else:
            st.info("No report details were entered.")
        perimeter_for_pdf = None
        if result is not None and st.session_state.get("spacing", 0) > 0:
            perimeter_for_pdf = contour_perimeter(result["mask"], st.session_state["spacing"])
        pdf = pdf_report(
            details,
            image=image,
            overlay=result["overlay"] if result is not None else None,
            perimeter=perimeter_for_pdf,
            spacing=st.session_state.get("spacing"),
            model_status="Loaded" if model_ready else "Not available",
        )
        st.download_button("Download patient-friendly summary (PDF)", pdf, "FetalAI_ultrasound_summary.pdf", "application/pdf", type="primary")
    else:
        st.caption("Enter report details above to create a downloadable summary.")


elif page == "How it works":
    st.markdown('<div class="eyebrow">THE WORKFLOW</div>', unsafe_allow_html=True)
    st.title("What the application can and cannot do")
    steps = [
        ("01", "Upload an image", "A still ultrasound image is loaded locally in the Streamlit session."),
        ("02", "Preprocessing", "The image is converted to grayscale, resized to 256 × 256 and normalized."),
        ("03", "U-Net segmentation", "The trained model predicts a pixel-level fetal-head-region mask."),
        ("04", "Visual review", "The app displays the original image, overlay, binary mask and probability heatmap."),
        ("05", "Optional geometry", "If exact source pixel spacing is known, a contour perimeter estimate can be calculated."),
        ("06", "Report summary", "Measurements can be transcribed from an existing clinical report and included in a PDF summary."),
    ]
    for number, title, desc in steps:
        st.markdown(f'<div class="workflow"><span>{number}</span><div><b>{title}</b><p>{desc}</p></div></div>', unsafe_allow_html=True)
    st.markdown("### Not currently inferred by the model")
    st.write("Fetal presentation, gestational age, fetal weight, growth percentile, heartbeat, anatomy, placenta location, amniotic-fluid level and fetal health are not predicted by this U-Net. These details must come from the signed ultrasound report or an appropriately validated system.")

else:
    st.markdown('<div class="eyebrow">PROJECT INFORMATION</div>', unsafe_allow_html=True)
    st.title("FetalAI research prototype")
    st.write("FetalAI is an academic ultrasound image-segmentation project using a U-Net architecture and the HC18 fetal-head ultrasound dataset.")
    st.markdown("### Model specifications")
    st.markdown("""
    - **Dataset:** HC18 image and annotation pairs
    - **Input:** Grayscale ultrasound image
    - **Architecture:** U-Net encoder-decoder
    - **Network input size:** 256 × 256 pixels
    - **Output:** Binary candidate fetal-head-region mask
    - **Training script:** `train_hc18.py`
    """)
    st.markdown("### Important safety and privacy notes")
    st.warning("""
    - This is an academic prototype, not a medical device.
    - It is not clinically validated and must not be used to diagnose, reassure, triage, or make pregnancy-care decisions.
    - A segmentation perimeter is not automatically equivalent to a clinically accepted head-circumference measurement.
    - Report values are manually entered by the user and are not verified by the application.
    - Do not upload identifying patient information for public demonstrations. This app does not intentionally save uploads to a database, but your local computer and hosting environment may have their own logs or storage.
    - Always use the original report and consult a qualified obstetric clinician for interpretation.
    """)
    st.markdown("### Model checkpoint")
    st.write(f"Expected location: `{CHECKPOINT.relative_to(BASE_DIR)}`")
    st.write("Checkpoint status: " + ("Loaded successfully." if model_ready else "Not found. Train the HC18 model first."))

st.markdown('<div class="footer">FetalAI · REVA University · Academic research prototype · Not for diagnosis</div>', unsafe_allow_html=True)
