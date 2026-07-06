"""
Low-Light Road Accident Detection — Thesis POC
Raw / Zero-DCE / Zero-3DCE enhancement + YOLO detection on CCTV frames.

Deployable on Streamlit Cloud. Weights are fetched on first launch:
  - YOLO best.pt (x3)      -> Google Drive via gdown (set file IDs below or in secrets)
  - Zero-DCE Epoch99.pth   -> official GitHub repo (no setup needed)
  - zero3dce_final.pth     -> Google Drive via gdown
"""

import os
import time
import urllib.request

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance

from models.zerodce import enhance_net_nopool
from models.zero3dce import Zero3DCE

# ══════════════════════════════════════════════════════════════════
# CONFIG — replace the placeholder Drive file IDs, or put them in
# .streamlit/secrets.toml under [drive_ids] (secrets win if present).
# File ID = the long string between /d/ and /view in the share link.
# ══════════════════════════════════════════════════════════════════
DEFAULT_DRIVE_IDS = {
    "raw_yolo":      "PASTE_RAW_BEST_PT_FILE_ID",
    "zerodce_yolo":  "PASTE_ZERODCE_BEST_PT_FILE_ID",
    "zero3dce_yolo": "PASTE_ZERO3DCE_BEST_PT_FILE_ID",
    "zero3dce_pth":  "PASTE_ZERO3DCE_FINAL_PTH_FILE_ID",
}

ZERODCE_GITHUB_URL = (
    "https://github.com/Li-Chongyi/Zero-DCE/raw/master/"
    "Zero-DCE_code/snapshots/Epoch99.pth"
)

WEIGHTS_DIR = "weights"

# Thesis defaults (must match the training notebook)
ENHANCE_RESOLUTION = 512
BLEND_ALPHA = 0.65
SHARPEN_FACTOR = 1.15

PIPELINES = {
    "Raw (no enhancement)": {"key": "raw",      "yolo_id_key": "raw_yolo",      "yolo_file": "raw_best.pt"},
    "Zero-DCE":             {"key": "zerodce",  "yolo_id_key": "zerodce_yolo",  "yolo_file": "zerodce_best.pt"},
    "Zero-3DCE":            {"key": "zero3dce", "yolo_id_key": "zero3dce_yolo", "yolo_file": "zero3dce_best.pt"},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def drive_id(key: str) -> str:
    try:
        return st.secrets["drive_ids"][key]
    except Exception:
        return DEFAULT_DRIVE_IDS[key]


# ══════════════════════════════════════════════════════════════════
# Weight fetching + cached model loaders
# ══════════════════════════════════════════════════════════════════
def fetch_from_drive(file_id: str, dest: str):
    if os.path.exists(dest):
        return dest
    if file_id.startswith("PASTE_"):
        st.error(
            f"Missing Google Drive file ID for `{os.path.basename(dest)}`. "
            "Set it in app.py (DEFAULT_DRIVE_IDS) or in Streamlit secrets under [drive_ids]."
        )
        st.stop()
    import gdown
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with st.spinner(f"Downloading {os.path.basename(dest)} (first launch only)..."):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", dest, quiet=True)
    return dest


@st.cache_resource(show_spinner=False)
def load_yolo(path: str, file_id: str):
    from ultralytics import YOLO
    fetch_from_drive(file_id, path)
    return YOLO(path)


@st.cache_resource(show_spinner=False)
def load_zerodce():
    path = os.path.join(WEIGHTS_DIR, "zerodce_epoch99.pth")
    if not os.path.exists(path):
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        with st.spinner("Downloading Zero-DCE weights (first launch only)..."):
            urllib.request.urlretrieve(ZERODCE_GITHUB_URL, path)
    net = enhance_net_nopool().to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net


@st.cache_resource(show_spinner=False)
def load_zero3dce():
    path = os.path.join(WEIGHTS_DIR, "zero3dce_final.pth")
    fetch_from_drive(drive_id("zero3dce_pth"), path)
    net = Zero3DCE().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    net.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    net.eval()
    return net


# ══════════════════════════════════════════════════════════════════
# Enhancement — identical logic to the training notebook
# ══════════════════════════════════════════════════════════════════
def enhance_zerodce(img: Image.Image) -> Image.Image:
    net = load_zerodce()
    orig_w, orig_h = img.size
    tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    t = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        result = net(t)
        enhanced = result[-2]
    out = enhanced.squeeze(0).cpu().clamp(0, 1)
    return transforms.ToPILImage()(out).resize((orig_w, orig_h), Image.LANCZOS)


def enhance_zero3dce(img: Image.Image, blend=BLEND_ALPHA,
                     resolution=ENHANCE_RESOLUTION, sharpen=SHARPEN_FACTOR) -> Image.Image:
    net = load_zero3dce()
    orig_w, orig_h = img.size
    tf = transforms.Compose([transforms.Resize((resolution, resolution)), transforms.ToTensor()])
    t = tf(img).unsqueeze(0)
    clip = torch.stack([t, t], dim=2).to(DEVICE)
    with torch.no_grad():
        enhanced, _ = net.forward_with_alphas(clip)
    out = enhanced[0, :, 0].cpu().clamp(0, 1)
    enhanced_img = transforms.ToPILImage()(out).resize((orig_w, orig_h), Image.LANCZOS)
    blended = Image.blend(img.resize((orig_w, orig_h)), enhanced_img, alpha=blend)
    if sharpen != 1.0:
        blended = ImageEnhance.Sharpness(blended).enhance(sharpen)
    return blended


# ══════════════════════════════════════════════════════════════════
# Detection
# ══════════════════════════════════════════════════════════════════
def run_pipeline(pipeline_name: str, img: Image.Image, conf: float,
                 blend: float, sharpen: float):
    cfg = PIPELINES[pipeline_name]

    t0 = time.perf_counter()
    if cfg["key"] == "zerodce":
        model_input = enhance_zerodce(img)
    elif cfg["key"] == "zero3dce":
        model_input = enhance_zero3dce(img, blend=blend, sharpen=sharpen)
    else:
        model_input = img
    enhance_ms = (time.perf_counter() - t0) * 1000

    yolo = load_yolo(os.path.join(WEIGHTS_DIR, cfg["yolo_file"]), drive_id(cfg["yolo_id_key"]))

    t1 = time.perf_counter()
    results = yolo.predict(model_input, imgsz=640, conf=conf, verbose=False)
    detect_ms = (time.perf_counter() - t1) * 1000

    r = results[0]
    annotated = Image.fromarray(r.plot()[..., ::-1])  # BGR -> RGB

    detections = []
    for box in r.boxes:
        detections.append({
            "Class": r.names[int(box.cls)],
            "Confidence": f"{float(box.conf):.3f}",
            "Box (x1, y1, x2, y2)": ", ".join(f"{v:.0f}" for v in box.xyxy[0].tolist()),
        })

    return {
        "input": model_input,
        "annotated": annotated,
        "detections": detections,
        "enhance_ms": enhance_ms,
        "detect_ms": detect_ms,
    }


# ══════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Low-Light Accident Detection POC",
    page_icon="🚨",
    layout="wide",
)

st.title("Low-Light Road Accident Detection")
st.caption(
    "Thesis proof of concept — low-light enhancement (Zero-DCE / Zero-3DCE) "
    "paired with YOLO for accident detection on Makati City CCTV footage."
)

with st.sidebar:
    st.header("Pipeline")
    mode = st.radio("Mode", ["Single pipeline", "Compare all three"], index=0)
    if mode == "Single pipeline":
        pipeline_choice = st.selectbox("Enhancement", list(PIPELINES.keys()), index=2)
    st.header("Detection")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    with st.expander("Zero-3DCE tuning"):
        blend = st.slider("Blend alpha", 0.0, 1.0, BLEND_ALPHA, 0.05,
                          help="0 = original frame, 1 = fully enhanced")
        sharpen = st.slider("Sharpen factor", 1.0, 2.0, SHARPEN_FACTOR, 0.05)
    st.caption(f"Device: `{DEVICE.type}`")

uploaded = st.file_uploader("Upload a CCTV frame", type=["jpg", "jpeg", "png"])

if not uploaded:
    st.info("Upload a frame to run detection. Low-light frames show the enhancement pipelines best.")
    st.stop()

img = Image.open(uploaded).convert("RGB")

if mode == "Single pipeline":
    out = run_pipeline(pipeline_choice, img, conf, blend, sharpen)

    m1, m2, m3 = st.columns(3)
    m1.metric("Detections", len(out["detections"]))
    m2.metric("Enhancement time", f"{out['enhance_ms']:.0f} ms")
    m3.metric("Inference time", f"{out['detect_ms']:.0f} ms")

    if PIPELINES[pipeline_choice]["key"] == "raw":
        c1, c2 = st.columns(2)
        c1.image(img, caption="Original frame", use_container_width=True)
        c2.image(out["annotated"], caption=f"Detections — {pipeline_choice}", use_container_width=True)
    else:
        c1, c2, c3 = st.columns(3)
        c1.image(img, caption="Original frame", use_container_width=True)
        c2.image(out["input"], caption=f"Enhanced ({pipeline_choice})", use_container_width=True)
        c3.image(out["annotated"], caption="Detections", use_container_width=True)

    if out["detections"]:
        st.subheader("Detections")
        st.dataframe(out["detections"], use_container_width=True, hide_index=True)
    else:
        st.warning("No detections above the confidence threshold.")

else:
    cols = st.columns(3)
    for col, name in zip(cols, PIPELINES.keys()):
        with col:
            st.subheader(name)
            out = run_pipeline(name, img, conf, blend, sharpen)
            st.image(out["annotated"], use_container_width=True)
            st.metric("Detections", len(out["detections"]))
            st.caption(f"Enhance {out['enhance_ms']:.0f} ms · Detect {out['detect_ms']:.0f} ms")
            if out["detections"]:
                st.dataframe(out["detections"], use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Mapúa Institute of Technology — undergraduate thesis · "
    "Comparison of low-light enhancement methods for CCTV-based road accident detection"
)
