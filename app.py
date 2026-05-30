# Streamlit entry point: python -m streamlit run app.py

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image

from precomputed import (
    DATASET_SIZE_TO_N,
    IMAGE_TYPES,
    IMAGE_TYPES_SIMULATOR,
    MODEL_IDS,
    MODEL_RESULTS_DF,
    PRIMARY_METRIC_NOTE,
    canonical_image_type,
    format_lr,
    get_results_page_data,
    lookup_recommendation,
)
from precomputed import DATASET_SIZES as DATASET_SIZE_CHOICES

TAB_LABELS = [
    "Overview",
    "Image Quality Simulator",
    "Recommendation Tool",
    "Results",
    "Biological Context",
]

# Default widget selections on first load.
DEFAULT_REC_DATASET_SIZE = "5000–10000"
DEFAULT_REC_IMAGE_TYPE_INDEX = 0
DEFAULT_IQSIM_IMAGE_TYPE_INDEX = 1

# Results page figure layout (fraction of column width).
RESULTS_TRAINING_CURVE_SCALE = 0.75
RESULTS_CONFUSION_MATRIX_SCALE = 0.90
RESULTS_CONFUSION_MATRIX_SHIFT = 0.0001

# Heuristic crop for iqsim side-by-side PNGs (title band above image panels).
IQSIM_TITLE_ROW_WHITE = 254
IQSIM_CONTENT_ROW_MAX = 250
IQSIM_TITLE_SCAN_Y_START = 30
IQSIM_TITLE_SCAN_Y_END = 80
IQSIM_DEFAULT_TOP_OFFSET = 49

_APP_DIR = Path(__file__).resolve().parent
_IQSIM_ASSETS_DIR = _APP_DIR / "iqsim_assets"
_BASELINE_IMAGE_TYPE = IMAGE_TYPES[0]


def _init_persistent_choice(key: str, options: list[str], default: str) -> None:
    """Init stored choice only (survives tab changes; not tied to widget keys)."""
    if key not in st.session_state:
        st.session_state[key] = default if default in options else options[0]
    elif st.session_state[key] not in options:
        st.session_state[key] = default if default in options else options[0]


def _persistent_selectbox(
    label: str,
    options: list[str],
    persistent_key: str,
    **kwargs,
) -> str:
    """Selectbox with a separate widget key so values persist across tab switches."""
    widget_key = f"_widget_{persistent_key}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state[persistent_key]
    st.selectbox(label, options, key=widget_key, **kwargs)
    st.session_state[persistent_key] = st.session_state[widget_key]
    return st.session_state[persistent_key]


def _invasive_recall_from_table(per_class_df) -> float | None:
    if per_class_df is None or per_class_df.empty:
        return None
    inv = per_class_df.loc[per_class_df["Class"] == "Invasive"]
    if inv.empty:
        return None
    val = inv.iloc[0].get("Recall")
    return float(val) if val is not None and pd.notna(val) else None


def _headline_metric_value(label: str, value: Any) -> str:
    if value is None:
        return "n/a"
    if label == "LR" and isinstance(value, (int, float)):
        return format_lr(float(value))
    if label == "Best Val Acc" and isinstance(value, (int, float)):
        return f"{value:.4f}"
    if label in ("Accuracy", "MCC", "Invasive Recall") and isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def _headline_kpi_items(metrics: dict, per_class_df) -> list[tuple[str, Any]]:
    """Label/value pairs for the Results KPI bar."""
    return [
        ("Accuracy", metrics.get("accuracy")),
        ("MCC", metrics.get("mcc")),
        ("LR", metrics.get("lr")),
        ("Invasive Recall", _invasive_recall_from_table(per_class_df)),
        ("Best Val Acc", metrics.get("best_val_acc")),
        ("Total Epochs", metrics.get("total_epochs")),
        ("Best Checkpoint", metrics.get("best_checkpoint")),
    ]


def _render_results_banner(
    image_type: str,
    model_name: str,
    n: int,
    metrics: dict,
    per_class_df,
) -> None:
    """Dark header + light-blue KPI bar in one HTML block (no white gaps)."""
    cells = []
    for label, value in _headline_kpi_items(metrics, per_class_df):
        display = html.escape(_headline_metric_value(label, value))
        cells.append(
            f'<div class="results-kpi-cell">'
            f'<p class="results-kpi-label">{html.escape(label)}</p>'
            f'<p class="results-kpi-value">{display}</p>'
            f"</div>"
        )

    st.markdown(
        f"""
<div class="results-banner">
  <div class="results-banner-header">
    <span>{html.escape(image_type)} | n = {n:,}</span>
    <span>{html.escape(model_name)}</span>
  </div>
  <div class="results-kpi-bar">
    <div class="results-kpi-grid">{"".join(cells)}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_per_class_table_notes(*, show_macro_avg: bool) -> None:
    notes = [
        f"{PRIMARY_METRIC_NOTE} The saved model corresponds to the best "
        "validation checkpoint, not the final epoch.",
    ]
    st.markdown(
        '<div class="results-table-notes">'
        + "".join(f"<p>{note}</p>" for note in notes)
        + "</div>",
        unsafe_allow_html=True,
    )


def _inject_app_styles() -> None:
    st.markdown(
        """
<style>
/* Section nav: current = #2563EB underline; hover = light blue; slide on change */
@keyframes nav-underline-slide {
    from { transform: scaleX(0); opacity: 0.5; }
    to { transform: scaleX(1); opacity: 1; }
}
div[data-testid="stMainBlockContainer"] {
    max-width: 1500px;
    margin: 0 auto;
}
div[data-testid="stRadio"] [role="radiogroup"] {
    display: flex !important;
    flex-wrap: wrap;
    gap: 0.65rem 1rem !important;
    justify-content: space-between !important;
    width: 100% !important;
    border-bottom: none !important;
    padding-bottom: 0.55rem !important;
    position: relative !important;
}
div[data-testid="stRadio"] [role="radiogroup"] > label {
    flex: 1 1 auto !important;
    justify-content: center !important;
    margin: 0 !important;
    padding: 0.35rem 0.5rem 0.6rem !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    position: relative !important;
    overflow: visible !important;
    box-shadow: none !important;
}
div[data-testid="stRadio"] [role="radiogroup"] > label::after {
    content: "" !important;
    display: block !important;
    position: absolute !important;
    left: 8% !important;
    width: 84% !important;
    bottom: 2px !important;
    height: 3px !important;
    border-radius: 2px !important;
    background: transparent !important;
    transform: scaleX(0) !important;
    transform-origin: center bottom !important;
    transition: background 0.18s ease, transform 0.26s ease, opacity 0.18s ease !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
div[data-testid="stRadio"] [role="radiogroup"] > label:hover::after {
    background: #4a77aa !important;
    transform: scaleX(1) !important;
    opacity: 1 !important;
}
div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked)::after {
    background: #1f4e79 !important;
    transform: scaleX(1) !important;
    opacity: 1 !important;
    animation: nav-underline-slide 0.28s ease forwards !important;
}
div[data-testid="stRadio"] [role="radiogroup"] > label:hover,
div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked),
div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked):hover {
    background: transparent !important;
    box-shadow: none !important;
}
div[data-testid="stRadio"] [role="radiogroup"] > label > div:first-child {
    display: none !important;
}
div[data-testid="stRadio"] [role="radiogroup"] > label p {
    font-size: 1.05rem;
}
/* Selectbox palette (unified blue; override Streamlit red accent) */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div {
    border-color: #D1D5DB !important;
    background-color: #FFFFFF !important;
    outline: none !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stSelectbox"] div[data-baseweb="select"] input,
div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: #374151 !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within > div,
div[data-baseweb="select"]:focus-within > div,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 1px #2563EB, 0 0 0 0.2rem rgba(37, 99, 235, 0.2) !important;
}
[data-testid="stSelectbox"] ul[role="listbox"] li,
[data-testid="stSelectbox"] li[role="option"],
ul[role="listbox"] li,
li[role="option"],
div[data-baseweb="popover"] li {
    color: #374151 !important;
    background-color: #FFFFFF !important;
}
[data-testid="stSelectbox"] ul[role="listbox"] li[aria-selected="true"],
[data-testid="stSelectbox"] li[role="option"][aria-selected="true"],
ul[role="listbox"] li[aria-selected="true"],
li[role="option"][aria-selected="true"],
div[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: #DBEAFE !important;
    color: #1E3A8A !important;
}
[data-testid="stSelectbox"] ul[role="listbox"] li:hover,
[data-testid="stSelectbox"] li[role="option"]:hover,
[data-testid="stSelectbox"] ul[role="listbox"] li[data-highlighted="true"],
[data-testid="stSelectbox"] li[role="option"][data-highlighted="true"],
ul[role="listbox"] li:hover,
li[role="option"]:hover,
ul[role="listbox"] li[data-highlighted="true"],
li[role="option"][data-highlighted="true"],
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] li[data-highlighted="true"] {
    background-color: #EFF6FF !important;
    color: #374151 !important;
}
[data-testid="stSelectbox"] ul[role="listbox"] li[aria-selected="true"]:hover,
[data-testid="stSelectbox"] li[role="option"][aria-selected="true"]:hover,
ul[role="listbox"] li[aria-selected="true"]:hover,
li[role="option"][aria-selected="true"]:hover,
div[data-baseweb="popover"] li[aria-selected="true"]:hover {
    background-color: #DBEAFE !important;
    color: #1E3A8A !important;
}

/* Results page: header + KPI banner */
div[data-testid="stMarkdownContainer"]:has(.results-banner) {
    background: transparent !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
.results-banner {
    margin-bottom: 10px;
    border: 1px solid #c5d9eb;
    border-radius: 6px;
    overflow: hidden;
    width: 100%;
    box-sizing: border-box;
}
.results-banner-header {
    background: #1f4e79;
    color: #fff;
    padding: 14px 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 1.35rem;
    font-weight: 700;
}
.results-kpi-bar {
    background: #d9eaf7;
    padding: 8px 12px 10px;
    border-radius: 0;
    margin: 0;
    border: none;
    width: 100%;
    box-sizing: border-box;
}
.results-kpi-grid {
    display: grid;
    grid-template-columns: repeat(7, minmax(0, 1fr));
    gap: 0.35rem 0.5rem;
    background: #d9eaf7;
}
.results-kpi-cell {
    background: #d9eaf7;
    min-width: 0;
}
.results-kpi-label {
    margin: 0;
    font-size: 0.94rem !important;
    color: #1f4e79;
    font-weight: 600;
    line-height: 1.2;
    text-align: center;
}
.results-kpi-value {
    margin: 2px 0 0;
    font-size: 1.05rem !important;
    color: #000;
    font-weight: 700;
    line-height: 1.25;
    text-align: center;
}
/* Results page: table notes below per-class table */
.results-table-notes {
    margin-top: 0.55rem;
    font-size: 0.96rem;
    color: #6b7280;
    line-height: 1.5;
}
.results-table-notes p {
    margin: 0.35rem 0 0;
}
/* Generate (primary) button */
div[data-testid="stButton"] button[kind="primary"],
button[kind="primary"][data-testid="stBaseButton-primary"] {
    background-color: #1f4e79 !important;
    color: #FFFFFF !important;
    border-color: #1f4e79 !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover,
button[kind="primary"][data-testid="stBaseButton-primary"]:hover {
    background-color: #163d5e  !important;
    border-color: #163d5e  !important;
    color: #FFFFFF !important;
}
div[data-testid="stButton"] button[kind="primary"]:focus,
button[kind="primary"][data-testid="stBaseButton-primary"]:focus {
    box-shadow: 0 0 0 0.15rem rgba(37, 99, 235, 0.35) !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _split_training_curves(curve_path: Path) -> tuple[Image.Image, Image.Image]:
    """Training-curve PNGs are two side-by-side plots; split for a wider display."""
    img = Image.open(curve_path).convert("RGB")
    w, h = img.size
    mid = w // 2
    return img.crop((0, 0, mid, h)), img.crop((mid, 0, w, h))


def _show_results_figure(
    image, scale: float, *, align: str = "center", shift: float = 0.0
) -> None:
    """Show a figure using a fixed fraction of the parent column width."""
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image

    if align == "left":
        shift = max(0.0, shift)
        rest = max(0.0, 1 - shift - scale)
        left_pad, body, _ = st.columns([shift, scale, rest])
        with body:
            st.image(img, use_container_width=True)
    else:
        pad = max(0.0, (1 - scale) / 2)
        left, body, right = st.columns([pad, scale, pad])
        with body:
            st.image(img, use_container_width=True)


def _prepare_per_class_display(df):
    """Format table for display: 2 dp on Precision/Recall/F1."""
    out = df.copy()
    for col in ("Precision", "Recall", "F1-Score"):
        if col in out.columns:
            out[col] = out[col].apply(
                lambda v: f"{float(v):.2f}"
                if v is not None and pd.notna(v) and isinstance(v, (int, float))
                else v
            )
    if "Support" in out.columns:
        out["Support"] = out["Support"].apply(
            lambda v: int(v) if v is not None and pd.notna(v) else v
        )
    return out


def _style_per_class_table(df):
    if df is None or df.empty:
        return df

    display = _prepare_per_class_display(df)

    def _highlight(row):
        n = len(row)
        cls = row.get("Class")
        if cls == "Invasive":
            return ["background-color:#fde8e8;font-weight:700"] * n
        if cls == "DCIS1":
            return ["background-color:#fde8e8"] * n
        return [""] * n

    return (
        display.style.apply(_highlight, axis=1)
        .set_table_styles(
            [
                {
                    "selector": "table",
                    "props": [("width", "100%")],
                },
                {
                    "selector": "thead th",
                    "props": [
                        ("background-color", "#1f4e79"),
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("text-align", "left"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [("text-align", "left")],
                },
            ],
            overwrite=False,
        )
    )


def _centered_black_label(text: str) -> None:
    st.markdown(
        f"""
<div style="text-align:center; color:#000; font-size:0.95rem; font-weight:600; line-height:1.2; margin:0.15rem 0 0.35rem 0;">{text}</div>
""",
        unsafe_allow_html=True,
    )


def _iqsim_comparison_top_offset(img: Image.Image) -> int:
    """Skip the baked-in title row above side-by-side comparison PNGs."""
    gray = img.convert("L")
    w, h = gray.size
    step = max(1, w // 40)
    pixels = gray.load()
    row_mean = [
        sum(pixels[x, y] for x in range(0, w, step)) / len(range(0, w, step))
        for y in range(h)
    ]
    for y in range(IQSIM_TITLE_SCAN_Y_START, min(IQSIM_TITLE_SCAN_Y_END, h - 2)):
        if (
            row_mean[y] > IQSIM_TITLE_ROW_WHITE
            and row_mean[y + 1] < IQSIM_CONTENT_ROW_MAX
        ):
            return y + 1
    return IQSIM_DEFAULT_TOP_OFFSET


def _split_iqsim_comparison(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    top = _iqsim_comparison_top_offset(img)
    w, h = img.size
    body = img.crop((0, top, w, h))
    mid = body.width // 2
    return body.crop((0, 0, mid, body.height)), body.crop((mid, 0, body.width, body.height))


def _discover_iqsim_invasive_dir() -> Path:
    """First iqsim subfolder that contains invasive-tumor comparison PNGs."""
    if not _IQSIM_ASSETS_DIR.is_dir():
        return _IQSIM_ASSETS_DIR / "drive-download-20260521T002204Z-3-001"
    for child in sorted(_IQSIM_ASSETS_DIR.iterdir()):
        if child.is_dir() and any(child.glob("Invasive_Tumor_*.png")):
            return child
    return _IQSIM_ASSETS_DIR / "drive-download-20260521T002204Z-3-001"


_init_persistent_choice("rec_dataset_size", DATASET_SIZE_CHOICES, DEFAULT_REC_DATASET_SIZE)
_init_persistent_choice("rec_image_type", IMAGE_TYPES, IMAGE_TYPES[DEFAULT_REC_IMAGE_TYPE_INDEX])
_init_persistent_choice(
    "iqsim_image_type", IMAGE_TYPES, IMAGE_TYPES[DEFAULT_IQSIM_IMAGE_TYPE_INDEX]
)
st.session_state.rec_image_type = canonical_image_type(st.session_state.rec_image_type)
st.session_state.iqsim_image_type = canonical_image_type(st.session_state.iqsim_image_type)
if "result_payload" not in st.session_state:
    st.session_state.result_payload = None

IMAGE_TYPE_SECTIONS: dict[str, tuple[str, str]] = {
    "Full Colour (Mayer's)": (
        "Mayer's H&E is the reference full-colour condition: baseline staining used "
        "for comparison against other image types in this study.",
        "Serves as the default reference when judging how much model performance shifts "
        "under Harris stain, resolution loss, noise, understained, or grayscale conversion.",
    ),
    "Full Colour (Harris)": (
        "Harris H&E staining variant applied to the same invasive tumor patch. "
        "Colour balance and contrast can differ from the Mayer's baseline while "
        "preserving overall tissue structure.",
        "Stain-protocol differences shift RGB distributions; models trained only on "
        "one stain may need retraining or stain-normalisation when Harris-stained "
        "slides are used.",
    ),
    "Low Resolution": (
        "Patch downsampled and upsampled to mimic loss of spatial detail while keeping "
        "the same field of view.",
        "Fine nuclear shape and texture are blurred; this often hurts models that "
        "depend on high-frequency spatial patterns for class separation.",
    ),
    "Noise": (
        "Additive noise overlaid on the patch, simulating sensor or compression artefacts.",
        "Noise disrupts local intensity patterns and can reduce CNN and handcrafted "
        "feature stability unless denoising or augmentation is used during training.",
    ),
    "Understained": (
        "Understained H&E where haematoxylin and/or eosin intensity is weaker than "
        "the Mayer's reference, producing a paler appearance.",
        "Weaker staining reduces visible nuclear and cytoplasmic detail. Feature "
        "extractors that rely on colour intensity may underperform unless the pipeline "
        "is robust to low stain.",
    ),
    "Grayscale": (
        "Colour channels removed (shown as RGB greyscale). Spatial layout is unchanged "
        "but H&E colour cues are absent.",
        "Models that lean on pink/blue stain ratios may lose discriminative signal; "
        "texture-only or HOG-based approaches can remain competitive.",
    ),
}

_IQSIM_INVASIVE_DIR = _discover_iqsim_invasive_dir()
_IQSIM_INVASIVE_FILES: dict[str, str] = {
    "Full Colour (Mayer's)": "Invasive_Tumor_combined_grid.png",
    "Full Colour (Harris)": "Invasive_Tumor_Harris_Stain.png",
    "Low Resolution": "Invasive_Tumor_Low_Resolution.png",
    "Noise": "Invasive_Tumor_Noise.png",
    "Understained": "Invasive_Tumor_Understained_Stain.png",
    "Grayscale": "Invasive_Tumor_Grayscale.png",
}
_QUALITY_SIMULATOR_IMAGE: dict[str, Path] = {
    label: _IQSIM_INVASIVE_DIR / fname for label, fname in _IQSIM_INVASIVE_FILES.items()
}

st.set_page_config(
    page_title="H&E Image Model Recommendation Tool",
    layout="wide",
)

_inject_app_styles()

st.title("H&E Image Model Recommendation Tool")

if "nav_radio" not in st.session_state:
    _legacy = st.session_state.get("nav_tab")
    st.session_state.nav_radio = (
        _legacy if _legacy in TAB_LABELS else "Recommendation Tool"
    )

_nav_pending = st.session_state.pop("nav_pending", None)
if _nav_pending in TAB_LABELS:
    st.session_state.nav_radio = _nav_pending

if st.session_state.nav_radio not in TAB_LABELS:
    st.session_state.nav_radio = "Recommendation Tool"

st.radio(
    "Section",
    TAB_LABELS,
    horizontal=True,
    label_visibility="collapsed",
    key="nav_radio",
)

active = st.session_state.nav_radio

if active == "Overview":
    st.subheader("Project Aim")
    st.markdown(
        "This project evaluates **classical and deep learning** models for classifying "
        "isolated **H&E-stained breast cell patches** across four tumor and immune "
        "classes. Experiments cover **six image-quality / staining conditions** "
        "(Mayer's baseline, Harris stain, low resolution, noise, understained, and "
        "grayscale) and **six training-set size bands**, with precomputed metrics and "
        "plots bundled in this app."
    )
    st.subheader("Research Question")
    st.markdown(
        "Which models remain most reliable under different **dataset sizes**, "
        "**staining protocols**, and **simulated degradations**, and how can those "
        "findings be turned into a practical **model recommendation** workflow for "
        "computational pathology?"
    )
    st.subheader("Target Audience")
    st.markdown(
        "Computational researchers and students working with histology image data who "
        "need to **compare candidate models**, interpret per-class behaviour "
        f"({PRIMARY_METRIC_NOTE}) and choose a model under a given "
        "imaging scenario without re-running the full training grid."
    )

    st.subheader("Models at a Glance (Study Roles)")
    html = MODEL_RESULTS_DF.style.set_table_styles([
        {
            "selector": "thead th",
            "props": [
                ("background-color", "#1f4e79"),
                ("color", "white"),
                ("font-weight", "bold"),
                ("text-align", "left"),
            ],
        },
        {
            "selector": "td",
            "props": [("text-align", "left"), ("font-size", "1rem")],
        },
        {
            "selector": "th",
            "props": [("font-size", "1rem")],
        },
        {
            "selector": "th:nth-child(1), td:nth-child(1)",
            "props": [("width", "120px")],
        },
        {
            "selector": "th:nth-child(2), td:nth-child(2)",
            "props": [("width", "45%")],
        },
        {
            "selector": "th:nth-child(3), td:nth-child(3)",
            "props": [("width", "40%")],
        },
    ]).hide(axis="index").to_html()
    html = html.replace('<table', '<table style="width:100%"')
    st.markdown(html, unsafe_allow_html=True)

    st.subheader("Using This App")
    st.markdown(
        """
1. **Image Quality Simulator:** Pick an image type to see what the degradation means,
   why it matters for modelling, and a visual comparison against the Mayer's baseline
   (invasive-tumor example patch).
2. **Recommendation Tool:** Choose **Dataset Size** and **Image Type**, then click
   **Generate** to store a recommendation and open **Results**.
3. **Results:** View the recommended model, training curves, confusion matrix,
   per-class metrics, and a short narrative summary for that scenario.
4. **Biological Context:** Background on the four classification classes used in the study.
        """
    )

    st.subheader("Experimental Grid (Precomputed)")
    col_exp1, col_exp2 = st.columns(2, gap="large")
    with col_exp1:
        st.markdown("**Models Compared**")
        for mid in MODEL_IDS:
            st.markdown(f"- {mid}")
        st.markdown("**Image Types** (same order as elsewhere in the app)")
        for it in IMAGE_TYPES:
            st.markdown(f"- {it}")
    with col_exp2:
        st.markdown("**Dataset Size Bands** (representative training *n*)")
        for band in DATASET_SIZE_CHOICES:
            n = DATASET_SIZE_TO_N[band]
            st.markdown(f"- **{band}** (n = {n:,})")
    st.markdown(
    '<p style="font-size:0.96rem; color:#6b7280;">Recommendations and plots are selected from the best-performing run for each (image type, <em>n</em>) pair in the study assets.</p>',
    unsafe_allow_html=True,
)

    st.subheader("Classification Task")
    st.markdown(
        "Four classes: **Invasive tumor**, **DCIS 1**, **CD8+ T cells**, and "
        "**Macrophages 1**. The source collection was larger; these four were retained "
        "for a manageable, biologically interpretable four-way task (see "
        "**Biological Context**). Class imbalance was addressed in training via "
        "class-balanced sampling where appropriate."
    )
    
    st.subheader("Training Details")
    st.markdown(
        "Hyperparameters were tuned with Optuna (median pruner, 10 trials per model) "
        "optimising learning rate on a log scale against validation accuracy. Final training used early "
        "stopping with patience 5 and a minimum improvement threshold of 1e-4 over a maximum of 30 epochs, "
        "with best-weight restoration on completion. Light augmentation was applied to improve model "
        "generalisation and robustness, comprising random horizontal and vertical flips (p=0.5) and colour "
        "jitter (brightness, contrast, saturation ±0.2; hue ±0.05)."
    )

elif active == "Image Quality Simulator":
    st.subheader("Image Quality Simulator")
    col_iq_left, col_iq_right = st.columns([1, 1.15], gap="large")

    with col_iq_left:
        st.markdown("**Image Type**")
        image_type_sim = _persistent_selectbox(
            "Image Type",
            IMAGE_TYPES,
            "iqsim_image_type",
            label_visibility="collapsed",
        )
        what_it_means, why_modelling = IMAGE_TYPE_SECTIONS[image_type_sim]
        with st.container(border=True):
            st.markdown("**What it means**")
            st.markdown(what_it_means)
            st.markdown("**Why it matters for modelling**")
            st.markdown(why_modelling)

    with col_iq_right:
        st.markdown("**Comparison Preview**")
        sim_path = _QUALITY_SIMULATOR_IMAGE.get(image_type_sim)
        if sim_path is not None and sim_path.is_file():
            if image_type_sim == _BASELINE_IMAGE_TYPE:
                st.caption("Overview of Mayer's baseline and derived image types.")
                st.image(str(sim_path), width=800)
            else:
                try:
                    img = Image.open(sim_path).convert("RGB")
                    left_img, right_img = _split_iqsim_comparison(img)

                    sub_left, sub_right = st.columns(2, gap="small")
                    with sub_left:
                        _centered_black_label(_BASELINE_IMAGE_TYPE)
                        st.image(left_img, use_container_width=True)
                    with sub_right:
                        _centered_black_label(image_type_sim)
                        st.image(right_img, use_container_width=True)
                except Exception:
                    st.image(str(sim_path), use_container_width=True)
        else:
            st.warning(
                f"Missing asset `{sim_path.name if sim_path else '?'}` in "
                f"`{_IQSIM_INVASIVE_DIR.name}/`."
            )

elif active == "Recommendation Tool":
    st.subheader("Model Recommendation Tool")
    col_rec_left, col_rec_right = st.columns([1.15, 1], gap="large")

    with col_rec_left:
        _persistent_selectbox(
            "Dataset Size",
            DATASET_SIZE_CHOICES,
            "rec_dataset_size",
        )
        _persistent_selectbox(
            "Image Type",
            IMAGE_TYPES,
            "rec_image_type",
        )

        if st.button("Generate", type="primary", use_container_width=True):
            rec = get_results_page_data(
                st.session_state.rec_dataset_size,
                st.session_state.rec_image_type,
            )
            st.session_state.result_payload = {
                "dataset_size": st.session_state.rec_dataset_size,
                "image_type": st.session_state.rec_image_type,
                "rec": rec,
            }
            st.session_state.nav_pending = "Results"
            st.rerun()

    with col_rec_right:
        rec_preview = lookup_recommendation(
            st.session_state.rec_dataset_size,
            st.session_state.rec_image_type,
        )
        with st.container(border=True):
            st.markdown("#### Recommendation Preview")
            st.markdown(rec_preview["summary"])

elif active == "Results":
    payload = st.session_state.result_payload
    if payload is None:
        st.subheader("Results")
        st.info(
            "No results yet. Open **Recommendation Tool**, choose **Dataset Size** "
            "and **Image Type**, then click **Generate**."
        )
    else:
        rec = get_results_page_data(payload["dataset_size"], payload["image_type"])
        metrics = rec.get("metrics") or {}
        n = rec.get("representative_n") or DATASET_SIZE_TO_N[payload["dataset_size"]]

        per_class_df = rec.get("per_class_df")

        _render_results_banner(
            payload["image_type"],
            rec.get("primary_model", "n/a"),
            n,
            metrics,
            per_class_df,
        )

        curve_path = rec.get("curve_path")
        confusion_path = rec.get("confusion_path")

        st.markdown("##### Loss & Accuracy Curves")
        col_curve_left, col_curve_right = st.columns(2, gap="medium")
        if curve_path is not None:
            try:
                curve_left, curve_right = _split_training_curves(curve_path)
                with col_curve_left:
                    _show_results_figure(curve_left, RESULTS_TRAINING_CURVE_SCALE)
                with col_curve_right:
                    _show_results_figure(curve_right, RESULTS_TRAINING_CURVE_SCALE)
            except Exception:
                with col_curve_left:
                    _show_results_figure(str(curve_path), RESULTS_TRAINING_CURVE_SCALE)
        else:
            with col_curve_left:
                st.warning("Training curve image not found for this selection.")

        show_macro_avg = (
            per_class_df is not None
            and not per_class_df.empty
            and (per_class_df["Class"] == "Macro avg").any()
        )

        col_cm, col_details = st.columns(2, gap="medium")
        with col_cm:
            st.markdown("##### Confusion Matrix")
            if confusion_path is not None:
                _show_results_figure(
                    str(confusion_path),
                    RESULTS_CONFUSION_MATRIX_SCALE,
                    align="left",
                    shift=RESULTS_CONFUSION_MATRIX_SHIFT,
                )
            else:
                st.warning("Confusion matrix image not found for this selection.")

        with col_details:
            st.markdown("##### Per-Class Results")
            if per_class_df is not None and not per_class_df.empty:
                html = _style_per_class_table(per_class_df).hide(axis="index").to_html()
                html = html.replace('<table', '<table style="width:100%"')
                st.markdown(html, unsafe_allow_html=True)
                _render_per_class_table_notes(show_macro_avg=show_macro_avg)
            else:
                st.info("Per-class metrics not available in Recommendation_Results.txt.")

            st.markdown(
                '<h5 style="margin-top:1.2rem; margin-bottom:-0.5rem;">Summary</h5>',
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                summary = rec.get("summary_text") or rec.get("summary", "")
                st.markdown(summary.replace("\n\n", " ").replace("\n", " "))

elif active == "Biological Context":
    st.subheader("Selected Cell / Tissue Classes")
    st.markdown(
        "The classification task uses **four H&E patch classes** from a broader "
        "annotated breast histology collection. Each patch is centred on a single "
        "cell or tissue category. Morphology is often subtle; global colour alone "
        "is usually insufficient, so models must learn fine-grained texture and structure."
    )

    st.markdown("#### Tumor Classes")
    st.markdown(
        f"""
**Invasive tumor**  
Cells that have penetrated surrounding stroma. This class marks progression beyond
in situ disease. In this project, {PRIMARY_METRIC_NOTE}
        
**DCIS 1**  
Pre-invasive tumor cells that remain within the ductal–lobular system. On H&E,
patches often show crowded epithelial cells with altered architecture but
without the stromal invasion pattern seen in invasive disease.
        """
    )

    st.markdown("#### Immune Classes")
    st.markdown(
        """
**CD8+ T cells**  
Cytotoxic T lymphocytes associated with anti-tumor activity and favourable prognosis
in many breast cancer settings. Patches tend to show smaller, more dispersed
lymphoid cells compared with dense tumor regions.

**Macrophages 1**  
Tumor-associated macrophages are linked to pro-tumorigenic processes (e.g. angiogenesis,
matrix remodelling, metastasis) and poorer outcomes in many studies. They can be
visually confused with other immune or stromal patterns, consistent with
**lower recall** for this class in several experiment summaries.
        """
    )

    st.markdown("#### Why These Classes Matter")
    st.markdown(
        """
Together, the four classes span **tumor progression** (DCIS 1 to Invasive) and
**immune context** (CD8+ vs Macrophages 1) within the breast tumor microenvironment.
They are biologically diverse yet still feasible for a single multi-class model,
which supports the robustness questions this app addresses: how performance shifts
under staining variants, resolution loss, noise, understained, and grayscale conversion
when dataset size changes.
        """
    )

    st.markdown("#### Class Distribution")
    st.markdown(
        """
Approximate class counts in the **selected four-class** set (before subsampling for training bands):
        """
    )

    class_dist_df = pd.DataFrame({
        "Class": ["Invasive tumor", "DCIS 1", "CD8+ T cells", "Macrophages 1"],
        "Approx. images": ["~34,000", "~10,000–13,000", "~7,000", "~10,000–13,000"],
    })

    html = class_dist_df.style.set_table_styles([
        {
            "selector": "thead th",
            "props": [
                ("background-color", "#1f4e79"),
                ("color", "white"),
                ("font-weight", "bold"),
                ("text-align", "left"),
            ],
        },
        {
            "selector": "td",
            "props": [("text-align", "left")],
        },
    ]).hide(axis="index").to_html()
    html = html.replace('<table', '<table style="width:350px"')
    st.markdown(html, unsafe_allow_html=True)

    st.markdown(
            """
    Invasive tumor is the majority class; imbalance can bias models toward predicting
    invasive unless training uses balancing or caps. EDA also noted overlapping
    brightness, contrast, and sharpness within and between classes, motivating
    the image-quality simulations elsewhere in the app.
            """
        )