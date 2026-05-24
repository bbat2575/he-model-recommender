"""Recommendation lookup and model table for the Streamlit app."""

from __future__ import annotations

import itertools
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

DATASET_SIZES = [
    "<1500",
    "1500–5000",
    "5000–10000",
    "10000–15000",
    "15000–22000",
    "22000+",
]

# Representative training-set sizes from the experimental grid (ShinyAppData).
DATASET_SIZE_TO_N: dict[str, int] = {
    "<1500": 500,
    "1500–5000": 2500,
    "5000–10000": 7500,
    "10000–15000": 12500,
    "15000–22000": 18000,
    "22000+": 25004,
}

IMAGE_TYPES = [
    "Full Colour (Mayer's)",
    "Full Colour (Harris)",
    "Low Resolution",
    "Noise",
    "Understained",
    "Grayscale",
]

# Same ordered list for Image Quality Simulator and Recommendation Tool.
IMAGE_TYPES_SIMULATOR = IMAGE_TYPES

IMAGE_TYPE_TO_PREFIX: dict[str, str] = {
    "Full Colour (Mayer's)": "Mayers",
    "Full Colour (Harris)": "Harris",
    "Low Resolution": "LowRes",
    "Noise": "Noise",
    "Understained": "Understained",
    "Grayscale": "Grayscale",
}

# Legacy / document labels to canonical IMAGE_TYPES label.
IMAGE_TYPE_DOC_ALIASES: dict[str, str] = {
    "Full Colour (Understained)": "Understained",
}


def canonical_image_type(label: str) -> str:
    label = IMAGE_TYPE_DOC_ALIASES.get(label, label)
    return label if label in IMAGE_TYPES else IMAGE_TYPES[0]

PREFIX_TO_IMAGE_TYPE = {v: k for k, v in IMAGE_TYPE_TO_PREFIX.items()}

# Section headers inside Recommendation_Results.txt
IMAGE_TYPE_TO_TXT_HEADER: dict[str, str] = {
    "Full Colour (Mayer's)": "Full Colour (Mayer's)",
    "Full Colour (Harris)": "Full Colour (Harris)",
    "Low Resolution": "Low Resolution",
    "Noise": "Noise",
    "Understained": "Full Colour (Understained)",
    "Grayscale": "Grayscale",
}

PER_CLASS_ROWS = ("Invasive", "DCIS1", "CD8", "Macrophage", "Macro avg")

PRIMARY_METRIC_NOTE = (
    "Invasive tumor is the primary clinical metric, followed by DCIS 1 - both are "
    "tumor classes where missed detections have clinical consequences."
)

# Used when curve/docx assets do not resolve a model for (image type, n).
DEFAULT_FALLBACK_MODEL = "EfficientNet-B0"

MODEL_IDS = [
    "SVM + HOG",
    "ResNet18",
    "EfficientNet-B0",
    "DenseNet121",
]

_FILE_MODEL_TO_APP: dict[str, str] = {
    "SVM-HOG": "SVM + HOG",
    "ResNet18": "ResNet18",
    "EfficientNet-B0": "EfficientNet-B0",
    "DenseNet121": "DenseNet121",
}

_APP_DIR = Path(__file__).resolve().parent
_FILENAME_RE = re.compile(r"^(.+)-(\d+)-(.+)$")


def _shiny_app_data_dir() -> Path | None:
    """Resolve ShinyAppData folder (supports dated zip extract names)."""
    for entry in sorted(_APP_DIR.glob("ShinyAppData*")):
        candidate = entry / "ShinyAppData" if (entry / "ShinyAppData").is_dir() else entry
        if (candidate / "model_curves").is_dir():
            return candidate
    return None


def _normalize_model(file_model: str) -> str:
    return _FILE_MODEL_TO_APP.get(file_model, file_model.replace("-", " "))


def _load_best_models_from_curves(data_dir: Path) -> dict[tuple[str, int], str]:
    """Map (image prefix, n) -> best model filename token from model_curves assets."""
    lookup: dict[tuple[str, int], str] = {}
    curves_dir = data_dir / "model_curves"
    for path in curves_dir.glob("*.png"):
        match = _FILENAME_RE.match(path.stem)
        if not match:
            continue
        prefix, n_str, file_model = match.groups()
        lookup[(prefix, int(n_str))] = _normalize_model(file_model)
    return lookup


def _load_metrics_from_docx(data_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Parse Model_Recommendations.docx for accuracy / MCC / training details."""
    docx_path = data_dir / "Model_Recommendations.docx"
    if not docx_path.is_file():
        return {}

    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(docx_path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))

    lines: list[str] = []
    for para in root.iter(f"{W}p"):
        text = "".join((node.text or "") for node in para.iter(f"{W}t")).strip()
        if text:
            lines.append(text)

    header_re = re.compile(
        r"^(Full Colour \(Mayer.s\)|Full Colour \(Harris\)|Full Colour \(Understained\)|"
        r"Understained|Low Resolution|Grayscale|Noise)\s*\|\s*n\s*=\s*([\d,]+)\s*$",
        re.IGNORECASE,
    )
    model_re = re.compile(
        r"^(DenseNet121|EfficientNet-B0|ResNet18|SVM\s*\(HOG\))\s*[—\-]\s*"
        r"Accuracy:\s*([\d.]+)\s*\|\s*MCC:\s*([\d.]+)\s*$",
        re.IGNORECASE,
    )
    detail_re = re.compile(
        r"^LR:\s*([\d.]+)\s*\|\s*Best val acc:\s*([\d.]+)\s*\|\s*"
        r"Total epochs:\s*(\d+)\s*\|\s*Best checkpoint:\s*(.+)$",
        re.IGNORECASE,
    )

    metrics: dict[tuple[str, int], dict[str, Any]] = {}
    current_key: tuple[str, int] | None = None

    for line in lines:
        hm = header_re.match(line.replace("Mayer's", "Mayer.s").replace("Mayers", "Mayer.s"))
        if hm:
            image_label = hm.group(1).replace("Mayer.s", "Mayer's")
            if image_label.startswith("Full Colour (Mayer"):
                image_label = "Full Colour (Mayer's)"
            image_label = canonical_image_type(image_label)
            n_val = int(hm.group(2).replace(",", ""))
            prefix = IMAGE_TYPE_TO_PREFIX.get(image_label)
            if prefix:
                current_key = (prefix, n_val)
            continue

        if current_key is None:
            continue

        mm = model_re.match(line)
        if mm:
            raw_model = mm.group(1).replace("(HOG)", "+ HOG").replace("SVM + HOG", "SVM + HOG")
            if "SVM" in raw_model and "HOG" in raw_model:
                model_name = "SVM + HOG"
            else:
                model_name = raw_model.replace(" ", "")
                if model_name == "SVM+HOG":
                    model_name = "SVM + HOG"
                else:
                    model_name = _normalize_model(model_name)

            metrics[current_key] = {
                "model": model_name,
                "accuracy": float(mm.group(2)),
                "mcc": float(mm.group(3)),
            }
            continue

        dm = detail_re.match(line)
        if dm and current_key in metrics:
            metrics[current_key].update(
                {
                    "lr": float(dm.group(1)),
                    "best_val_acc": float(dm.group(2)),
                    "total_epochs": int(dm.group(3)),
                    "best_checkpoint": dm.group(4).strip(),
                }
            )

    return metrics

def _load_summaries_from_docx(data_dir: Path) -> dict[tuple[str, int], str]:
    """Parse Recommendation_Summaries.docx for per-(image_type, n) summary text."""
    docx_path = data_dir / "Recommendation_Summaries.docx"
    if not docx_path.is_file():
        return {}

    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(docx_path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))

    lines: list[str] = []
    for para in root.iter(f"{W}p"):
        text = "".join((node.text or "") for node in para.iter(f"{W}t")).strip()
        if text:
            lines.append(text)

    header_re = re.compile(
        r"^(.+?)\s*\|\s*n\s*=\s*([\d,]+)\s*→\s*.+$"
    )

    summaries: dict[tuple[str, int], str] = {}
    current_key: tuple[str, int] | None = None
    current_lines: list[str] = []

    for line in lines:
        m = header_re.match(line)
        if m:
            if current_key is not None and current_lines:
                summaries[current_key] = " ".join(current_lines)
            image_label = canonical_image_type(m.group(1).strip())
            n_val = int(m.group(2).replace(",", ""))
            current_key = (image_label, n_val)
            current_lines = []
        elif current_key is not None and not line.startswith("#"):
            current_lines.append(line)

    if current_key is not None and current_lines:
        summaries[current_key] = " ".join(current_lines)

    return summaries

def format_lr(value: float) -> str:
    """Learning rate in scientific notation (avoids rounding to 0.000)."""
    return f"{float(value):.2e}"


def format_training_n(n: int) -> str:
    """Training set size label, consistent with Results page header."""
    return f"n = {n:,}"


def _resolve_recommendation_row(
    dataset_size: str,
    image_type: str,
    best_by_prefix_n: dict[tuple[str, int], str],
    metrics_by_prefix_n: dict[tuple[str, int], dict[str, Any]],
    data_dir: Path | None,
) -> dict[str, Any]:
    prefix = IMAGE_TYPE_TO_PREFIX[image_type]
    n = DATASET_SIZE_TO_N[dataset_size]
    key = (prefix, n)

    primary = best_by_prefix_n.get(key)
    meta = metrics_by_prefix_n.get(key, {})
    if primary is None and meta:
        primary = meta.get("model")
    if primary is None:
        primary = DEFAULT_FALLBACK_MODEL

    curve_path = None
    confusion_path = None
    if data_dir is not None:
        file_model = {v: k for k, v in _FILE_MODEL_TO_APP.items()}.get(primary, primary)
        curve_candidate = data_dir / "model_curves" / f"{prefix}-{n}-{file_model}.png"
        if curve_candidate.is_file():
            curve_path = curve_candidate
        confusion_candidate = data_dir / "confusion_matrix_heatmaps" / f"{prefix}-{n}-{file_model}.png"
        if confusion_candidate.is_file():
            confusion_path = confusion_candidate

    acc = meta.get("accuracy")
    mcc = meta.get("mcc")
    metrics_line = ""
    if acc is not None and mcc is not None:
        metrics_line = f"  \n**Test accuracy:** {acc:.3f} | **MCC:** {mcc:.3f}"
        if meta.get("best_val_acc") is not None:
            metrics_line += f" | **Best Val Acc:** {meta['best_val_acc']:.3f}"

    detail_line = ""
    if meta.get("lr") is not None:
        lr_display = format_lr(meta["lr"])
        epochs = meta.get("total_epochs", "n/a")
        checkpoint = meta.get("best_checkpoint", "n/a")
        detail_line = (
            f"  \n**LR:** {lr_display} · **Epochs:** {epochs} · **Best checkpoint:** {checkpoint}"
        )

    return {
        "primary_model": primary,
        "representative_n": n,
        "metrics": meta,
        "curve_path": curve_path,
        "confusion_path": confusion_path,
        "image_type": image_type,
        "dataset_size": dataset_size,
        "summary": (
            f"**Model:** {primary}  \n\n"
            f"_Recommended from project experiments for **{image_type}** with "
            f"dataset size **{dataset_size}** (n = {n:,})._"
            f"{metrics_line}{detail_line}"
        ),
    }


def build_recommendation_cache() -> dict[tuple[str, str], dict[str, Any]]:
    data_dir = _shiny_app_data_dir()
    best_by_prefix_n: dict[tuple[str, int], str] = {}
    metrics_by_prefix_n: dict[tuple[str, int], dict[str, Any]] = {}
    summaries_by_key: dict[tuple[str, int], str] = {}

    if data_dir is not None:
        best_by_prefix_n = _load_best_models_from_curves(data_dir)
        metrics_by_prefix_n = _load_metrics_from_docx(data_dir)
        summaries_by_key = _load_summaries_from_docx(data_dir)

    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for ds, img_t in itertools.product(DATASET_SIZES, IMAGE_TYPES):
        row = _resolve_recommendation_row(
            ds, img_t, best_by_prefix_n, metrics_by_prefix_n, data_dir
        )
        n = DATASET_SIZE_TO_N[ds]
        doc_summary = summaries_by_key.get((img_t, n))
        if doc_summary:
            row["summary_text"] = doc_summary
        cache[(ds, img_t)] = row
    return cache


# Built at import time; restart Streamlit after changing assets under ShinyAppData*.
RECOMMENDATION_CACHE = build_recommendation_cache()

import os
print("APP_DIR:", _APP_DIR)
print("Contents:", list(_APP_DIR.iterdir()))
print("ShinyAppData dir:", _shiny_app_data_dir())

def lookup_recommendation(dataset_size: str, image_type: str) -> dict[str, Any]:
    return RECOMMENDATION_CACHE[(dataset_size, image_type)]


def _txt_header_for(image_type: str, n: int) -> list[str]:
    label = IMAGE_TYPE_TO_TXT_HEADER.get(image_type, image_type)
    return [f"{label} | n = {n}", f"{label} | n = {n:,}"]


def _parse_results_txt_section(block: str) -> dict[str, Any]:
    """Parse one experiment block from Recommendation_Results.txt."""
    per_class_acc: dict[str, float] = {}
    for cls in PER_CLASS_ROWS[:4]:
        m = re.search(rf"^\s*{cls}:\s*([\d.]+)\s*$", block, re.MULTILINE)
        if m:
            per_class_acc[cls] = float(m.group(1))

    table_rows: list[dict[str, Any]] = []
    row_re = re.compile(
        r"^\s*(Invasive|DCIS1|CD8|Macrophage)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
        re.MULTILINE,
    )
    macro_re = re.compile(
        r"^\s*macro\s+avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
        re.MULTILINE,
    )
    for m in row_re.finditer(block):
        cls = m.group(1)
        table_rows.append(
            {
                "Class": cls,
                "Precision": float(m.group(2)),
                "Recall": float(m.group(3)),
                "F1-Score": float(m.group(4)),
                "Support": int(float(m.group(5))),
                "Per-Class Acc": (
                    f"{per_class_acc[cls] * 100:.1f}%"
                    if cls in per_class_acc
                    else "n/a"
                ),
            }
        )
    mm = macro_re.search(block)
    if mm:
        table_rows.append(
            {
                "Class": "Macro avg",
                "Precision": float(mm.group(1)),
                "Recall": float(mm.group(2)),
                "F1-Score": float(mm.group(3)),
                "Support": int(float(mm.group(4))),
                "Per-Class Acc": "n/a",
            }
        )

    per_class_df = pd.DataFrame(table_rows) if table_rows else pd.DataFrame()
    return {"per_class_df": per_class_df, "per_class_acc": per_class_acc}


def load_result_detail_from_txt(
    data_dir: Path | None,
    image_type: str,
    dataset_size: str,
) -> dict[str, Any]:
    if data_dir is None:
        return {"per_class_df": pd.DataFrame(), "per_class_acc": {}, "summary_text": ""}

    txt_path = data_dir / "Recommendation_Results.txt"
    if not txt_path.is_file():
        return {"per_class_df": pd.DataFrame(), "per_class_acc": {}, "summary_text": ""}

    n = DATASET_SIZE_TO_N[dataset_size]
    headers = set(_txt_header_for(image_type, n))
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("=" * 20):
            i += 1
            continue
        if i + 1 >= len(lines) or "| n =" not in lines[i + 1]:
            i += 1
            continue
        header = lines[i + 1].strip()
        i += 2
        while i < len(lines) and lines[i].strip().startswith("=" * 20):
            i += 1
        body_start = i
        while i < len(lines):
            if lines[i].strip().startswith("=" * 20):
                if i + 1 < len(lines) and "| n =" in lines[i + 1]:
                    break
            i += 1
        body = "\n".join(lines[body_start:i])
        if header in headers:
            parsed = _parse_results_txt_section(body)
            return parsed

    return {"per_class_df": pd.DataFrame(), "per_class_acc": {}, "summary_text": ""}


def get_results_page_data(dataset_size: str, image_type: str) -> dict[str, Any]:
    rec = lookup_recommendation(dataset_size, image_type)
    data_dir = _shiny_app_data_dir()
    detail = load_result_detail_from_txt(data_dir, image_type, dataset_size)
    merged = {**rec, **detail}
    # Prefer the hand-written summary from Recommendation_Summaries.docx
    if rec.get("summary_text"):
        merged["summary_text"] = rec["summary_text"]
    return merged


# Summary table for Overview (static study roles; not tied to RECOMMENDATION_CACHE).
MODEL_RESULTS_DF = pd.DataFrame(
    {
        "Model": MODEL_IDS,
        "Role in study": [
            "Baseline fallback at n=500 when all NNs collapse. Non-viable above n=500.",
            "Wins at Grayscale n=18,000 and Low Resolution n=2,500. Stable plateau but hard accuracy ceiling.",
            "Best on large clean-colour datasets. Collapses below n=2,500 under degradation.",
            "Best on degraded/stain-variant types. Most robust across small-to-large dataset sizes.",
        ],
        "Typical use": [
            "n=500 on Harris, Low Resolution, and Grayscale only",
            "Fallback when EfficientNet and DenseNet both fail on stability",
            "Full Colour (Mayer's) and Low Resolution at n >= 7,500",
            "Harris, Noise, Understained, and Grayscale at most dataset sizes",
        ],
    }
)
