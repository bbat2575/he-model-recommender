"""
One-off generator: left = baseline H&E, right = degraded view.
Reads ../100/CD4+_T_Cells/cell_361_100.png, writes PNGs next to app.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "100" / "CD4+_T_Cells" / "cell_361_100.png"
OUT_DIR = Path(__file__).resolve().parent / "iqsim_assets"
HALF = 400  # upscale each half for clearer display


def _square_base(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((HALF, HALF), Image.Resampling.LANCZOS)


def _concat_lr(left: Image.Image, right: Image.Image) -> Image.Image:
    assert left.size == right.size
    w, h = left.size
    out = Image.new("RGB", (w * 2, h))
    out.paste(left, (0, 0))
    out.paste(right, (w, 0))
    return out


def _blur(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=2.2))


def _low_brightness(img: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(0.42)


def _low_contrast(img: Image.Image) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(0.38)


def _low_resolution(img: Image.Image) -> Image.Image:
    w, h = img.size
    tiny = max(12, w // 6)
    down = img.resize((tiny, tiny), Image.Resampling.BILINEAR)
    return down.resize((w, h), Image.Resampling.BILINEAR)


def _staining_variation(img: Image.Image) -> Image.Image:
    """Mild H&E-style colour shift (eosin / haematoxylin emphasis)."""
    arr = np.asarray(img, dtype=np.float32)
    # stronger pink in cytoplasm-like tones, slightly cooler nuclei
    arr[..., 0] *= 1.18
    arr[..., 1] *= 0.88
    arr[..., 2] *= 1.05
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"Missing source image: {SOURCE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _square_base(Image.open(SOURCE))
    specs: list[tuple[str, Image.Image]] = [
        ("iqsim_blur.png", _blur(base)),
        ("iqsim_low_brightness.png", _low_brightness(base)),
        ("iqsim_low_contrast.png", _low_contrast(base)),
        ("iqsim_low_resolution.png", _low_resolution(base)),
        ("iqsim_staining_variation.png", _staining_variation(base)),
    ]

    for name, right in specs:
        left = base
        out = _concat_lr(left, right)
        path = OUT_DIR / name
        out.save(path, format="PNG", optimize=True)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
