#!/usr/bin/env python
"""Generate the demo app's icon set: favicon, PWA icons (incl. maskable), apple-touch-icon.

Motif: a descending "quantization staircase" (FP32 -> INT8 -> INT6 -> INT4), four bars of decreasing
height in the app's accent teal on the app's dark background, matching demo/app/static/styles.css tokens.
Deterministic (no external assets); re-run if the palette in styles.css changes.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "app" / "static" / "icons"
BG = (11, 16, 32, 255)       # --bg
BAR = (94, 234, 212, 255)    # --accent
BAR_DIM = (125, 211, 252, 255)  # --accent-2, for the shortest/last bar


def draw_mark(size: int, pad_frac: float = 0.0) -> Image.Image:
    """Draw the staircase mark on a size x size canvas. pad_frac shrinks the mark for maskable safe zones."""
    img = Image.new("RGBA", (size, size), BG)
    d = ImageDraw.Draw(img)
    radius = int(size * 0.18)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=BG)

    n = 4
    pad = size * (0.20 + pad_frac)
    gap = size * 0.045
    usable_w = size - 2 * pad
    bar_w = (usable_w - (n - 1) * gap) / n
    base_y = size - pad
    max_h = size - 2 * pad
    heights = [max_h * f for f in (1.0, 0.72, 0.5, 0.3)]
    colors = [BAR, BAR, BAR, BAR_DIM]
    x = pad
    for h, c in zip(heights, colors):
        top = base_y - h
        bar_radius = max(2, int(bar_w * 0.22))
        d.rounded_rectangle([x, top, x + bar_w, base_y], radius=bar_radius, fill=c)
        x += bar_w + gap
    return img


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # Standard (non-maskable) icons: mark fills most of the canvas.
    for size, name in [(512, "icon-512.png"), (192, "icon-192.png"), (180, "apple-touch-icon.png"), (32, "favicon-32.png")]:
        draw_mark(size).save(OUT / name)

    # Maskable icons: Android applies its own mask, so keep the mark inside the ~80% safe zone.
    for size, name in [(512, "icon-maskable-512.png"), (192, "icon-maskable-192.png")]:
        draw_mark(size, pad_frac=0.12).save(OUT / name)

    # SVG favicon (crisp at any size; used by <link rel="icon">).
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="12" fill="rgb{BG[:3]}"/>
  <rect x="10" y="14" width="9" height="40" rx="2" fill="rgb{BAR[:3]}"/>
  <rect x="23" y="23" width="9" height="31" rx="2" fill="rgb{BAR[:3]}"/>
  <rect x="36" y="30" width="9" height="24" rx="2" fill="rgb{BAR[:3]}"/>
  <rect x="49" y="36" width="9" height="18" rx="2" fill="rgb{BAR_DIM[:3]}"/>
</svg>
"""
    (OUT / "favicon.svg").write_text(svg, encoding="utf-8")
    print(f"wrote icons to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
