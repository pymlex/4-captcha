import numpy as np
from PIL import Image, ImageDraw

from data.fonts import load_fonts


def render_digits(
    combo: np.ndarray,
    width: int,
    height: int,
    rng: np.random.Generator,
    fonts: list,
) -> np.ndarray:
    """Render four digits onto a grayscale canvas."""
    bg_value = int(rng.integers(230, 256))
    canvas = Image.new("L", (width, height), color=bg_value)
    slot_width = width // 4
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))

    for pos, digit in enumerate(combo):
        font = fonts[int(rng.integers(0, len(fonts)))]
        char = str(int(digit))
        bbox = measure.textbbox((0, 0), char, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        stretch_x = float(rng.uniform(0.85, 1.25))
        stretch_y = float(rng.uniform(0.85, 1.25))
        new_w = max(8, int(tw * stretch_x))
        new_h = max(8, int(th * stretch_y))
        digit_layer = Image.new("L", (tw, th), color=bg_value)
        digit_draw = ImageDraw.Draw(digit_layer)
        digit_draw.text((-bbox[0], -bbox[1]), char, fill=0, font=font)
        if (new_w, new_h) != (tw, th):
            digit_layer = digit_layer.resize(
                (new_w, new_h),
                Image.Resampling.BILINEAR,
            )
        angle = float(rng.uniform(-15.0, 15.0))
        rotated = digit_layer.rotate(angle, expand=True, fillcolor=bg_value)
        ox = int(rng.integers(-6, 7))
        oy = int(rng.integers(-6, 7))
        overlap = int(rng.integers(-4, 5))
        paste_x = pos * slot_width + (slot_width - rotated.width) // 2 + ox + overlap
        paste_y = (height - rotated.height) // 2 + oy
        canvas.paste(rotated, (paste_x, paste_y))
    return np.array(canvas, dtype=np.uint8)
