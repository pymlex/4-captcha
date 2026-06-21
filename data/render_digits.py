import numpy as np
from PIL import Image, ImageDraw


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
    max_digit_width = int(slot_width * 0.9)
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))

    for pos, digit in enumerate(combo):
        font = fonts[int(rng.integers(0, len(fonts)))]
        char = str(int(digit))
        bbox = measure.textbbox((0, 0), char, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        digit_layer = Image.new("L", (tw, th), color=bg_value)
        digit_draw = ImageDraw.Draw(digit_layer)
        digit_draw.text((-bbox[0], -bbox[1]), char, fill=0, font=font)

        target_h = int(height * float(rng.uniform(0.45, 0.55)))
        stretch_x = float(rng.uniform(0.9, 1.1))
        stretch_y = float(rng.uniform(0.9, 1.1))
        scaled_w = max(8, int(digit_layer.width * stretch_x))
        scaled_h = max(8, int(digit_layer.height * stretch_y))
        digit_layer = digit_layer.resize(
            (scaled_w, scaled_h),
            Image.Resampling.BILINEAR,
        )

        fit_scale = target_h / digit_layer.height
        fit_w = max(8, int(digit_layer.width * fit_scale))
        fit_h = target_h
        if fit_w > max_digit_width:
            fit_scale = max_digit_width / digit_layer.width
            fit_w = max_digit_width
            fit_h = max(8, int(digit_layer.height * fit_scale))
        digit_layer = digit_layer.resize(
            (fit_w, fit_h),
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
