import numpy as np
from PIL import Image, ImageDraw

from data.fonts import font_for_glyph_height


def render_digits(
    combo: np.ndarray,
    width: int,
    height: int,
    rng: np.random.Generator,
    font_paths: list,
) -> np.ndarray:
    """Render four digits onto a grayscale canvas."""
    bg_value = int(rng.integers(230, 256))
    canvas = Image.new("L", (width, height), color=bg_value)
    slot_width = width // 4
    max_digit_width = int(slot_width * 0.92)
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))

    for pos, digit in enumerate(combo):
        font_path = font_paths[int(rng.integers(0, len(font_paths)))]
        char = str(int(digit))
        glyph_height = int(height * float(rng.uniform(0.45, 0.55)))
        while True:
            font = font_for_glyph_height(font_path, glyph_height)
            bbox = measure.textbbox((0, 0), char, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            if tw <= max_digit_width or glyph_height <= 14:
                break
            glyph_height -= 2

        digit_layer = Image.new("L", (tw, th), color=bg_value)
        digit_draw = ImageDraw.Draw(digit_layer)
        digit_draw.text((-bbox[0], -bbox[1]), char, fill=0, font=font)

        angle = float(rng.uniform(-15.0, 15.0))
        rotated = digit_layer.rotate(angle, expand=True, fillcolor=bg_value)
        ox = int(rng.integers(-6, 7))
        oy = int(rng.integers(-6, 7))
        overlap = int(rng.integers(-4, 5))
        paste_x = pos * slot_width + (slot_width - rotated.width) // 2 + ox + overlap
        paste_y = (height - rotated.height) // 2 + oy
        canvas.paste(rotated, (paste_x, paste_y))
    return np.array(canvas, dtype=np.uint8)
