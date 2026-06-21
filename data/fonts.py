import os
from pathlib import Path

from PIL import ImageFont


FONT_CANDIDATES = [
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "LiberationSans-Regular.ttf",
    "arial.ttf",
    "times.ttf",
    "cour.ttf",
    "comic.ttf",
    "impact.ttf",
    "verdana.ttf",
    "georgia.ttf",
    "calibri.ttf",
    "tahoma.ttf",
    "consola.ttf",
    "Arial.ttf",
    "Times.ttf",
    "Cour.ttf",
    "Comic.ttf",
    "Impact.ttf",
    "Verdana.ttf",
    "Georgia.ttf",
    "Calibri.ttf",
    "Tahoma.ttf",
    "Consola.ttf",
]

FONT_SIZE_TO_GLYPH_HEIGHT = 1.45


def font_directories() -> list[Path]:
    dirs = []
    windir = os.environ.get("WINDIR")
    if windir:
        dirs.append(Path(windir) / "Fonts")
    dirs.append(Path("/usr/share/fonts/truetype/dejavu"))
    dirs.append(Path("/usr/share/fonts/truetype/liberation"))
    dirs.append(Path("/usr/share/fonts/truetype/msttcorefonts"))
    dirs.append(Path("/System/Library/Fonts/Supplemental"))
    return dirs


def collect_font_paths() -> list[Path]:
    paths = []
    seen = set()
    for name in FONT_CANDIDATES:
        for directory in font_directories():
            path = directory / name
            resolved = str(path.resolve())
            if path.exists() and resolved not in seen:
                paths.append(path)
                seen.add(resolved)
                break
    return paths


def font_for_glyph_height(path: Path, glyph_height: int) -> ImageFont.FreeTypeFont:
    size = max(16, int(glyph_height * FONT_SIZE_TO_GLYPH_HEIGHT))
    return ImageFont.truetype(str(path), size=size)
