import os
from pathlib import Path

from PIL import ImageFont


FONT_CANDIDATES = [
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


def load_fonts(size: int = 56) -> list[ImageFont.FreeTypeFont]:
    fonts = []
    for name in FONT_CANDIDATES:
        for directory in font_directories():
            path = directory / name
            if path.exists():
                fonts.append(ImageFont.truetype(str(path), size=size))
                break
    if len(fonts) < 4:
        fonts = [ImageFont.load_default() for _ in range(10)]
    return fonts
