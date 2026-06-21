import os
import shutil
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_FONT_DIR = PROJECT_ROOT / "assets" / "fonts"

SYSTEM_DEJAVU_SOURCES = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
]


def font_directories() -> list[Path]:
    dirs = [BUNDLED_FONT_DIR]
    windir = os.environ.get("WINDIR")
    if windir:
        dirs.append(Path(windir) / "Fonts")
    dirs.extend(
        [
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            Path.home() / ".fonts",
            Path.home() / ".local" / "share" / "fonts",
            Path("/System/Library/Fonts"),
            Path("/System/Library/Fonts/Supplemental"),
        ]
    )
    return [path for path in dirs if path.exists()]


def ensure_project_fonts() -> None:
    BUNDLED_FONT_DIR.mkdir(parents=True, exist_ok=True)
    bundled_regular = BUNDLED_FONT_DIR / "DejaVuSans.ttf"
    if bundled_regular.exists():
        return
    for source in SYSTEM_DEJAVU_SOURCES:
        if not source.exists():
            continue
        shutil.copy(source, bundled_regular)
        bold = source.parent / "DejaVuSans-Bold.ttf"
        if bold.exists():
            shutil.copy(bold, BUNDLED_FONT_DIR / "DejaVuSans-Bold.ttf")
        return


def collect_named_font_paths() -> list[Path]:
    paths = []
    seen = set()
    for name in FONT_CANDIDATES:
        for directory in font_directories():
            path = directory / name
            if not path.is_file():
                matches = list(directory.rglob(name))
                if matches:
                    path = matches[0]
            resolved = str(path.resolve())
            if path.is_file() and resolved not in seen:
                paths.append(path)
                seen.add(resolved)
                break
    return paths


def collect_scanned_font_paths(limit: int = 32) -> list[Path]:
    paths = []
    seen = set()
    scan_roots = [
        BUNDLED_FONT_DIR,
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
    ]
    for root in scan_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.ttf")):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            paths.append(path)
            seen.add(resolved)
            if len(paths) >= limit:
                return paths
    return paths


def collect_font_paths() -> list[Path]:
    ensure_project_fonts()
    paths = collect_named_font_paths()
    if paths:
        return paths
    return collect_scanned_font_paths()


def font_for_glyph_height(path: Path, glyph_height: int) -> ImageFont.FreeTypeFont:
    size = max(16, int(glyph_height * FONT_SIZE_TO_GLYPH_HEIGHT))
    return ImageFont.truetype(str(path), size=size)
