import _path
from pathlib import Path

from config import get_settings
from eval.plots import generate_all_plots


if __name__ == "__main__":
    settings = get_settings()
    generate_all_plots(settings.output_dir)
