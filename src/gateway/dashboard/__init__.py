"""Static dashboard shipped inside the package."""

from pathlib import Path


def load_html() -> str:
    return Path(__file__).with_name("index.html").read_text(encoding="utf-8")
