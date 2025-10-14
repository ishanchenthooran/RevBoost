from __future__ import annotations
import os, logging, tomllib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def cfg() -> dict:
    with open("config.toml", "rb") as f:
        return tomllib.load(f)

def ensure_dirs():
    Path("data/processed").mkdir(parents=True, exist_ok=True)
