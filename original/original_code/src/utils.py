"""Shared helpers: configuration, paths, logging, seeding, version recording."""
from __future__ import annotations

import json
import logging
import platform
import random
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_config() -> dict:
    cfg = {}
    for name in ("data", "model", "experiment"):
        with open(ROOT / "configs" / f"{name}_config.yaml") as fh:
            cfg[name] = yaml.safe_load(fh)
    return cfg


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:  # torch is only needed for the neural models
        pass


def out(kind: str, name: str) -> Path:
    """Path inside results/<kind>/, creating the directory."""
    p = RESULTS / kind
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def save_table(df, name: str, index: bool = False) -> None:
    """Write a table as CSV and Markdown (results/tables)."""
    df.to_csv(out("tables", name + ".csv"), index=index)
    with open(out("tables", name + ".md"), "w") as fh:
        fh.write(df.to_markdown(index=index, floatfmt=".4g") + "\n")


def record_versions() -> dict:
    import importlib
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "torch", "statsmodels", "matplotlib", "xlrd"):
        try:
            v[m] = importlib.import_module(m).__version__
        except Exception:  # noqa: BLE001
            v[m] = "not installed"
    with open(out("diagnostics", "software_versions.json"), "w") as fh:
        json.dump(v, fh, indent=2)
    return v
