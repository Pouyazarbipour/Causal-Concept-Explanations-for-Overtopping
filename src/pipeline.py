"""Shared loading of the modelling table, folds, predictors and concept labels."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .concept_ontology import concept_labels
from .features import features_from_raw
from .geometry_grouping import GROUP_COLUMN
from .utils import ROOT, load_config


@lru_cache(maxsize=1)
def load_model_data():
    """Returns dict with the regression table (positive q, usable inputs), folds, views, labels."""
    cfg = load_config()
    d = cfg["data"]
    master = pd.read_csv(ROOT / d["master_file"], dtype={"series": str, "name": str}, keep_default_na=True, low_memory=False)
    for c in ("remark", "reference", "unusable_reason"):
        master[c] = master[c].fillna("")
    folds_all = pd.read_csv(ROOT / d["folds_file"], dtype={"name": str})
    use = master["use_regression"].to_numpy()
    df = master[use].reset_index(drop=True)
    folds = folds_all[use].reset_index(drop=True)
    assert (df["name"] == folds["name"]).all()
    X = features_from_raw(df, d["physical_features"], d["log1p_features"], d["gravity"])
    return {"cfg": cfg, "master": master, "df": df, "folds": folds, "X": X,
            "y": df["y"].to_numpy(float), "labels": concept_labels(df), "features": d["physical_features"]}


def groups_for(df: pd.DataFrame, scheme: str) -> np.ndarray:
    if scheme == "random":
        return df["name"].to_numpy()
    return df[GROUP_COLUMN[scheme]].to_numpy()


def load_pred(tag: str) -> np.ndarray:
    """Out-of-fold predictions [seeds, n] of one model."""
    return np.load(ROOT / "results" / "predictions" / f"{tag}.npz")["pred"]


def have(tag: str) -> bool:
    return (ROOT / "results" / "predictions" / f"{tag}.npz").exists()
