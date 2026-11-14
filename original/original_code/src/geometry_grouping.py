"""Geometry, family, source and laboratory grouping; fold assignment; leakage checks.

Available identifiers in the workbook
-------------------------------------
* `Name` = "SSS-TTT".  SSS is the CLASH test-series number: a genuine identifier.
* `Reference` = literature reference: a genuine but incomplete source identifier
  (blank for confidential series).
* There is NO structure identifier and NO laboratory identifier.  Geometry groups
  are therefore *reconstructed* and laboratories are *inferred* from keywords in
  the reference; both are approximations and are labelled as such.

Geometry reconstruction
-----------------------
Rc, Ac, h, ht and hb are all measured relative to the still-water level, so they
change when the same physical model is tested at another water level.  The key is
built from water-level-invariant combinations:
    crest level above toe bed      h + Rc
    armour crest level             h + Ac
    toe-berm height                h - ht
    berm level                     h - hb   (only if a berm exists)
plus Bt, B, Gc, cot(alpha_d), cot(alpha_u), gamma_f, tan(alpha_B), foreshore m,
all rounded to `sig` significant digits, within a test series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

LAB_KEYWORDS = [  # (keyword in lower-case reference, inferred laboratory)
    ("wallingford", "HR Wallingford"), ("waterloopkundig laboratorium borgerhout", "Flanders Hydraulics"),
    ("waterloopkundig", "Delft Hydraulics"), ("delft", "Delft Hydraulics"), ("aalborg", "Aalborg University"),
    ("leichtwei", "LWI Braunschweig"), ("braunschweig", "LWI Braunschweig"), ("edinburgh", "Univ. Edinburgh"),
    ("ghent", "Ghent University"), ("vicksburg", "USACE WES/CERC"), ("cerc", "USACE WES/CERC"),
    ("port and harbour research", "PHRI Japan"), ("upv", "UPV Valencia"), ("modimar", "Modimar"),
    ("kyoto", "NRC Canada (Cornett)"), ("icce 1988", "Univ. Florence/Rome (Aminti-Franco)"),
    ("franzius", "Franzius Hannover"), ("hannover", "Franzius Hannover"), ("danish hydraulic", "DHI"),
    ("dhi", "DHI"), ("enel", "ENEL"), ("barcelona", "UPC Barcelona"), ("lnec", "LNEC"), ("cedex", "CEDEX"),
]


def _sig(x, n):
    if pd.isna(x):
        return "nan"
    return f"{0.0 if abs(x) < 1e-12 else x:.{n}g}"


def structural_family(df: pd.DataFrame) -> pd.Series:
    """Coarse structural family from documented geometry descriptors (rule-based)."""
    steep_d, steep_u = df["cotad"] < 1, df["cotau"] < 1
    fam = pd.Series("smooth_slope", index=df.index)
    fam[df["gf_raw"] < 0.9] = "rough_slope"
    fam[df["gf_raw"].isna()] = "slope_unknown_roughness"
    fam[df["b_berm"] > 0] = "bermed_slope"
    fam[~steep_d & steep_u] = "slope_with_wall"       # sloping/mound lower part, (near-)vertical upper part
    fam[steep_d & ~steep_u] = "wall_with_slope_above"  # (near-)vertical lower part, sloping upper part
    fam[steep_d & steep_u] = "vertical_wall"
    return fam


def infer_lab(reference: pd.Series) -> pd.Series:
    ref = reference.str.lower()
    lab = pd.Series("unknown", index=reference.index)
    for kw, name in reversed(LAB_KEYWORDS):  # earlier entries take precedence
        lab[ref.str.contains(kw, regex=False)] = name
    lab[ref.str.strip() == ""] = "unknown"
    return lab


def add_groups(df: pd.DataFrame, sig: int = 3) -> pd.DataFrame:
    df = df.copy()
    inv = pd.DataFrame({
        "crest": df["h"] + df["rc"], "acrest": df["h"] + df["ac"], "toe": df["h"] - df["ht"],
        "berm": np.where(df["b_berm"] > 0, df["h"] - df["hb"], 0.0),
        "bt": df["bt"], "B": df["b_berm"], "gc": df["gc"], "cd": df["cotad"], "cu": df["cotau"],
        "gf": df["gf_raw"], "tb": df["tanab"], "m": df["m"]})
    fine = inv.apply(lambda c: c.map(lambda v: _sig(v, sig)))
    df["geometry"] = df["series"] + "|" + fine.agg("|".join, axis=1)
    # sub-structure: everything except crest levels (same model, different crest height / water level)
    df["substructure"] = df["series"] + "|" + fine.drop(columns=["crest", "acrest"]).agg("|".join, axis=1)
    coarse = pd.DataFrame({"cd": df["cotad"].map(lambda v: _sig(v, 2)), "cu": df["cotau"].map(lambda v: _sig(v, 2)),
                           "gf": df["gf_raw"].map(lambda v: _sig(v, 2)), "berm": (df["b_berm"] > 0).astype(str),
                           "toe": (df["bt"] > 0).astype(str)})
    df["geometry_coarse"] = df["series"] + "|" + coarse.agg("|".join, axis=1)
    df["family"] = structural_family(df)
    ref = df["reference"].str.strip()
    df["source"] = np.where(ref == "", "series-" + df["series"], "ref-" + pd.factorize(ref)[0].astype(str))
    df["lab"] = infer_lab(df["reference"])
    # laboratory-or-series unit for hierarchical models (unknown lab -> its own series)
    df["lab_unit"] = np.where(df["lab"] == "unknown", "series-" + df["series"], df["lab"])
    return df


GROUP_COLUMN = {"geometry": "geometry", "geometry_coarse": "geometry_coarse", "series": "series",
                "source": "source", "lofo_family": "family"}


def make_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """Fold index per record for every validation scheme (same folds for all models)."""
    rng = np.random.RandomState(seed)
    folds = pd.DataFrame(index=df.index)
    folds["random"] = -1
    for k, (_, te) in enumerate(KFold(n_folds, shuffle=True, random_state=seed).split(df)):
        folds.iloc[te, folds.columns.get_loc("random")] = k
    for scheme in ("geometry", "geometry_coarse", "series", "source"):
        g = df[GROUP_COLUMN[scheme]].to_numpy()
        # shuffle group labels so GroupKFold's size-balancing is seed dependent but deterministic
        uniq = np.unique(g)
        perm = dict(zip(uniq, rng.permutation(len(uniq))))
        gi = np.array([perm[v] for v in g])
        folds[scheme] = -1
        for k, (_, te) in enumerate(GroupKFold(n_folds).split(df, groups=gi)):
            folds.iloc[te, folds.columns.get_loc(scheme)] = k
    fam = df["family"]
    folds["lofo_family"] = pd.factorize(fam, sort=True)[0]
    return folds


def inner_group_folds(groups: np.ndarray, n_folds: int, seed: int):
    """Grouped inner folds on the training partition (indices relative to the partition)."""
    uniq = np.unique(groups)
    rng = np.random.RandomState(seed)
    perm = dict(zip(uniq, rng.permutation(len(uniq))))
    gi = np.array([perm[v] for v in groups])
    n = min(n_folds, len(uniq))
    return list(GroupKFold(n).split(gi, groups=gi))


def check_no_leakage(train_groups, test_groups) -> None:
    """Raise if a group appears on both sides of a split."""
    inter = set(np.unique(train_groups)) & set(np.unique(test_groups))
    if inter:
        raise AssertionError(f"Group leakage: {len(inter)} groups on both sides, e.g. {sorted(inter)[:3]}")
