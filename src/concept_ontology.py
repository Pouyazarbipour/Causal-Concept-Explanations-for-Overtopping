"""Physical concept ontology (predictor side only; never uses q, RF or CF).

Every concept *group* owns one vector slot in the CCW-Net.  A group has K ordered
or nominal states.  Where an engineering "absent / neutral" state exists (no berm,
normal incidence, smooth surface, ...) it is state 0 and is called the *null*
state: the slot length is trained towards 0 there, which reproduces the zero
reference of the original CCW-Net.  Groups that are always expressed (freeboard,
steepness, depth, profile) have no null state and are never "inactive".

All thresholds are fixed a-priori constants taken from engineering practice; none
is tuned on the response, so no per-fold fitting is required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Concept:
    cid: str
    name: str
    kind: str            # "ordinal" | "nominal"
    has_null: bool       # state 0 is an engineering "absent/neutral" state
    states: tuple
    definition: str
    inputs: str
    rule: str
    interpretation: str
    interactions: str
    limitations: str


CONCEPTS = [
    Concept("C1", "relative_freeboard", "ordinal", False, ("Rc/Hm0<0.5", "0.5-1", "1-2", ">=2"),
            "Crest freeboard relative to the incident wave height: degree of overtopping protection.",
            "Rc, Hm0,toe", "bins of Rc/Hm0,toe at 0.5, 1, 2",
            "Low classes: frequent green-water overtopping; high classes: rare overtopping events.",
            "wave steepness/breaking (C2,C3), roughness (C7), obliquity (C4), crest (C9)",
            "Bin edges are round engineering values, not physical regime boundaries; within-bin variation is carried by the slot direction."),
    Concept("C2", "wave_steepness", "ordinal", False, ("s<0.01", "0.01-0.04", ">=0.04"),
            "Fictitious wave steepness s_m-1,0 = 2 pi Hm0,toe / (g Tm-1,0,toe^2).",
            "Hm0,toe, Tm-1,0,toe", "bins at 0.01 (swell-like or broken waves) and 0.04 (wind sea), EurOtop convention",
            "Low steepness: long/surging or depth-broken waves; high: steep wind sea.",
            "profile slope (C5) -> breaker parameter; depth (C3)",
            "Steepness at the toe mixes offshore sea state and foreshore transformation."),
    Concept("C3", "relative_depth", "ordinal", False, ("h/Hm0<1", "1-4", ">=4"),
            "Toe water depth relative to wave height (foreshore depth regime).",
            "h, Hm0,toe", "bins at 1 and 4 (very shallow / shallow / deep, after Hofland et al. 2017)",
            "Shallow classes imply depth-limited, broken waves at the toe.",
            "steepness (C2), foreshore (C10), profile (C5: impulsive conditions at walls)",
            "The <1 class is small; emergent toes (h<=0) fall in it."),
    Concept("C4", "wave_obliquity", "ordinal", True, ("normal (0 deg)", "0-20", "20-45", ">45"),
            "Angle of wave attack relative to the structure normal.",
            "beta", "0; (0,20]; (20,45]; >45 degrees",
            "Oblique attack reduces overtopping (gamma_beta).",
            "freeboard (C1), profile (C5)",
            "Short-crestedness is not recorded in the database."),
    Concept("C5", "profile_type", "nominal", False,
            ("uniform slope", "composite slope", "slope with wall on top", "wall with slope above", "vertical/steep wall"),
            "Type of seaward profile from the lower and upper slope cotangents.",
            "cot(alpha_d), cot(alpha_u)",
            "steep = cot<1; uniform if both >=1 and equal within 2 %; composite if both >=1 and different; mixed and wall classes by which part is steep",
            "Separates sloping, vertical and mixed structures, whose overtopping physics differ.",
            "steepness (C2) and depth (C3)",
            "cot<1 as 'steep' follows the usual 1:1 convention; recurves/overhangs (negative cot) are merged with walls."),
    Concept("C6", "berm", "nominal", True, ("no berm", "submerged berm", "berm at/above SWL"),
            "Presence and position of a berm.",
            "B, hb", "B=0; B>0 and hb>0; B>0 and hb<=0",
            "Berms dissipate energy; effect depends on berm level relative to SWL.",
            "freeboard (C1), steepness (C2)",
            "Berm width is carried by the slot direction, not by the label."),
    Concept("C7", "roughness_permeability", "ordinal", True, ("smooth (gf=1)", "moderately rough (0.6<gf<1)", "rough armour (gf<=0.6)"),
            "Surface roughness / permeability expressed by gamma_f.",
            "gamma_f", "gf=1; 0.6<gf<1; gf<=0.6",
            "Lower gamma_f = more dissipation on the slope.",
            "freeboard (C1), crest (C9)",
            "gamma_f in CLASH is itself an assigned (partly estimated) value, not a measurement."),
    Concept("C8", "toe_berm", "nominal", True, ("no toe", "toe berm"),
            "Presence of a toe berm in front of the structure.", "Bt", "Bt=0; Bt>0",
            "A toe can trigger breaking or have little effect.", "depth (C3), profile (C5)",
            "Toe significance (ht/Hm0) is carried by the slot direction."),
    Concept("C9", "crest_configuration", "nominal", True,
            ("plain crest", "armour crest, wall not protruding", "protruding crown wall"),
            "Configuration of the crest: armour crest berm and crown wall.",
            "Rc, Ac, Gc", "Gc=0 and Ac=Rc; Rc<=Ac with Gc>0 or Ac>Rc; Rc>Ac",
            "Permeable crest width and wall protrusion reduce overtopping.",
            "roughness (C7), freeboard (C1)", "Wall shape (recurve/bullnose) is not recorded."),
    Concept("C10", "foreshore", "ordinal", True, ("flat (m>=1000)", "gentle (30<m<1000)", "steep (m<=30)"),
            "Foreshore slope in front of the structure.", "m", "m>=1000; 30<m<1000; m<=30",
            "Steep foreshores change breaker type at the structure.", "depth (C3), steepness (C2)",
            "m=1000 is a database convention for flat/deep foreshore."),
]
CONCEPT_NAMES = [c.name for c in CONCEPTS]
N_STATES = [len(c.states) for c in CONCEPTS]
HAS_NULL = [c.has_null for c in CONCEPTS]
IS_ORDINAL = [c.kind == "ordinal" for c in CONCEPTS]
MIN_STATE_SUPPORT = 100  # a counterfactual state must have at least this many training records


def concept_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Integer state per concept group; -1 where the defining variable is missing."""
    lab = pd.DataFrame(index=df.index)
    rc_h, h_h = df["rc"] / df["hm0_toe"], df["h"] / df["hm0_toe"]
    lab["relative_freeboard"] = np.digitize(rc_h, [0.5, 1.0, 2.0])
    lab["wave_steepness"] = np.digitize(df["s0"], [0.01, 0.04])
    lab["relative_depth"] = np.digitize(h_h, [1.0, 4.0])
    b = df["beta_deg"]
    lab["wave_obliquity"] = np.where(b.isna(), -1, np.where(b <= 0, 0, 1 + np.digitize(b, [20.0, 45.0], right=True)))
    sd, su = df["cotad"] < 1, df["cotau"] < 1
    same = np.isclose(df["cotad"], df["cotau"], rtol=0.02)
    lab["profile_type"] = np.select([sd & su, sd & ~su, ~sd & su, same], [4, 3, 2, 0], default=1)
    lab["berm"] = np.where(df["b_berm"] <= 0, 0, np.where(df["hb"] > 0, 1, 2))
    gf = df["gf_raw"]
    lab["roughness_permeability"] = np.where(gf.isna(), -1, np.where(gf >= 1, 0, np.where(gf > 0.6, 1, 2)))
    lab["toe_berm"] = (df["bt"] > 0).astype(int)
    plain = (df["gc"] <= 0) & np.isclose(df["ac"], df["rc"])
    lab["crest_configuration"] = np.where(plain, 0, np.where(df["rc"] > df["ac"] + 1e-9, 2, 1))
    m = df["m"]
    lab["foreshore"] = np.where(m.isna() | (m <= 0), -1, np.where(m >= 1000, 0, np.where(m > 30, 1, 2)))
    return lab[CONCEPT_NAMES].astype(int)


def ontology_table(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CONCEPTS:
        vc = labels[c.name].value_counts()
        counts = "; ".join(f"{s}: {int(vc.get(k, 0))}" for k, s in enumerate(c.states))
        if vc.get(-1, 0):
            counts += f"; undefined: {int(vc[-1])}"
        rows.append({"id": c.cid, "concept": c.name, "type": c.kind + (" + null state" if c.has_null else ""),
                     "engineering_definition": c.definition, "input_variables": c.inputs,
                     "label_rule": c.rule, "n_per_state": counts, "physical_interpretation": c.interpretation,
                     "expected_interactions": c.interactions, "limitations": c.limitations})
    return pd.DataFrame(rows)
