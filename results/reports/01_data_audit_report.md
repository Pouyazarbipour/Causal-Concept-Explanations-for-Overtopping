# 1. Repository and original-data audit

All numbers below are produced by `python -m src.data_audit` and stored in `results/tables/table01*`,
`table02*`, `tableS_concept_ontology` and `results/diagnostics/data_audit_summary.json`.

## 1.1 What the repository contained

| File | Role |
|---|---|
| `20250402 - Overtopping Database.xls` (now `data/raw/`) | The CLASH database. Sheet `20050101`: 10 532 tests x 33 columns, one header row and one units row. Sheet `cross-section`: a plotting helper that redraws one pasted row; it holds no data. |
| `2611.14407v1.pdf` (now `docs/`) | Rountree et al. (2026), the CCW-Net reference. |
| `Prompt.md` (now `docs/`) | The first research brief (superseded by the data-quality brief). |
| `README.md` | Title only. |

There were **no** experimental reports, the two previous papers were not in the repository, and there were no previous scripts, processed or filtered datasets, data dictionary, laboratory reports,
measurement protocols or earlier geometry-grouping procedures. Consequences:

* The variable dictionary (Table 1) was written from the workbook's header and units rows and the
  published CLASH parameter definitions.
* **The RF/CF category wording is not documented inside the repository.** The scale direction
  (1 = best, 4 = worst, for both) was confirmed from secondary literature; the per-category wording in
  Table 2 follows the published CLASH documentation (Verhaeghe et al., CLASH WP2 report; Steendam et al.
  2004; Van der Meer et al. 2009) as recalled by the author of this analysis and **should be checked
  against the WP2 report before publication**.
* No independent quality information exists, so RF/CF can be evaluated for predictive validity and
  internal consistency only (Section 9.4 of the brief).

Is this the original, unfiltered database? The evidence is consistent with it: 10 532 tests (the size
published for the final CLASH database), 163 test series, all four RF and CF categories present, and
509 records still carrying the remark "do not use for neural network". The file cannot be proven
identical to the 2005 release from the repository alone.

## 1.2 Structure and ranges

* 10 532 rows, 33 columns; test names are unique and all match `SSS-TTT`.
* Units row: lengths in m, periods in s, `b` (wave obliquity) in degrees, `q` in m3/s per m, `Pow` in %.
  q and Hm0 are therefore compatible and q* = q / sqrt(g Hm0,toe^3) is dimensionless.
* Model scale and prototype data are mixed: Hm0,toe ranges from 0 to 3.77 m.
* Missing values per variable are in Table 1 (largest: `Pow` 93 %, `Reference` 26.7 %, `Hm0 deep` 4.1 %).
* 85 rows are exact duplicates of another row apart from the name; 196 records fall in 83 sets with
  identical physical inputs and identical q (all within one series). One such set carries two different
  RF values and one carries two different CF values.

## 1.3 Response

| Class | n | Treatment |
|---|---|---|
| q > 0 | 9 442 | positive-discharge regression (9 333 after input checks) |
| q = 0 | 1 069 | separate wet/dry model (1 056 with usable inputs); never log-transformed, no epsilon |
| q < 0 | 1 | kept, flagged, not modelled |
| q blank or a text marker (`.` x7, `-` x5, `?` x1, blank x7) | 20 | kept, flagged, not modelled |

No detection limit is recorded anywhere in the workbook, so "below detection limit" cannot be separated
from "zero"; the smallest positive q is 2.0e-9 m3/s/m. A censored-response model would need a
per-series detection limit that the database does not provide; the two-part (wet/dry + positive) model is
used instead.

Records that cannot enter the positive regression (all retained in `data/processed/clash_master.csv`
with `unusable_reason`): Hm0,toe missing or zero 120; q missing/marker 20; q negative 1; Tm-1,0,toe
missing 1; h missing 1; zero overtopping 1 056. **All 143 records with unusable inputs have RF = 4.**

## 1.4 RF and CF

| | 1 | 2 | 3 | 4 | missing | out of range |
|---|---|---|---|---|---|---|
| RF, all records | 3 386 (32.1 %) | 3 598 (34.2 %) | 2 642 (25.1 %) | 906 (8.6 %) | 0 | 0 |
| CF, all records | 7 989 (75.9 %) | 1 405 (13.3 %) | 884 (8.4 %) | 254 (2.4 %) | 0 | 0 |
| RF, regression set | 3 189 | 3 174 | 2 333 | 637 | | |
| CF, regression set | 7 067 | 1 229 | 825 | 212 | | |

Joint counts (all records; Figure 2, `table02b`): RF=4 and CF=4 together 31; RF=4 or CF=4 1 129 (10.7 %).
Thin cells in the regression set: (RF2,CF4) 11, (RF3,CF4) 8, (RF4,CF4) 10, (RF4,CF3) 30; and CF=4 as a
whole comes from only **9 series** (7 in the regression set), 62 % of it being one group of chute tests
flagged "do not use". Conclusions about CF = 4 are therefore conclusions about a handful of test series.

Observations made directly from the data:

* RF = 4 is where incomplete or physically questionable records sit: every record with a missing
  foreshore slope, obliquity or roughness factor, every emergent toe (h <= 0) and 152 of the 154 records
  with Hm0,toe > h have RF = 4.
* The share of zero-overtopping tests rises with RF (5.8 %, 11.8 %, 11.7 %, 16.5 % for RF 1-4) and does
  not rise with CF.
* A spectral period equal to exactly Tp/1.1 (the standard CLASH estimate when Tm-1,0 was not measured)
  occurs in 42 % of RF=1 tests and 91 %, 84 %, 63 % of RF=2,3,4 tests. This is an inferred indicator of an
  estimated parameter, not a recorded flag.
* **CF is not constant within a structure.** With geometry reconstructed from water-level-invariant
  descriptors, 32 of 1 100 multi-test geometries (311 records) carry more than one CF value (difference
  of 1 in 14, 2 in 16, 3 in 2); 34 of 163 test series carry more than one CF. Some of this is legitimate
  (the same model is easier or harder to parameterise at another water level, for example when a berm
  emerges), some may be a limitation of the reconstruction. The cases are listed in
  `results/diagnostics/cf_discrepancies_within_geometry.csv` and are not resolved silently: geometry-level
  analyses use the maximum CF and report the number of inconsistent geometries.
* RF varies within 166 geometries, as expected for a test-level rating.

## 1.5 Identifiers that actually exist

| Identifier | Status | Units (regression set) |
|---|---|---|
| Test series (`Name` prefix) | original | 162 |
| Source (`Reference` text) | original but blank for 26.7 % of records (confidential data); blank references fall back to the series | 182 |
| Laboratory | **not recorded**; inferred by keyword from the reference; "unknown" for 2 819 regression records | 13 |
| Structure / geometry | **not recorded**; reconstructed | 2 046 (fine), 1 175 (sub-structure), 554 (coarse) |
| Structural family | rule-based from slopes, berm and roughness | 6 |

The fine geometry key combines, within a series, crest level (h + Rc), armour crest level (h + Ac), toe
height (h - ht), berm level, Bt, B, Gc, both slopes, gamma_f, berm slope and foreshore slope at three
significant digits. Limitations: rounding can split one model into two keys; two different models with the
same descriptors in one series are merged; half of the fine groups hold a single test.

## 1.6 Flags (never used to exclude)

`flag_rf4`, `flag_cf4`, `flag_rf4_and_cf4`, `flag_rf4_or_cf4`, `flag_remark_do_not_use` (509: wind 223,
prototype 132, chute 154), `flag_reference_missing`, `flag_deep_missing`, `flag_tm10_default_ratio`,
`flag_toe_equals_deep`, `flag_depth_limited_gt1`, `flag_toe_emergent`, `flag_duplicate_physical`,
`flag_questionable`, `flag_y_outlier_mad4` (no record exceeds 4 robust SD). Counts: `table02e`.
