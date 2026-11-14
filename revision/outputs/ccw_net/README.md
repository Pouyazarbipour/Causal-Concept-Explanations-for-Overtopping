# CCW-Net analysis (preserved, audited)

The complete CCW-Net analysis is unchanged and fully reproducible. It is not repeated here.

| Item | Location |
|---|---|
| Implementation | `src/ccw_net.py`, `src/mediator_model.py`, `src/concept_ontology.py`, `experiments/run_ccw.py` |
| Reference-code check (errors E1–E5 fixed, deliberate adaptations, remaining differences) | `results/reports/08_reference_implementation_check.md` |
| Alignment and ablation (CCW-Net, unaligned wrapper, without necessity, sufficiency, independence, interaction or conditional counterfactual; concept bottleneck) | `results/tables/table09_ccw_alignment_and_ablation.*` |
| Concept attribution; agreement with grouped TreeSHAP | `results/tables/table10_*`, `table11_concept_vs_feature_attribution.*` |
| Explanation quality vs RF/CF; quality probe | `results/tables/table12a-c_*` |
| Quality layer, interaction pairs | `results/tables/tableS_ccw_quality_layer.*`, `tableS_interaction_pairs_*` |
| Figures | `results/figures/fig11_ccw_net_architecture.*`, `fig12_concept_attribution_and_alignment.*` |
| Per-fold models and caches | `results/models/ccw_cache/` |

## Audit result

- **Accuracy.** CCW-Net does not improve predictive accuracy: pooled RMSE 0.601 against 0.501 for
  XGBoost.
- **Concept faithfulness.**
  - The alignment losses improve necessity (0.038 vs 0.055) and sufficiency (0.108 vs 0.273), the
    criteria they target.
  - Additivity is not improved (0.062 vs 0.049).
  - Matched concept routing is not improved: the share carried by the changed concept is 0.171 vs 0.169,
    where 0.10 means indiscriminate routing.
- **Agreement with feature attribution.** Rank agreement with grouped TreeSHAP is 0.07 over ten concepts.
  CCW-Net ranks the concepts differently from established formulae.
- **Quality dependence.** No explanation-quality trend survives Holm correction.

These are negative methodological results. CCW-Net examines model-internal representations, not physical
causal mechanisms. Recommended placement: Supplementary Information, with a one-paragraph summary in the
main text.

**Bibliography.** The arXiv identifier 2611.14407 used in earlier drafts could not be verified. The
reference repository cites the paper as NeurIPS 2026 (`revision/audit/reference_audit.csv`).
