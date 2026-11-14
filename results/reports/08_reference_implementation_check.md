# 8. Check of the CCW-Net code against the reference implementation

Reference: `squarebush/ccw-net-neurips2026` (public repository of Rountree et al. 2026, commit `8fb1283`),
cloned read-only and compared file by file with `src/ccw_net.py` and `src/mediator_model.py` of this
study. Files compared: `models/squash_attention.py`, `losses/causal_alignment.py`, `losses/supervised.py`,
`flows/invertible_nf.py`, `flows/training.py`, `train.py`, `scoring/independent.py`,
`scoring/attribution.py`, `models/cbm.py`, `configs/carracing/width64/ccwnet.yaml`.

## 8.1 Errors found and corrected

| # | Reference behaviour | Earlier version of this study | Effect | Correction |
|---|---|---|---|---|
| E1 | Necessity contrast `PE- = h(M with slot j <- donor) − y_self.detach()` | factual output not detached | gradient of the necessity loss also pulled the factual prediction toward the counterfactual one, mixing necessity with the task objective | factual output detached |
| E2 | Zero reference `y_zero = h(0).detach()` in sufficiency and in the total effect | output bias received gradient from the sufficiency loss | bias partly fitted to the sufficiency target rather than to the task | `b` detached in the alignment terms |
| E3 | Alignment losses computed only on rows with every concept label defined | rows with an undefined concept label (−1) treated as active | small (12 rows) | alignment on fully labelled rows only |
| E4 | Presence loss active with lambda 1 × align weight 0.1, beta = 1/(number of active concepts) | presence switched off | one or two active concepts could carry the whole effect | presence enabled (0.1) |
| E5 | Evaluation estimator: matches drawn **independently for every mediator group**, 96 draws, all group orderings, effect = **norm of the subject-averaged** effect, per label combination | one shared A-match and B-match per subject for all slots, mean of **absolute** per-subject effects, comparisons pooled | the mean of absolute values does not let noise cancel, which pushes every slot's share toward 1/J; the reported "13.5 % via the changed concept" was biased toward chance and is withdrawn | estimator re-implemented: independent per-group matches, 32 draws with a random ordering each (sampled Shapley), norm of the mean, scored within each active/absent pattern of the held concepts; reports the changed-concept share and the share per held absent concept |

All CCW-Net results were regenerated after these corrections; the earlier CCW-Net tables are superseded.

## 8.2 Components that agree with the reference

* Concept encoder: MLP from the frozen backbone latent to J × D, capsule squash per slot.
* Head: single-head attention with a learned query, bias-free query/key/value/output projections, so a zero
  slot contributes zero value and h(0) is a constant (Prop. 1); single-slot effects PE+ measured from the
  all-zero vector with the other slots at zero.
* Necessity on absent concepts with donor representations drawn from a conditional flow given the flipped
  labels and the covariates, without gradient into the flow; sufficiency rebuilt from h(0) plus the effects of
  the present concepts against the target; independence as total effect minus the sum of single-slot effects.
* Effective weights 0.1 for necessity, sufficiency, independence and presence (lambda 1 × align weight 0.1).
* Schedule: warm-up on task and concept losses, mediator fit, alignment with periodic mediator refits.
* Mediator: per-slot conditional affine-coupling flow with alternating masks, bounded log-scale (tanh),
  zero-initialised conditioner heads; per-slot conditional affine maps when D = 1.
* Zero-ablation attribution shares (reference `scoring/attribution.py`), with the CBM reference set to
  probability zero (the reference uses logit −inf).

## 8.3 Deliberate adaptations (documented, not errors)

| Reference | This study | Reason |
|---|---|---|
| tanh output, MSE to actions or one-hot labels | linear output on standardised log10 q* | continuous unbounded regression target |
| binary concepts; slot norm trained by BCE as presence | multi-state concept groups; length hinge for presence (null-capable groups only) plus prototype cross-entropy for the state | ordinal/nominal engineering concepts; no artificial present/absent coding |
| counterfactual label 1 − c for every slot | a supported non-null alternative state | multi-state groups; unsupported states excluded |
| strict additivity (independence on the full output) | independence on the attention read-out A(M); explicit pairwise interaction term I(M) with an L1 penalty, included in sufficiency | physical interactions between hydraulic concepts |
| flow fitted on squashed concepts | flow fitted on pre-squash activations, samples squashed | every draw is a valid concept vector |
| frozen pre-trained backbone supplied by the task | tabular MLP pre-trained per outer fold on training geometries | tabular data; leakage control |
| final-epoch model | early stopping on a grouped validation split; Optuna tuning of size, learning rate and alignment strength | small data set, grouped generalisation |
| label combinations of binary concepts | active/absent pattern of the held concepts; remaining state differences absorbed by covariate matching | ten multi-state concepts make exact label combinations too sparse |

## 8.4 Remaining differences kept for runtime (sensitivity not run)

Gradient clip 5 (reference 1); 4 counterfactual draws per step (reference 8); flow width 64 with 6 couplings
(reference 256 and 8); mediator refits warm-started for 8 epochs every 15 alignment epochs (reference: a new
flow trained for 50 epochs every 10 epochs); estimator with 32 draws (reference 96) and >= 30 inputs per arm
(reference 100).
