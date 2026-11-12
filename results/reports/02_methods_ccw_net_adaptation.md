# 2. Regression-oriented CCW-Net: adaptation and derivation

Reference: Rountree et al. (2026), *Causal Concept Explanations for Deep Neural Models*,
arXiv:2611.14407. Code: `src/ccw_net.py`, `src/mediator_model.py`, `src/concept_ontology.py`.

## 2.1 What is kept from the original

* Frozen pre-trained backbone f: X -> Z; concept encoder g: Z -> J vectors of dimension D, each passed
  through the capsule squash; bias-free single-head attention read-out, so a zero slot contributes zero
  value and h(0) is a constant (Prop. 1 of the paper).
* No direct path from input to output: X reaches the prediction only through the concept slots.
* Three training stages (warm-up, mediator fit, alignment with periodic mediator refits) and the three
  alignment criteria (necessity, sufficiency, independence).
* A mediator-free estimator of interventional effects for evaluation (Sec. 5.4 / App. F of the paper).

## 2.2 What is changed, and why

| Original | Here | Reason |
|---|---|---|
| Image/control backbones | Tabular MLP (3 x 128 SiLU, latent 64) pre-trained per fold on the 15 dimensionless predictors | CLASH is tabular; the backbone sees training geometries only |
| tanh output on bounded targets | Linear output on the fold-standardised Y = log10 q* | The target is continuous and unbounded |
| Binary concepts; slot length = presence | One slot per concept *group* with K ordered or nominal states. Length = presence only for the six groups with a physical null state (no berm, no toe, normal incidence, smooth, plain crest, flat foreshore). The state is read from the slot direction through learned prototypes (cross-entropy) | Freeboard, steepness, depth and profile type are never "absent"; forcing present/absent on them would be artificial |
| Flip label c -> 1 - c | Counterfactual state drawn among the *other non-null states with at least 100 training records* | Multi-state groups; unsupported states are never used |
| Necessity for every inactive concept | Necessity for null-capable groups in their null state | "Inactive" has no engineering meaning for an always-expressed group |
| Strict additivity (independence) | Additivity required of the attention read-out A(M) only; an explicit low-rank pairwise term I(M) carries interactions and is L1-penalised | Hydraulic concepts interact (freeboard x roughness, depth x steepness, ...) |
| Flow on the squashed vectors | Conditional RealNVP on the pre-squash activations, samples squashed afterwards | Every counterfactual draw is then a valid concept vector (length < 1) |

Ontology: ten groups (Table 4). Thresholds are fixed engineering constants (0.5/1/2 for Rc/Hm0; 0.01/0.04
for steepness; 1/4 for h/Hm0; 20/45 degrees; cot = 1 for "steep"; gamma_f = 0.6 and 1; m = 30 and 1000).
They are not tuned, use no response information and need no per-fold fitting.

## 2.3 Head

With slot vectors M_j in R^D (j = 1..J):

    s_j = (W_Q q)^T W_K M_j / sqrt(D),   a_j = softmax_j(s_j)
    A(M) = sum_j a_j w^T W_O W_V M_j                       (bias-free attention read-out)
    I(M) = sum_{j<k} G_jk <U_j M_j, U_k M_k>               (rank-4 pairwise interaction)
    y_hat = b + A(M) + I(M)

If M_j = 0 the slot contributes nothing to A (zero value) and every pair term containing j vanishes, so
the all-zero vector is a valid reference with h(0) = b.

## 2.4 Loss terms (standardised target units)

Single-slot contribution from the zero reference (closed form, because all other slots are zero):

    D0_j = h(0 with slot j <- M_j) - b = [e^{s_j} / (e^{s_j} + J - 1)] w^T W_O W_V M_j

Let act(j) = 1 unless group j is null-capable and in its null state.

    L_task = E (y_hat - Y)^2
    L_concept = E mean_j [ length loss_j + CE(prototype logits_j, state_j) ]
        length loss = max(0, 0.9 - |M_j|)^2 if act(j), max(0, |M_j| - 0.1)^2 otherwise
    L_nec = E sum_{j: null state} ( h(M with slot j <- M~_j) - y_hat )^2,  M~_j ~ p(M_j | c~_j, X)
    L_suf = E ( b + sum_{j: act} D0_j + sum_{j<k: act} I_jk(M) - Y )^2
    L_ind = E ( A(M) - sum_j D0_j )^2
    L_int = E sum_{j<k} |I_jk(M)|
    L = L_task + l_cls L_concept + l_nec L_nec + l_suf L_suf + l_ind L_ind + l_int L_int

Reading: *necessity* - a concept that is physically absent must not move the prediction even if its slot
is replaced by a plausible "present" representation; *sufficiency* - the active slots, each measured from
zero, plus their mutual interactions, reproduce the target; *independence* - whatever is not additive in
single-slot contributions must appear in the explicit pair terms, which are kept sparse by L_int. In the
notation of interventional mediation analysis, the residual interaction term IE_mu is not driven to zero
(as in the paper) but made explicit and inspectable. The paper's optional presence term is implemented
(`pres`) and switched off.

Default weights follow the paper (l_cls = 1, l_nec = l_suf = l_ind = 0.1) with l_int = 0.01; a sensitivity
run scales the three alignment weights by 0.1 and 10.

## 2.5 Mediator model

One independent conditional RealNVP per concept (6 affine couplings, hidden width 64; weights batched
over concepts), conditioned on the one-hot state of that concept and the 15 standardised inputs, fitted by
maximum likelihood on detached pre-squash activations of the training partition. Benchmark: a conditional
diagonal Gaussian (the same model with zero couplings), also used when D = 1. A third option draws the
counterfactual from other rows of the batch ("no conditional counterfactual").

## 2.6 Evaluation quantities (held-out geometries)

* Necessity score: mean |change of y_hat| when a null-state slot is replaced by a counterfactual draw.
* Sufficiency: RMSE between the active-only reconstruction and (a) the target, (b) the model output.
* Independence: RMSE of A(M) - sum_j D0_j. Interaction share: mean |I| / mean |y_hat - b|.
* Zero-ablation effect shares (mediator-free): e_j = |y_hat - h(M with slot j <- 0)|, rho_j = e_j / sum e;
  reported per null-state slot and per active slot.
* Matched label-swap estimator: for concept j and two of its states, test inputs are paired with matches
  that share all other concept labels (Gaussian kernel on z-scored inputs); hybrid concept vectors and
  sampled Shapley orderings give the share of the output change that passes through slot j
  (`rho_changed_matched`; 1/J = 0.10 would be indiscriminate routing).
* Concept read-out: accuracy, macro-F1, ordinal MAE, expected calibration error.
* Representation interventions (audit type B): move the freeboard, obliquity or roughness slot to the next
  state by a mediator draw and record the sign of the output change.

**These are effects of the model's concept representations on the model's output. None of them is a
physical causal effect of changing a structure or a sea state.** Physical input interventions are a
different analysis (audit type A, `src/counterfactual_audit.py`).

## 2.7 Baselines sharing the folds

Random Forest, XGBoost, monotone XGBoost (non-increasing in Rc/Hm0, Ac/Hm0, Gc/L0; non-decreasing in
gamma_f), MLP, concept-bottleneck regression (Koh et al. 2020, joint training, prediction from concept
state probabilities only), unaligned concept wrapper (same architecture, task + concept loss only).
Hyper-parameters of the tree models and the MLP are selected on grouped inner folds; the concept models use
one pre-specified configuration with a grouped validation split for early stopping.
