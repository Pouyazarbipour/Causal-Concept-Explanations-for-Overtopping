"""Regression-oriented Causal Concept Wrapper Network (CCW-Net) for tabular overtopping data.

Adaptation of Rountree et al. (2026, arXiv:2611.14407).  Differences from the original, and why:

1. Output.  The original head ends in tanh (bounded actions / one-hot targets).  Here the target is the
   unbounded, standardised Y = log10(q*), so the head output is linear:  y = w.c + b.  The zero-value
   property (Prop. 1 of the paper) is kept: all projections are bias-free, hence h(0) = b.
2. Concepts.  The original uses binary concepts whose slot length is "presence".  Here each slot is a
   concept GROUP with K ordered/nominal states.  Slot length still encodes presence, but only for groups
   that have a physical null state (state 0: no berm, normal incidence, ...).  The state itself is read
   from the slot direction through learned prototypes (cross-entropy), so no artificial present/absent
   coding is forced on continuous properties.
3. Necessity is applied only where "inactive" has an engineering meaning (null-capable groups): replacing
   the slot of a null-state concept by a counterfactual draw for a supported non-null state must not
   change the prediction.
4. Interaction.  Hydraulic concepts interact, so strict additivity is not imposed.  The head is
   y = b + A(M) + I(M): A is the bias-free attention read-out and I(M) = sum_{j<k} G_jk <U_j M_j, U_k M_k>
   is an explicit low-rank pairwise term, which vanishes when either slot is zero.  The independence
   loss requires A to be additive in single-slot contributions; all non-additivity must therefore pass
   through the explicit, inspectable I, which is itself penalised (L_interaction, L1 on pair terms).

Loss (standardised target units):
    L = L_task + l_cls L_concept + l_nec L_nec + l_suf L_suf + l_ind L_ind + l_int L_interaction
    D0_j  = h(0 with slot j <- M_j) - b                      (single-slot contribution from zero)
    L_task = E (y_hat - Y)^2
    L_nec  = E sum_{j: c_j = null} (h(M with slot j <- M~_j) - y_hat)^2,   M~_j ~ p(M_j | c~_j, X)
    L_suf  = E (b + sum_{j active} D0_j + I_active(M) - Y)^2
    L_ind  = E (A(M) - sum_j D0_j)^2
    L_int  = E sum_{j<k} |G_jk <U_j M_j, U_k M_k>|
All quantities are effects of the MODEL's concept representations on the MODEL's output.  They are not
physical causal effects of structural or hydraulic changes.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .baseline_models import FoldPrep, train_mlp, val_split
from .concept_ontology import HAS_NULL, IS_ORDINAL, MIN_STATE_SUPPORT, N_STATES
from .mediator_model import ConceptFlow, fit_flow, make_cond

torch.set_num_threads(1)
J = len(N_STATES)
KMAX = max(N_STATES)
NULLCAP = torch.tensor(HAS_NULL)


def squash(v, eps=1e-8):
    n2 = (v ** 2).sum(-1, keepdim=True)
    return n2 / (1 + n2) * v / torch.sqrt(n2 + eps)


class Wrapper(nn.Module):
    def __init__(self, zdim, D=8, hidden=128, rank=4, interaction=True):
        super().__init__()
        self.D, self.use_int = D, interaction
        self.enc = nn.Sequential(nn.Linear(zdim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, J * D))
        lin = lambda: nn.Linear(D, D, bias=False)  # noqa: E731
        self.WQ, self.WK, self.WV, self.WO = lin(), lin(), lin(), lin()
        self.query = nn.Parameter(torch.randn(D) / math.sqrt(D))
        self.w = nn.Parameter(torch.randn(D) / math.sqrt(D))
        self.b = nn.Parameter(torch.zeros(()))
        self.U = nn.Parameter(torch.randn(J, rank, D) / math.sqrt(D))
        self.G = nn.Parameter(torch.zeros(J, J))
        self.register_buffer("triu", torch.triu(torch.ones(J, J), 1))
        self.proto = nn.Parameter(torch.randn(J, KMAX, D) / math.sqrt(D))
        self.proto_b = nn.Parameter(torch.zeros(J, KMAX))
        kmask = torch.zeros(J, KMAX)
        for j, k in enumerate(N_STATES):
            kmask[j, :k] = 1
        self.register_buffer("kmask", kmask)

    # -- encoder ---------------------------------------------------------------------------
    def pre(self, z):
        return self.enc(z).view(-1, J, self.D)

    def state_logits(self, M):
        lg = 4.0 * torch.einsum("bjd,jkd->bjk", M, self.proto) + self.proto_b
        return lg.masked_fill(self.kmask == 0, -1e9)

    # -- head ------------------------------------------------------------------------------
    def _value(self, M):  # scalar value of each slot: w . WO WV M_j   [B, J]
        return self.WO(self.WV(M)) @ self.w

    def _logit(self, M):
        return (self.WK(M) @ self.WQ(self.query)) / math.sqrt(self.D)

    def pairs(self, M):
        if not self.use_int:
            return M.new_zeros(M.shape[0], J, J)
        u = torch.einsum("jrd,bjd->bjr", self.U, M)
        return torch.einsum("bjr,bkr->bjk", u, u) * (self.G * self.triu)

    def head(self, M):
        """Returns y_hat, additive attention part A(M) (without b), pair terms [B, J, J]."""
        a = torch.softmax(self._logit(M), -1)
        add = (a * self._value(M)).sum(-1)
        P = self.pairs(M)
        return self.b + add + P.sum((1, 2)), add, P

    def single(self, M):
        """D0_j: output change when slot j alone is switched on from the all-zero reference."""
        s = self._logit(M)
        a = torch.exp(s) / (torch.exp(s) + (J - 1))
        return a * self._value(M)


def concept_loss(model, M, labels):
    """Slot length encodes presence (null-capable groups), prototypes encode the state."""
    valid = labels >= 0
    length = M.norm(dim=-1)
    present = ((labels != 0) | ~NULLCAP).float()
    l_len = present * F.relu(0.9 - length) ** 2 + (1 - present) * F.relu(length - 0.1) ** 2
    ce = F.cross_entropy(model.state_logits(M).reshape(-1, KMAX), labels.clamp_min(0).reshape(-1), reduction="none").view_as(labels)
    return ((l_len + ce) * valid).sum() / valid.sum()


def active_mask(labels):
    """A slot is active unless its group is null-capable and in its null state (undefined -> active)."""
    return ((labels != 0) | ~NULLCAP).float()


def alt_states(labels, support, generator):
    """Counterfactual state per (row, concept): a supported non-null state different from the current one."""
    B = labels.shape[0]
    alt = torch.zeros_like(labels)
    for j in range(J):
        cand = [k for k in range(1 if HAS_NULL[j] else 0, N_STATES[j]) if support[j][k]]
        if not cand:
            continue
        c = torch.tensor(cand)
        pick = c[torch.randint(len(c), (B,), generator=generator)]
        if len(c) > 1:  # avoid drawing the current state
            same = pick == labels[:, j]
            pick[same] = c[(torch.searchsorted(c, pick[same]) + 1) % len(c)]
        alt[:, j] = pick
    return alt


def losses(model, z, x, y, labels, flow, support, w, generator, draws=4, cf_mode="flow"):
    """Task, concept and alignment losses.  Conventions follow the reference implementation
    (squarebush/ccw-net-neurips2026, losses/causal_alignment.py): the zero reference h(0) and the factual
    output in the counterfactual contrast are detached, and the alignment terms use fully labelled rows only."""
    M = squash(model.pre(z))
    yhat, add, P = model.head(M)
    out = {"task": ((yhat - y) ** 2).mean(), "cls": concept_loss(model, M, labels)}
    zero = yhat.new_zeros(())
    full = (labels >= 0).all(1)                                 # rows with every concept label defined
    if full.sum() < 4:
        out.update({"suf": zero, "ind": zero, "int": zero, "nec": zero, "pres": zero})
        return out, M
    Mf, yf, yhf, addf, Pf, Lf, xf = M[full], y[full], yhat[full], add[full], P[full], labels[full], x[full]
    y0 = model.b.detach()                                       # h(0), detached as y_zero in the reference
    d0 = model.single(Mf)                                       # PE+ : slot j alone, measured from zero
    act = active_mask(Lf)
    I_act = (Pf * act.unsqueeze(1) * act.unsqueeze(2)).sum((1, 2))
    out["suf"] = ((y0 + (d0 * act).sum(1) + I_act - yf) ** 2).mean()
    out["ind"] = ((addf - d0.sum(1)) ** 2).mean()           # total additive effect vs sum of PE+
    out["int"] = Pf.abs().sum((1, 2)).mean()
    beta = 1.0 / act.sum(1, keepdim=True).clamp_min(1)          # presence: each active slot >= beta * |total|
    out["pres"] = (act * F.relu(beta * (yhf - y0).abs().unsqueeze(1) - d0.abs()) ** 2).sum(1).mean()
    out["nec"] = zero
    if w["nec"] > 0 and cf_mode != "none":
        null = ((Lf == 0) & NULLCAP)                            # [B, J] slots whose concept is absent
        jn = torch.nonzero(NULLCAP).squeeze(1)
        y_ref = yhf.detach()                                    # PE- is measured against the detached factual output
        tot = 0.0
        for _ in range(draws):
            if cf_mode == "flow":
                alt = alt_states(Lf, support, generator)
                Mt = squash(flow.sample(make_cond(alt, xf, KMAX), generator))
            else:  # "permute": unconditional draw from other rows of the batch (no mediator model)
                Mt = Mf.detach()[torch.randperm(len(Mf), generator=generator)]
            Mrep = Mf.unsqueeze(1).repeat(1, len(jn), 1, 1)     # [B, Jn, J, D]
            Mrep[:, torch.arange(len(jn)), jn] = Mt[:, jn]
            ycf = model.head(Mrep.view(-1, J, model.D))[0].view(len(Mf), len(jn))
            tot = tot + (((ycf - y_ref.unsqueeze(1)) ** 2) * null[:, jn]).sum(1).mean()
        out["nec"] = tot / draws
    return out, M


def total(l, w, align=True):
    t = l["task"] + w["cls"] * l["cls"]
    if align:
        t = t + w["nec"] * l["nec"] + w["suf"] * l["suf"] + w["ind"] * l["ind"] + w["int"] * l["int"]
        t = t + w.get("pres", 0.0) * l["pres"]
    return t


VARIANTS = {  # name -> overrides
    "ccw_full": {},
    "ccw_unaligned": {"align": False},
    "ccw_no_nec": {"w": {"nec": 0.0}},
    "ccw_no_suf": {"w": {"suf": 0.0}},
    "ccw_no_ind": {"w": {"ind": 0.0}},
    "ccw_no_interaction": {"interaction": False},
    "ccw_no_cond_cf": {"cf_mode": "permute"},
}


def train_wrapper(Ztr, Xtr, ytr, Ltr, groups, C, seed, variant, final_flow=True):
    """Stages 2-4: warm-up, mediator fit, alignment with early stopping.  Returns model, flow, history."""
    ov = VARIANTS[variant]
    w = dict(C["weights"])
    w.update(ov.get("w", {}))
    align, cf_mode = ov.get("align", True), ov.get("cf_mode", "flow")
    D = ov.get("D", C["concept_dim"])
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = Wrapper(Ztr.shape[1], D, C["encoder_hidden"], C["interaction_rank"], ov.get("interaction", True))
    flow = ConceptFlow(J, D, KMAX + Xtr.shape[1], ov.get("flow_layers", C["flow"]["layers"]), C["flow"]["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=C["lr"])
    tr, va = val_split(groups, seed)
    itr, iva = torch.tensor(np.flatnonzero(tr)), torch.tensor(np.flatnonzero(va))
    support = [[int((Ltr[itr, j] == k).sum()) >= MIN_STATE_SUPPORT for k in range(KMAX)] for j in range(J)]
    bs = C["batch_size"]
    hist = []
    best, best_state, bad = np.inf, None, 0

    def fit_mediator(epochs, reset):
        with torch.no_grad():
            pre = model.pre(Ztr[itr])
        return fit_flow(flow, pre, Ltr[itr], Xtr[itr], KMAX, epochs, C["flow_lr"], bs, seed, reset)

    n_ep = C["warmup_epochs"] + C["align_epochs"]
    for ep in range(n_ep):
        in_align = align and ep >= C["warmup_epochs"]
        if align and ep == C["warmup_epochs"]:
            fit_mediator(C["flow_epochs"], True)            # stage 3 (intermission)
            best, bad = np.inf, 0                           # early stopping restarts on the full objective
        elif in_align and (ep - C["warmup_epochs"]) % C["refit_interval"] == 0:
            fit_mediator(C["flow_refit_epochs"], False)
        model.train()
        perm = itr[torch.randperm(len(itr), generator=g)]
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            l, _ = losses(model, Ztr[b], Xtr[b], ytr[b], Ltr[b], flow, support, w if in_align else {**w, "nec": 0.0}, g,
                          C["cf_draws"], cf_mode)
            loss = total(l, w, in_align)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            gv = torch.Generator().manual_seed(1234)
            lv, _ = losses(model, Ztr[iva], Xtr[iva], ytr[iva], Ltr[iva], flow, support, w if in_align else {**w, "nec": 0.0},
                           gv, 2, cf_mode)
            v = float(total(lv, w, in_align))
        hist.append({"epoch": ep, "stage": "align" if in_align else "warmup", "val_objective": v,
                     **{f"val_{k}": float(t) for k, t in lv.items()}})
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= C["patience"] and (ep >= C["warmup_epochs"] + 10 or not align):
                break
    model.load_state_dict(best_state)
    model.eval()
    if not final_flow:      # hyper-parameter search: no evaluation mediator needed
        return model, None, support, hist, {"epochs_run": len(hist)}
    # the mediator used for EVALUATION is always refitted on the final encoder (all variants alike)
    flow = ConceptFlow(J, D, KMAX + Xtr.shape[1], ov.get("flow_layers", C["flow"]["layers"]), C["flow"]["hidden"])
    with torch.no_grad():
        pre = model.pre(Ztr[itr])
    fh = fit_flow(flow, pre, Ltr[itr], Xtr[itr], KMAX, C["flow_epochs"], C["flow_lr"], bs, seed, True)
    return model, flow, support, hist, {"flow_nll_final": fh[-1], "epochs_run": len(hist), "val_idx": iva}


# ------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, flow, support, Z, X, y_std, L, ysd, seed):
    """Model-internal alignment metrics on held-out records (all in log10 units unless a share)."""
    g = torch.Generator().manual_seed(seed)
    M = squash(model.pre(Z))
    yhat, add, P = model.head(M)
    d0 = model.single(M)
    act = active_mask(L)
    I = P.sum((1, 2))
    I_act = (P * act.unsqueeze(1) * act.unsqueeze(2)).sum((1, 2))
    res = {}
    suf_pred = model.b + (d0 * act).sum(1) + I_act
    res["suf_rmse_vs_target"] = float(((suf_pred - y_std) ** 2).mean().sqrt()) * ysd
    res["suf_rmse_vs_output"] = float(((suf_pred - yhat) ** 2).mean().sqrt()) * ysd
    res["ind_rmse"] = float(((add - d0.sum(1)) ** 2).mean().sqrt()) * ysd
    res["ind_rel"] = float((add - d0.sum(1)).abs().mean() / (yhat - model.b).abs().mean())
    res["interaction_share"] = float(I.abs().mean() / ((yhat - model.b).abs().mean() + 1e-12))
    res["interaction_var_share"] = float(I.var() / (yhat.var() + 1e-12))
    # necessity: counterfactual swap of null-state slots
    null = (L == 0) & NULLCAP
    dn, cnt = torch.zeros(J), torch.zeros(J)
    rec_nec = torch.zeros(len(M))
    for _ in range(8):
        alt = alt_states(L, support, g)
        Mt = squash(flow.sample(make_cond(alt, X, KMAX), g))
        for j in torch.nonzero(NULLCAP).squeeze(1).tolist():
            Mr = M.clone()
            Mr[:, j] = Mt[:, j]
            d = (model.head(Mr)[0] - yhat).abs()
            dn[j] += (d * null[:, j]).sum()
            cnt[j] += null[:, j].sum()
            rec_nec += d * null[:, j]
    res["nec_abs_effect"] = float(dn.sum() / cnt.sum().clamp_min(1)) * ysd
    # zero-ablation effect shares (mediator-free, as in the paper's per-input evaluation)
    e = torch.zeros(len(M), J)
    for j in range(J):
        Mr = M.clone()
        Mr[:, j] = 0
        e[:, j] = (yhat - model.head(Mr)[0]).abs()
    rho = e / e.sum(1, keepdim=True).clamp_min(1e-12)
    has_null_slot = null.any(1)
    res["rho_inactive"] = float((rho * null).sum(1)[has_null_slot].mean())
    res["rho_inactive_per_null_slot"] = float(((rho * null).sum(1) / null.sum(1).clamp_min(1))[has_null_slot].mean())
    res["rho_active_per_active_slot"] = float(((rho * (~null)).sum(1) / (~null).sum(1)).mean())
    per = {"abl_effect": (e.mean(0) * ysd).tolist(), "d0_abs": (d0.abs().mean(0) * ysd).tolist(),
           "abl_effect_null": ((e * null).sum(0) / null.sum(0).clamp_min(1) * ysd).tolist(),
           "abl_effect_active": ((e * (~null)).sum(0) / (~null).sum(0).clamp_min(1) * ysd).tolist(),
           "pair_abs": (P.abs().mean(0) * ysd).tolist()}
    # concept read-out quality
    lg = model.state_logits(M)
    prob = torch.softmax(lg, -1)
    pred = lg.argmax(-1)
    acc, f1, omae, ece = [], [], [], []
    for j in range(J):
        v = L[:, j] >= 0
        t, p_ = L[v, j], pred[v, j]
        acc.append(float((t == p_).float().mean()))
        fs = []
        for k in range(N_STATES[j]):
            tp = float(((p_ == k) & (t == k)).sum()); fp = float(((p_ == k) & (t != k)).sum()); fn = float(((p_ != k) & (t == k)).sum())
            if tp + fn > 0:
                fs.append(2 * tp / max(2 * tp + fp + fn, 1e-12))
        f1.append(float(np.mean(fs)))
        omae.append(float((t - p_).abs().float().mean()) if IS_ORDINAL[j] else np.nan)
        conf, hit = prob[v, j].max(-1).values, (t == p_).float()
        bins = torch.clamp((conf * 10).long(), max=9)
        ece.append(float(sum((bins == b).float().mean() * (conf[bins == b].mean() - hit[bins == b].mean()).abs()
                             for b in range(10) if (bins == b).any())))
    per.update({"concept_acc": acc, "concept_f1": f1, "concept_ordinal_mae": omae, "concept_ece": ece})
    res.update({"concept_acc": float(np.mean(acc)), "concept_macro_f1": float(np.mean(f1)),
                "concept_ordinal_mae": float(np.nanmean(omae)), "concept_ece": float(np.mean(ece))})
    # per-record explanation-quality quantities (used to relate explanations to RF / CF categories)
    rec = {"nec_effect": (torch.where(null.any(1), rec_nec / null.sum(1).clamp_min(1) / 8, torch.full((len(M),), float("nan"))) * ysd).numpy(),
           "suf_gap": ((suf_pred - yhat).abs() * ysd).numpy(), "ind_resid": ((add - d0.sum(1)).abs() * ysd).numpy(),
           "interaction_abs": (I.abs() * ysd).numpy(), "rho_inactive": torch.where(has_null_slot, (rho * null).sum(1), torch.full((len(M),), float("nan"))).numpy(),
           "concept_correct": ((pred == L) | (L < 0)).float().mean(1).numpy(), "abl_effect": (e * ysd).numpy()}
    return res, per, M, pred, rec


@torch.no_grad()
def matched_shapley(model, M, Xs, L, seed, n_draws=32, min_n=30, max_subjects=300, tau=0.25, eps=1.5):
    """Interventional-effect estimator of the reference implementation (scoring/independent.py), adapted to
    multi-state concept groups.

    A comparison changes concept j from state a to state b (adjacent states for ordinal groups, every pair for
    nominal ones) within ONE active/absent pattern of the other nine concepts (the reference scores each
    binary label combination separately; with multi-state concepts the active/absent pattern is the analogue,
    and the remaining state differences are absorbed by matching on the inputs); both arms must hold
    >= min_n held-out inputs.  Subjects come from both arms.  Every
    concept slot is its own mediator group.  For every subject, group and draw, an a-arm and a b-arm match are
    drawn INDEPENDENTLY (Gaussian kernel, bandwidth tau, caliper eps on z-scored inputs, self excluded, nearest
    row if none is in the caliper).  Hybrid concept vectors take each group's slot from its a- or b-match; each
    draw uses a random group ordering, so the average over draws is a sampled Shapley value.  |IE_g| is the
    absolute value of the subject-averaged effect (norm of the mean, as in the reference).
    Returned per comparison: shares of the changed slot, of the held active slots and of the held absent
    (null-state) slots, and the mean share per held absent slot (1/J = 0.10 would be indiscriminate)."""
    rng = np.random.RandomState(seed)
    Xz = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-8)
    Ln = L.numpy()
    active = np.where(np.array(HAS_NULL)[None, :], (Ln != 0).astype(int), 1)   # active/absent pattern per concept
    rows = []
    for j in range(J):
        others = np.delete(np.arange(J), j)
        key = np.array(["".join(map(str, r)) for r in active[:, others]])
        states = range(N_STATES[j])
        pairs = [(a, a + 1) for a in range(N_STATES[j] - 1)] if IS_ORDINAL[j] else [(a, b) for a in states for b in states if a < b]
        for a, b in pairs:
            for kv in np.unique(key[(Ln[:, j] == a) | (Ln[:, j] == b)]):
                ia = np.flatnonzero((Ln[:, j] == a) & (key == kv))
                ib = np.flatnonzero((Ln[:, j] == b) & (key == kv))
                if len(ia) < min_n or len(ib) < min_n:
                    continue
                subj = np.concatenate([ia, ib])
                subj = rng.choice(subj, min(max_subjects, len(subj)), replace=False)

                def draws_for(i, pool):
                    pool = pool[pool != i]
                    d2 = ((Xz[pool] - Xz[i]) ** 2).mean(1)
                    w = np.where(d2 <= eps ** 2, np.exp(-d2 / (2 * tau ** 2)), 0.0)
                    if w.sum() <= 1e-12:
                        return np.full((J, n_draws), pool[np.argmin(d2)])
                    return rng.choice(pool, size=(J, n_draws), p=w / w.sum())     # independent per group and draw
                A_idx = np.stack([draws_for(i, ia) for i in subj])                 # [S, J, draws]
                B_idx = np.stack([draws_for(i, ib) for i in subj])
                S = len(subj)
                phi = np.zeros((S, J))
                for d in range(n_draws):
                    MA = M[A_idx[:, :, d]][:, np.arange(J), np.arange(J)]          # slot g from its own a-match
                    MB = M[B_idx[:, :, d]][:, np.arange(J), np.arange(J)]
                    cur = MA.clone()
                    prev = model.head(cur)[0]
                    for g in rng.permutation(J):
                        cur[:, g] = MB[:, g]
                        nxt = model.head(cur)[0]
                        phi[:, g] += (nxt - prev).numpy()
                        prev = nxt
                phi /= n_draws
                ie = np.abs(phi.mean(0))                                           # norm of the subject-averaged effect
                share = ie / max(ie.sum(), 1e-12)
                held_null = active[ia[0]][others] == 0
                rows.append({"concept": j, "a": a, "b": b, "n": S, "label_combination": kv, "rho_changed": float(share[j]),
                             "rho_held_active": float(share[others][~held_null].sum()),
                             "rho_held_null": float(share[others][held_null].sum()),
                             "rho_per_held_null_slot": float(share[others][held_null].mean()) if held_null.any() else np.nan,
                             "n_held_null": int(held_null.sum()), "total_abs": float(np.abs(phi.sum(1).mean()))})
    return rows


# ------------------------------------------------------------------------------- CBM baseline
class CBM(nn.Module):
    """Concept-bottleneck regression (Koh et al. 2020, joint training): Z -> concept-state probabilities
    -> small MLP -> y.  The prediction can only use the categorical concept states."""

    def __init__(self, zdim, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(zdim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, sum(N_STATES)))
        self.head = nn.Sequential(nn.Linear(sum(N_STATES), 64), nn.ReLU(), nn.Linear(64, 1))
        self.splits = list(N_STATES)

    def probs(self, z):
        return [torch.softmax(l, -1) for l in self.enc(z).split(self.splits, -1)]

    def forward(self, z, zero: int | None = None):
        pr = self.probs(z)
        if zero is not None:
            pr[zero] = torch.zeros_like(pr[zero])
        return self.head(torch.cat(pr, -1)).squeeze(-1), pr


def train_cbm(Ztr, ytr, Ltr, groups, C, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = CBM(Ztr.shape[1], C["encoder_hidden"])
    opt = torch.optim.Adam(m.parameters(), lr=C["lr"])
    tr, va = val_split(groups, seed)
    itr, iva = torch.tensor(np.flatnonzero(tr)), torch.tensor(np.flatnonzero(va))

    def loss_fn(idx):
        yh, pr = m(Ztr[idx])
        ce = sum((F.nll_loss(torch.log(p + 1e-9), Ltr[idx, j].clamp_min(0), reduction="none") * (Ltr[idx, j] >= 0)).mean()
                 for j, p in enumerate(pr)) / J
        return ((yh - ytr[idx]) ** 2).mean() + C["weights"]["cls"] * ce
    best, state, bad = np.inf, None, 0
    for _ in range(C["warmup_epochs"] + C["align_epochs"]):
        m.train()
        perm = itr[torch.randperm(len(itr), generator=g)]
        for s in range(0, len(perm), C["batch_size"]):
            loss = loss_fn(perm[s:s + C["batch_size"]])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = float(loss_fn(iva))
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: t.clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= C["patience"]:
                break
    m.load_state_dict(state)
    return m.eval()


# ------------------------------------------------------------------------------- fold driver
def tune_ccw(Ztr, Xt, yt, Lt, groups, C, n_trials, seed):
    """Optuna (TPE) for the wrapper on the training partition only.  Objective: validation RMSE of the
    aligned wrapper on a grouped validation split (shortened schedule).  The alignment-weight multiplier is
    part of the search, so alignment strength is chosen without looking at held-out geometries."""
    import copy
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def build(params):
        c = copy.deepcopy(C)
        c.update({"concept_dim": params["concept_dim"], "encoder_hidden": params["encoder_hidden"], "lr": params["lr"],
                  "interaction_rank": params["interaction_rank"]})
        for k in ("nec", "suf", "ind", "pres"):
            c["weights"][k] = C["weights"][k] * params["align_mult"]
        return c

    def objective(trial):
        params = {"concept_dim": trial.suggest_categorical("concept_dim", [4, 8, 16]),
                  "encoder_hidden": trial.suggest_categorical("encoder_hidden", [64, 128, 256]),
                  "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
                  "interaction_rank": trial.suggest_categorical("interaction_rank", [2, 4, 8]),
                  "align_mult": trial.suggest_float("align_mult", 0.1, 10.0, log=True)}
        c = build(params)
        t = C["tune_epochs"]
        c.update({"warmup_epochs": t["warmup"], "flow_epochs": t["flow"], "align_epochs": t["align"]})
        _, _, _, hist, _ = train_wrapper(Ztr, Xt, yt, Lt, groups, c, seed, "ccw_full", final_flow=False)
        return float(np.sqrt(min(h["val_task"] for h in hist if h["stage"] == "align")))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return build(study.best_params), dict(study.best_params), float(study.best_value), study.trials_dataframe(attrs=("number", "value", "params", "duration"))


def run_fold(k, seed, tr, te, X, y, labels, groups, rfcf, mcfg, variants, base_seed, tuned=None):
    """Stage 1 backbone, (optional) Optuna for the wrapper, then every requested variant on one fold/seed.
    `tuned` = (C, params, value) from an earlier seed of the same fold re-uses the selected configuration."""
    torch.set_num_threads(1)
    C = mcfg["ccw"]
    s = base_seed + 1000 * seed + k
    prep = FoldPrep(X[tr])
    ymu, ysd = float(y[tr].mean()), float(y[tr].std())
    Xtr = prep.scale(X[tr])
    ytr = (y[tr] - ymu) / ysd
    Xte = torch.tensor(prep.scale(X[te]), dtype=torch.float32)
    bb, _ = train_mlp(Xtr, ytr, groups[tr], C["backbone"], s, mcfg["mlp"]["epochs"], mcfg["mlp"]["patience"], C["batch_size"],
                      latent=C["backbone"]["latent"])
    for q in bb.parameters():
        q.requires_grad_(False)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    Lt = torch.tensor(labels[tr])
    with torch.no_grad():
        Ztr, Zte = bb.latent(Xt), bb.latent(Xte)
        bb_pred = bb(Xte).numpy() * ysd + ymu
    Lte = torch.tensor(labels[te])
    yte_std = torch.tensor((y[te] - ymu) / ysd, dtype=torch.float32)
    outp = {"fold": k, "seed": seed, "te": te, "preds": {"backbone": bb_pred}, "metrics": [], "per_concept": [], "shapley": [],
            "history": {}, "quality": [], "records": {}, "reps": {}}
    if tuned is None:
        Ct, params, val, trials = tune_ccw(Ztr, Xt, yt, Lt, groups[tr], C, mcfg["optuna_trials"]["ccw"], s)
        outp["tuning"] = {"params": params, "value": val, "trials": trials}
    else:
        Ct = tuned
    outp["C"] = Ct
    for name in variants:
        if name == "cbm":
            m = train_cbm(Ztr, yt, Lt, groups[tr], Ct, s)
            with torch.no_grad():
                y0, pr0 = m(Zte)
                outp["preds"][name] = y0.numpy() * ysd + ymu
                acc = [float((p_.argmax(-1) == Lte[:, j])[Lte[:, j] >= 0].float().mean()) for j, p_ in enumerate(pr0)]
                e = torch.stack([(y0 - m(Zte, zero=j)[0]).abs() for j in range(J)], 1)
                rho = e / e.sum(1, keepdim=True).clamp_min(1e-12)
                null = (Lte == 0) & NULLCAP
                hn = null.any(1)
            outp["metrics"].append({"variant": name, "fold": k, "seed": seed, "concept_acc": float(np.mean(acc)),
                                    "rho_inactive": float((rho * null).sum(1)[hn].mean()),
                                    "rho_inactive_per_null_slot": float(((rho * null).sum(1) / null.sum(1).clamp_min(1))[hn].mean()),
                                    "rho_active_per_active_slot": float(((rho * (~null)).sum(1) / (~null).sum(1)).mean())})
            continue
        model, flow, support, hist, extra = train_wrapper(Ztr, Xt, yt, Lt, groups[tr], Ct, s, name)
        with torch.no_grad():
            outp["preds"][name] = model.head(squash(model.pre(Zte)))[0].numpy() * ysd + ymu
        res, per, Mte, pred, rec = evaluate(model, flow, support, Zte, Xte, yte_std, Lte, ysd, s)
        sh = matched_shapley(model, Mte, Xte.numpy(), Lte, s)
        tot_n = sum(r["n"] for r in sh)
        res["rho_changed_matched"] = float(sum(r["n"] * r["rho_changed"] for r in sh) / tot_n) if tot_n else np.nan
        hn = [r for r in sh if r["n_held_null"] > 0]
        res["rho_held_null_matched"] = float(sum(r["n"] * r["rho_held_null"] for r in hn) / sum(r["n"] for r in hn)) if hn else np.nan
        res["rho_per_held_null_slot_matched"] = float(sum(r["n"] * r["rho_per_held_null_slot"] for r in hn) / sum(r["n"] for r in hn)) if hn else np.nan
        res["n_matched_comparisons"] = len(sh)
        res.update({"variant": name, "fold": k, "seed": seed, "epochs_run": extra["epochs_run"], "flow_nll": extra["flow_nll_final"]})
        outp["metrics"].append(res)
        outp["per_concept"].append({"variant": name, "fold": k, "seed": seed, **per})
        outp["shapley"] += [{"variant": name, "fold": k, "seed": seed, **r} for r in sh]
        outp["history"][name] = hist
        outp.setdefault("states", {})[name] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if name in ("ccw_full", "ccw_unaligned"):
            outp["records"][name] = rec
            outp["reps"][name] = Mte.flatten(1).numpy()
        if name == "ccw_full":
            outp["quality"] += quality_branch(model, Ztr, yt, extra["val_idx"], Zte, yte_std, rfcf[tr], rfcf[te], ysd, s, k, seed)
    return outp


def quality_branch(model, Ztr, ytr, iva, Zte, yte, q_tr, q_te, ysd, seed, k, sd):
    """Separate quality-assessment layer: a variance head on top of the FROZEN mean prediction.

    The mean is detached, so RF/CF cannot alter the physical mean predictor by construction.  Heads are
    fitted on the grouped validation records of the training partition (never used for gradient steps of
    the mean model) and scored on the held-out fold by Gaussian negative log-likelihood and coverage."""
    def onehot(q):
        return torch.tensor(np.concatenate([np.eye(4)[q[:, 0].astype(int) - 1], np.eye(4)[q[:, 1].astype(int) - 1]], 1), dtype=torch.float32)
    with torch.no_grad():
        r_va = (model.head(squash(model.pre(Ztr[iva])))[0] - ytr[iva])
        r_te = (model.head(squash(model.pre(Zte)))[0] - yte)
        Mva = squash(model.pre(Ztr[iva])).flatten(1)
        Mte = squash(model.pre(Zte)).flatten(1)
    Qva, Qte = onehot(q_tr[iva.numpy()]), onehot(q_te)
    rows = []
    for name, use_m, use_q in (("constant", False, False), ("quality_only(RF,CF)", False, True),
                               ("physical_concepts_only", True, False), ("physical+quality_layer", True, True)):
        torch.manual_seed(seed)
        fm = nn.Sequential(nn.Linear(Mva.shape[1], 32), nn.ReLU(), nn.Linear(32, 1)) if use_m else None
        fq = nn.Linear(8, 1, bias=False) if use_q else None
        b0 = nn.Parameter(torch.log(r_va.var()).clone())
        params = [b0] + (list(fm.parameters()) if fm else []) + (list(fq.parameters()) if fq else [])
        opt = torch.optim.Adam(params, lr=5e-3, weight_decay=1e-3)

        def logvar(Mx, Qx):
            lv = b0.expand(len(Mx))
            if fm:
                lv = lv + fm(Mx).squeeze(-1)
            if fq:
                lv = lv + fq(Qx).squeeze(-1)
            return lv.clamp(-8, 6)
        for _ in range(300):
            lv = logvar(Mva, Qva)
            loss = 0.5 * (lv + r_va ** 2 / torch.exp(lv)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            lv = logvar(Mte, Qte)
            nll = 0.5 * (lv + r_te ** 2 / torch.exp(lv) + math.log(2 * math.pi)).mean() + math.log(ysd)
            zs = (r_te.abs() / torch.exp(0.5 * lv))
            rows.append({"head": name, "fold": k, "seed": sd, "test_nll": float(nll), "cov90": float((zs <= 1.645).float().mean()),
                         "mean_sigma": float(torch.exp(0.5 * lv).mean() * ysd)})
    return rows
