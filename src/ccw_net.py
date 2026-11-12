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
    M = squash(model.pre(z))
    yhat, add, P = model.head(M)
    out = {"task": ((yhat - y) ** 2).mean(), "cls": concept_loss(model, M, labels)}
    d0 = model.single(M)
    act = active_mask(labels)
    I_act = (P * act.unsqueeze(1) * act.unsqueeze(2)).sum((1, 2))
    out["suf"] = ((model.b + (d0 * act).sum(1) + I_act - y) ** 2).mean()
    out["ind"] = ((add - d0.sum(1)) ** 2).mean()
    out["int"] = P.abs().sum((1, 2)).mean()
    out["nec"] = yhat.new_zeros(())
    if w.get("pres", 0) > 0:  # optional presence term of the original paper (off by default)
        beta = 1.0 / act.sum(1, keepdim=True).clamp_min(1)
        out["pres"] = (act * F.relu(beta * (yhat - model.b).abs().unsqueeze(1) - d0.abs()) ** 2).sum(1).mean()
    if w["nec"] > 0 and cf_mode != "none":
        null = ((labels == 0) & NULLCAP)                       # [B, J] slots whose concept is absent
        jn = torch.nonzero(NULLCAP).squeeze(1)
        tot = 0.0
        for _ in range(draws):
            if cf_mode == "flow":
                alt = alt_states(labels, support, generator)
                Mt = squash(flow.sample(make_cond(alt, x, KMAX), generator))
            else:  # "permute": unconditional draw from other rows of the batch (no mediator model)
                Mt = M.detach()[torch.randperm(len(M), generator=generator)]
            Mrep = M.unsqueeze(1).repeat(1, len(jn), 1, 1)       # [B, Jn, J, D]
            Mrep[:, torch.arange(len(jn)), jn] = Mt[:, jn]
            ycf = model.head(Mrep.view(-1, J, model.D))[0].view(len(M), len(jn))
            tot = tot + (((ycf - yhat.unsqueeze(1)) ** 2) * null[:, jn]).sum(1).mean()
        out["nec"] = tot / draws
    return out, M


def total(l, w, align=True):
    t = l["task"] + w["cls"] * l["cls"]
    if align:
        t = t + w["nec"] * l["nec"] + w["suf"] * l["suf"] + w["ind"] * l["ind"] + w["int"] * l["int"]
        if "pres" in l:
            t = t + w["pres"] * l["pres"]
    return t


VARIANTS = {  # name -> overrides
    "ccw_full": {},
    "ccw_unaligned": {"align": False},
    "ccw_no_nec": {"w": {"nec": 0.0}},
    "ccw_no_suf": {"w": {"suf": 0.0}},
    "ccw_no_ind": {"w": {"ind": 0.0}},
    "ccw_no_interaction": {"interaction": False},
    "ccw_scalar": {"D": 1},
    "ccw_no_cond_cf": {"cf_mode": "permute"},
    "ccw_gaussian_mediator": {"flow_layers": 0},
    "ccw_weights_x0.1": {"wmult": 0.1},
    "ccw_weights_x10": {"wmult": 10.0},
}


def train_wrapper(Ztr, Xtr, ytr, Ltr, groups, C, seed, variant):
    """Stages 2-4: warm-up, mediator fit, alignment with early stopping.  Returns model, flow, history."""
    ov = VARIANTS[variant]
    w = dict(C["weights"])
    for k in ("nec", "suf", "ind"):
        w[k] *= ov.get("wmult", 1.0)
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
    # the mediator used for EVALUATION is always refitted on the final encoder (all variants alike)
    flow = ConceptFlow(J, D, KMAX + Xtr.shape[1], ov.get("flow_layers", C["flow"]["layers"]), C["flow"]["hidden"])
    with torch.no_grad():
        pre = model.pre(Ztr[itr])
    fh = fit_flow(flow, pre, Ltr[itr], Xtr[itr], KMAX, C["flow_epochs"], C["flow_lr"], bs, seed, True)
    return model, flow, support, hist, {"flow_nll_final": fh[-1], "epochs_run": len(hist), "val_idx": iva}


# ------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, flow, support, Z, X, y_std, L, ysd, seed, tol):
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
    for _ in range(8):
        alt = alt_states(L, support, g)
        Mt = squash(flow.sample(make_cond(alt, X, KMAX), g))
        for j in torch.nonzero(NULLCAP).squeeze(1).tolist():
            Mr = M.clone()
            Mr[:, j] = Mt[:, j]
            d = (model.head(Mr)[0] - yhat).abs()
            dn[j] += (d * null[:, j]).sum()
            cnt[j] += null[:, j].sum()
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
    # ---- type B audit: representation interventions (NOT physical interventions) ------------
    # move an ordinal slot to the next state whose expected physical direction is "less overtopping"
    for name, j in (("freeboard", 0), ("obliquity", 3), ("roughness", 6)):
        nxt = L[:, j] + 1
        ok = (L[:, j] >= 0) & (nxt < N_STATES[j])
        ok &= torch.tensor([support[j][min(int(k), KMAX - 1)] for k in nxt])
        if ok.sum() < 20:
            res[f"repr_{name}_reversal_rate"], res[f"repr_{name}_median_dy"] = np.nan, np.nan
            continue
        alt = L.clone()
        alt[:, j] = nxt.clamp(max=N_STATES[j] - 1)
        d = []
        for _ in range(8):
            Mt = squash(flow.sample(make_cond(alt, X, KMAX), g))
            Mr = M.clone()
            Mr[:, j] = Mt[:, j]
            d.append((model.head(Mr)[0] - yhat) * ysd)
        d = torch.stack(d).mean(0)[ok]
        res[f"repr_{name}_reversal_rate"] = float((d > tol).float().mean())
        res[f"repr_{name}_median_dy"] = float(d.median())
    return res, per, M, pred


@torch.no_grad()
def matched_shapley(model, M, Xs, L, seed, n_perm=16, min_n=30, tau=0.25, eps=1.5, max_inputs=300):
    """Mediator-free interventional-effect estimator (adapted from Sec. 5.4 / App. F of the paper).

    For concept j and two of its states (a -> b; adjacent states for ordinal groups) every test input i
    in state a is paired with an A-match (state a) and a B-match (state b) that share ALL other concept
    labels with i, drawn with a Gaussian kernel on z-scored inputs.  Hybrid concept vectors take each
    slot from the A- or the B-match; the Shapley value of slot k is the share of the output change that
    passes through slot k.  A faithful model routes the change through the slot whose label changed.
    Returns rows: concept, a, b, n, rho_changed, mean |total effect| (standardised units)."""
    rng = np.random.RandomState(seed)
    Xz = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
    Ln = L.numpy()
    rows = []
    for j in range(J):
        others = np.delete(np.arange(J), j)
        key = np.array(["_".join(map(str, r)) for r in Ln[:, others]])
        states = range(N_STATES[j])
        pairs = [(a, a + 1) for a in range(N_STATES[j] - 1)] if IS_ORDINAL[j] else [(a, b) for a in states for b in states if a < b]
        for a, b in pairs:
            ia, ib = np.flatnonzero(Ln[:, j] == a), np.flatnonzero(Ln[:, j] == b)
            if len(ia) < min_n or len(ib) < min_n:
                continue
            trip = []
            for i in rng.permutation(ia):
                cb = ib[key[ib] == key[i]]
                if len(cb) == 0:
                    continue
                ca = ia[(key[ia] == key[i]) & (ia != i)]
                ca = ca if len(ca) else np.array([i])

                def draw(c):
                    d = np.sqrt(((Xz[c] - Xz[i]) ** 2).mean(1))
                    inr = d <= eps
                    if not inr.any():
                        return c[np.argmin(d)]
                    pw = np.exp(-d[inr] ** 2 / (2 * tau ** 2)) + 1e-300
                    return rng.choice(c[inr], p=pw / pw.sum())
                trip.append((draw(ca), draw(cb)))
                if len(trip) >= max_inputs:
                    break
            if len(trip) < min_n:
                continue
            A = M[[t[0] for t in trip]]
            Bm = M[[t[1] for t in trip]]
            phi = torch.zeros(len(trip), J)
            for _ in range(n_perm):
                order = rng.permutation(J)
                cur = A.clone()
                prev = model.head(cur)[0]
                for k in order:
                    cur[:, k] = Bm[:, k]
                    nxt = model.head(cur)[0]
                    phi[:, k] += nxt - prev
                    prev = nxt
            phi /= n_perm
            mag = phi.abs().mean(0)
            rows.append({"concept": j, "a": a, "b": b, "n": len(trip), "rho_changed": float(mag[j] / mag.sum().clamp_min(1e-12)),
                         "total_abs": float(phi.sum(1).abs().mean())})
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
def run_fold(k, seed, tr, te, X_views, y, labels, groups, rfcf, mcfg, variants, base_seed, tol, want_reps=False):
    """Train backbone (stage 1) once, then every requested variant, on one outer fold and one seed."""
    torch.set_num_threads(1)
    C = mcfg["ccw"]
    s = base_seed + 1000 * seed + k
    prep = FoldPrep(X_views[tr, 0])
    ymu, ysd = float(y[tr].mean()), float(y[tr].std())
    Xtr = prep.scale(X_views[tr, 0])
    ytr = (y[tr] - ymu) / ysd
    n, V, p = X_views[te].shape
    Xte_all = torch.tensor(prep.scale(X_views[te].reshape(-1, p)), dtype=torch.float32)
    bb, _ = train_mlp(Xtr, ytr, groups[tr], C["backbone"], s, mcfg["mlp"]["epochs"], mcfg["mlp"]["patience"], C["batch_size"],
                      latent=C["backbone"]["latent"])
    for q in bb.parameters():
        q.requires_grad_(False)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    Lt = torch.tensor(labels[tr])
    with torch.no_grad():
        Ztr, Zte_all = bb.latent(Xt), bb.latent(Xte_all)
        bb_pred = (bb(Xte_all).numpy() * ysd + ymu).reshape(n, V)
    Zte, Xte = Zte_all.view(n, V, -1)[:, 0], Xte_all.view(n, V, -1)[:, 0]
    Lte = torch.tensor(labels[te])
    yte_std = torch.tensor((y[te] - ymu) / ysd, dtype=torch.float32)
    outp = {"fold": k, "seed": seed, "te": te, "preds": {"backbone": bb_pred}, "metrics": [], "per_concept": [], "shapley": [],
            "history": {}, "quality": []}
    for name in variants:
        if name == "cbm":
            m = train_cbm(Ztr, yt, Lt, groups[tr], C, s)
            with torch.no_grad():
                yh, pr = m(Zte_all)
                outp["preds"][name] = (yh.numpy() * ysd + ymu).reshape(n, V)
                y0, pr0 = m(Zte)
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
        model, flow, support, hist, extra = train_wrapper(Ztr, Xt, yt, Lt, groups[tr], C, s, name)
        with torch.no_grad():
            outp["preds"][name] = (model.head(squash(model.pre(Zte_all)))[0].numpy() * ysd + ymu).reshape(n, V)
        res, per, Mte, pred = evaluate(model, flow, support, Zte, Xte, yte_std, Lte, ysd, s, tol)
        sh = matched_shapley(model, Mte, Xte.numpy(), Lte, s)
        tot_n = sum(r["n"] for r in sh)
        res["rho_changed_matched"] = float(sum(r["n"] * r["rho_changed"] for r in sh) / tot_n) if tot_n else np.nan
        res["n_matched_comparisons"] = len(sh)
        res.update({"variant": name, "fold": k, "seed": seed, "epochs_run": extra["epochs_run"], "flow_nll": extra["flow_nll_final"]})
        outp["metrics"].append(res)
        outp["per_concept"].append({"variant": name, "fold": k, "seed": seed, **per})
        outp["shapley"] += [{"variant": name, "fold": k, "seed": seed, **r} for r in sh]
        outp["history"][name] = hist
        if name == "ccw_full":
            if want_reps:
                outp["reps"] = Mte.numpy()
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
