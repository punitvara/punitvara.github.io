"""Friend recommendation - link prediction as a two-stage recommender.

What this module teaches (each section below is tagged with the interview concept):
  1. Synthetic world      - stochastic-block-model communities + latent real-world ties that
                            turn into online friendships over TIME (triadic closure hazard).
  2. Snapshots            - point-in-time feature computation (the cure for label leakage).
  3. Candidate generation - friends-of-friends (common neighbours, Adamic-Adar, Jaccard),
                            spectral/SVD embeddings + cosine (brute force and a hand-written
                            IVF index to show the ANN recall/cost trade-off), and a
                            contacts/attribute fallback for cold-start users.
  4. Ranking              - HistGradientBoosting on pairwise features, negative downsampling
                            with prior-correction calibration, occlusion contributions.
  5. Pitfall: leakage     - random split with "today's" features vs time split.
  6. Pitfall: hub bias    - popularity feedback loop + celebrity cap / dismiss mitigation.

Everything is deterministic (fixed seeds) and small (1,000 users) so it builds in ~2 s.
"""
import threading
import time
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.extmath import randomized_svd

META = {
    "slug": "friend-recommendation",
    "title": "Friend Recommendation",
    "tagline": "Link prediction at 1B users: FoF + embedding retrieval, GBDT ranker, leakage and hub-bias pitfalls",
    "icon": "\U0001F465",
    "order": 1,
}
router = APIRouter()

# --------------------------------------------------------------------------------------
# 0. Constants
# --------------------------------------------------------------------------------------
N = 1000                 # users
NC = 20                  # communities (SBM blocks)
K_STEPS = 20             # simulation steps; time runs 0 -> 1
T0, T1, T2, T3 = 0.55, 0.70, 0.85, 1.00
# Timeline used everywhere:
#   features@T0 -> labels (T0,T1]  = TRAIN
#   features@T1 -> labels (T1,T2]  = VALIDATION (time split)
#   features@T2 -> labels (T2,T3]  = "PRODUCTION": the hidden friendships the demo tries to find
VIEW_TAU = 0.10          # decay constant of the profile-view recency feature
EMB_DIM = 32
N_CELLS = 32             # IVF cells for the ANN demo
NEG_PER_POS = 10         # negative downsampling ratio for the ranker
HUB_PCTL = 95            # "hub" = top 5% by degree

FEATURES = [
    ("mutual_friends", "count of common neighbours"),
    ("adamic_adar", "sum over common neighbours w of 1/log(deg(w)); rare mutual friends count more"),
    ("jaccard", "common / union of the two friend lists"),
    ("same_school", "1 if same school"),
    ("same_employer", "1 if same employer"),
    ("same_city", "1 if same city"),
    ("contact_overlap", "0/1/2: how many of the two address books contain the other person"),
    ("view_recency", "exp(-(now - last profile view between the pair)/tau), 0 if never"),
    ("age_gap", "absolute age difference in years"),
    ("min_degree", "friend count of the less connected of the two"),
    ("max_degree", "friend count of the more connected of the two (popularity)"),
    ("emb_cosine", "cosine of the two users' graph embeddings (retrieval score reused as a feature)"),
]
FNAMES = [f[0] for f in FEATURES]

_STATE = None
_LOCK = threading.Lock()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------------------
# 1. Synthetic world.  Concept: homophily (attributes) + triadic closure (graph) are the two
#    forces behind friendship formation, and every edge has a TIMESTAMP.
# --------------------------------------------------------------------------------------
def _simulate():
    rng = np.random.default_rng(7)
    comm = rng.choice(NC, N, p=rng.dirichlet(np.ones(NC) * 4))
    # attributes are correlated with community but not identical to it
    pref_school = rng.integers(0, 15, (NC, 2))
    pref_emp = rng.integers(0, 30, (NC, 2))
    pref_city = rng.integers(0, 8, NC)
    school = np.where(rng.random(N) < 0.7, pref_school[comm, rng.integers(0, 2, N)], rng.integers(0, 15, N))
    employer = np.where(rng.random(N) < 0.6, pref_emp[comm, rng.integers(0, 2, N)], rng.integers(0, 30, N))
    city = np.where(rng.random(N) < 0.8, pref_city[comm], rng.integers(0, 8, N))
    age = np.clip(rng.uniform(22, 50, NC)[comm] + rng.normal(0, 4, N), 18, 70).round()
    activity = rng.lognormal(0, 0.55, N)
    celeb = np.zeros(N, bool)
    celeb[rng.choice(N, 10, replace=False)] = True       # a few celebrities / super-connectors
    activity[celeb] *= 5

    # cold-start users join right at T2: zero friends when we must serve them
    joined = np.zeros(N)
    cold = rng.choice(np.where(~celeb)[0], 50, replace=False)
    joined[cold] = T2

    same_school = school[:, None] == school[None, :]
    same_emp = employer[:, None] == employer[None, :]
    same_city = city[:, None] == city[None, :]
    age_gap = np.abs(age[:, None] - age[None, :])
    upper = np.triu(np.ones((N, N), bool), 1)

    # latent real-world ties R (people who actually know each other offline)
    logit = (-5.5 + 2.6 * (comm[:, None] == comm[None, :]) + 1.2 * same_school + 1.5 * same_emp
             + 0.7 * same_city - 0.10 * age_gap)
    p = _sigmoid(logit) * np.sqrt(activity[:, None] * activity[None, :])
    p *= (N * 17) / p[upper].sum()                         # ~34 real-world ties per user
    R = (rng.random((N, N)) < p) & upper
    R = R | R.T

    # online friendships appear over time; hazard grows with mutual friends (triadic closure)
    A = np.zeros((N, N), np.float32)
    ET = np.full((N, N), np.inf)                           # edge time (inf = never)
    fan = np.maximum(celeb[:, None], celeb[None, :])
    for k in range(1, K_STEPS + 1):
        t = k / K_STEPS
        cn = A @ A
        haz = np.where(R, 0.035 * (1 + 0.6 * cn), np.where(cn >= 2, 0.0012 * cn, 0.0))
        haz = haz + 0.004 * fan * (~R)                     # strangers follow celebrities
        elig = (joined[:, None] < t) & (joined[None, :] < t)
        newbie = (joined[:, None] > 0) | (joined[None, :] > 0)
        haz = np.clip(np.where(newbie, haz * 5, haz), 0, 0.9)   # new users add friends fast
        new = (rng.random((N, N)) < haz) & upper & (A == 0) & elig
        tt = t - rng.random((N, N)) / K_STEPS
        ET[new] = tt[new]
        ET = np.minimum(ET, ET.T)
        A = np.isfinite(ET).astype(np.float32)

    # address books: directional, uploaded by ~2/3 of users, noisy (plumbers, landlords...)
    upload = rng.random(N) < np.where(joined > 0, 0.8, 0.65)
    C = R & upload[:, None] & (rng.random((N, N)) < 0.4)
    for u in np.where(upload)[0]:
        pool = np.where((city == city[u]) & (np.arange(N) != u))[0]
        C[u, rng.choice(pool, min(3, len(pool)), replace=False)] = True
    csym = C.astype(np.int8) + C.T.astype(np.int8)

    # profile views.  NOTE the trap: most views happen AFTER two people become friends.
    vu, vv, vt = [], [], []
    ru, rv = np.where(R & upper)
    n = rng.poisson(0.9, len(ru))                          # curiosity views (legit signal)
    for a, b in ((ru, rv),):
        rep_u, rep_v = np.repeat(a, n), np.repeat(b, n)
        flip = rng.random(len(rep_u)) < 0.5
        vu.append(np.where(flip, rep_v, rep_u)); vv.append(np.where(flip, rep_u, rep_v))
        vt.append(rng.random(len(rep_u)))
    eu, ev = np.where(np.isfinite(ET) & upper)
    n = rng.poisson(3, len(eu))                            # post-friendship views (leak!)
    rep_u, rep_v = np.repeat(eu, n), np.repeat(ev, n)
    tpost = np.repeat(ET[eu, ev], n) + rng.exponential(0.06, len(rep_u))
    keep = tpost < 1
    vu.append(rep_u[keep]); vv.append(rep_v[keep]); vt.append(tpost[keep])
    deg_final = A.sum(1)
    pv = (deg_final + 1) ** 1.5
    vu.append(rng.integers(0, N, 8000)); vv.append(rng.choice(N, 8000, p=pv / pv.sum()))
    vt.append(rng.random(8000))                            # noise: people browse hubs
    vu, vv, vt = np.concatenate(vu), np.concatenate(vv), np.concatenate(vt)
    ok = vu != vv

    return dict(comm=comm, school=school, employer=employer, city=city, age=age, celeb=celeb,
                joined=joined, cold=np.sort(cold), R=R, ET=ET, csym=csym, upper=upper,
                same_school=same_school, same_emp=same_emp, same_city=same_city, age_gap=age_gap,
                attr_match=same_school.astype(np.int8) + same_emp + same_city,
                views=(vu[ok], vv[ok], vt[ok]))


# --------------------------------------------------------------------------------------
# 2. Point-in-time snapshot.  Concept: every feature is computed "as of" time t, using only
#    edges/events with timestamp <= t.  This is what a feature store with time travel gives you.
# --------------------------------------------------------------------------------------
class Snap:
    def __init__(self, W, t):
        self.t = t
        self.A = (W["ET"] <= t).astype(np.float32)
        self.refresh_graph()
        vu, vv, vt = W["views"]
        m = vt <= t
        last = np.full((N, N), -np.inf)
        np.maximum.at(last, (vu[m], vv[m]), vt[m])
        last = np.maximum(last, last.T)
        self.rec = np.where(np.isfinite(last), np.exp(-(t - last) / VIEW_TAU), 0.0).astype(np.float32)
        self.E = _embed(self.A)
        self.COS = self.E @ self.E.T

    def refresh_graph(self):
        A = self.A
        self.deg = A.sum(1)
        self.CN = A @ A                                          # common neighbours (2-hop paths)
        self.AA = (A / np.log(np.maximum(self.deg, 2.0))[None, :]) @ A   # Adamic-Adar
        d = self.deg[:, None] + self.deg[None, :] - self.CN
        self.JAC = self.CN / np.maximum(d, 1.0)


def _embed(A):
    """Spectral embedding: truncated SVD of D^-1/2 A D^-1/2, rows L2-normalised.
    Stand-in for node2vec / GraphSAGE / two-tower embeddings used in production."""
    deg = A.sum(1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1)), 0.0)
    U, S, _ = randomized_svd(A * dinv[:, None] * dinv[None, :], EMB_DIM, n_iter=3, random_state=0)
    E = U * np.sqrt(S)[None, :]
    E[deg == 0] = 0.0                                            # cold-start: no embedding
    nrm = np.linalg.norm(E, axis=1, keepdims=True)
    return (E / np.maximum(nrm, 1e-9)).astype(np.float32)


def _feats(W, S, U, V, drop_self_edge=False):
    """Pairwise feature matrix for pairs (U[i], V[i]) as of snapshot S (symmetric in u,v)."""
    cn = S.CN[U, V]
    du, dv = S.deg[U].copy(), S.deg[V].copy()
    if drop_self_edge:        # leaky pipeline "carefully" removes the target edge itself
        e = S.A[U, V]
        du, dv = du - e, dv - e
    jac = cn / np.maximum(du + dv - cn, 1.0)
    return np.column_stack([
        cn, S.AA[U, V], jac,
        W["same_school"][U, V], W["same_emp"][U, V], W["same_city"][U, V],
        W["csym"][U, V], S.rec[U, V], W["age_gap"][U, V],
        np.minimum(du, dv), np.maximum(du, dv), S.COS[U, V],
    ]).astype(np.float32)


# --------------------------------------------------------------------------------------
# 3. Training data.  Concept: the ranker is trained on the distribution it will SERVE
#    (the candidate pool), positives = edges formed in the label window AFTER the snapshot,
#    negatives downsampled, probabilities re-calibrated with the prior-correction formula.
# --------------------------------------------------------------------------------------
def _pool_mask(W, S):
    j = W["joined"] <= S.t
    return ((S.CN >= 1) | (W["csym"] >= 1) | (W["attr_match"] >= 2)) & W["upper"] & j[:, None] & j[None, :]


def _dataset(W, S, t_lo, t_hi, rng, leaky=False):
    pool = _pool_mask(W, S)
    lab = (W["ET"] > t_lo) & (W["ET"] <= t_hi)
    if leaky:      # features at the END of the label window: pair may already be connected
        pos = pool & lab
        neg = pool & (S.A == 0)
    else:          # correct: only pairs not yet connected at the snapshot
        pos = pool & lab & (S.A == 0)
        neg = pool & (S.A == 0) & ~lab
    pu, pv = np.where(pos)
    nu, nv = np.where(neg)
    take = rng.choice(len(nu), min(len(nu), NEG_PER_POS * len(pu)), replace=False)
    w = len(take) / max(len(nu), 1)                              # negative sampling rate
    U = np.concatenate([pu, nu[take]]); V = np.concatenate([pv, nv[take]])
    y = np.concatenate([np.ones(len(pu)), np.zeros(len(take))])
    X = _feats(W, S, U, V, drop_self_edge=leaky)
    return X, y, w, dict(pool_pairs=int(pool.sum()), positives=int(len(pu)), negatives_in_pool=int(len(nu)),
                         negatives_sampled=int(len(take)), base_rate=float(len(pu) / max(len(pu) + len(nu), 1)))


def _gbdt():
    return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=0)


def _calibrate(p, w):
    """Prior correction after negative downsampling at rate w:  p' = p / (p + (1-p)/w)."""
    return p / (p + (1.0 - p) / w)


def _perm_importance(model, X, y, rng):
    """Permutation importance: AUC drop when one feature column is shuffled."""
    base = roc_auc_score(y, model.decision_function(X))
    out = {}
    for j, name in enumerate(FNAMES):
        Xp = X.copy()
        Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
        out[name] = float(base - roc_auc_score(y, model.decision_function(Xp)))
    return out


def _metrics(model, X, y):
    s = model.decision_function(X)
    return {"auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}


# --------------------------------------------------------------------------------------
# Build everything once (lazy, ~2 s)
# --------------------------------------------------------------------------------------
def _build():
    t_start = time.perf_counter()
    W = _simulate()
    S0, S1, S2 = Snap(W, T0), Snap(W, T1), Snap(W, T2)
    rng = np.random.default_rng(11)
    Xtr, ytr, w_tr, info_tr = _dataset(W, S0, T0, T1, rng)
    Xva, yva, _, info_va = _dataset(W, S1, T1, T2, rng)
    Xpr, ypr, _, info_pr = _dataset(W, S2, T2, T3, rng)
    model = _gbdt().fit(Xtr, ytr)

    hidden = (W["ET"] > T2) & (W["ET"] <= T3)                    # ground truth to be discovered
    # Safety: block list.  Pairs with several mutual friends who must NEVER be suggested
    # (think: harasser / ex-partner).  Enforced as a hard filter before ranking.
    cu, cv = np.where((S2.CN >= 3) & W["upper"] & ~np.isfinite(W["ET"]))
    pick = rng.choice(len(cu), min(600, len(cu)), replace=False)
    blocked = np.zeros((N, N), bool)
    blocked[cu[pick], cv[pick]] = True
    blocked = blocked | blocked.T

    # attribute / contact fallback generator score (works with zero friends)
    att = 3.0 * W["csym"] + W["attr_match"] + 0.5 * np.exp(-W["age_gap"] / 5.0)
    att = np.where((W["csym"] >= 1) | (W["attr_match"] >= 2), att, 0.0).astype(np.float32)

    # IVF index for the ANN demo: k-means cells, search only the nprobe nearest cells
    warm = np.where(S2.deg > 0)[0]
    km = KMeans(N_CELLS, n_init=2, random_state=0).fit(S2.E[warm])
    cell = np.full(N, -1)
    cell[warm] = km.labels_

    hub_deg = float(np.percentile(S2.deg, HUB_PCTL))
    st = dict(W=W, S0=S0, S1=S1, S2=S2, model=model, w=w_tr, hidden=hidden, blocked=blocked, att=att,
              centroids=km.cluster_centers_.astype(np.float32), cell=cell,
              members=[np.where(cell == c)[0] for c in range(N_CELLS)],
              med=np.median(Xtr, axis=0), hub_deg=hub_deg, hubs=S2.deg >= hub_deg,
              data=dict(train=(Xtr, ytr, info_tr), val=(Xva, yva, info_va), prod=(Xpr, ypr, info_pr)),
              cache={})
    st["metrics"] = {"train": _metrics(model, Xtr, ytr), "validation_time_split": _metrics(model, Xva, yva),
                     "production": _metrics(model, Xpr, ypr)}
    st["importance"] = _perm_importance(model, Xpr, ypr, np.random.default_rng(3))
    st["build_seconds"] = time.perf_counter() - t_start
    return st


def _st():
    global _STATE
    if _STATE is None:
        with _LOCK:
            if _STATE is None:
                _STATE = _build()
    return _STATE


# --------------------------------------------------------------------------------------
# 4. Candidate generation + ranking for all users at once (used for aggregate recall and
#    hub-bias stats).  Concept: retrieval optimises RECALL cheaply; the ranker optimises
#    precision on a few hundred candidates.
# --------------------------------------------------------------------------------------
GENS = ["fof_mutual_friends", "fof_adamic_adar", "fof_jaccard", "embedding_cosine", "contacts_attributes"]


def _gen_scores(st, S):
    return {"fof_mutual_friends": S.CN, "fof_adamic_adar": S.AA, "fof_jaccard": S.JAC,
            "embedding_cosine": np.where(S.CN >= 0, S.COS, 0), "contacts_attributes": st["att"]}


def _hub_discount(st, deg_v):
    """Celebrity cap as a soft score discount: sqrt(hub_deg / deg) for candidates above the cap."""
    return np.minimum(1.0, np.sqrt(st["hub_deg"] / np.maximum(deg_v, 1.0)))


def _topk_mask(score, excl, k):
    s = np.where(excl | (score <= 0), -np.inf, score)
    idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
    m = np.zeros(s.shape, bool)
    rows = np.repeat(np.arange(s.shape[0]), k)
    m[rows, idx.ravel()] = True
    return m & np.isfinite(s)


def _stage_eval(st, per_gen=50, k=10):
    key = ("stage", per_gen, k)
    if key in st["cache"]:
        return st["cache"][key]
    W, S = st["W"], st["S2"]
    hidden = st["hidden"] & ~st["blocked"]
    excl = (S.A > 0) | np.eye(N, dtype=bool) | st["blocked"]
    masks = {g: _topk_mask(sc, excl, per_gen) for g, sc in _gen_scores(st, S).items()}
    union = np.zeros((N, N), bool)
    for m in masks.values():
        union |= m
    U, V = np.where(union)
    p = _calibrate(st["model"].predict_proba(_feats(W, S, U, V))[:, 1], st["w"])
    P = np.full((N, N), -1.0)
    P[U, V] = p
    Pd = P.copy()
    Pd[U, V] = p * _hub_discount(st, S.deg[V])
    top = lambda M: np.argsort(-M, axis=1)[:, :k]
    nh = hidden.sum(1)
    is_cold = W["joined"] > 0
    groups = {"warm users": (nh > 0) & ~is_cold, "cold-start users": (nh > 0) & is_cold}

    def rec(mask, sel):
        return float(((mask & hidden).sum(1)[sel] / nh[sel]).mean()) if sel.any() else 0.0

    def topmask(idx, M):
        m = np.zeros((N, N), bool)
        m[np.repeat(np.arange(N), k), idx.ravel()] = True
        return m & (M > -1)

    final, final_d = topmask(top(P), P), topmask(top(Pd), Pd)
    naive = _topk_mask(S.CN, excl, k)
    rows = []
    for g in GENS:
        rows.append({"stage": f"retrieval: {g} (top {per_gen})", "avg_candidates": float(masks[g].sum(1).mean()),
                     **{f"recall ({n})": rec(masks[g], sel) for n, sel in groups.items()}})
    rows.append({"stage": "union of generators", "avg_candidates": float(union.sum(1).mean()),
                 **{f"recall ({n})": rec(union, sel) for n, sel in groups.items()}})
    rows.append({"stage": f"baseline: top {k} by mutual friends", "avg_candidates": float(k),
                 **{f"recall ({n})": rec(naive, sel) for n, sel in groups.items()}})
    rows.append({"stage": f"GBDT ranker (top {k})", "avg_candidates": float(k),
                 **{f"recall ({n})": rec(final, sel) for n, sel in groups.items()}})
    rows.append({"stage": f"GBDT + hub discount (top {k})", "avg_candidates": float(k),
                 **{f"recall ({n})": rec(final_d, sel) for n, sel in groups.items()}})

    # hub bias: how many of the top-k slots go to the top-5% most connected users?
    hubs = st["hubs"]
    share = lambda m: float(m[:, hubs].sum() / max(m.sum(), 1))
    hub = {"hub_definition": f"top {100 - HUB_PCTL}% by friend count (degree >= {st['hub_deg']:.0f})",
           "share_of_true_future_friendships": float(hidden[:, hubs].sum() / max(hidden.sum(), 1)),
           "share_of_slots_mutual_friend_ranking": share(naive),
           "share_of_slots_gbdt": share(final),
           "share_of_slots_gbdt_hub_discount": share(final_d),
           "share_of_population": float(hubs.mean())}
    out = {"per_gen": per_gen, "k": k, "rows": rows, "hub_bias": hub,
           "eval_users": {n: int(sel.sum()) for n, sel in groups.items()}}
    st["cache"][key] = out
    return out


# --------------------------------------------------------------------------------------
# Endpoint: overview
# --------------------------------------------------------------------------------------
@router.get("/overview")
def overview():
    st = _st()
    W, S = st["W"], st["S2"]
    nh = (st["hidden"] & ~st["blocked"]).sum(1)
    warm = np.where((W["joined"] == 0) & (nh >= 3) & ~st["hubs"])[0]
    hubs = np.argsort(-S.deg)[:8]
    cold = [int(u) for u in W["cold"] if nh[u] >= 1][:8]
    rng = np.random.default_rng(5)
    stage = _stage_eval(st)
    return {
        "dataset": {
            "users": N, "communities": NC,
            "friendships_at_serving_time": int(S.A.sum() / 2),
            "hidden_future_friendships": int(st["hidden"].sum() / 2),
            "avg_degree": float(S.deg.mean()), "median_degree": float(np.median(S.deg)),
            "max_degree": float(S.deg.max()),
            "cold_start_users": int((W["joined"] > 0).sum()),
            "blocked_pairs": int(st["blocked"].sum() / 2),
            "real_world_ties": int(W["R"].sum() / 2),
            "profile_view_events": int(len(W["views"][0])),
        },
        "timeline": {"T0": T0, "T1": T1, "T2": T2, "T3": T3,
                     "train": "features@T0 -> label = friendship formed in (T0,T1]",
                     "validation": "features@T1 -> label in (T1,T2]",
                     "production": "features@T2 -> label in (T2,T3] (hidden from the system)"},
        "model": {
            "type": "sklearn HistGradientBoostingClassifier (120 trees, 15 leaves, lr 0.08)",
            "training_rows": int(len(st["data"]["train"][1])),
            "train_info": st["data"]["train"][2],
            "negative_sampling_rate_w": st["w"],
            "calibration": "p' = p / (p + (1-p)/w)",
            "metrics": st["metrics"],
            "feature_importance": st["importance"],
        },
        "features": [{"name": n, "description": d} for n, d in FEATURES],
        "stage_recall": stage,
        "sample_users": {"typical": [int(u) for u in rng.choice(warm, min(8, len(warm)), replace=False)],
                         "hubs": [int(u) for u in hubs], "cold_start": cold},
        "build_seconds": st["build_seconds"],
    }


# --------------------------------------------------------------------------------------
# Endpoint: recommend for one user (the serving path, with per-stage internals)
# --------------------------------------------------------------------------------------
class RecommendReq(BaseModel):
    user_id: int = 0
    k: int = Field(10, ge=1, le=50)
    per_gen: int = Field(50, ge=5, le=300)
    hub_discount: bool = False                 # celebrity cap (soft)
    dismissed: List[int] = []                  # negative feedback: "X" clicks, never show again
    nprobe: Optional[int] = None               # None = brute-force cosine; else IVF with nprobe cells


def _ivf_search(st, q, nprobe):
    """Hand-written IVF: scan only members of the nprobe cells whose centroid is closest to q."""
    cells = np.argsort(-(st["centroids"] @ q))[:nprobe]
    ids = np.concatenate([st["members"][c] for c in cells]) if len(cells) else np.array([], int)
    return ids, st["S2"].E[ids] @ q


@router.post("/recommend")
def recommend(req: RecommendReq):
    st = _st()
    if not 0 <= req.user_id < N:
        raise HTTPException(400, f"user_id must be in 0..{N - 1}")
    W, S, u = st["W"], st["S2"], req.user_id
    t_all = time.perf_counter()
    dismissed = np.zeros(N, bool)
    dismissed[[d for d in req.dismissed if 0 <= d < N]] = True
    friends = S.A[u] > 0
    excl = friends | dismissed | st["blocked"][u]
    excl[u] = True
    hidden = np.where(st["hidden"][u] & ~st["blocked"][u])[0]
    hidden_set = set(int(h) for h in hidden)
    cold = bool(S.deg[u] == 0)

    # ---- stage 1: candidate generators -------------------------------------------------
    gens, sources, blocked_caught = [], {}, set()
    for g in GENS:
        t0 = time.perf_counter()
        scanned = N
        if g == "embedding_cosine" and req.nprobe and not cold:
            ids, sc = _ivf_search(st, S.E[u], min(req.nprobe, N_CELLS))
            s = np.zeros(N, np.float32)
            s[ids] = sc
            scanned = int(len(ids))
        else:
            s = _gen_scores(st, S)[g][u].astype(np.float32)
        raw = np.where(friends | (np.arange(N) == u) | (s <= 0), -np.inf, s)
        order_raw = np.argsort(-raw)[:req.per_gen]
        blocked_caught |= {int(v) for v in order_raw if np.isfinite(raw[v]) and st["blocked"][u, v]}
        s = np.where(excl | (s <= 0), -np.inf, s)
        order = [int(v) for v in np.argsort(-s)[:req.per_gen] if np.isfinite(s[v])]
        ms = (time.perf_counter() - t0) * 1000
        for v in order:
            sources.setdefault(v, []).append(g)
        hit = len(hidden_set & set(order))
        gens.append({"name": g, "n_candidates": len(order), "vectors_scanned": scanned,
                     "hidden_found": hit, "recall": hit / len(hidden) if len(hidden) else None,
                     "latency_ms": round(ms, 3),
                     "top": [{"id": v, "score": float(s[v]), "is_hidden_friend": v in hidden_set} for v in order[:8]]})

    # ---- stage 2: feature assembly + GBDT ranker -----------------------------------------
    cand = np.array(sorted(sources), dtype=int)
    ranked, t_feat, t_rank = [], 0.0, 0.0
    if len(cand):
        t0 = time.perf_counter()
        X = _feats(W, S, np.full(len(cand), u), cand)
        t_feat = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        logit = st["model"].decision_function(X)
        p = _calibrate(_sigmoid(logit), st["w"])
        disc = _hub_discount(st, S.deg[cand]) if req.hub_discount else np.ones(len(cand))
        final = p * disc
        order = np.argsort(-final)[:req.k]
        # Occlusion contributions: change in logit when ONE feature is replaced by its
        # training-set median (cheap local explanation; TreeSHAP in production).
        Xo = np.repeat(X[order], len(FNAMES), axis=0)
        for j in range(len(FNAMES)):
            Xo[j::len(FNAMES), j] = st["med"][j]
        contrib = logit[order][:, None] - st["model"].decision_function(Xo).reshape(len(order), len(FNAMES))
        t_rank = (time.perf_counter() - t0) * 1000
        for r, i in enumerate(order):
            v = int(cand[i])
            ranked.append({
                "rank": r + 1, "id": v, "accept_prob": float(p[i]), "final_score": float(final[i]),
                "hub_discount": float(disc[i]), "is_hub": bool(st["hubs"][v]),
                "is_hidden_friend": v in hidden_set, "sources": sources[v],
                "degree": int(S.deg[v]), "community": int(W["comm"][v]),
                "features": {n: float(X[i, j]) for j, n in enumerate(FNAMES)},
                "contributions": {n: float(contrib[r, j]) for j, n in enumerate(FNAMES)},
            })
        rank_of = {int(cand[i]): r + 1 for r, i in enumerate(np.argsort(-final))}
    else:
        rank_of = {}

    union_hit = len(hidden_set & set(int(c) for c in cand))
    topk_hit = sum(1 for r in ranked if r["is_hidden_friend"])
    nh = len(hidden)
    stage_recall = [{"stage": g["name"], "recall": g["recall"] or 0.0, "n": g["n_candidates"]} for g in gens]
    stage_recall.append({"stage": "union", "recall": union_hit / nh if nh else 0.0, "n": int(len(cand))})
    stage_recall.append({"stage": f"ranker top {req.k}", "recall": topk_hit / nh if nh else 0.0, "n": len(ranked)})
    return {
        "user": {"id": u, "community": int(W["comm"][u]), "school": int(W["school"][u]),
                 "employer": int(W["employer"][u]), "city": int(W["city"][u]), "age": int(W["age"][u]),
                 "friends": int(S.deg[u]), "is_cold_start": cold, "is_hub": bool(st["hubs"][u]),
                 "uploaded_contacts_matches": int((W["csym"][u] > 0).sum()),
                 "hidden_future_friends": nh},
        "generators": gens,
        "filters": {"blocked_removed": len(blocked_caught), "dismissed_removed": int(dismissed.sum()),
                    "already_friends_removed": int(friends.sum())},
        "union_size": int(len(cand)),
        "stage_recall": stage_recall,
        "ranked": ranked,
        "hidden_friend_ranks": [{"id": h, "rank_in_union": rank_of.get(int(h))} for h in hidden.tolist()],
        "latency_ms": {"retrieval": round(sum(g["latency_ms"] for g in gens), 3), "features": round(t_feat, 3),
                       "ranking_and_explanations": round(t_rank, 3),
                       "total": round((time.perf_counter() - t_all) * 1000, 3)},
        "cold_start_note": ("User has no friends: FoF and embedding generators return nothing; "
                            "candidates come from the contacts/attributes fallback.") if cold else None,
    }


# --------------------------------------------------------------------------------------
# Endpoint: ANN trade-off.  Concept: at 1B users you cannot brute-force cosine; an IVF/HNSW
# index trades a little recall for scanning a tiny fraction of the vectors.
# --------------------------------------------------------------------------------------
@router.get("/ann_tradeoff")
def ann_tradeoff(k: int = 20):
    st = _st()
    k = int(min(max(k, 1), 100))
    key = ("ann", k)
    if key not in st["cache"]:
        S = st["S2"]
        warm = np.where(S.deg > 0)[0]
        cos = S.COS[warm].copy()
        cos[np.arange(len(warm)), warm] = -np.inf
        cos[:, S.deg == 0] = -np.inf
        exact = np.argsort(-cos, axis=1)[:, :k]
        cell_rank = np.argsort(-(S.E[warm] @ st["centroids"].T), axis=1)
        rows = []
        for nprobe in (1, 2, 4, 8, 16, N_CELLS):
            rec, scanned = 0.0, 0
            t0 = time.perf_counter()
            for i, u in enumerate(warm):
                ids = np.concatenate([st["members"][c] for c in cell_rank[i, :nprobe]])
                ids = ids[ids != u]
                top = ids[np.argsort(-(S.E[ids] @ S.E[u]))[:k]]
                rec += len(np.intersect1d(top, exact[i])) / k
                scanned += len(ids)
            rows.append({"nprobe": nprobe, "recall_vs_exact": rec / len(warm),
                         "fraction_scanned": scanned / (len(warm) * (len(warm) - 1)),
                         "ms_per_query": (time.perf_counter() - t0) * 1000 / len(warm)})
        st["cache"][key] = {"k": k, "cells": N_CELLS, "vectors": int(len(warm)), "dim": EMB_DIM, "rows": rows}
    return st["cache"][key]


# --------------------------------------------------------------------------------------
# 5. PITFALL: label leakage.  "random" = wrong way: take today's graph, label edges created
#    in the last window as positives, compute features from TODAY's tables (after the edge
#    exists) and split rows randomly.  "time" = right way: features strictly before labels,
#    validate on a later window.  Both are then scored on the same future (T2,T3].
# --------------------------------------------------------------------------------------
class LeakReq(BaseModel):
    split: str = "random"      # "random" (leaky) | "time" (correct)


@router.post("/leakage")
def leakage(req: LeakReq):
    st = _st()
    if req.split not in ("random", "time"):
        raise HTTPException(400, "split must be 'random' or 'time'")
    key = ("leak", req.split)
    if key in st["cache"]:
        return st["cache"][key]
    Xpr, ypr, _ = st["data"]["prod"]
    if req.split == "time":
        Xva, yva, _ = st["data"]["val"]
        offline, model, info = _metrics(st["model"], Xva, yva), st["model"], st["data"]["train"][2]
        imp = _perm_importance(model, Xva, yva, np.random.default_rng(3))
        how = ("Features as of T0, labels = friendships formed in (T0,T1]. Offline metric measured on a LATER "
               "window: features as of T1, labels in (T1,T2].")
    else:
        rng = np.random.default_rng(21)
        X, y, _, info = _dataset(st["W"], st["S2"], T0, T2, rng, leaky=True)
        idx = rng.permutation(len(y))
        cut = int(0.8 * len(y))
        model = _gbdt().fit(X[idx[:cut]], y[idx[:cut]])
        offline = _metrics(model, X[idx[cut:]], y[idx[cut:]])
        imp = _perm_importance(model, X[idx[cut:]], y[idx[cut:]], np.random.default_rng(3))
        how = ("Features computed from TODAY's tables (as of T2) for friendships formed in (T0,T2], target edge "
               "removed from the graph, rows split 80/20 at random. Profile views, triangles and embeddings "
               "created AFTER the friendship leak the label.")
    prod = _metrics(model, Xpr, ypr)
    out = {"split": req.split, "how": how, "offline": offline, "production": prod,
           "offline_minus_production_auc": offline["auc"] - prod["auc"],
           "offline_minus_production_pr_auc": offline["pr_auc"] - prod["pr_auc"],
           "feature_importance_offline": imp, "data": info,
           "verdict": ("Offline number is trustworthy: it predicts production." if req.split == "time" else
                       "Offline number is inflated by leakage: the model learned consequences of friendship, "
                       "not causes.")}
    st["cache"][key] = out
    return out


# --------------------------------------------------------------------------------------
# 6. PITFALL: popularity / hub bias feedback loop, and the mitigation.
#    Each round every user sees k suggestions.  A request is SENT with a probability that
#    depends on the real-world tie (and popularity attraction), and ACCEPTED by the other
#    side with a probability that drops when that person is flooded with requests
#    (two-sided marketplace).  Accepted edges change the graph -> more mutual friends for
#    hubs -> hubs get suggested even more.
#    Mitigation = per-candidate impression cap + hub score discount + dismiss memory.
# --------------------------------------------------------------------------------------
class LoopReq(BaseModel):
    mitigate: bool = False
    rounds: int = Field(5, ge=1, le=8)
    k: int = Field(5, ge=1, le=10)
    cap: int = Field(12, ge=1, le=200)         # max impressions per candidate per round


def _gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() > 0 else 0.0


@router.post("/feedback_loop")
def feedback_loop(req: LoopReq):
    st = _st()
    key = ("loop", req.mitigate, req.rounds, req.k, req.cap)
    if key in st["cache"]:
        return st["cache"][key]
    W, S2 = st["W"], st["S2"]
    rng = np.random.default_rng(99)
    S = Snap.__new__(Snap)                     # mutable copy of the serving snapshot
    S.t, S.A, S.rec, S.E, S.COS = S2.t, S2.A.copy(), S2.rec, S2.E, S2.COS   # embeddings stay stale (daily refresh)
    hubs = st["hubs"]
    deg0 = S2.deg
    pop = np.minimum(1.0, deg0 / np.percentile(deg0, 99))
    shown_before = np.zeros((N, N), bool)
    dismissed = np.zeros((N, N), bool)
    M = 30
    rows = []
    for r in range(1, req.rounds + 1):
        S.refresh_graph()
        excl = (S.A > 0) | np.eye(N, dtype=bool) | st["blocked"] | dismissed
        s = np.where(excl | (S.CN <= 0), -np.inf, S.CN + 1e-3 * S.AA)
        cand = np.argpartition(-s, M - 1, axis=1)[:, :M]
        U = np.repeat(np.arange(N), M); V = cand.ravel()
        valid = np.isfinite(s[U, V])
        p = np.zeros(len(U))
        p[valid] = _calibrate(st["model"].predict_proba(_feats(W, S, U[valid], V[valid]))[:, 1], st["w"])
        if req.mitigate:
            p = p * _hub_discount(st, deg0[V])
        p[~valid] = -1
        P = p.reshape(N, M)
        order = np.argsort(-P, axis=1)
        imp = np.zeros(N, int)
        su, sv = [], []
        for u in rng.permutation(N):
            n_shown = 0
            for j in order[u]:
                if P[u, j] < 0 or n_shown >= req.k:
                    break
                v = cand[u, j]
                if req.mitigate and imp[v] >= req.cap:
                    continue                   # celebrity cap: this person has had enough exposure
                imp[v] += 1; n_shown += 1
                su.append(u); sv.append(v)
        su, sv = np.array(su, int), np.array(sv, int)
        tie = W["R"][su, sv]
        sent = rng.random(len(su)) < np.where(tie, 0.55, 0.03 + 0.20 * pop[sv])
        recv = np.bincount(sv[sent], minlength=N)
        fatigue = 1.0 / (1.0 + np.maximum(recv[sv] - 3, 0) / 5.0)
        acc = sent & (rng.random(len(su)) < np.where(tie, 0.9, 0.25) * fatigue)
        repeat = shown_before[su, sv]
        shown_before[su, sv] = True
        if req.mitigate:                       # negative feedback: some ignored suggestions get an explicit X
            dis = ~sent & (rng.random(len(su)) < 0.4)
            dismissed[su[dis], sv[dis]] = True
        S.A[su[acc], sv[acc]] = 1; S.A[sv[acc], su[acc]] = 1
        n_imp = max(len(su), 1)
        rows.append({
            "round": r, "impressions": int(len(su)),
            "hub_impression_share": float(imp[hubs].sum() / n_imp),
            "gini_impressions": _gini(imp),
            "max_impressions_one_person": int(imp.max()),
            "users_never_suggested_pct": float((imp == 0).mean() * 100),
            "requests_sent": int(sent.sum()),
            "request_accept_rate": float(acc.sum() / max(sent.sum(), 1)),
            "friendships_per_100_impressions": float(100 * acc.sum() / n_imp),
            "new_friendships": int(acc.sum()),
            "new_friendships_real_tie_pct": float(100 * (acc & tie).sum() / max(acc.sum(), 1)),
            "repeat_impression_pct": float(100 * repeat.sum() / n_imp),
            "hub_avg_requests_received": float(recv[hubs].mean()),
        })
    tot = lambda f: sum(x[f] for x in rows)
    out = {"mitigate": req.mitigate, "rounds": rows,
           "policy": ("GBDT score x hub discount, <= %d impressions per person per round, dismissed pairs never "
                      "re-shown" % req.cap) if req.mitigate else "GBDT score only, no caps, no dismiss memory",
           "summary": {"total_new_friendships": int(tot("new_friendships")),
                       "total_impressions": int(tot("impressions")),
                       "friendships_per_100_impressions": float(100 * tot("new_friendships") / max(tot("impressions"), 1)),
                       "hub_impression_share_first": rows[0]["hub_impression_share"],
                       "hub_impression_share_last": rows[-1]["hub_impression_share"],
                       "hub_share_of_population": float(hubs.mean()),
                       "final_degree_gini": _gini(S.A.sum(1)),
                       "start_degree_gini": _gini(deg0)}}
    st["cache"][key] = out
    return out
