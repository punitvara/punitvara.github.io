"""Account Bot Detection - "Design a system to detect bot / fake accounts on a social platform."

What this module teaches (each section below is tagged with the interview concept):
  1. DATA      - synthetic accounts of 4 kinds (human / naive bot / sophisticated bot / coordinated ring)
                 at a ~1% base rate.  Raw behaviour is simulated (inter-post gaps, post token sets,
                 device/IP edges) and features are *computed* from it (entropy, burstiness, MinHash, union-find).
  2. MODELS    - rules baseline vs logistic regression vs GBDT vs IsolationForest (no labels)
                 vs graph ring detection (hand-written union-find) vs a noisy-OR ensemble.
  3. IMBALANCE - accuracy trap, PR curve / PR-AUC, precision@K (review-queue capacity).
  4. DECISIONS - cost-sensitive threshold + tiered response (allow / CAPTCHA / human review / ban)
                 derived from expected cost on calibrated probabilities.
  5. ADVERSARY - "bot operators adapt" knob: model decay, and what retraining / hard-to-fake
                 features / unsupervised ensembles buy back.
  6. PITFALL   - label noise + selection bias: training on rule-generated or review-queue-only labels
                 relearns the rules; random audit + exploration fixes training AND prevalence estimates.
"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

META = {
    "slug": "bot-detection",
    "title": "Account Bot Detection",
    "tagline": "1% base rate, adversaries that adapt, rings found by union-find, and cost-based tiered enforcement",
    "icon": "\U0001F916",
    "order": 7,
}
router = APIRouter()

TYPES = ["human", "naive_bot", "sophisticated_bot", "ring_bot"]
N_PERIOD = 12000        # accounts per snapshot (train = month t, test = month t+1)
PREVALENCE = 0.01       # 1% bots -> the whole problem is the imbalance
N_POSTS, N_TOK, N_HASH, N_GAPS, VOCAB = 8, 12, 16, 30, 4000
HASH_MOD = 1048573      # prime just under 2^20 for the MinHash permutations
IP_DEGREE_CAP = 20      # IPs shared by more accounts than this are NAT hubs, not evidence
QUEUE_K = 150           # default human-review capacity per period

# ---------------------------------------------------------------------------------------------
# FEATURE CATALOGUE (interview concept: feature engineering; which signals exist at which stage,
# and how expensive each is for an attacker to fake - cheap-to-fake features decay first).
# ---------------------------------------------------------------------------------------------
FEATURES = [
    # name, group, log-transform?, stage, cost to fake, description
    ("account_age_days", "profile", True, "signup", "medium", "Days since signup (aged accounts can be bought)"),
    ("follow_ratio_log", "profile", False, "batch", "low", "log10((following+1)/(followers+1)) - bots follow many, are followed by few"),
    ("profile_completeness", "profile", False, "signup", "low", "Share of profile fields filled (bio, avatar, links)"),
    ("posts_per_day", "temporal", True, "action", "low", "Posting rate"),
    ("interval_entropy", "temporal", False, "action", "medium", "Normalised entropy of the histogram of log inter-post gaps (0 = clockwork)"),
    ("burstiness", "temporal", False, "action", "medium", "B = (sigma - mu)/(sigma + mu) of inter-post gaps; -1 periodic, 0 Poisson, >0 bursty (human)"),
    ("night_frac", "temporal", False, "action", "medium", "Share of activity between 01:00-06:00 local time (humans sleep)"),
    ("dup_ratio", "content", False, "action", "low", "Share of own post pairs with MinHash Jaccard >= 0.5 (self near-duplicates)"),
    ("shared_text_frac", "content", False, "batch", "medium", "Share of posts whose LSH bucket is shared with >= 2 other accounts (campaign templates)"),
    ("reciprocity", "network", False, "batch", "high", "Share of outgoing follows that follow back - needs real humans to cooperate"),
    ("actions_per_min", "session", True, "action", "low", "Median actions per minute inside a session"),
    ("device_share", "network", True, "login", "high", "Accounts seen on the same device fingerprint"),
    ("ip_share", "network", True, "login", "medium", "Max accounts seen on any of this account's IPs (NAT inflates it)"),
    # graph features: need the whole graph -> batch layer
    ("comp_size", "graph", True, "batch", "high", "Size of the connected component over shared devices / non-hub IPs"),
    ("comp_age_spread", "graph", True, "batch", "high", "Std-dev of account age inside the component (rings sign up together); 999 = singleton"),
]
FNAMES = [f[0] for f in FEATURES]
N_BEH = 13                              # first 13 = per-account behaviour; last 2 = graph
LOG_COLS = np.array([f[2] for f in FEATURES])


def _lerp(a, b, t):
    return a + (b - a) * t


# ---------------------------------------------------------------------------------------------
# GRAPH (interview concept: coordinated-ring detection).  Union-find written by hand: every device
# or IP shared by several accounts unions them.  Hub IPs (university / carrier NAT) must be
# dropped or thousands of humans collapse into one "ring".
# ---------------------------------------------------------------------------------------------
def union_find(n: int, groups: list, caps: list):
    """groups: list of (member-lists, is_ip) ; returns (root label per node, #entities skipped as hubs)."""
    parent = list(range(n))
    size = [1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]      # path halving
            x = parent[x]
        return x

    skipped = 0
    for members_list, cap in zip(groups, caps):
        for members in members_list:
            if len(members) < 2:
                continue
            if cap is not None and len(members) > cap:
                skipped += 1
                continue
            r = find(members[0])
            for m in members[1:]:
                q = find(m)
                if q == r:
                    continue
                if size[q] > size[r]:           # union by size
                    r, q = q, r
                parent[q] = r
                size[r] += size[q]
    return np.array([find(i) for i in range(n)]), skipped


def component_stats(labels, age):
    _, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    s1 = np.bincount(inv, weights=age)
    s2 = np.bincount(inv, weights=age * age)
    var = np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0)
    spread = np.where(cnt > 1, np.sqrt(var), 999.0)
    return inv, cnt[inv].astype(float), spread[inv]


# ---------------------------------------------------------------------------------------------
# CONTENT (interview concept: near-duplicate detection at scale).  MinHash signatures per post;
# within-account pairs give dup_ratio, LSH banding (8 bands x 2 rows) gives cross-account sharing.
# ---------------------------------------------------------------------------------------------
_HRNG = np.random.default_rng(1234)
_HA = _HRNG.integers(1, HASH_MOD, N_HASH)
_HB = _HRNG.integers(0, HASH_MOD, N_HASH)


def minhash_features(tok):
    n = tok.shape[0]
    sig = np.empty((n, N_POSTS, N_HASH), dtype=np.int64)
    for h in range(N_HASH):
        sig[:, :, h] = ((tok * _HA[h] + _HB[h]) % HASH_MOD).min(axis=2)
    # self near-duplicates: estimated Jaccard = share of equal signature rows
    dup, pairs = np.zeros(n), 0
    for i in range(N_POSTS):
        for j in range(i + 1, N_POSTS):
            dup += (sig[:, i] == sig[:, j]).mean(axis=1) >= 0.5
            pairs += 1
    # LSH: a bucket = (band id, 2 signature rows).  Count DISTINCT accounts per bucket.
    keys = np.stack([(sig[:, :, 2 * b] * HASH_MOD + sig[:, :, 2 * b + 1]) * 8 + b for b in range(8)], axis=2)
    acct = np.broadcast_to(np.arange(n)[:, None, None], keys.shape)
    uniq = np.unique((keys * n + acct).ravel())
    ukeys, cnt = np.unique(uniq // n, return_counts=True)
    c = cnt[np.searchsorted(ukeys, keys)]
    shared = (c >= 3).any(axis=2).mean(axis=1)
    return dup / pairs, shared


# ---------------------------------------------------------------------------------------------
# DATA GENERATION (interview concept: what bots actually look like, and how they adapt).
# `adapt` in [0,1] moves bot behaviour toward human: slower posting, jittered gaps, paraphrased
# text, filled profiles, fewer shared proxies.  Cheap signals move a lot, costly ones barely.
# ---------------------------------------------------------------------------------------------
def gen(n: int, seed: int, adapt: float = 0.0):
    rng = np.random.default_rng(seed)
    a = float(adapt)
    n_bot = int(round(n * PREVALENCE))
    n_naive, n_ring = int(round(n_bot * 0.42)), int(round(n_bot * 0.25))
    n_soph = n_bot - n_naive - n_ring
    typ = np.zeros(n, dtype=int)
    bots = rng.choice(n, n_bot, replace=False)
    naive, soph, ringm = bots[:n_naive], bots[n_naive:n_naive + n_soph], bots[n_naive + n_soph:]
    typ[naive], typ[soph], typ[ringm] = 1, 2, 3
    rings = [list(map(int, r)) for r in np.array_split(ringm, max(1, int(round(n_ring / 10))))]
    ring_id = -np.ones(n, dtype=int)
    for k, r in enumerate(rings):
        ring_id[r] = k
    T = lambda v: np.asarray(v, dtype=float)[typ]          # per-type parameter -> per-account
    L = math.log

    # -- temporal: simulate 30 inter-post gaps ~ Gamma(k); k>>1 clockwork, k<1 bursty
    rate = np.exp(rng.normal(T([L(1.5), _lerp(L(60), L(6), a), _lerp(L(10), L(3), a), _lerp(L(4), L(2.5), a)]),
                             T([1.0, 0.5, 0.6, 0.6])))
    k = np.exp(rng.normal(T([L(0.45), _lerp(L(30), L(0.9), a), _lerp(L(1.5), L(0.6), a), _lerp(L(1.0), L(0.6), a)]), 0.35))
    gaps = rng.gamma(k[:, None], 1.0, (n, N_GAPS)) * (24.0 / (rate * k))[:, None] + 1e-6   # hours
    mu, sd = gaps.mean(1), gaps.std(1)
    burst = (sd - mu) / (sd + mu)
    b = np.clip(np.floor((np.log10(gaps / mu[:, None]) + 3) / (4 / 12)), 0, 11).astype(int)
    pr = np.stack([(b == j).mean(1) for j in range(12)], axis=1)
    ent = -(pr * np.log(np.where(pr > 0, pr, 1))).sum(1) / math.log(12)
    night_p = np.where(typ == 0, rng.beta(1.2, 14, n),
                       np.clip(rng.normal(T([0, _lerp(.21, .12, a), _lerp(.17, .11, a), _lerp(.15, .11, a)]), 0.03), 0.01, 0.5))
    night = rng.binomial(60, night_p) / 60

    # -- profile / social graph counters
    followers = np.exp(rng.normal(T([L(150), _lerp(L(8), L(40), a), _lerp(L(80), L(120), a), L(120)]), T([1.2, 1.0, 0.9, 0.8])))
    following = np.exp(rng.normal(T([L(200), _lerp(L(1500), L(500), a), _lerp(L(600), L(300), a), L(300)]), T([0.9, 0.6, 0.7, 0.7])))
    fr = np.log10((following + 1) / (followers + 1))
    recip = rng.beta(T([4, _lerp(1, 1.5, a), _lerp(2, 2.5, a), 5]), T([6, _lerp(25, 15, a), _lerp(12, 10, a), 5]))
    prof = rng.beta(T([5, _lerp(1.5, 4.5, a), 5, 4]), T([2.5, _lerp(5, 3, a), 3, 3]))
    apm = np.exp(rng.normal(T([L(4), _lerp(L(40), L(8), a), _lerp(L(9), L(5), a), _lerp(L(6), L(4.5), a)]), T([.5, .4, .45, .45])))
    age = np.clip(rng.exponential(600, n), 1, 3000)
    age[naive] = rng.uniform(1, _lerp(30, 120, a), len(naive))
    age[soph] = rng.uniform(20, 200, len(soph))
    for r in rings:                                         # rings = batches of aged accounts bought together
        age[r] = rng.uniform(200, 900) + rng.normal(0, 4, len(r))

    # -- content: posts are 12-token sets; bots re-use a (mutated) template
    campaign = rng.integers(0, VOCAB, (8, N_TOK))
    memes = rng.integers(0, VOCAB, (5, N_TOK))
    tpl = rng.integers(0, VOCAB, (n, N_TOK))                # humans: a personal catch-phrase
    ind = np.concatenate([naive, soph])
    tpl[ind] = campaign[rng.integers(0, 8, len(ind))]
    for r in rings:
        tpl[r] = rng.integers(0, VOCAB, N_TOK)              # one shared script per ring
    d = T([.06, _lerp(.8, .4, a), _lerp(.45, .2, a), _lerp(.4, .25, a)])     # P(post from template)
    mp = T([.1, _lerp(.08, .3, a), _lerp(.2, .4, a), _lerp(.15, .3, a)])     # per-token paraphrase rate
    use = rng.random((n, N_POSTS)) < d[:, None]
    mut = rng.random((n, N_POSTS, N_TOK)) < mp[:, None, None]
    tok = np.where(use[:, :, None] & ~mut, tpl[:, None, :], rng.integers(0, VOCAB, (n, N_POSTS, N_TOK)))
    memers = np.where((typ == 0) & (rng.random(n) < 0.08))[0]   # humans who repost a viral meme verbatim
    tok[memers, rng.integers(0, N_POSTS, len(memers))] = memes[rng.integers(0, 5, len(memers))]
    dup, shared = minhash_features(tok)

    # -- infrastructure: account-device and account-IP edges
    dev_of, ip_of = [[] for _ in range(n)], [[] for _ in range(n)]
    nd = ni = 0
    humans = rng.permutation(np.where(typ == 0)[0])
    sizes = rng.choice([1, 2, 3, 4], size=len(humans), p=[.55, .25, .12, .08])
    home_ip = np.zeros(n, dtype=int)
    i = h = 0
    while i < len(humans):                                  # households share one IP, sometimes a device
        m = humans[i:i + sizes[h]]
        for j, x in enumerate(m):
            ip_of[x].append(ni)
            home_ip[x] = ni
            if j == 1 and rng.random() < 0.25:
                dev_of[x].append(nd - 1)
            else:
                dev_of[x].append(nd)
                nd += 1
        ni += 1
        i += sizes[h]
        h += 1
    for x in rng.choice(humans, int(0.05 * len(humans)), replace=False):    # visiting friends / cafes
        ip_of[x].append(int(home_ip[rng.choice(humans)]))
    hubs = set()
    for hs in [90, 45, 25]:                                 # university, office, carrier-grade NAT
        for x in rng.choice(humans, max(3, int(hs * n / 12000)), replace=False):
            ip_of[x].append(ni)
        hubs.add(ni)
        ni += 1
    j = 0
    sloppy = naive[: int(0.4 * len(naive))]                 # naive bots: some share a datacenter IP
    while j < len(sloppy):
        g = int(rng.integers(3, 7))
        for x in sloppy[j:j + g]:
            ip_of[x].append(ni)
        ni += 1
        j += g
    for x in np.concatenate([naive[len(sloppy):], soph]):   # sophisticated: residential proxies, 1 IP each
        ip_of[x].append(ni)
        ni += 1
    for x in np.concatenate([naive, soph]):
        dev_of[x].append(nd)
        nd += 1
    for r in rings:                                         # device farm: 2 accounts / phone, small proxy pool
        pool = list(range(ni, ni + max(2, len(r) // 2)))
        ni += len(pool)
        for j, x in enumerate(r):
            dev_of[x].append(nd + j // 2)
            n_ip = 1 if rng.random() < 0.6 * a else 2
            ip_of[x].extend(int(v) for v in rng.choice(pool, n_ip, replace=False))
        nd += (len(r) + 1) // 2
    dev_groups, ip_groups = [[] for _ in range(nd)], [[] for _ in range(ni)]
    for x in range(n):
        ip_of[x] = sorted(set(ip_of[x]))
        for dv in dev_of[x]:
            dev_groups[dv].append(x)
        for ip in ip_of[x]:
            ip_groups[ip].append(x)
    dev_share = np.array([max(len(dev_groups[dv]) for dv in dev_of[x]) for x in range(n)], dtype=float)
    ip_share = np.array([max(len(ip_groups[ip]) for ip in ip_of[x]) for x in range(n)], dtype=float)

    labels, _ = union_find(n, [dev_groups, ip_groups], [None, IP_DEGREE_CAP])
    comp, csize, cspread = component_stats(labels, age)

    raw = np.column_stack([age, fr, prof, rate, ent, burst, night, dup, shared, recip, apm, dev_share, ip_share, csize, cspread])
    X = raw.copy()
    X[:, LOG_COLS] = np.log10(1 + X[:, LOG_COLS])
    return SimpleNamespace(n=n, raw=raw, X=X, y=(typ > 0).astype(int), typ=typ, ring_id=ring_id, gaps=gaps.astype(np.float32),
                           dev_of=dev_of, ip_of=ip_of, dev_groups=dev_groups, ip_groups=ip_groups, hubs=hubs, comp=comp)


# ---------------------------------------------------------------------------------------------
# SCORERS that need no labels
# ---------------------------------------------------------------------------------------------
RULES = [
    ("posts_per_day > 30", lambda r: r[:, 3] > 30),
    ("dup_ratio >= 0.5", lambda r: r[:, 7] >= 0.5),
    ("age < 14d and follow_ratio_log > 0.8", lambda r: (r[:, 0] < 14) & (r[:, 1] > 0.8)),
    ("ip_share >= 40", lambda r: r[:, 12] >= 40),
]


def rule_score(raw):
    """Legacy hand-written rules (the baseline every platform starts with)."""
    return sum(f(raw).astype(float) for _, f in RULES)


def ring_score(raw, min_size=2):
    """Graph score: big component AND synchronised signup. log(size) / (1 + spread/15)."""
    size, spread = raw[:, 13], raw[:, 14]
    return np.where(size >= min_size, np.log(size) / (1 + spread / 15.0), 0.0)


def make_gbdt(iters=80):
    return HistGradientBoostingClassifier(max_iter=iters, learning_rate=0.08, max_depth=4, l2_regularization=1.0,
                                          early_stopping=False, random_state=0)


# ---------------------------------------------------------------------------------------------
# METRICS (interview concept: class imbalance -> PR-AUC and precision@K, never accuracy)
# ---------------------------------------------------------------------------------------------
def metrics(y, s, typ, k=QUEUE_K, p=None):
    order = np.argsort(-s, kind="stable")
    top = np.zeros(len(y), dtype=bool)
    top[order[:k]] = True
    prec, rec, _ = precision_recall_curve(y, s)
    out = {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        f"precision@{k}": float(y[top].mean()),
        f"recall@{k}": float(y[top].sum() / max(1, y.sum())),
        "recall@P>=0.9": float(rec[prec >= 0.9].max()),
    }
    for t in (1, 2, 3):
        out[f"recall_{TYPES[t]}"] = float(top[typ == t].mean()) if (typ == t).any() else 0.0
    if p is not None:
        out["accuracy@0.5"] = float(((p >= 0.5).astype(int) == y).mean())
    return out


def pr_points(y, s, m=70):
    prec, rec, _ = precision_recall_curve(y, s)
    idx = np.unique(np.linspace(0, len(prec) - 1, m).astype(int))
    return {"recall": [round(float(v), 4) for v in rec[idx]], "precision": [round(float(v), 4) for v in prec[idx]]}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------------------------
# BUILD (runs once at import): train every model on snapshot t, calibrate on out-of-fold scores,
# evaluate on snapshot t+1 (time-based split).
# ---------------------------------------------------------------------------------------------
def _build():
    t0 = time.time()
    tr, te = gen(N_PERIOD, 7), gen(N_PERIOD, 11)
    S = SimpleNamespace(tr=tr, te=te, raw_tr={}, raw_te={}, iso={}, p_te={}, p_tr={})
    S.scaler = StandardScaler().fit(tr.X[:, :N_BEH])
    zb = lambda d: S.scaler.transform(d.X[:, :N_BEH])
    S.zb = zb
    S.lr = LogisticRegression(C=1.0, max_iter=1000).fit(zb(tr), tr.y)
    S.gbdt = make_gbdt().fit(tr.X[:, :N_BEH], tr.y)
    S.gbdtg = make_gbdt().fit(tr.X, tr.y)
    S.iforest = IsolationForest(n_estimators=100, max_samples=512, random_state=0).fit(tr.X[:, :N_BEH])

    # out-of-fold scores on train -> isotonic calibration maps that were never fit on their own predictions
    oof = {m: np.zeros(tr.n) for m in ("logreg", "gbdt", "gbdt_graph")}
    ztr = zb(tr)
    for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(tr.X, tr.y):
        oof["logreg"][b] = LogisticRegression(C=1.0, max_iter=1000).fit(ztr[a], tr.y[a]).decision_function(ztr[b])
        oof["gbdt"][b] = make_gbdt(60).fit(tr.X[a, :N_BEH], tr.y[a]).decision_function(tr.X[b, :N_BEH])
        oof["gbdt_graph"][b] = make_gbdt(60).fit(tr.X[a], tr.y[a]).decision_function(tr.X[b])
    S.raw_tr = {"rules": rule_score(tr.raw), "logreg": oof["logreg"], "gbdt": oof["gbdt"], "gbdt_graph": oof["gbdt_graph"],
                "iforest": -S.iforest.score_samples(tr.X[:, :N_BEH]), "graph_rings": ring_score(tr.raw)}
    S.raw_te = score_all(S, te)
    for m in S.raw_tr:
        S.iso[m] = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(S.raw_tr[m], tr.y)
        S.p_tr[m], S.p_te[m] = S.iso[m].predict(S.raw_tr[m]), S.iso[m].predict(S.raw_te[m])
    # ensemble = noisy-OR of a supervised model, an unsupervised one and the graph detector
    S.raw_tr["ensemble"] = noisy_or(S.p_tr)
    S.raw_te["ensemble"] = noisy_or(S.p_te)
    S.iso["ensemble"] = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(S.raw_tr["ensemble"], tr.y)
    S.p_te["ensemble"] = S.iso["ensemble"].predict(S.raw_te["ensemble"])
    S.if_sorted = np.sort(S.raw_tr["iforest"])

    S.metrics = {m: metrics(te.y, S.raw_te[m], te.typ, p=S.p_te[m]) for m in S.raw_te}
    S.human_med = np.median(te.raw[te.typ == 0], axis=0)
    S.human_sorted = np.sort(te.raw[te.typ == 0], axis=0)
    # permutation importance (PR-AUC drop) for the GBDT+graph model
    rng = np.random.default_rng(0)
    base = average_precision_score(te.y, S.raw_te["gbdt_graph"])
    S.perm = []
    for j in range(len(FNAMES)):
        Xp = te.X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        S.perm.append(float(base - average_precision_score(te.y, S.gbdtg.decision_function(Xp))))
    S.build_s = round(time.time() - t0, 2)
    return S


def score_all(S, d):
    return {"rules": rule_score(d.raw), "logreg": S.lr.decision_function(S.zb(d)),
            "gbdt": S.gbdt.decision_function(d.X[:, :N_BEH]), "gbdt_graph": S.gbdtg.decision_function(d.X),
            "iforest": -S.iforest.score_samples(d.X[:, :N_BEH]), "graph_rings": ring_score(d.raw)}


def noisy_or(p):
    return 1 - (1 - p["gbdt_graph"]) * (1 - p["iforest"]) * (1 - p["graph_rings"])


S = _build()
MODEL_INFO = {
    "rules": "Hand-written rules (score = #rules fired)",
    "logreg": "Logistic regression on 13 standardised behaviour features",
    "gbdt": "Gradient-boosted trees (sklearn HistGradientBoosting) on behaviour features",
    "gbdt_graph": "GBDT + graph features (component size, signup synchrony)",
    "iforest": "IsolationForest - unsupervised, uses no labels",
    "graph_rings": "Union-find over shared device/IP: log(size)/(1+age_spread/15)",
    "ensemble": "Noisy-OR of GBDT+graph, IsolationForest and graph rings",
}


# ---------------------------------------------------------------------------------------------
# COST MODEL (interview concept: cost-sensitive decisions + tiered enforcement).
# For calibrated p = P(bot): pick the action with minimum expected cost.
#   allow     : p * C_miss
#   challenge : (1-p) * friction + p * (1-q) * C_miss      q = share of bots a CAPTCHA stops
#   review    : C_review   (a human gets it right)
#   ban       : (1-p) * C_false_ban
# ---------------------------------------------------------------------------------------------
class Costs(BaseModel):
    cost_false_ban: float = Field(100, ge=0)
    cost_missed_bot: float = Field(10, ge=0)
    review_cost: float = Field(1, ge=0)
    challenge_friction: float = Field(0.3, ge=0)
    captcha_stop_rate: float = Field(0.6, ge=0, le=1)
    review_capacity: int = Field(QUEUE_K, ge=0)


TIERS = ["allow", "challenge", "review", "ban"]


def expected_costs(p, c: Costs):
    p = np.asarray(p, dtype=float)
    return np.stack([p * c.cost_missed_bot,
                     (1 - p) * c.challenge_friction + p * (1 - c.captcha_stop_rate) * c.cost_missed_bot,
                     np.full_like(p, c.review_cost),
                     (1 - p) * c.cost_false_ban], axis=1)


def assign_tiers(p, c: Costs):
    ec = expected_costs(p, c)
    act = ec.argmin(axis=1)
    rev = np.where(act == 2)[0]
    overflow = 0
    if len(rev) > c.review_capacity:                         # reviewers are finite: overflow takes the next-best action
        rev = rev[np.argsort(-p[rev], kind="stable")]
        over = rev[c.review_capacity:]
        ec2 = ec[over].copy()
        ec2[:, 2] = np.inf
        act[over] = ec2.argmin(axis=1)
        overflow = len(over)
    return act, overflow


def realised_cost(act, y, c: Costs):
    cost = np.zeros(len(y))
    cost[act == 0] = y[act == 0] * c.cost_missed_bot
    m = act == 1
    cost[m] = (1 - y[m]) * c.challenge_friction + y[m] * (1 - c.captcha_stop_rate) * c.cost_missed_bot
    cost[act == 2] = c.review_cost
    m = act == 3
    cost[m] = (1 - y[m]) * c.cost_false_ban
    return float(cost.sum())


# ---------------------------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------------------------
@router.get("/overview")
def overview():
    te, tr = S.te, S.tr
    med = [{"feature": FNAMES[j], **{TYPES[t]: round(float(np.median(te.raw[te.typ == t, j])), 3) for t in range(4)},
            "stage": FEATURES[j][3], "cost_to_fake": FEATURES[j][4]} for j in range(len(FNAMES))]
    all_human_acc = float((te.y == 0).mean())
    return {
        "dataset": {
            "train_accounts": tr.n, "test_accounts": te.n, "split": "time-based: train on snapshot t, test on t+1",
            "prevalence": float(te.y.mean()), "by_type": {TYPES[t]: int((te.typ == t).sum()) for t in range(4)},
            "devices": len(te.dev_groups), "ips": len(te.ip_groups), "hub_ips": len(te.hubs),
            "components_ge2": int(len(np.unique(te.comp[te.raw[:, 13] >= 2]))), "build_seconds": S.build_s,
        },
        "features": [{"name": f[0], "group": f[1], "stage": f[3], "cost_to_fake": f[4], "desc": f[5]} for f in FEATURES],
        "feature_medians": med,
        "models": [{"model": m, "desc": MODEL_INFO[m], **S.metrics[m]} for m in S.metrics],
        "accuracy_trap": {"always_predict_human_accuracy": all_human_acc, "its_recall": 0.0, "its_pr_auc": float(te.y.mean()),
                          "note": "Random ranking has PR-AUC = prevalence, so 0.01 is the floor - not 0.5."},
        "pr_curves": {m: pr_points(te.y, S.raw_te[m]) for m in S.raw_te},
        "importance": {
            "logreg_coef": dict(zip(FNAMES[:N_BEH], [round(float(v), 3) for v in S.lr.coef_[0]])),
            "gbdt_graph_permutation_pr_auc_drop": dict(zip(FNAMES, [round(v, 4) for v in S.perm])),
        },
        "rules": [r[0] for r in RULES],
        "queue_k": QUEUE_K,
    }


class ThresholdReq(Costs):
    model: str = "gbdt_graph"
    threshold: Optional[float] = Field(None, ge=0, le=1)     # None -> Bayes-optimal C_fb / (C_fb + C_miss)


@router.post("/threshold")
def threshold(req: ThresholdReq):
    """Imbalance + cost-sensitive thresholding + tiered response, all on the test snapshot."""
    if req.model not in S.p_te:
        raise HTTPException(404, f"unknown model; pick one of {list(S.p_te)}")
    p, y, typ = S.p_te[req.model], S.te.y, S.te.typ
    bayes = req.cost_false_ban / max(1e-9, req.cost_false_ban + req.cost_missed_bot)

    def at(t):
        f = p >= t
        tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
        fn = int(y.sum()) - tp
        return {"threshold": round(float(t), 4), "flagged": int(f.sum()), "tp": tp, "fp": fp, "fn": fn,
                "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                "accuracy": float((f.astype(int) == y).mean()),
                "cost": fp * req.cost_false_ban + fn * req.cost_missed_bot}

    grid = [at(t) for t in np.round(np.concatenate([[0.002, 0.005], np.linspace(0.01, 1.0, 67)]), 4)]
    best = min(grid, key=lambda g: g["cost"])
    t_used = bayes if req.threshold is None else req.threshold
    point = at(t_used)
    point["recall_by_type"] = {TYPES[k]: float((p[typ == k] >= t_used).mean()) for k in (1, 2, 3)}

    # tiered response
    act, overflow = assign_tiers(p, req)
    tiers = []
    for k, name in enumerate(TIERS):
        m = act == k
        tiers.append({"tier": name, "accounts": int(m.sum()), "bots": int(y[m].sum()), "humans": int((1 - y[m]).sum()),
                      "bot_rate": float(y[m].mean()) if m.any() else 0.0,
                      **{TYPES[t]: int((typ[m] == t).sum()) for t in (1, 2, 3)}})
    gp = np.linspace(0, 1, 2001)
    ga = expected_costs(gp, req).argmin(axis=1)
    bounds = {TIERS[k]: (round(float(gp[ga == k].min()), 4) if (ga == k).any() else None) for k in range(4)}
    allow_all = float(y.sum() * req.cost_missed_bot)
    policies = {
        "allow everyone": allow_all,
        "ban at p>=0.5": at(0.5)["cost"],
        "ban at Bayes threshold": at(bayes)["cost"],
        "ban at empirical best threshold": best["cost"],
        "tiered (allow/challenge/review/ban)": realised_cost(act, y, req),
    }
    edges = [0, 0.01, 0.05, 0.2, 0.5, 0.9, 1.0001]
    calib = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            calib.append({"p_bin": f"[{lo:g}, {min(hi, 1):g})", "accounts": int(m.sum()), "mean_predicted": float(p[m].mean()),
                          "observed_bot_rate": float(y[m].mean())})
    return {"model": req.model, "bayes_threshold": bayes, "formula": "t* = C_false_ban / (C_false_ban + C_missed_bot)",
            "point": point, "empirical_best": best, "curve": grid, "tiers": tiers, "tier_starts_at_p": bounds,
            "review_overflow": overflow, "policies": policies, "calibration": calib,
            "n": int(len(y)), "bots": int(y.sum())}


def _tier_names(p):
    act, _ = assign_tiers(np.asarray(p), Costs(review_capacity=10 ** 9))
    return [TIERS[a] for a in act]


@router.get("/accounts")
def accounts(view: str = "queue", limit: int = 15):
    """Pick-lists for the inspector: review queue, missed bots, false positives, by type, random."""
    te, p = S.te, S.p_te["ensemble"]
    s = S.raw_te["ensemble"]
    limit = max(1, min(limit, 50))
    rng = np.random.default_rng(3)
    if view == "queue":
        idx = np.argsort(-s, kind="stable")[:limit]
    elif view == "missed":
        b = np.where(te.y == 1)[0]
        idx = b[np.argsort(s[b], kind="stable")][:limit]
    elif view == "false_pos":
        h = np.where(te.y == 0)[0]
        idx = h[np.argsort(-s[h], kind="stable")][:limit]
    elif view in TYPES:
        c = np.where(te.typ == TYPES.index(view))[0]
        idx = rng.choice(c, min(limit, len(c)), replace=False)
    elif view == "random":
        idx = rng.choice(te.n, limit, replace=False)
    else:
        raise HTTPException(400, "view must be queue|missed|false_pos|random|" + "|".join(TYPES))
    tiers = _tier_names(p[idx])
    return {"view": view, "accounts": [
        {"id": int(i), "type": TYPES[te.typ[i]], "p_bot": round(float(p[i]), 4), "tier": tiers[k],
         "posts_per_day": round(float(te.raw[i, 3]), 1), "dup_ratio": round(float(te.raw[i, 7]), 2),
         "comp_size": int(te.raw[i, 13]), "rules_fired": int(S.raw_te["rules"][i])} for k, i in enumerate(idx)]}


def _component_view(d, members, p=None, cap=IP_DEGREE_CAP, max_nodes=40):
    """Bipartite account-device-IP subgraph for the canvas."""
    members = [int(m) for m in members]
    truncated = len(members) > max_nodes
    members = members[:max_nodes]
    ms = set(members)
    ents = {}
    for x in members:
        for dv in d.dev_of[x]:
            ents.setdefault(("device", dv), []).append(x)
        for ip in d.ip_of[x]:
            ents.setdefault(("ip", ip), []).append(x)
    out = []
    for (kind, eid), mem in ents.items():
        deg = len(d.dev_groups[eid]) if kind == "device" else len(d.ip_groups[eid])
        hub = kind == "ip" and (eid in d.hubs or (cap is not None and deg > cap))
        if len(mem) >= 2 or len(ms) <= 2:
            out.append({"key": f"{kind[0]}{eid}", "kind": kind, "members": mem, "degree": deg, "hub": bool(hub)})
    return {"nodes": [{"id": x, "type": TYPES[d.typ[x]], "p_bot": None if p is None else round(float(p[x]), 3)} for x in members],
            "entities": out, "truncated": truncated}


@router.get("/account/{aid}")
def account(aid: int):
    """Inspector: features vs human baseline, every model's score, per-feature contributions, ring neighbours."""
    te = S.te
    if not 0 <= aid < te.n:
        raise HTTPException(404, "account id out of range")
    raw, x = te.raw[aid], te.X[aid]
    feats = []
    for j, name in enumerate(FNAMES):
        pct = float(np.searchsorted(S.human_sorted[:, j], raw[j], side="right") / len(S.human_sorted))
        feats.append({"feature": name, "value": round(float(raw[j]), 3), "human_median": round(float(S.human_med[j]), 3),
                      "human_percentile": round(pct, 3), "group": FEATURES[j][1]})
    # linear contribution: coef_j * z_j  (log-odds relative to the average account)
    z = S.scaler.transform(x[None, :N_BEH])[0]
    lr_contrib = {FNAMES[j]: round(float(S.lr.coef_[0][j] * z[j]), 3) for j in range(N_BEH)}
    # SHAP-like single-feature ablation for the GBDT: reset feature j to the human median, measure logit drop
    hm = np.median(te.X[te.typ == 0], axis=0)
    Xa = np.repeat(x[None, :], len(FNAMES) + 1, axis=0)
    for j in range(len(FNAMES)):
        Xa[j + 1, j] = hm[j]
    lg = S.gbdtg.decision_function(Xa)
    gb_contrib = {FNAMES[j]: round(float(lg[0] - lg[j + 1]), 3) for j in range(len(FNAMES))}
    scores = {m: {"raw": round(float(S.raw_te[m][aid]), 4), "p_bot": round(float(S.p_te[m][aid]), 4)} for m in S.p_te}
    p = float(S.p_te["ensemble"][aid])
    c = Costs()
    ec = expected_costs(np.array([p]), c)[0]
    members = np.where(te.comp == te.comp[aid])[0]
    members = [aid] + [int(m) for m in members if m != aid]
    return {
        "id": aid, "type": TYPES[te.typ[aid]], "is_bot": bool(te.y[aid]), "ring_id": int(te.ring_id[aid]),
        "features": feats, "scores": scores,
        "rules_fired": [name for name, f in RULES if f(raw[None, :])[0]],
        "contrib_logreg": lr_contrib, "contrib_gbdt_graph": gb_contrib,
        "decision": {"p_bot": p, "tier": TIERS[int(ec.argmin())], "expected_cost": dict(zip(TIERS, [round(float(v), 3) for v in ec])),
                     "costs": c.model_dump()},
        "gaps_hours": [round(float(g), 3) for g in te.gaps[aid]],
        "component": {"size": len(members), "age_spread_days": round(float(raw[14]), 1),
                      **_component_view(te, members, S.p_te["ensemble"])},
    }


class RingReq(BaseModel):
    min_size: int = Field(6, ge=2, le=200)
    cap_hub_ips: bool = True             # False = the pitfall: NAT hubs glue humans into giant "rings"
    max_ip_degree: int = Field(IP_DEGREE_CAP, ge=2, le=500)
    require_sync: bool = True            # also require synchronised signup (age spread small)
    max_spread_days: float = Field(30, ge=0, le=2000)


@router.post("/rings")
def rings(req: RingReq):
    """Graph-based ring detection: union-find over shared devices / IPs, re-run with the chosen knobs."""
    te = S.te
    t0 = time.perf_counter()
    labels, skipped = union_find(te.n, [te.dev_groups, te.ip_groups], [None, req.max_ip_degree if req.cap_hub_ips else None])
    comp, size, spread = component_stats(labels, te.raw[:, 0])
    ms = (time.perf_counter() - t0) * 1000
    flag = size >= req.min_size
    if req.require_sync:
        flag &= spread <= req.max_spread_days
    y, typ = te.y, te.typ
    comps = []
    for cid in np.unique(comp[flag]):
        m = np.where(comp == cid)[0]
        nb = int(y[m].sum())
        comps.append({"size": len(m), "bots": nb, "humans": len(m) - nb, "age_spread_days": round(float(spread[m[0]]), 1),
                      "ring_bots": int((typ[m] == 3).sum()), "naive_bots": int((typ[m] == 1).sum()),
                      "kind": "bot ring" if nb / len(m) > 0.8 else ("human cluster (false positive)" if nb == 0 else "mixed"),
                      "members": m})
    comps.sort(key=lambda c: -c["size"])
    viz = [{"label": f"{c['kind']} - {c['size']} accounts", **_component_view(te, c["members"], S.p_te["gbdt"],
            cap=req.max_ip_degree if req.cap_hub_ips else None, max_nodes=36)} for c in comps[:8]]
    for c in comps:
        c["sample_ids"] = [int(v) for v in c.pop("members")[:6]]
    nf = int(flag.sum())
    # what the per-account behaviour model thinks of the ring members the graph caught
    ringm = typ == 3
    return {
        "params": req.model_dump(), "union_find_ms": round(ms, 1),
        "graph": {"accounts": te.n, "devices": len(te.dev_groups), "ips": len(te.ip_groups), "hub_ips_skipped": skipped,
                  "components_ge2": int(len(np.unique(comp[size >= 2]))), "largest_component": int(size.max())},
        "flagged_accounts": nf, "flagged_components": len(comps),
        "precision": float(y[flag].mean()) if nf else 0.0,
        "recall_ring_bots": float(flag[ringm].mean()), "recall_all_bots": float(flag[y == 1].mean()),
        "humans_flagged": int((flag & (y == 0)).sum()),
        "behaviour_gbdt_recall_on_ring_bots_at_queue": S.metrics["gbdt"]["recall_ring_bot"],
        "mean_p_bot_of_ring_members_behaviour_gbdt": float(S.p_te["gbdt"][ringm].mean()),
        "components": comps[:25], "viz": viz,
    }


# ---------------------------------------------------------------------------------------------
# ADVERSARIAL DRIFT (interview concept: the positive class fights back -> decay, monitoring that
# cannot see it, and the responses: retrain, hard-to-fake features, label-free ensemble members).
# ---------------------------------------------------------------------------------------------
class DriftReq(BaseModel):
    adapt: float = Field(0.5, ge=0, le=1)


_DRIFT_CACHE: dict = {}


def _psi(a, b, bins=10):
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)[1:-1]))
    pa = np.bincount(np.searchsorted(q, a), minlength=len(q) + 1) / len(a)
    pb = np.bincount(np.searchsorted(q, b), minlength=len(q) + 1) / len(b)
    pa, pb = np.clip(pa, 1e-4, None), np.clip(pb, 1e-4, None)
    return float(((pb - pa) * np.log(pb / pa)).sum())


@router.post("/drift")
def drift(req: DriftReq):
    a = round(req.adapt * 20) / 20
    if a in _DRIFT_CACHE:
        return _DRIFT_CACHE[a]
    t0 = time.perf_counter()
    half = N_PERIOD // 2
    d1 = gen(half, 23, a)          # month t+2: bots already adapted; labels arrive (late) -> usable for retraining
    d2 = gen(N_PERIOD, 29, a)      # month t+3: evaluation
    k = int(1.5 * d2.y.sum())
    raw = score_all(S, d2)
    p = {m: S.iso[m].predict(raw[m]) for m in raw}
    # label-free response: refit IsolationForest on the new snapshot, quantile-map onto the old score scale
    if2 = IsolationForest(n_estimators=100, max_samples=512, random_state=0).fit(d2.X[:, :N_BEH])
    s2 = -if2.score_samples(d2.X[:, :N_BEH])
    rank = np.argsort(np.argsort(s2)) / (len(s2) - 1)
    p_if = S.iso["iforest"].predict(np.quantile(S.if_sorted, rank))
    ens = 1 - (1 - p["gbdt_graph"]) * (1 - p_if) * (1 - p["graph_rings"])
    # labelled response: retrain on old + new labels, recent rows weighted 3x
    Xr, yr = np.vstack([S.tr.X, d1.X]), np.concatenate([S.tr.y, d1.y])
    w = np.concatenate([np.ones(S.tr.n), 3 * np.ones(d1.n)])
    re_g = make_gbdt().fit(Xr, yr, sample_weight=w)
    s_re = re_g.decision_function(d2.X)
    p_re = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(re_g.decision_function(Xr), yr).predict(s_re)
    ens_re = 1 - (1 - p_re) * (1 - p_if) * (1 - p["graph_rings"])
    strategies = [
        ("rules (static)", raw["rules"]),
        ("stale logistic regression", raw["logreg"]),
        ("stale GBDT, behaviour only", raw["gbdt"]),
        ("stale GBDT + graph features", raw["gbdt_graph"]),
        ("graph rings only (no labels)", raw["graph_rings"]),
        ("stale ensemble + refit IsolationForest", ens),
        ("retrained GBDT + graph", s_re),
        ("retrained ensemble", ens_re),
    ]
    rows = []
    for name, s in strategies:
        m = metrics(d2.y, s, d2.typ, k=k)
        rows.append({"strategy": name, "pr_auc": m["pr_auc"], f"precision@{k}": m[f"precision@{k}"],
                     "recall_naive": m["recall_naive_bot"], "recall_sophisticated": m["recall_sophisticated_bot"],
                     "recall_ring": m["recall_ring_bot"]})
    hstd = S.tr.X[S.tr.typ == 0].std(axis=0) + 1e-9
    hmean = S.tr.X[S.tr.typ == 0].mean(axis=0)
    feat = []
    for j, name in enumerate(FNAMES):
        feat.append({"feature": name, "cost_to_fake": FEATURES[j][4],
                     "psi_all_accounts": round(_psi(S.tr.X[:, j], d2.X[:, j]), 4),
                     "bot_gap_before_sd": round(float((S.tr.X[S.tr.y == 1, j].mean() - hmean[j]) / hstd[j]), 2),
                     "bot_gap_after_sd": round(float((d2.X[d2.y == 1, j].mean() - hmean[j]) / hstd[j]), 2)})
    feat.sort(key=lambda r: -abs(r["bot_gap_before_sd"] - r["bot_gap_after_sd"]))
    t_rev = 0.19
    out = {
        "adapt": a, "k": k, "eval_accounts": d2.n, "eval_bots": int(d2.y.sum()), "strategies": rows, "features": feat,
        "monitors": {
            "max_feature_psi_all_accounts": max(f["psi_all_accounts"] for f in feat),
            "psi_alert_threshold": 0.2,
            "flag_rate_stale_model": float((p["gbdt_graph"] >= t_rev).mean()),
            "flag_rate_at_train_time": float((S.p_te["gbdt_graph"] >= t_rev).mean()),
            "true_prevalence": float(d2.y.mean()),
            "note": "Bots are 1% of accounts, so population PSI stays ~0 while the bot class moves. "
                    "The falling flag rate is ambiguous (fewer bots or better evasion?) - only an audited random sample tells.",
        },
        "ms": round((time.perf_counter() - t0) * 1000),
    }
    _DRIFT_CACHE[a] = out
    return out


# ---------------------------------------------------------------------------------------------
# PITFALL (interview concept: label noise from past rules + selection bias + biased offline eval).
# Wrong: (1) labels = "the rules fired", (2) labels only from the rule-driven review queue.
# Right: spend part of the review budget on exploration (anomaly / graph candidates) and on a
# uniform random audit, which is also the only unbiased prevalence estimate.
# ---------------------------------------------------------------------------------------------
class LabelReq(BaseModel):
    audit_frac: float = Field(0.03, ge=0, le=0.3)
    explore_k: int = Field(300, ge=0, le=3000)


_LABEL_CACHE: dict = {}


def _wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def _fit_eval(name, y_lab, reviews, note, wrong):
    te = S.te
    if y_lab.sum() < 2:
        return {"training_labels": name, "wrong_way": wrong, "note": "too few positives to train"}
    g = make_gbdt(60).fit(S.tr.X, y_lab)
    s = g.decision_function(te.X)
    m = metrics(te.y, s, te.typ)
    top = np.argsort(-s, kind="stable")[:QUEUE_K]
    rule_te = rule_score(te.raw) >= 1
    return {"training_labels": name, "wrong_way": wrong, "reviews_spent": int(reviews), "positives_in_labels": int(y_lab.sum()),
            "true_bots_labelled": int((y_lab.astype(bool) & (S.tr.y == 1)).sum()),
            "pr_auc_true_labels": m["pr_auc"],
            "pr_auc_vs_rule_labels": float(average_precision_score(rule_te.astype(int), s)),
            f"precision@{QUEUE_K}": m[f"precision@{QUEUE_K}"],
            "recall_naive": m["recall_naive_bot"], "recall_sophisticated": m["recall_sophisticated_bot"],
            "recall_ring": m["recall_ring_bot"],
            "queue_overlap_with_rules": float(rule_te[top].mean()), "note": note}


@router.post("/pitfall/labels")
def pitfall_labels(req: LabelReq):
    key = (round(req.audit_frac, 3), req.explore_k)
    if key in _LABEL_CACHE:
        return _LABEL_CACHE[key]
    tr = S.tr
    y = tr.y
    rule = rule_score(tr.raw) >= 1
    rng = np.random.default_rng(5)
    audit = rng.random(tr.n) < req.audit_frac
    explore = np.zeros(tr.n, dtype=bool)
    if req.explore_k:
        explore[np.argsort(-S.raw_tr["iforest"], kind="stable")[: req.explore_k]] = True      # anomaly candidates
    ring_c = (tr.raw[:, 13] >= 6) & (tr.raw[:, 14] <= 30)                                       # graph candidates
    reviewed = rule | audit | explore | ring_c
    if "static" not in _LABEL_CACHE:
        _LABEL_CACHE["static"] = [
            _fit_eval("oracle: every account truly labelled", y, tr.n, "Unattainable upper bound.", False),
            _fit_eval("rule output used as label", rule.astype(int), 0,
                      "Model distils the rules: near-perfect against rule labels, blind to what rules never caught.", True),
            _fit_eval("only rule-flagged accounts reviewed", (y.astype(bool) & rule).astype(int), int(rule.sum()),
                      "Clean labels, biased sample: unreviewed bots are silently 'negative' (selection bias).", True),
        ]
    rows = list(_LABEL_CACHE["static"]) + [
        _fit_eval("rules + anomaly/graph exploration + random audit", (y.astype(bool) & reviewed).astype(int), int(reviewed.sum()),
                  "Review budget is split: exploit (rules), explore (IsolationForest top-K, ring candidates), uniform audit.", False)]
    na, ka = int(audit.sum()), int(y[audit].sum())
    lo, hi = _wilson(ka, na)
    nr, kr = int(rule.sum()), int(y[rule].sum())
    out = {
        "params": {"audit_frac": req.audit_frac, "explore_k": req.explore_k},
        "review_budget": {"rule_queue": nr, "random_audit": na, "anomaly_explore": int(explore.sum()), "ring_candidates": int(ring_c.sum()),
                          "total_distinct": int(reviewed.sum())},
        "rows": rows,
        "prevalence": {
            "true": float(y.mean()),
            "naive_from_review_queue": kr / tr.n,
            "naive_explanation": f"{kr} confirmed bots in the rule queue / {tr.n} accounts - counts only what the rules can see",
            "random_audit_estimate": ka / max(1, na), "random_audit_ci95": [lo, hi],
            "audit_sample": na, "audit_bots_found": ka,
            "sample_size_note": "At 1% prevalence a +-0.2pp (95%) estimate needs n ~ 1.96^2 * 0.01 * 0.99 / 0.002^2 ~ 9.5k audited accounts; "
                                "stratify by model score and re-weight to cut that 5-10x.",
        },
        "rule_queue_precision": kr / max(1, nr),
        "rule_recall_true": kr / max(1, int(y.sum())),
    }
    _LABEL_CACHE[key] = out
    return out
