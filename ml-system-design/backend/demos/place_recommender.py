"""Place / Location Recommender ("restaurants and attractions near you").

Pipeline shown end to end, all on a small synthetic city:

  geo retrieval (grid index, radius)  ->  two-tower-style relevance (BPR matrix
  factorisation written in numpy)  ->  GBDT re-ranker over relevance + geo +
  context features  ->  exploration bonus (UCB / epsilon-greedy)  ->  MMR diversity

The synthetic world has a known ground truth ("true satisfaction"), so the demo can
show things you can never see in production: how far visits are from satisfaction,
and how much each bias (popularity, distance, position, random split) distorts
what you measure.
"""
import math
import time
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import HistGradientBoostingClassifier

META = {
    "slug": "place-recommender",
    "title": "Nearby Place Recommendations",
    "tagline": "Geo retrieval + embeddings + context-aware GBDT re-rank, with diversity, exploration and bias pitfalls",
    "icon": "\U0001F4CD",
    "order": 3,
}
router = APIRouter()

# --------------------------------------------------------------------------
# 1. SYNTHETIC WORLD
#    Concept: implicit feedback. A check-in happens when *utility* is high, and
#    utility mixes taste with things that are not taste at all: distance, fame
#    (exposure), time of day, weather. "Satisfaction" is utility without distance
#    and fame - the thing we would like to optimise but never observe directly.
# --------------------------------------------------------------------------
SEED = 7
CITY = 10.0            # city is a 10 km x 10 km square
N_PLACES, N_USERS, N_DAYS, D_TRUE = 600, 800, 90, 6
T_MF, T_VAL, T_TEST = 54, 69, 72   # time split: [0,54) MF | [54,69) ranker | [69,72) ranker val | [72,90) test
TODAY = N_DAYS - 1
DIST_SCALE = 0.7       # km; true distance decay is exp(-d / 0.7)

CATS = ["cafe", "restaurant", "fast_food", "bar", "park", "museum", "bakery", "shopping"]
NC = len(CATS)
BUCKETS = ["morning", "lunch", "afternoon", "evening", "night"]
#                     morning lunch  aftn  evening night
HOUR_FIT = np.array([[1.0, 0.3, 0.5, -0.3, -1.5],    # cafe
                     [-1.0, 1.0, -0.2, 1.2, -0.5],   # restaurant
                     [-0.3, 0.8, 0.2, 0.5, 0.6],     # fast_food
                     [-2.5, -1.5, -0.5, 1.0, 1.5],   # bar
                     [0.5, 0.3, 0.8, -0.3, -2.5],    # park
                     [0.2, 0.5, 0.8, -1.0, -3.0],    # museum
                     [1.3, 0.2, 0.0, -1.0, -2.5],    # bakery
                     [-0.3, 0.4, 0.8, 0.2, -2.5]])   # shopping
RAIN_FIT = np.array([0.3, 0.1, 0.0, 0.0, -2.0, 0.6, 0.0, 0.4])
WKND_FIT = np.array([0.0, 0.2, 0.0, 0.4, 0.5, 0.6, 0.0, 0.3])
THETA_ETA = 0.8        # position-bias strength used by the feedback-loop simulator


def hour_bucket(h):
    h = np.asarray(h)
    return np.where((h >= 6) & (h <= 10), 0, np.where((h >= 11) & (h <= 14), 1,
           np.where((h >= 15) & (h <= 17), 2, np.where((h >= 18) & (h <= 21), 3, 4))))


def _unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def _generate():
    rng = np.random.default_rng(SEED)
    centers = np.array([[5, 5], [2.5, 7.5], [7.5, 7], [7, 2.5], [2.5, 2.5], [5, 8.5]], float)
    mix = rng.dirichlet(np.ones(NC) * 1.5, size=len(centers))       # each neighbourhood has its own category mix
    nb = rng.choice(len(centers), N_PLACES, p=[0.3, 0.14, 0.14, 0.14, 0.14, 0.14])
    xy = centers[nb] + rng.normal(0, 0.8, (N_PLACES, 2))
    scatter = rng.random(N_PLACES) < 0.2
    xy[scatter] = rng.random((int(scatter.sum()), 2)) * CITY
    xy = np.clip(xy, 0.05, CITY - 0.05)
    cat = np.array([rng.choice(NC, p=mix[b]) for b in nb])
    proto = _unit(rng.normal(size=(NC, D_TRUE)))
    v_true = _unit(proto[cat] + 0.45 * rng.normal(size=(N_PLACES, D_TRUE)))
    quality = rng.normal(size=N_PLACES)
    fame = rng.normal(size=N_PLACES)                                # exposure / brand: drives visits, not satisfaction
    price = np.clip(np.round(2.2 + 0.5 * quality + rng.normal(0, 0.8, N_PLACES)), 1, 4).astype(int)
    price[cat == 4] = 1                                             # parks are free-ish
    rating = np.clip(3.9 + 0.4 * quality + rng.normal(0, 0.2, N_PLACES), 1, 5).round(1)
    open_day = np.where(rng.random(N_PLACES) < 0.15, rng.integers(10, 86, N_PLACES), 0)   # freshness: new places
    close_day = np.where(rng.random(N_PLACES) < 0.05, rng.integers(30, 90, N_PLACES), 999)  # ... and closures
    close_day = np.where(close_day <= open_day + 5, 999, close_day)
    adj = ["Blue", "Golden", "Old Town", "Corner", "Sunny", "Little", "Grand", "Urban", "Lucky", "Green", "Copper", "Night Owl"]
    noun = ["Cafe", "Kitchen", "Burger", "Bar", "Park", "Museum", "Bakery", "Market"]
    names = [f"{adj[int(rng.integers(len(adj)))]} {noun[c]} #{i}" for i, c in enumerate(cat)]

    # users: home + work location, latent taste = blend of two favourite categories, price preference
    hnb = rng.integers(0, len(centers), N_USERS)
    home = np.clip(centers[hnb] + rng.normal(0, 1.3, (N_USERS, 2)), 0.1, CITY - 0.1)
    work = np.where(rng.random((N_USERS, 1)) < 0.6, centers[0] + rng.normal(0, 0.9, (N_USERS, 2)),
                    centers[rng.integers(1, len(centers), N_USERS)] + rng.normal(0, 0.9, (N_USERS, 2)))
    work = np.clip(work, 0.1, CITY - 0.1)
    fav = np.stack([rng.integers(0, NC, N_USERS), rng.integers(0, NC, N_USERS)], 1)
    taste = _unit(proto[fav[:, 0]] + 0.6 * proto[fav[:, 1]] + 0.3 * rng.normal(size=(N_USERS, D_TRUE)))
    price_pref = np.clip(rng.normal(2.2, 0.8, N_USERS), 1, 4)
    aff = (taste @ v_true.T).astype(np.float32)

    # check-in events
    n_ev = np.maximum(8, rng.poisson(32, N_USERS))
    eu = np.repeat(np.arange(N_USERS), n_ev)
    n = len(eu)
    day = rng.integers(0, N_DAYS, n)
    hb = rng.choice(5, n, p=[0.2, 0.25, 0.15, 0.3, 0.1])
    hours = [np.arange(6, 11), np.arange(11, 15), np.arange(15, 18), np.arange(18, 22), np.array([22, 23, 0, 1])]
    hour = np.array([hours[b][int(r * len(hours[b]))] for b, r in zip(hb, rng.random(n))])
    wk = (day % 7 >= 5).astype(int)
    rain_by_day = (rng.random(N_DAYS) < 0.3).astype(int)
    rain = rain_by_day[day]
    at_work = (wk == 0) & (hour >= 9) & (hour <= 17) & (rng.random(n) < 0.7)
    loc = np.where(at_work[:, None], work[eu], home[eu]) + rng.normal(0, 0.4, (n, 2))
    away = rng.random(n) < 0.12                                      # travelling / somewhere unusual
    loc[away] = rng.random((int(away.sum()), 2)) * CITY
    loc = np.clip(loc, 0, CITY)
    ep = np.empty(n, int)
    for s in range(0, n, 4000):                                      # multinomial-logit choice via Gumbel-max
        sl = slice(s, s + 4000)
        d = np.linalg.norm(loc[sl, None, :] - xy[None, :, :], axis=2)
        util = (1.6 * aff[eu[sl]] + 0.6 * quality + 0.8 * fame
                - 0.7 * np.abs(price[None, :] - price_pref[eu[sl], None]) - d / DIST_SCALE
                + HOUR_FIT[cat][:, hb[sl]].T + rain[sl, None] * RAIN_FIT[cat] + wk[sl, None] * WKND_FIT[cat])
        closed = (open_day[None, :] > day[sl, None]) | (close_day[None, :] <= day[sl, None])
        util[closed] = -np.inf
        ep[sl] = np.argmax(util + rng.gumbel(size=util.shape), axis=1)
    order = np.lexsort((hour, day))                                  # global time order
    ev = dict(u=eu[order], p=ep[order], day=day[order], hour=hour[order], hb=hb[order], wk=wk[order],
              rain=rain[order], loc=loc[order], away=away[order])
    return dict(xy=xy, cat=cat, quality=quality, fame=fame, price=price, rating=rating, open=open_day,
                close=close_day, names=names, home=home, work=work, taste=taste, price_pref=price_pref,
                aff=aff, ev=ev, centers=centers)


W = _generate()
XY, CAT, PRICE, RATING, OPEN, CLOSE = W["xy"], W["cat"], W["price"], W["rating"], W["open"], W["close"]
EV = W["ev"]
N_EV = len(EV["u"])
LIVE = (OPEN <= TODAY) & (CLOSE > TODAY)


def true_sat(u, p, hb, wk, rain):
    """Oracle satisfaction (exists only because the world is synthetic): taste + quality - price mismatch + context."""
    c = CAT[p]
    return (1.6 * W["aff"][u, p] + 0.6 * W["quality"][p] - 0.7 * np.abs(PRICE[p] - W["price_pref"][u])
            + HOUR_FIT[c, hb] + rain * RAIN_FIT[c] + wk * WKND_FIT[c])


# --------------------------------------------------------------------------
# 2. GEO CANDIDATE RETRIEVAL
#    Concept: never score the whole catalogue. A uniform grid (stand-in for
#    geohash / S2 / H3 cells) maps cell -> place ids; a radius query scans only
#    the cells the circle touches, then applies the exact distance filter.
# --------------------------------------------------------------------------
CELL = 0.5
G = int(CITY / CELL)
_cell = np.clip((XY // CELL).astype(int), 0, G - 1)
GRID = {}
for _i, (_cx, _cy) in enumerate(_cell):
    GRID.setdefault((_cx, _cy), []).append(_i)
GRID = {k: np.array(v) for k, v in GRID.items()}


def geo_retrieve(x, y, r):
    cx0, cx1 = int(max(0, (x - r) // CELL)), int(min(G - 1, (x + r) // CELL))
    cy0, cy1 = int(max(0, (y - r) // CELL)), int(min(G - 1, (y + r) // CELL))
    cells, ids = [], []
    for cx in range(cx0, cx1 + 1):
        for cy in range(cy0, cy1 + 1):
            # skip cells whose nearest corner is already outside the circle
            nx, ny = min(max(x, cx * CELL), (cx + 1) * CELL), min(max(y, cy * CELL), (cy + 1) * CELL)
            if (nx - x) ** 2 + (ny - y) ** 2 > r * r:
                continue
            cells.append([cx, cy])
            if (cx, cy) in GRID:
                ids.append(GRID[(cx, cy)])
    ids = np.concatenate(ids) if ids else np.array([], int)
    d = np.linalg.norm(XY[ids] - np.array([x, y]), axis=1) if len(ids) else np.array([])
    keep = d <= r
    return ids[keep], d[keep], cells, len(ids)


# neighbour lists (places within 2.5 km of a place) used for geo-matched negative sampling
_PD = np.linalg.norm(XY[:, None, :] - XY[None, :, :], axis=2)


# --------------------------------------------------------------------------
# 3. RELEVANCE MODEL: BPR MATRIX FACTORISATION (numpy, SGD, negative sampling)
#    Concept: two-tower in its simplest form - a user vector and a place vector
#    whose dot product ranks visited places above sampled negatives.
#    Negative sampling is the lesson: "uniform" negatives are mostly far-away
#    places, so the model learns geography; "geo" negatives are places the user
#    could plausibly have gone to instead, so it has to learn taste.
# --------------------------------------------------------------------------
def train_bpr(eu, ep, known, neg_mode="geo", d=16, epochs=30, lr=0.05, reg=0.02, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.1, (N_USERS, d))
    V = rng.normal(0, 0.1, (N_PLACES, d))
    b = np.zeros(N_PLACES)
    known_ids = np.nonzero(known)[0]
    if neg_mode == "geo":
        near = (_PD <= 2.5) & known[None, :]
        np.fill_diagonal(near, False)
        cnt = near.sum(1)
        width = int(cnt.max())
        nbr = np.zeros((N_PLACES, width), int)
        for i in range(N_PLACES):
            nbr[i, :cnt[i]] = np.nonzero(near[i])[0]
    n, B = len(eu), 512
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, B):
            idx = perm[s:s + B]
            u, i = eu[idx], ep[idx]
            if neg_mode == "geo":
                j = nbr[i, (rng.random(len(i)) * np.maximum(cnt[i], 1)).astype(int)]
            else:
                j = known_ids[rng.integers(0, len(known_ids), len(i))]
            Uu, Vi, Vj = U[u], V[i], V[j]
            x = (Uu * (Vi - Vj)).sum(1) + b[i] - b[j]
            g = 1.0 / (1.0 + np.exp(np.clip(x, -30, 30)))          # d/dx of -log sigmoid(x)
            np.add.at(U, u, lr * (g[:, None] * (Vi - Vj) - reg * Uu))
            np.add.at(V, i, lr * (g[:, None] * Uu - reg * Vi))
            np.add.at(V, j, lr * (-g[:, None] * Uu - reg * Vj))
            np.add.at(b, i, lr * (g - reg * b[i]))
            np.add.at(b, j, lr * (-g - reg * b[j]))
    return U, V, b


class Snapshot:
    """Everything the serving system knows as of day `t`: embeddings + counting features.
    Concept: point-in-time correctness. Features for an example at time t use data < t only."""

    def __init__(self, mask, t, neg_mode="geo", seed=0):
        u, p = EV["u"][mask], EV["p"][mask]
        self.t = t
        self.n_events = int(mask.sum())
        known = OPEN < t
        self.U, self.V, self.b = train_bpr(u, p, known, neg_mode, seed=seed)
        self.count = np.bincount(p, minlength=N_PLACES).astype(float)
        days_open = np.clip(np.minimum(t, CLOSE) - OPEN, 0, None)
        self.pop_rate = self.count / (days_open + 7.0) * 30.0       # visits / month, shrunk for young places
        # cold-start places: no interactions -> content fallback = mean embedding of their category
        self.cold_place = self.count < 1
        for c in range(NC):
            warm = (CAT == c) & (self.count >= 5)
            cold = (CAT == c) & self.cold_place
            if warm.any() and cold.any():
                self.V[cold] = self.V[warm].mean(0)
                self.b[cold] = self.b[warm].mean()
        # user counting features (+1 row = cold-start default)
        gcat = np.bincount(CAT[p], minlength=NC) / max(1, len(p))
        uc = np.zeros((N_USERS + 1, NC))
        np.add.at(uc, (u, CAT[p]), 1)
        un = uc.sum(1, keepdims=True)
        self.ucat = (uc + 3 * gcat) / (un + 3)                      # smoothed share of user's visits per category
        ps = np.bincount(u, weights=PRICE[p], minlength=N_USERS + 1)
        gprice = PRICE[p].mean()
        self.uprice = (ps + 3 * gprice) / (un[:, 0] + 3)
        self.un = un[:, 0]
        self.gcat = gcat
        # context x category lifts (target-encoding style crosses)
        def lift(ctx, k):
            m = np.zeros((NC, k))
            np.add.at(m, (CAT[p], ctx), 1)
            return ((m + 1) / (m.sum(0, keepdims=True) + NC)) / gcat[:, None]
        self.hour_lift = lift(EV["hb"][mask], 5)
        self.rain_lift = lift(EV["rain"][mask], 2)
        self.wk_lift = lift(EV["wk"][mask], 2)
        self.cold_u = self.U.mean(0)


# --------------------------------------------------------------------------
# 4. RE-RANKER FEATURES + GBDT
#    Concept: the embedding score is one feature among many. Distance, context
#    crosses and popularity go into a GBDT that predicts P(visit | shown nearby).
#    Groups below drive the per-item score breakdown in the UI.
# --------------------------------------------------------------------------
FEATURES = ["mf_score", "user_cat_share", "price_gap", "dist_km", "log_pop_rate", "rating", "place_age_days",
            "cat_hour_lift", "cat_rain_lift", "cat_weekend_lift", "hour_bucket", "price"]
GROUPS = {"relevance": [0, 1, 2], "distance": [3], "popularity": [4, 5, 6], "context": [7, 8, 9, 10, 11]}


def feats(S, mf, ucs, uprice, p, dist, hb, wk, rain, asof):
    c = CAT[p]
    hb = np.broadcast_to(hb, len(p))
    return np.column_stack([mf, ucs, PRICE[p] - uprice, dist, np.log1p(S.pop_rate[p]), RATING[p],
                            np.clip(asof - OPEN[p], 0, 60), S.hour_lift[c, hb], S.rain_lift[c, rain],
                            S.wk_lift[c, wk], hb, PRICE[p]])


def _dist_matrix(ev):
    return np.linalg.norm(EV["loc"][ev][:, None, :] - XY[None, :, :], axis=2)


def _open_matrix(ev):
    d = EV["day"][ev][:, None]
    return (OPEN[None, :] <= d) & (CLOSE[None, :] > d)


def _pair_feats(S, ev, rows, cols, D):
    e = ev[rows]
    u = EV["u"][e]
    mf = (S.U[u] * S.V[cols]).sum(1) + S.b[cols]
    return feats(S, mf, S.ucat[u, CAT[cols]], S.uprice[u], cols, D[rows, cols], EV["hb"][e], EV["wk"][e],
                 EV["rain"][e], EV["day"][e])


def _ranker_rows(S, ev, rng, n_neg=6, radius=3.0):
    """Positive = the visited place; negatives = sampled nearby open places not visited (geo-matched)."""
    D = _dist_matrix(ev)
    cand = (D <= radius) & _open_matrix(ev)
    R = rng.random(D.shape)
    R[~cand] = -1
    R[np.arange(len(ev)), EV["p"][ev]] = -1
    neg = np.argsort(-R, axis=1)[:, :n_neg]
    ok = np.take_along_axis(R, neg, 1) > 0
    rows = np.concatenate([np.arange(len(ev)), np.repeat(np.arange(len(ev)), n_neg)[ok.ravel()]])
    cols = np.concatenate([EV["p"][ev], neg.ravel()[ok.ravel()]])
    y = np.concatenate([np.ones(len(ev)), np.zeros(int(ok.sum()))])
    return _pair_feats(S, ev, rows, cols, D), y


def _auc(y, s):
    r = np.argsort(np.argsort(s)) + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(ev, S, k=10, radius=3.0, gbdt=None, gamma=1.0):
    """Offline ranking metrics. Protocol mirrors serving: candidates = open places within `radius`
    of where the user was; the visited place must be retrieved AND ranked in the top k."""
    D = _dist_matrix(ev)
    openm = _open_matrix(ev)
    cand = (D <= radius) & openm
    ar = np.arange(len(ev))
    ep, eu = EV["p"][ev], EV["u"][ev]
    mf = S.U[eu] @ S.V.T + S.b
    scores = {"popularity_global": (np.broadcast_to(S.pop_rate, D.shape), openm),
              "popularity_nearby": (np.broadcast_to(S.pop_rate, D.shape), cand),
              "mf": (mf, cand), "mf_distance": (mf - gamma * D, cand)}
    if gbdt is not None:
        rows, cols = np.nonzero(cand)
        s = np.full(D.shape, -np.inf)
        s[rows, cols] = gbdt.decision_function(_pair_feats(S, ev, rows, cols, D))
        scores["mf_geo_context_gbdt"] = (s, cand)
    out = {}
    for name, (s, m) in scores.items():
        s = np.where(m, s, -np.inf)
        st = s[ar, ep]
        rank = (s > st[:, None]).sum(1)
        hit = m[ar, ep] & (rank < k)
        out[name] = {"hit": hit, "ndcg": hit / np.log2(rank + 2), "retrieved": m[ar, ep]}
    return out


def _summ(res, sel=None):
    rows = []
    for name, r in res.items():
        m = slice(None) if sel is None else sel
        rows.append({"model": name, "recall": float(r["hit"][m].mean()), "ndcg": float(r["ndcg"][m].mean()),
                     "retrieval_recall": float(r["retrieved"][m].mean())})
    return rows


# ---- train everything at import (time-based split) ----
_t0 = time.perf_counter()
_days = EV["day"]
SNAP_A = Snapshot(_days < T_MF, T_MF, seed=1)       # used only to build ranker training features (no leakage)
SNAP = Snapshot(_days < T_TEST, T_TEST, seed=2)     # serving / test-time snapshot
_rng = np.random.default_rng(SEED + 1)
_tr = np.nonzero((_days >= T_MF) & (_days < T_VAL))[0]
_va = np.nonzero((_days >= T_VAL) & (_days < T_TEST))[0]
_Xtr, _ytr = _ranker_rows(SNAP_A, _tr, _rng)
_Xva, _yva = _ranker_rows(SNAP_A, _va, _rng)
GBDT = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, max_leaf_nodes=15, l2_regularization=1.0,
                                      random_state=0).fit(_Xtr, _ytr)
VAL_AUC = _auc(_yva, GBDT.decision_function(_Xva))
# permutation importance on the validation window (AUC drop when one feature is shuffled)
IMPORTANCE = []
for _j, _f in enumerate(FEATURES):
    _Xp = _Xva.copy()
    _Xp[:, _j] = _rng.permutation(_Xp[:, _j])
    IMPORTANCE.append({"feature": _f, "auc_drop": VAL_AUC - _auc(_yva, GBDT.decision_function(_Xp))})
IMPORTANCE.sort(key=lambda r: -r["auc_drop"])
# hand-tuned-style distance decay for the "MF + distance" rung: grid search on the ranker window
_sub = _tr[:: max(1, len(_tr) // 800)]
GAMMA = max([0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
            key=lambda g: evaluate(_sub, SNAP_A, 10, 3.0, None, g)["mf_distance"]["hit"].mean())
TEST = np.nonzero(_days >= T_TEST)[0]
TEST_SUB = TEST[:: max(1, len(TEST) // 1200)]
TRAIN_SECONDS = time.perf_counter() - _t0


@lru_cache(maxsize=16)
def _metrics(k, radius):
    res = evaluate(TEST_SUB, SNAP, k, radius, GBDT, GAMMA)
    ep, eu = EV["p"][TEST_SUB], EV["u"][TEST_SUB]
    home_d = np.linalg.norm(EV["loc"][TEST_SUB] - W["home"][eu], axis=1)
    slices = {"all": None, "new place (no history at train time)": SNAP.cold_place[ep],
              "warm place": ~SNAP.cold_place[ep], "away from home (>3 km)": home_d > 3,
              "near home": home_d <= 3, "light user (<20 check-ins)": SNAP.un[eu] < 20}
    out = {"k": k, "radius": radius, "n_events": int(len(TEST_SUB)), "overall": _summ(res), "slices": []}
    for name, sel in slices.items():
        if sel is not None and sel.sum() < 5:
            continue
        for r in _summ(res, sel):
            out["slices"].append({"slice": name, "n": int(len(TEST_SUB) if sel is None else sel.sum()), **r})
    return out


# --------------------------------------------------------------------------
# 5. ENDPOINTS: overview, map data, user profile, metrics
# --------------------------------------------------------------------------
@router.get("/overview")
def overview():
    m = _metrics(10, 3.0)
    cnt = SNAP.count[OPEN < T_TEST]
    top = np.sort(cnt)[::-1]
    return {
        "dataset": {"places": N_PLACES, "live_places_today": int(LIVE.sum()), "users": N_USERS, "check_ins": N_EV,
                    "days": N_DAYS, "city_km": CITY, "categories": CATS,
                    "places_opened_during_log": int((OPEN > 0).sum()), "places_closed": int((CLOSE < 999).sum()),
                    "median_checkin_distance_km": float(np.median(np.linalg.norm(EV["loc"] - XY[EV["p"]], axis=1))),
                    "share_of_visits_to_top10pct_places": float(top[: len(top) // 10].sum() / top.sum())},
        "split": {"mf_train_days": [0, T_MF], "ranker_train_days": [T_MF, T_VAL], "ranker_val_days": [T_VAL, T_TEST],
                  "test_days": [T_TEST, N_DAYS], "test_events": int(len(TEST)), "evaluated_events": int(len(TEST_SUB))},
        "models": {"retrieval": f"uniform grid, {CELL} km cells ({G}x{G}), radius query + exact filter",
                   "relevance": "BPR matrix factorisation, d=16, numpy SGD, geo-matched negative sampling",
                   "ranker": "HistGradientBoosting (120 trees), 1 positive : 6 nearby negatives",
                   "ranker_rows": int(len(_ytr)), "ranker_val_auc": VAL_AUC, "tuned_distance_decay_gamma": GAMMA,
                   "train_seconds": TRAIN_SECONDS, "features": FEATURES, "feature_groups": {g: [FEATURES[i] for i in ix] for g, ix in GROUPS.items()}},
        "importance": IMPORTANCE,
        "metrics": m,
    }


@router.get("/places")
def places():
    return {"city_km": CITY, "cell_km": CELL, "categories": CATS, "places": [
        {"id": i, "name": W["names"][i], "x": float(XY[i, 0]), "y": float(XY[i, 1]), "cat": CATS[CAT[i]],
         "price": int(PRICE[i]), "rating": float(RATING[i]), "visits": int(SNAP.count[i]),
         "live": bool(LIVE[i]), "new": bool(SNAP.cold_place[i] and LIVE[i])} for i in range(N_PLACES)]}


@router.get("/user/{uid}")
def user(uid: int):
    if not 0 <= uid < N_USERS:
        raise HTTPException(404, "user_id must be 0..%d" % (N_USERS - 1))
    idx = np.nonzero((EV["u"] == uid) & (EV["day"] < T_TEST))[0]
    share = SNAP.ucat[uid]
    recent = [{"day": int(EV["day"][e]), "hour": int(EV["hour"][e]), "place": W["names"][EV["p"][e]],
               "cat": CATS[CAT[EV["p"][e]]], "dist_km": float(np.linalg.norm(EV["loc"][e] - XY[EV["p"][e]]))}
              for e in idx[-8:][::-1]]
    return {"user_id": uid, "home": W["home"][uid].tolist(), "work": W["work"][uid].tolist(),
            "check_ins": int(len(idx)), "avg_price": float(SNAP.uprice[uid]),
            "category_share": [{"cat": CATS[c], "share": float(share[c])} for c in np.argsort(-share)],
            "recent": recent}


@router.get("/metrics")
def metrics(k: int = 10, radius: float = 3.0):
    """Offline recall@K / NDCG@K per model variant and per slice on the time-based test window."""
    return _metrics(int(min(max(k, 1), 50)), float(min(max(radius, 0.5), 8.0)))


# --------------------------------------------------------------------------
# 6. /recommend : the full serving funnel with per-stage counts and latency
#    Concepts: retrieval -> filter -> score -> re-rank -> explore -> diversify;
#    cold-start user (no history) and cold-start place (no interactions);
#    occlusion-style score breakdown (relevance vs distance vs popularity vs context).
# --------------------------------------------------------------------------
class RecReq(BaseModel):
    user_id: int = 0                      # -1 = brand-new user
    onboard_cats: list[str] = []          # cold-start onboarding answers
    x: float = 5.0
    y: float = 5.0
    hour: int = Field(19, ge=0, le=23)
    weekend: bool = False
    rain: bool = False
    budget: int = Field(4, ge=1, le=4)    # max price level
    radius: float = Field(2.0, ge=0.3, le=8.0)
    k: int = Field(10, ge=1, le=30)
    variant: str = "gbdt"                 # popularity | mf | mf_distance | gbdt
    diversity: float = Field(0.0, ge=0, le=1)   # MMR: 0 = pure relevance
    explore: str = "none"                 # none | ucb | epsilon
    explore_strength: float = Field(0.3, ge=0, le=1)
    loc_precision_km: float = Field(0.0, ge=0, le=5)   # privacy: snap location to a coarse cell
    seed: int = 0


def _minmax(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


@router.post("/recommend")
def recommend(q: RecReq):
    S = SNAP
    if q.user_id >= N_USERS or q.user_id < -1:
        raise HTTPException(404, "user_id must be -1 (new user) or 0..%d" % (N_USERS - 1))
    if q.variant not in ("popularity", "mf", "mf_distance", "gbdt"):
        raise HTTPException(400, "unknown variant")
    x, y = min(max(q.x, 0), CITY), min(max(q.y, 0), CITY)
    if q.loc_precision_km > 0:           # location privacy: server only sees the cell centre
        x = (math.floor(x / q.loc_precision_km) + 0.5) * q.loc_precision_km
        y = (math.floor(y / q.loc_precision_km) + 0.5) * q.loc_precision_km
    hb, wk, rain = int(hour_bucket(q.hour)), int(q.weekend), int(q.rain)
    stages = []

    t = time.perf_counter()
    ids, dist, cells, scanned = geo_retrieve(x, y, q.radius)
    stages.append({"stage": "geo retrieval (grid cells)", "in": N_PLACES, "out": int(len(ids)),
                   "ms": (time.perf_counter() - t) * 1e3, "detail": f"{len(cells)} cells, {scanned} places scanned"})

    t = time.perf_counter()
    keep = LIVE[ids] & (PRICE[ids] <= q.budget)
    n_before = len(ids)
    ids, dist = ids[keep], dist[keep]
    stages.append({"stage": "filters (open today, budget)", "in": n_before, "out": int(len(ids)),
                   "ms": (time.perf_counter() - t) * 1e3, "detail": "hard business rules before any model"})
    base = {"location_used": [x, y], "cells": cells, "context": {"bucket": BUCKETS[hb], "weekend": bool(wk), "rain": bool(rain)}}
    if len(ids) == 0:
        return {**base, "stages": stages, "items": [], "summary": {}, "cold_user": q.user_id < 0}

    # --- user tower (with cold-start fallbacks) ---
    t = time.perf_counter()
    cold_user = q.user_id < 0
    if cold_user:
        chosen = [CATS.index(c) for c in q.onboard_cats if c in CATS]
        if chosen:                       # onboarding: average embedding of popular places in the chosen categories
            m = np.isin(CAT, chosen) & (S.count >= 5)
            w = S.count[m]
            uvec = (S.V[m] * w[:, None]).sum(0) / w.sum()
            ucat = 0.4 * S.gcat.copy()
            ucat[chosen] += 0.6 / len(chosen)
        else:                            # nothing known: mean user -> effectively popularity + context
            uvec, ucat = S.cold_u, S.ucat[N_USERS]
        uprice = float(S.uprice[N_USERS])
    else:
        uvec, ucat, uprice = S.U[q.user_id], S.ucat[q.user_id], float(S.uprice[q.user_id])
    mf = S.V[ids] @ uvec + S.b[ids]
    stages.append({"stage": "embedding score (user . place)", "in": int(len(ids)), "out": int(len(ids)),
                   "ms": (time.perf_counter() - t) * 1e3, "detail": "cold user fallback" if cold_user else "learned user vector"})

    # --- ranker ---
    t = time.perf_counter()
    X = feats(S, mf, ucat[CAT[ids]], uprice, ids, dist, hb, wk, rain, TODAY)
    logit = GBDT.decision_function(X)
    contrib = {}
    med = np.median(X, axis=0)
    for g, ix in GROUPS.items():         # occlusion: replace one feature group by the candidate-set median
        Xo = X.copy()
        Xo[:, ix] = med[ix]
        contrib[g] = logit - GBDT.decision_function(Xo)
    raw = {"popularity": np.log1p(S.pop_rate[ids]), "mf": mf, "mf_distance": mf - GAMMA * dist, "gbdt": logit}[q.variant]
    score = _minmax(raw)
    stages.append({"stage": f"rank ({q.variant})", "in": int(len(ids)), "out": int(len(ids)),
                   "ms": (time.perf_counter() - t) * 1e3, "detail": f"{len(FEATURES)} features + 4 occlusion passes"})

    # --- exploration (cold-start places need impressions to ever get data) ---
    t = time.perf_counter()
    rng = np.random.default_rng(q.seed)
    n_p = S.count[ids]
    bonus = np.zeros(len(ids))
    if q.explore == "ucb":               # optimism under uncertainty: bonus shrinks as 1/sqrt(visits)
        bonus = q.explore_strength * 0.3 * np.sqrt(math.log(1 + S.n_events) / (1 + n_p))
    final = score + bonus
    order0 = np.argsort(-score)
    base_top = set(order0[: q.k].tolist())

    # --- MMR diversity re-rank over the head of the list ---
    lam = 1.0 - q.diversity
    pool = np.argsort(-final)[: max(60, q.k * 3)]
    Vn = _unit(S.V[ids[pool]])
    sim = 0.5 * np.clip(Vn @ Vn.T, 0, 1) + 0.5 * (CAT[ids[pool]][:, None] == CAT[ids[pool]][None, :])
    rel = final[pool]
    chosen_ix, max_sim = [], np.zeros(len(pool))
    avail = np.ones(len(pool), bool)
    for _ in range(min(q.k, len(pool))):
        mmr = np.where(avail, lam * rel - (1 - lam) * max_sim, -np.inf)
        j = int(np.argmax(mmr))
        chosen_ix.append(j)
        avail[j] = False
        max_sim = np.maximum(max_sim, sim[j])
    sel = pool[chosen_ix].tolist()
    explored = set()
    if q.explore == "epsilon":           # epsilon-greedy: each slot below the top is swapped with prob eps
        rest = [i for i in range(len(ids)) if i not in set(sel)]
        for slot in range(1, len(sel)):
            if rest and rng.random() < q.explore_strength:
                w = 1.0 / (1.0 + n_p[rest])          # prefer places we know least about
                pick = rest.pop(int(rng.choice(len(rest), p=w / w.sum())))
                sel[slot] = pick
                explored.add(pick)
    elif q.explore == "ucb":
        explored = {i for i in sel if i not in base_top and bonus[i] > 0.05}
    stages.append({"stage": "explore + MMR diversity", "in": int(len(pool)), "out": len(sel),
                   "ms": (time.perf_counter() - t) * 1e3, "detail": f"lambda={lam:.2f}, explore={q.explore}"})

    rank0 = {int(i): r + 1 for r, i in enumerate(order0)}
    sat = None if cold_user else true_sat(q.user_id, ids, hb, wk, rain)
    pop_pct = np.argsort(np.argsort(S.pop_rate)) / (N_PLACES - 1)
    items = []
    for i in sel:
        p = int(ids[i])
        items.append({"id": p, "name": W["names"][p], "cat": CATS[CAT[p]], "price": int(PRICE[p]),
                      "rating": float(RATING[p]), "x": float(XY[p, 0]), "y": float(XY[p, 1]),
                      "dist_km": float(dist[i]), "score": float(final[i]), "p_visit": float(1 / (1 + math.exp(-logit[i]))),
                      "mf_score": float(mf[i]), "visits": int(n_p[i]), "pop_percentile": float(pop_pct[p]),
                      "explore_bonus": float(bonus[i]), "cold_place": bool(S.cold_place[p]),
                      "explored": i in explored, "rank_before_rerank": rank0[int(i)],
                      "contrib": {g: float(c[i]) for g, c in contrib.items()},
                      "true_satisfaction": None if sat is None else float(sat[i])})
    sel_a = np.array(sel)
    pair = sim_full = None
    Vs = _unit(S.V[ids[sel_a]])
    sim_full = 0.5 * np.clip(Vs @ Vs.T, 0, 1) + 0.5 * (CAT[ids[sel_a]][:, None] == CAT[ids[sel_a]][None, :])
    iu = np.triu_indices(len(sel_a), 1)
    pair = sim_full[iu]
    summary = {"categories_in_list": int(len(set(CAT[ids[sel_a]].tolist()))),
               "intra_list_diversity": float(1 - pair.mean()) if len(pair) else 0.0,
               "mean_distance_km": float(dist[sel_a].mean()),
               "mean_popularity_percentile": float(pop_pct[ids[sel_a]].mean()),
               "new_places_in_list": int(S.cold_place[ids[sel_a]].sum()),
               "new_places_in_candidates": int(S.cold_place[ids].sum())}
    if sat is not None:
        best = np.sort(sat)[::-1][: len(sel_a)]
        summary["mean_true_satisfaction"] = float(sat[sel_a].mean())
        summary["oracle_best_satisfaction"] = float(best.mean())
    return {**base, "stages": stages, "items": items, "summary": summary, "cold_user": cold_user,
            "total_ms": float(sum(s["ms"] for s in stages))}


# --------------------------------------------------------------------------
# 7. PITFALL: random vs time-based split
#    Concept: a random split lets the model train on the future (new places'
#    later visits, a user's later habits). Offline numbers look better than
#    anything production will deliver.
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _split_pitfall():
    rng = np.random.default_rng(11)
    is_test = rng.random(N_EV) < 0.2
    S_rand = Snapshot(~is_test, N_DAYS, seed=3)      # "knows" every place and every day
    te = np.nonzero(is_test)[0]
    te = te[:: max(1, len(te) // 1200)]
    r_rand = evaluate(te, S_rand, 10, 3.0, None, GAMMA)
    r_time = evaluate(TEST_SUB, SNAP, 10, 3.0, None, GAMMA)
    rows = []
    for name in ("popularity_nearby", "mf", "mf_distance"):
        a, b = float(r_rand[name]["hit"].mean()), float(r_time[name]["hit"].mean())
        rows.append({"model": name, "random_split_recall@10": a, "time_split_recall@10": b,
                     "optimism": a - b, "relative_inflation_pct": 100 * (a - b) / max(b, 1e-9)})
    return {"rows": rows,
            "unseen_place_share": {"random": float(S_rand.cold_place[EV["p"][te]].mean()),
                                   "time": float(SNAP.cold_place[EV["p"][TEST_SUB]].mean())},
            "wrong": "Random 80/20 split of check-ins: the model trains on visits that happen after the ones it is tested on.",
            "right": f"Time split: train on days < {T_TEST}, test on days {T_TEST}-{N_DAYS - 1}; features are point-in-time.",
            "explain": "The time-split number is the one that predicts launch performance. In the random split almost no test visit lands on a place the model has never seen; in the time split the places that opened after the cutoff are exactly where the model is weakest."}


@router.get("/pitfall/split")
def pitfall_split():
    return _split_pitfall()


# --------------------------------------------------------------------------
# 8. PITFALL: distance bias in implicit labels
#    Concept: users only visit nearby places, so "not visited" mostly means
#    "too far", not "disliked". Uniform negatives turn the embedding into a
#    map of where people live. Geo-matched negatives + explicit distance feature
#    keep taste and geography separate.
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _distance_pitfall():
    S_uni = Snapshot(_days < T_TEST, T_TEST, neg_mode="uniform", seed=2)
    rng = np.random.default_rng(5)
    u = rng.integers(0, N_USERS, 20000)
    p = rng.choice(np.nonzero(~SNAP.cold_place)[0], 20000)
    hd = np.linalg.norm(W["home"][u] - XY[p], axis=1)
    eu = EV["u"][TEST_SUB]
    away = np.linalg.norm(EV["loc"][TEST_SUB] - W["home"][eu], axis=1) > 3
    out = {}
    for key, S in (("wrong", S_uni), ("right", SNAP)):
        s = (S.U[u] * S.V[p]).sum(1) + S.b[p]
        r = evaluate(TEST_SUB, S, 10, 3.0, None, GAMMA)
        out[key] = {"corr(score, -distance_from_home)": float(np.corrcoef(s, -hd)[0, 1]),
                    "corr(score, true_taste_affinity)": float(np.corrcoef(s, W["aff"][u, p])[0, 1]),
                    "recall@10 MF (all)": float(r["mf"]["hit"].mean()),
                    "recall@10 MF (away from home)": float(r["mf"]["hit"][away].mean()),
                    "recall@10 MF+distance (all)": float(r["mf_distance"]["hit"].mean()),
                    "recall@10 MF+distance (away from home)": float(r["mf_distance"]["hit"][away].mean())}
    d = np.linalg.norm(EV["loc"] - XY[EV["p"]], axis=1)
    hist, edges = np.histogram(d, bins=[0, 0.25, 0.5, 1, 1.5, 2, 3, 5, 15])
    return {**out, "n_away_events": int(away.sum()),
            "checkin_distance_hist": [{"km": f"{edges[i]:g}-{edges[i + 1]:g}", "share": float(hist[i] / hist.sum())} for i in range(len(hist))],
            "wrong_label": "BPR with uniform random negatives (any place in the city)",
            "right_label": "BPR with geo-matched negatives (places within 2.5 km of the visited one) + distance as a ranker feature",
            "explain": "With uniform negatives the cheapest way to separate positives from negatives is to learn where the user lives. The embedding score then tracks distance from home rather than taste, and it breaks as soon as the user is somewhere else. Matching negatives on geography removes that shortcut; distance is then modelled explicitly where it can use the user's current location."}


@router.get("/pitfall/distance_bias")
def pitfall_distance():
    return _distance_pitfall()


# --------------------------------------------------------------------------
# 9. PITFALL: position bias in click logs
#    Concept: position-based model, P(click) = P(examined | position) * P(attractive).
#    Naive CTR rewards whatever the old ranker put on top. Fix: estimate the
#    examination curve from a small slice of randomised traffic, then weight
#    clicks by 1 / propensity (IPW).
# --------------------------------------------------------------------------
def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


class PosReq(BaseModel):
    eta: float = Field(1.0, ge=0, le=2)             # examination ~ 1 / rank^eta
    random_frac: float = Field(0.1, ge=0.01, le=0.5)
    sessions: int = Field(6000, ge=500, le=20000)
    seed: int = 0


@router.post("/pitfall/position_bias")
def pitfall_position(q: PosReq):
    rng = np.random.default_rng(100 + q.seed)
    pool = np.nonzero(LIVE & ~SNAP.cold_place)[0]
    pool = pool[np.argsort(-SNAP.count[pool])[:200]]
    attract = 1 / (1 + np.exp(-(1.2 * W["quality"][pool] - 1.0)))         # true P(click | examined)
    prod = np.log1p(SNAP.pop_rate[pool])                                   # production ranker = popularity
    theta = 1.0 / np.arange(1, 11) ** q.eta
    n = q.sessions
    shown = np.argsort(rng.random((n, len(pool))), axis=1)[:, :10]         # 10 distinct candidates per session
    is_rand = rng.random(n) < q.random_frac
    key = np.where(is_rand[:, None], rng.random((n, 10)), -(prod[shown] + rng.normal(0, 0.3, (n, 10))))
    shown = np.take_along_axis(shown, np.argsort(key, axis=1), 1)          # final display order
    clicks = (rng.random((n, 10)) < theta[None, :]) & (rng.random((n, 10)) < attract[shown])
    pos_ctr_all = clicks.mean(0)
    pos_ctr_rand = clicks[is_rand].mean(0)
    theta_hat = np.clip(pos_ctr_rand / max(pos_ctr_rand[0], 1e-9), 0.02, 1.0)   # randomisation -> examination curve
    imp = np.bincount(shown.ravel(), minlength=len(pool)).astype(float)
    clk = np.bincount(shown.ravel(), weights=clicks.ravel().astype(float), minlength=len(pool))
    clk_ipw = np.bincount(shown.ravel(), weights=(clicks / theta_hat[None, :]).ravel(), minlength=len(pool))
    naive = clk / np.maximum(imp, 1)
    ipw = clk_ipw / np.maximum(imp, 1)
    avg_pos = np.bincount(shown.ravel(), weights=np.tile(np.arange(1, 11), n), minlength=len(pool)) / np.maximum(imp, 1)
    best = set(np.argsort(-attract)[:20].tolist())

    def stats(est):
        return {"spearman_vs_true_attractiveness": _spearman(est, attract),
                "mean_abs_error": float(np.abs(est - attract).mean()),
                "top20_overlap_with_truly_best": len(best & set(np.argsort(-est)[:20].tolist())) / 20,
                "corr(estimate, avg_display_position)": float(np.corrcoef(est, avg_pos)[0, 1])}

    worst = np.argsort(naive - attract)[:6]
    return {"wrong": stats(naive), "right": stats(ipw),
            "positions": list(range(1, 11)), "ctr_by_position_logged": pos_ctr_all.tolist(),
            "ctr_by_position_randomised": pos_ctr_rand.tolist(),
            "theta_true": theta.tolist(), "theta_estimated": theta_hat.tolist(),
            "randomised_sessions": int(is_rand.sum()),
            "most_underrated": [{"place": W["names"][pool[i]], "avg_position": float(avg_pos[i]),
                                 "true_attractiveness": float(attract[i]), "naive_ctr": float(naive[i]),
                                 "ipw_ctr": float(ipw[i])} for i in worst],
            "wrong_label": "naive CTR = clicks / impressions", "right_label": "IPW CTR = sum(click / examination propensity) / impressions",
            "explain": "The old ranker sorted by popularity, so popular places sit in slots that get examined and good-but-obscure places sit where nobody looks. Naive CTR mostly measures the slot. Dividing each click by the estimated examination probability of its slot recovers attractiveness, at the price of higher variance for places that only ever appeared low."}


# --------------------------------------------------------------------------
# 10. PITFALL: popularity bias as a feedback loop
#     Concept: the ranker's output becomes its own training data. Raw visit
#     counts as a feature -> shown more -> more visits -> shown more. Fix:
#     position-debiased rate instead of raw count, plus a UCB exploration bonus.
# --------------------------------------------------------------------------
class LoopReq(BaseModel):
    rounds: int = Field(12, ge=2, le=25)
    pop_weight: float = Field(0.6, ge=0, le=2)
    users_per_round: int = Field(250, ge=50, le=500)
    fix: bool = False
    seed: int = 0


def _gini(x):
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() > 0 else 0.0


@lru_cache(maxsize=1)
def _home_candidates():
    D = np.linalg.norm(W["home"][:, None, :] - XY[None, :, :], axis=2)
    return [np.nonzero((D[u] <= 3.0) & LIVE)[0] for u in range(N_USERS)]


@router.post("/pitfall/popularity_loop")
def pitfall_popularity(q: LoopReq):
    rng = np.random.default_rng(200 + q.seed)
    cands = _home_candidates()
    S = SNAP
    theta = 1.0 / np.arange(1, 11) ** THETA_ETA
    count = S.count.copy()                     # wrong: raw historical visit counts keep accumulating
    imp_w = np.zeros(N_PLACES)                 # right: impressions and propensity-weighted clicks
    clk_w = np.zeros(N_PLACES)
    prior = 0.25
    head = np.zeros(N_PLACES, bool)
    head[np.argsort(-S.count)[: N_PLACES // 10]] = True
    ever = np.zeros(N_PLACES, bool)
    hb, wk, rain = 3, 0, 0
    series = []
    for r in range(q.rounds):
        imp_r = np.zeros(N_PLACES)
        sat_sum = clicks = new_imp = 0.0
        for u in rng.integers(0, N_USERS, q.users_per_round):
            c = cands[u]
            if len(c) < 10:
                continue
            mf = S.V[c] @ S.U[u] + S.b[c]
            rel = (mf - mf.mean()) / (mf.std() + 1e-9)
            if q.fix:
                rate = (clk_w[c] + 5 * prior) / (imp_w[c] + 5)
                ucb = 0.5 * np.sqrt(math.log(2 + imp_w.sum()) / (1 + imp_w[c]))
                s = rel + q.pop_weight * (rate - prior) / 0.1 + ucb
            else:
                s = rel + q.pop_weight * np.log1p(count[c])
            top = c[np.argsort(-s)[:10]]
            sat = true_sat(u, top, hb, wk, rain)
            click = (rng.random(10) < theta) & (rng.random(10) < 1 / (1 + np.exp(-(sat - 1.0))))
            imp_r[top] += 1
            imp_w[top] += 1
            clk_w[top] += click / theta
            count[top] += click
            sat_sum += float(sat[click].sum())
            clicks += int(click.sum())
            new_imp += int(S.cold_place[top].sum())
        ever |= imp_r > 0
        live_imp = imp_r[LIVE]
        series.append({"round": r + 1, "gini_impressions": _gini(live_imp),
                       "head_share": float(imp_r[head].sum() / max(imp_r.sum(), 1)),
                       "coverage": float(ever[LIVE].mean()),
                       "new_place_impression_share": float(new_imp / max(imp_r.sum(), 1)),
                       "clicks_per_session": clicks / q.users_per_round,
                       "mean_true_satisfaction_of_clicks": sat_sum / max(clicks, 1)})
    return {"fix": q.fix, "series": series, "final": series[-1],
            "label": ("debiased rate (IPW clicks / impressions) + UCB exploration" if q.fix
                      else "raw visit count as a ranking feature, no exploration"),
            "explain": "Raw counts reward places for having been shown. Each round the head gets a larger share of impressions, catalogue coverage stalls and places that opened recently never collect the data that would let them compete. Rates instead of counts, corrected for position, plus an uncertainty bonus keep the loop open."}
