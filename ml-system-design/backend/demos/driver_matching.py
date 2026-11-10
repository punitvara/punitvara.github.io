"""Matching Drivers with Customers (ride-hailing / delivery dispatch).

What this module teaches (each section below is tagged with the interview concept):
  1. WORLD        - a synthetic 10x10 km city with ground-truth ETA / acceptance / cancellation
                    functions. Policies never see the truth, only model predictions.
  2. MODELS       - GBDT ETA regressor + GBDT acceptance classifier (time-based split, calibration).
  3. GEO INDEX    - grid / geohash / H3-style cell index for candidate retrieval (k-ring lookup).
  4. DISPATCH     - greedy nearest-driver vs batched global assignment (Hungarian algorithm)
                    maximising  EV = P(accept) * value - eta_cost * ETA.
  5. SIMULATION   - minute-by-minute marketplace sim: batching-window trade-off, surge, cold regions.
  6. PITFALL      - selection bias / feedback loop in acceptance logs (we only observe offers that
                    were sent) and how exploration + inverse-propensity weighting fixes it.
"""
import time
from functools import lru_cache

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment  # ships with scikit-learn's scipy dependency
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score

META = {
    "slug": "driver-matching",
    "title": "Ride Dispatch Matching",
    "tagline": "Two-sided marketplace dispatch: ETA + acceptance models, greedy vs batched Hungarian matching, surge, geo index.",
    "icon": "🚕",
    "order": 2,
}
router = APIRouter()

# =====================================================================================
# 1. WORLD  (concept: the simulator's ground truth is hidden from the policy -> the
#    policy must act on *predictions*, exactly like production)
# =====================================================================================
SIZE = 10.0            # city is SIZE x SIZE km
SPEED = 0.5            # free-flow km / minute (30 km/h)
ZONE = 2.0             # surge zones are 2 km squares (5 x 5)
NZ = int(SIZE / ZONE)
MAX_RADIUS = 6.0       # dispatch radius (km, Manhattan) - pairs beyond it are never scored
OFFER_TIMEOUT = 30.0   # seconds a declined offer wastes for both rider and driver
P_ABANDON_AFTER_DECLINE = 0.15   # rider gives up ("no driver found") after each declined offer
HOTSPOTS = [(5.0, 5.0, 1.0, "downtown"), (8.3, 1.7, 0.7, "airport"), (2.0, 8.0, 0.8, "university")]


def rush(hour):
    hour = np.asarray(hour, float)
    return np.exp(-((hour - 8.5) / 1.5) ** 2) + np.exp(-((hour - 17.5) / 1.8) ** 2)


def traffic_index(x, y, hour):
    """Congestion multiplier >= 1. Worst downtown at rush hour."""
    core = np.exp(-((x - 5.0) ** 2 + (y - 5.0) ** 2) / (2 * 2.0 ** 2))
    r = rush(hour)
    return 1.0 + 0.15 * core + 0.25 * r + 0.9 * r * core


def true_eta_mean(dist, traffic):
    """Minutes for a driver to reach the rider (non-linear in traffic: the model must learn it)."""
    return 1.0 + dist / SPEED * traffic ** 1.15


def true_accept_prob(eta, hist_latent, surge, fare, hour):
    """Driver accepts more when: close, well paid (surge/fare), habitually accepting, not late night."""
    night = ((np.asarray(hour) >= 23) | (np.asarray(hour) < 5)).astype(float)
    z = 2.2 + 5.0 * (hist_latent - 0.6) - 0.20 * eta + 1.3 * (surge - 1.0) + 0.05 * (fare - 14.0) - 0.4 * night
    return 1.0 / (1.0 + np.exp(-z))


def true_cancel_prob(eta):
    """Rider cancels after being matched if the pickup ETA is long: 4min->6%, 8->22%, 12->43%."""
    return 1.0 - np.exp(-(np.asarray(eta) / 16.0) ** 2)


def sample_drivers(rng, shape):
    """Drivers are heterogeneous: a latent acceptance propensity (hidden) and the noisy historical
    acceptance rate we can actually measure and feed to the model."""
    latent = rng.beta(3, 2, shape)
    hist = np.clip(latent + rng.normal(0, 0.06, shape), 0.05, 0.99)
    return latent, hist


def manhattan(ax, ay, bx, by):
    return np.abs(ax - bx) + np.abs(ay - by)


def sample_demand(rng, n, hot_share):
    """Rider pickup points: `hot_share` of demand comes from hotspots, the rest is uniform."""
    xy = rng.uniform(0, SIZE, size=(n, 2))
    hot = rng.random(n) < hot_share
    w = np.array([h[2] for h in HOTSPOTS]); w = w / w.sum()
    which = rng.choice(len(HOTSPOTS), size=n, p=w)
    for i in np.where(hot)[0]:
        hx, hy, _, _ = HOTSPOTS[which[i]]
        xy[i] = np.clip(rng.normal([hx, hy], 0.8), 0.05, SIZE - 0.05)
    return xy


def zone_of(x, y):
    return (np.minimum(np.asarray(x) // ZONE, NZ - 1)).astype(int), (np.minimum(np.asarray(y) // ZONE, NZ - 1)).astype(int)


def surge_map(demand_xy, supply_xy):
    """Supply/demand imbalance per zone -> surge multiplier in [1, 2.5]."""
    D = np.zeros((NZ, NZ)); S = np.zeros((NZ, NZ))
    if len(demand_xy):
        zx, zy = zone_of(demand_xy[:, 0], demand_xy[:, 1]); np.add.at(D, (zx, zy), 1)
    if len(supply_xy):
        zx, zy = zone_of(supply_xy[:, 0], supply_xy[:, 1]); np.add.at(S, (zx, zy), 1)
    return np.clip(1.0 + 0.35 * np.maximum(D - S, 0) / (S + 1.0), 1.0, 2.5), D, S


# =====================================================================================
# 2. MODELS  (concept: ETA regression + acceptance classification, time-based split,
#    beat a simple baseline, check calibration because the score is used as a probability)
# =====================================================================================
ETA_FEATS = ["distance_km", "traffic_index", "hour"]
ACC_FEATS = ["distance_km", "traffic_index", "hour", "driver_accept_hist", "surge", "fare"]


def _make_offers(rng, n):
    """Historical offer log from a *randomised* dispatch period (unbiased over distance)."""
    hour = rng.choice(24, size=n, p=_HOUR_P).astype(float) + rng.random(n)
    rider = sample_demand(rng, n, 0.6)
    d = rng.exponential(2.2, n) + 0.1
    ang = rng.uniform(0, 2 * np.pi, n)
    drv = np.clip(rider + np.c_[d * np.cos(ang), d * np.sin(ang)], 0, SIZE)
    dist = manhattan(rider[:, 0], rider[:, 1], drv[:, 0], drv[:, 1])
    mid = (rider + drv) / 2
    traffic = traffic_index(mid[:, 0], mid[:, 1], hour)
    latent, hist = sample_drivers(rng, n)
    surge = np.where(rng.random(n) < 0.65, 1.0, np.clip(1 + rng.exponential(0.4, n), 1, 2.5))
    fare = 6 + 1.8 * rng.gamma(2.0, 2.0, n)
    eta_mu = true_eta_mean(dist, traffic)
    eta = eta_mu * rng.lognormal(0, 0.15, n)
    acc = (rng.random(n) < true_accept_prob(eta_mu, latent, surge, fare, hour)).astype(int)
    X = np.c_[dist, traffic, hour, hist, surge, fare]
    return X, eta, acc


_HOUR_P = 0.4 + rush(np.arange(24) + 0.5); _HOUR_P = _HOUR_P / _HOUR_P.sum()
_rng0 = np.random.default_rng(7)
N_OFFERS = 8000
_X, _ETA, _ACC = _make_offers(_rng0, N_OFFERS)
_CUT = int(N_OFFERS * 0.8)          # rows are in time order -> time-based split, never random
_Xtr, _Xte = _X[:_CUT], _X[_CUT:]

ETA_MODEL = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=0)
ETA_MODEL.fit(_Xtr[:, :3], _ETA[:_CUT])
ETA_BASE = LinearRegression().fit(_Xtr[:, :1], _ETA[:_CUT])   # baseline: distance / average speed
ACC_MODEL = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=0)
ACC_MODEL.fit(_Xtr, _ACC[:_CUT])


def _calibration(y, p, bins=8):
    edges = np.linspace(0, 1, bins + 1); out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum() >= 15:
            out.append({"bucket": f"{a:.2f}-{b:.2f}", "n": int(m.sum()), "predicted": float(p[m].mean()), "observed": float(y[m].mean())})
    return out


def _model_summary():
    eta_p = ETA_MODEL.predict(_Xte[:, :3]); eta_b = ETA_BASE.predict(_Xte[:, :1])
    acc_p = ACC_MODEL.predict_proba(_Xte)[:, 1]
    yte, ete = _ACC[_CUT:], _ETA[_CUT:]
    prior = np.full(len(yte), _ACC[:_CUT].mean())
    return {
        "eta": {
            "model": "GradientBoostingRegressor(120 trees, depth 3)", "features": ETA_FEATS,
            "mae_min": float(mean_absolute_error(ete, eta_p)),
            "baseline_mae_min (linear on distance)": float(mean_absolute_error(ete, eta_b)),
            "mape": float(np.mean(np.abs(ete - eta_p) / ete)),
            "p90_abs_err_min": float(np.quantile(np.abs(ete - eta_p), 0.9)),
            "importance": dict(zip(ETA_FEATS, map(float, ETA_MODEL.feature_importances_))),
        },
        "accept": {
            "model": "GradientBoostingClassifier(120 trees, depth 3)", "features": ACC_FEATS,
            "auc": float(roc_auc_score(yte, acc_p)),
            "logloss": float(log_loss(yte, acc_p)),
            "baseline_logloss (base rate)": float(log_loss(yte, prior)),
            "brier": float(brier_score_loss(yte, acc_p)),
            "importance": dict(zip(ACC_FEATS, map(float, ACC_MODEL.feature_importances_))),
            "calibration": _calibration(yte, acc_p),
        },
    }


_SUMMARY = _model_summary()


def predict_pairs(dist, traffic, hour, hist, surge, fare):
    """Score a batch of (rider, driver) pairs -> predicted ETA (min), P(accept)."""
    hour = np.broadcast_to(hour, dist.shape)
    X = np.c_[dist, traffic, hour, hist, surge, fare]
    return ETA_MODEL.predict(X[:, :3]), ACC_MODEL.predict_proba(X)[:, 1]


@router.get("/overview")
def overview():
    return {
        "city": {"size_km": SIZE, "surge_zone_km": ZONE, "dispatch_radius_km": MAX_RADIUS,
                 "hotspots": [{"x": h[0], "y": h[1], "name": h[3]} for h in HOTSPOTS]},
        "dataset": {
            "historical_offers": N_OFFERS, "train": _CUT, "test (later in time)": N_OFFERS - _CUT,
            "accept_rate": float(_ACC.mean()), "mean_eta_min": float(_ETA.mean()),
            "mean_distance_km": float(_X[:, 0].mean()), "share_surged": float((_X[:, 4] > 1).mean()),
        },
        "models": _SUMMARY,
        "objective": "EV(rider, driver) = P(accept) * fare * surge - eta_cost * ETA_minutes",
    }


class PredictReq(BaseModel):
    distance_km: float = Field(2.0, ge=0, le=15)
    hour: float = Field(18.0, ge=0, le=24)
    traffic_index: float = Field(1.5, ge=1, le=3)
    driver_accept_hist: float = Field(0.7, ge=0, le=1)
    surge: float = Field(1.0, ge=1, le=2.5)
    fare: float = Field(14.0, ge=3, le=60)
    eta_cost: float = Field(0.6, ge=0, le=5)


@router.post("/predict")
def predict(q: PredictReq):
    """What-if: one pair's scores + how the scores move with distance (a partial-dependence slice)."""
    one = lambda a: np.array([a], float)
    eta, p = predict_pairs(one(q.distance_km), one(q.traffic_index), q.hour, one(q.driver_accept_hist), one(q.surge), one(q.fare))
    ds = np.linspace(0.2, 8, 30); o = np.ones_like(ds)
    eta_c, p_c = predict_pairs(ds, o * q.traffic_index, q.hour, o * q.driver_accept_hist, o * q.surge, o * q.fare)
    true_p = true_accept_prob(true_eta_mean(ds, q.traffic_index), q.driver_accept_hist, q.surge, q.fare, q.hour)
    return {
        "eta_min": float(eta[0]), "p_accept": float(p[0]),
        "expected_value": float(p[0] * q.fare * q.surge - q.eta_cost * eta[0]),
        "true_eta_min (hidden from policy)": float(true_eta_mean(q.distance_km, q.traffic_index)),
        "curve": {"distance_km": ds.tolist(), "eta_pred": eta_c.tolist(), "p_accept_pred": p_c.tolist(),
                  "p_accept_true": true_p.tolist(), "ev": (p_c * q.fare * q.surge - q.eta_cost * eta_c).tolist()},
    }


# =====================================================================================
# 3. GEO INDEX  (concept: you cannot score every driver for every request. Bucket drivers
#    into cells (geohash / S2 / H3); a request looks up its cell + neighbours (k-ring),
#    then exact-filters. Cell size trades candidates scanned vs cells touched.)
# =====================================================================================
def morton(cx, cy, bits=8):
    """Interleave bits of (cx, cy): the z-order key that geohash is built on."""
    k = 0
    for b in range(bits):
        k |= ((int(cx) >> b) & 1) << (2 * b) | ((int(cy) >> b) & 1) << (2 * b + 1)
    return k


class GridIndex:
    """CSR-style cell index: drivers sorted by cell id + offsets, so a cell lookup is a slice."""

    def __init__(self, xy, cell_km):
        self.cell = cell_km
        self.n = int(np.ceil(SIZE / cell_km))
        cx = np.minimum((xy[:, 0] // cell_km).astype(int), self.n - 1)
        cy = np.minimum((xy[:, 1] // cell_km).astype(int), self.n - 1)
        cid = cx * self.n + cy
        self.order = np.argsort(cid, kind="stable")
        self.counts = np.bincount(cid, minlength=self.n * self.n)
        self.start = np.concatenate([[0], np.cumsum(self.counts)])
        self.xy = xy

    def ring(self, x, y, radius):
        """Cells overlapping the bounding box of the query circle (a k-ring, k = ceil(r / cell))."""
        c = self.cell
        x0, x1 = max(int((x - radius) // c), 0), min(int((x + radius) // c), self.n - 1)
        y0, y1 = max(int((y - radius) // c), 0), min(int((y + radius) // c), self.n - 1)
        return [(i, j) for i in range(x0, x1 + 1) for j in range(y0, y1 + 1)]

    def query(self, x, y, radius):
        cells = self.ring(x, y, radius)
        parts = [self.order[self.start[i * self.n + j]:self.start[i * self.n + j + 1]] for i, j in cells]
        cand = np.concatenate(parts) if parts else np.empty(0, int)
        d = np.hypot(self.xy[cand, 0] - x, self.xy[cand, 1] - y)
        return cells, cand, cand[d <= radius]


class GeoReq(BaseModel):
    n_drivers: int = Field(3000, ge=50, le=20000)
    cell_km: float = Field(1.0, ge=0.2, le=5.0)
    radius_km: float = Field(1.5, ge=0.2, le=6.0)
    x: float = Field(5.0, ge=0, le=SIZE)
    y: float = Field(5.0, ge=0, le=SIZE)
    seed: int = 1


@router.post("/geo_index")
def geo_index(q: GeoReq):
    rng = np.random.default_rng(q.seed)
    xy = np.r_[sample_demand(rng, q.n_drivers // 2, 0.8), rng.uniform(0, SIZE, (q.n_drivers - q.n_drivers // 2, 2))]
    t0 = time.perf_counter(); idx = GridIndex(xy, q.cell_km); build_ms = (time.perf_counter() - t0) * 1e3
    cells, cand, hits = idx.query(q.x, q.y, q.radius_km)
    brute = np.where(np.hypot(xy[:, 0] - q.x, xy[:, 1] - q.y) <= q.radius_km)[0]
    # latency: average over repeated queries at random points (same numpy code path for both)
    pts = rng.uniform(0, SIZE, (200, 2))
    t0 = time.perf_counter()
    for px, py in pts: idx.query(px, py, q.radius_km)
    idx_us = (time.perf_counter() - t0) / 200 * 1e6
    t0 = time.perf_counter()
    for px, py in pts: np.where(np.hypot(xy[:, 0] - px, xy[:, 1] - py) <= q.radius_km)[0]
    brute_us = (time.perf_counter() - t0) / 200 * 1e6
    occ = idx.counts
    show = rng.choice(q.n_drivers, size=min(q.n_drivers, 3000), replace=False)
    hit_set = set(hits.tolist()); cand_set = set(cand.tolist())
    qc = (min(int(q.x // q.cell_km), idx.n - 1), min(int(q.y // q.cell_km), idx.n - 1))
    return {
        "grid": {"cell_km": q.cell_km, "cells_per_side": idx.n, "total_cells": idx.n ** 2},
        "query_cell": {"cx": qc[0], "cy": qc[1], "morton_key": morton(*qc), "morton_bits": format(morton(*qc), "012b")},
        "ring_cells": [{"cx": i, "cy": j, "drivers": int(occ[i * idx.n + j])} for i, j in cells],
        "counts": {
            "drivers_total (brute-force distance checks)": q.n_drivers,
            "cells_touched": len(cells),
            "candidates_from_cells (distance checks with index)": int(len(cand)),
            "within_radius (exact filter)": int(len(hits)),
            "brute_force_within_radius": int(len(brute)),
            "results_match": bool(set(brute.tolist()) == hit_set),
            "pruned_pct": float(100 * (1 - len(cand) / q.n_drivers)),
            "wasted_checks_pct": float(100 * (1 - len(hits) / max(len(cand), 1))),
        },
        "latency": {"index_build_ms": build_ms, "index_query_us": idx_us, "brute_force_query_us": brute_us},
        "occupancy": {"max_per_cell": int(occ.max()), "mean_per_cell": float(occ.mean()), "empty_cells_pct": float(100 * (occ == 0).mean()),
                      "p99_per_cell": float(np.quantile(occ, 0.99))},
        "drivers": [[round(float(xy[i, 0]), 2), round(float(xy[i, 1]), 2), 2 if i in hit_set else (1 if i in cand_set else 0)] for i in show],
    }


# =====================================================================================
# 4. DISPATCH POLICIES  (concept: dispatch is an *assignment* problem, not per-request ranking.
#    Greedy = each rider grabs the nearest driver (myopic). Batched = collect riders for a
#    window, build the rider x driver value matrix, solve max-weight bipartite matching.)
# =====================================================================================
INFEASIBLE = -1e6


def greedy_assign(dist, banned=None):
    """Riders in FIFO order each take the nearest free driver inside the dispatch radius."""
    nr, nd = dist.shape
    taken = np.zeros(nd, bool); pairs = []
    for r in range(nr):
        d = np.where(taken, np.inf, dist[r])
        if banned is not None: d = np.where(banned[r], np.inf, d)
        j = int(np.argmin(d)) if nd else -1
        if nd and d[j] <= MAX_RADIUS:
            taken[j] = True; pairs.append((r, j))
    return pairs


def batched_assign(score, feasible):
    """Hungarian algorithm on the value matrix; infeasible pairs get a huge negative value."""
    if score.size == 0: return []
    M = np.where(feasible, score, INFEASIBLE)
    rows, cols = linear_sum_assignment(M, maximize=True)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if feasible[r, c]]


def pair_matrices(rxy, dxy, hour, hist, surge_r, fare_r, eta_cost, use_accept=True):
    """Feature build + model scoring for every rider x driver candidate pair."""
    nr, nd = len(rxy), len(dxy)
    dist = manhattan(rxy[:, None, 0], rxy[:, None, 1], dxy[None, :, 0], dxy[None, :, 1])
    mx = (rxy[:, None, 0] + dxy[None, :, 0]) / 2; my = (rxy[:, None, 1] + dxy[None, :, 1]) / 2
    traffic = traffic_index(mx, my, hour)
    feasible = dist <= MAX_RADIUS
    eta = np.zeros((nr, nd)); p = np.zeros((nr, nd))
    if feasible.any():   # only score pairs that survive geo pruning
        H = np.broadcast_to(hist[None, :], (nr, nd)); S = np.broadcast_to(surge_r[:, None], (nr, nd)); F = np.broadcast_to(fare_r[:, None], (nr, nd))
        e, a = predict_pairs(dist[feasible], traffic[feasible], hour, H[feasible], S[feasible], F[feasible])
        eta[feasible] = e; p[feasible] = a
    value = (fare_r * surge_r)[:, None]
    ev = p * value - eta_cost * eta
    score = ev if use_accept else -eta      # pitfall: ignore acceptance -> pure min-ETA matching
    return dist, traffic, feasible, eta, p, ev, score


class TickReq(BaseModel):
    n_drivers: int = Field(30, ge=3, le=120)
    n_riders: int = Field(24, ge=1, le=120)
    hour: float = Field(18.0, ge=0, le=24)
    hot_share: float = Field(0.7, ge=0, le=1, description="share of demand in hotspots (supply is uniform)")
    eta_cost: float = Field(0.6, ge=0, le=5)
    use_accept: bool = True
    surge_on: bool = True
    seed: int = 0


def _tick_kpis(pairs, nr, nd, eta, p, ev, t_eta, t_p, value):
    if not pairs:
        return {"matched": 0, "unmatched_riders": nr}
    r = np.array([a for a, _ in pairs]); d = np.array([b for _, b in pairs])
    te, tp = t_eta[r, d], t_p[r, d]
    tc = true_cancel_prob(te)
    return {
        "matched": len(pairs), "unmatched_riders": nr - len(pairs),
        "avg_pred_eta_min": float(eta[r, d].mean()), "avg_true_eta_min": float(te.mean()),
        "max_true_eta_min": float(te.max()),
        "avg_p_accept (true)": float(tp.mean()),
        "expected_declines": float((1 - tp).sum()),
        "expected_completed_trips": float((tp * (1 - tc)).sum()),
        "expected_cancel_rate": float((tp * tc).sum() / max(tp.sum(), 1e-9)),
        "sum_model_EV": float(ev[r, d].sum()),
        "true_expected_value": float((tp * (1 - tc) * value[r]).sum()),
    }


@router.post("/tick")
def tick(q: TickReq):
    """One matching round on a frozen snapshot: same riders/drivers, two policies, internals exposed."""
    rng = np.random.default_rng(1000 + q.seed)
    rxy = sample_demand(rng, q.n_riders, q.hot_share)
    dxy = rng.uniform(0, SIZE, (q.n_drivers, 2))             # supply spread out -> imbalance vs hot demand
    latent, hist = sample_drivers(rng, q.n_drivers)
    fare = 6 + 1.8 * rng.gamma(2.0, 2.0, q.n_riders)
    smap, D, S = surge_map(rxy, dxy)
    if not q.surge_on: smap = np.ones_like(smap)
    zx, zy = zone_of(rxy[:, 0], rxy[:, 1]); surge_r = smap[zx, zy]

    t0 = time.perf_counter()
    dist0 = manhattan(rxy[:, None, 0], rxy[:, None, 1], dxy[None, :, 0], dxy[None, :, 1])
    t1 = time.perf_counter()
    dist, traffic, feas, eta, p, ev, score = pair_matrices(rxy, dxy, q.hour, hist, surge_r, fare, q.eta_cost, q.use_accept)
    t2 = time.perf_counter()
    g_pairs = greedy_assign(dist0); t3 = time.perf_counter()
    b_pairs = batched_assign(score, feas); t4 = time.perf_counter()

    # ground truth (hidden from the policies) used only to grade them
    t_eta = true_eta_mean(dist, traffic)
    t_p = true_accept_prob(t_eta, latent[None, :], surge_r[:, None], fare[:, None], q.hour)
    value = fare * surge_r

    def rows(pairs):
        return [{"rider": r, "driver": d, "dist_km": float(dist[r, d]), "eta_pred": float(eta[r, d]), "p_accept": float(p[r, d]),
                 "ev": float(ev[r, d]), "eta_true": float(t_eta[r, d]), "p_accept_true": float(t_p[r, d])} for r, d in pairs]

    return {
        "drivers": [{"x": float(x), "y": float(y), "hist": float(h)} for (x, y), h in zip(dxy, hist)],
        "riders": [{"x": float(x), "y": float(y), "fare": float(f), "surge": float(s)} for (x, y), f, s in zip(rxy, fare, surge_r)],
        "surge_zones": [{"zx": i, "zy": j, "surge": float(smap[i, j]), "demand": int(D[i, j]), "supply": int(S[i, j])} for i in range(NZ) for j in range(NZ)],
        "zone_km": ZONE, "size_km": SIZE,
        "greedy": {"pairs": rows(g_pairs), "kpis": _tick_kpis(g_pairs, q.n_riders, q.n_drivers, eta, p, ev, t_eta, t_p, value)},
        "batched": {"pairs": rows(b_pairs), "kpis": _tick_kpis(b_pairs, q.n_riders, q.n_drivers, eta, p, ev, t_eta, t_p, value)},
        "candidates": {"all_pairs": int(dist.size), "after_radius_pruning": int(feas.sum()), "scored_by_models": int(feas.sum()),
                       "objective": "P(accept)*fare*surge - eta_cost*ETA" if q.use_accept else "-ETA  (acceptance ignored)"},
        "latency_ms": {"greedy: distance matrix": (t1 - t0) * 1e3, "greedy: assign": (t3 - t2) * 1e3,
                       "batched: features + 2 GBDT scoring": (t2 - t1) * 1e3, "batched: Hungarian solve": (t4 - t3) * 1e3},
    }


# =====================================================================================
# 5. SIMULATION  (concept: matching quality is a *system* outcome. Longer batching window ->
#    better global assignment (shorter pickups) but riders wait for the batch to close.
#    Surge both raises driver acceptance and sheds demand; idle drivers drift to surged zones.)
# =====================================================================================
DT = 5.0  # seconds per simulation step


def simulate(policy, window_s, minutes, n_drivers, rate_per_min, hour, hot_share, eta_cost, use_accept, surge_on, seed):
    rng = np.random.default_rng(5000 + seed)      # world randomness (identical across policies)
    T = minutes * 60.0
    n_arr = rng.poisson(rate_per_min * minutes)
    t_req = np.sort(rng.uniform(0, T, n_arr))
    rxy = sample_demand(rng, n_arr, hot_share)
    dest = sample_demand(rng, n_arr, 0.25)        # trips end all over town -> drivers leak to cold regions
    trip_km = manhattan(rxy[:, 0], rxy[:, 1], dest[:, 0], dest[:, 1])
    fare = 5 + 1.6 * trip_km
    patience = np.clip(rng.exponential(150, n_arr) + 60, 60, 480)     # seconds a rider waits unmatched
    u_price = rng.random(n_arr); u_cancel = rng.random(n_arr); eta_noise = rng.lognormal(0, 0.15, n_arr)
    u_abandon = rng.random((n_arr, 8))            # one draw per decline a rider suffers
    dxy = rng.uniform(0, SIZE, (n_drivers, 2))
    latent, hist = sample_drivers(rng, n_drivers)
    arng = np.random.default_rng(9000 + seed)     # acceptance coin flips

    # warm start: 60% of drivers are mid-trip at t=0 so the sim starts near steady state
    free_at = np.where(rng.random(n_drivers) < 0.6, rng.uniform(0, 720, n_drivers), 0.0)   # available when free_at <= t
    status = np.zeros(n_arr, int)                 # 0 not arrived, 1 waiting, 2 done, 3 cancelled, 4 priced out
    avail_at = t_req.copy()                       # rider can be (re)offered at this time
    r_surge = np.ones(n_arr)
    banned = np.zeros((n_arr, n_drivers), bool)   # pairs already declined
    n_decl = np.zeros(n_arr, int)
    smap = np.ones((NZ, NZ))
    waits, etas, queue_delays = [], [], []
    declines = offers = cancel_wait = cancel_post = cancel_decl = priced_out = completed = 0
    revenue = 0.0; on_trip_s = 0.0; deadhead_s = 0.0
    series = {"minute": [], "waiting": [], "free_drivers": [], "max_surge": []}
    nxt = 0; step = 0; win_steps = max(int(round(window_s / DT)), 1)

    t = 0.0
    while t < T:
        # --- surge refresh every 30 s from recent demand vs free supply ---
        if step % 6 == 0 and surge_on:
            recent = (t_req <= t) & (t_req > t - 180) & (status != 4)
            smap, _, _ = surge_map(rxy[recent], dxy[free_at <= t])
        # --- arrivals (price-sensitive riders drop out when surged = demand shedding) ---
        while nxt < n_arr and t_req[nxt] <= t:
            zx, zy = zone_of(rxy[nxt, 0], rxy[nxt, 1]); s = smap[zx, zy]; r_surge[nxt] = s
            if u_price[nxt] < 0.45 * (s - 1.0): status[nxt] = 4; priced_out += 1
            else: status[nxt] = 1
            nxt += 1
        # --- unmatched riders run out of patience ---
        w = np.where(status == 1)[0]
        gone = w[(t - t_req[w]) > patience[w]]
        status[gone] = 3; cancel_wait += len(gone)
        # --- matching round ---
        if policy == "greedy" or step % win_steps == 0:
            R = np.where((status == 1) & (avail_at <= t))[0]
            Dv = np.where(free_at <= t)[0]
            if len(R) and len(Dv):
                if policy == "greedy":
                    dist = manhattan(rxy[R, None, 0], rxy[R, None, 1], dxy[None, Dv, 0], dxy[None, Dv, 1])
                    pairs = greedy_assign(dist, banned[np.ix_(R, Dv)])
                else:
                    dist, _, feas, _, _, _, score = pair_matrices(rxy[R], dxy[Dv], hour, hist[Dv], r_surge[R], fare[R], eta_cost, use_accept)
                    pairs = batched_assign(score, feas & ~banned[np.ix_(R, Dv)])
                for a, b in pairs:
                    r, d = R[a], Dv[b]; offers += 1
                    dkm = dist[a, b]
                    tr = traffic_index((rxy[r, 0] + dxy[d, 0]) / 2, (rxy[r, 1] + dxy[d, 1]) / 2, hour)
                    eta_mu = true_eta_mean(dkm, tr); eta = eta_mu * eta_noise[r]
                    if arng.random() > true_accept_prob(eta_mu, latent[d], r_surge[r], fare[r], hour):
                        declines += 1; banned[r, d] = True            # decline: both sides lose OFFER_TIMEOUT
                        free_at[d] = t + OFFER_TIMEOUT; avail_at[r] = t + OFFER_TIMEOUT
                        if u_abandon[r, min(n_decl[r], 7)] < P_ABANDON_AFTER_DECLINE: status[r] = 3; cancel_decl += 1
                        n_decl[r] += 1
                        continue
                    if u_cancel[r] < true_cancel_prob(eta):            # rider bails on a long pickup
                        status[r] = 3; cancel_post += 1
                        free_at[d] = t + 0.4 * eta * 60; deadhead_s += 0.4 * eta * 60
                        dxy[d] = dxy[d] + 0.4 * (rxy[r] - dxy[d])
                        continue
                    trip_min = trip_km[r] / SPEED * traffic_index((rxy[r, 0] + dest[r, 0]) / 2, (rxy[r, 1] + dest[r, 1]) / 2, hour)
                    status[r] = 2; completed += 1
                    queue_delays.append((t - t_req[r]) / 60); etas.append(eta); waits.append((t - t_req[r]) / 60 + eta)
                    revenue += fare[r] * r_surge[r]
                    start = t + eta * 60; end = start + trip_min * 60
                    deadhead_s += max(min(start, T) - t, 0); on_trip_s += max(min(end, T) - min(start, T), 0)
                    free_at[d] = end; dxy[d] = dest[r]
        # --- idle drivers reposition toward surged zones (the supply-side response to surge) ---
        if surge_on and step % 6 == 0 and smap.max() > 1.15:
            idle = np.where(free_at <= t)[0]
            hz = np.argwhere(smap > 1.15)
            if len(idle) and len(hz):
                centers = (hz + 0.5) * ZONE
                dd = manhattan(dxy[idle, None, 0], dxy[idle, None, 1], centers[None, :, 0], centers[None, :, 1])
                tgt = centers[np.argmin(dd, axis=1)]; vec = tgt - dxy[idle]
                norm = np.maximum(np.abs(vec).sum(1, keepdims=True), 1e-9)
                dxy[idle] = dxy[idle] + vec / norm * np.minimum(norm, SPEED * 0.5 * 0.7)   # 30 s at 70% speed
        if step % 12 == 0:
            series["minute"].append(t / 60); series["waiting"].append(int((status == 1).sum()))
            series["free_drivers"].append(int((free_at <= t).sum())); series["max_surge"].append(float(smap.max()))
        t += DT; step += 1

    unmatched = int((status == 1).sum())
    req = max(n_arr - priced_out, 1)
    return {
        "kpis": {
            "requests": int(n_arr - priced_out), "completed_trips": completed,
            "avg_pickup_eta_min": float(np.mean(etas)) if etas else 0.0,
            "avg_queue_delay_min": float(np.mean(queue_delays)) if queue_delays else 0.0,
            "avg_total_wait_min": float(np.mean(waits)) if waits else 0.0,
            "p90_total_wait_min": float(np.quantile(waits, 0.9)) if waits else 0.0,
            "cancel_rate": (cancel_wait + cancel_post + cancel_decl) / req,
            "cancel_while_waiting": cancel_wait, "cancel_after_match": cancel_post, "cancel_after_decline": cancel_decl,
            "decline_rate": declines / max(offers, 1), "offers_sent": offers,
            "unmatched_at_end": unmatched, "priced_out": priced_out,
            "utilization": on_trip_s / (n_drivers * T), "deadhead_share": deadhead_s / (n_drivers * T),
            "revenue": revenue, "completion_rate": completed / req,
        },
        "series": series,
    }


def _replicate(policy, window_s, use_accept, reps, kw):
    """Average KPIs over `reps` seeds (same seeds for every policy = paired comparison); series from the first."""
    kw = dict(kw); seed = kw.pop("seed")
    runs = [simulate(policy, window_s, use_accept=use_accept, seed=seed * 10 + i, **kw) for i in range(reps)]
    kp = {k: float(np.mean([r["kpis"][k] for r in runs])) for k in runs[0]["kpis"]}
    return {"kpis": kp, "series": runs[0]["series"]}


class SimReq(BaseModel):
    minutes: int = Field(30, ge=5, le=45)
    n_drivers: int = Field(60, ge=5, le=120)
    rate_per_min: float = Field(3.0, ge=0.5, le=10)
    window_s: int = Field(20, ge=5, le=120)
    replications: int = Field(3, ge=1, le=5, description="independent seeds averaged (KPIs are noisy)")
    hour: float = Field(18.0, ge=0, le=24)
    hot_share: float = Field(0.7, ge=0, le=1)
    eta_cost: float = Field(0.6, ge=0, le=5)
    use_accept: bool = True
    surge_on: bool = True
    seed: int = 0


@router.post("/simulate")
def run_sim(q: SimReq):
    """Greedy (instant dispatch) vs batched Hungarian on the *same* stream of riders."""
    kw = dict(minutes=q.minutes, n_drivers=q.n_drivers, rate_per_min=q.rate_per_min, hour=q.hour, hot_share=q.hot_share,
              eta_cost=q.eta_cost, surge_on=q.surge_on, seed=q.seed)
    t0 = time.perf_counter()
    g = _replicate("greedy", DT, True, q.replications, kw)
    b = _replicate("batched", q.window_s, q.use_accept, q.replications, kw)
    return {"greedy": g, "batched": b, "window_s": q.window_s, "replications": q.replications, "sim_ms": (time.perf_counter() - t0) * 1e3,
            "objective": "P(accept)*value - eta_cost*ETA" if q.use_accept else "-ETA (acceptance ignored)"}


class SweepReq(BaseModel):
    minutes: int = Field(25, ge=5, le=30)
    n_drivers: int = Field(60, ge=5, le=120)
    rate_per_min: float = Field(3.0, ge=0.5, le=10)
    replications: int = Field(3, ge=1, le=4)
    hour: float = Field(18.0, ge=0, le=24)
    hot_share: float = Field(0.7, ge=0, le=1)
    eta_cost: float = Field(0.6, ge=0, le=5)
    use_accept: bool = True
    surge_on: bool = True
    seed: int = 0


SWEEP_WINDOWS = [5, 10, 20, 30, 45, 60, 90]


@router.post("/window_sweep")
def window_sweep(q: SweepReq):
    """Batching-window trade-off curve: pickup ETA falls, queueing delay rises, total wait is U-shaped."""
    kw = dict(minutes=q.minutes, n_drivers=q.n_drivers, rate_per_min=q.rate_per_min, hour=q.hour, hot_share=q.hot_share,
              eta_cost=q.eta_cost, surge_on=q.surge_on, seed=q.seed)
    g = _replicate("greedy", DT, True, q.replications, kw)["kpis"]
    keys = ["avg_pickup_eta_min", "avg_queue_delay_min", "avg_total_wait_min", "cancel_rate", "utilization", "decline_rate", "completion_rate", "completed_trips"]
    rows = []
    for w in SWEEP_WINDOWS:
        k = _replicate("batched", w, q.use_accept, q.replications, kw)["kpis"]
        rows.append({"window_s": w, **{key: k[key] for key in keys}})
    best = min(rows, key=lambda r: r["avg_total_wait_min"])
    return {"windows": SWEEP_WINDOWS, "batched": rows, "greedy": {key: g[key] for key in keys}, "best_window_s (min total wait)": best["window_s"]}


# =====================================================================================
# 6. PITFALL: SELECTION BIAS / FEEDBACK LOOP IN ACCEPTANCE LOGS
#    We only observe accept/decline for offers the *current policy* chose to send. Under
#    nearest-driver dispatch, far offers are only ever sent when supply is scarce - exactly
#    when surge is high - so the log confounds "far" with "well paid". A model trained on it
#    over-predicts acceptance for far, un-surged pairs, which a batched matcher then starts
#    sending (and the new log is shaped by that model: a feedback loop).
#    Fix: epsilon-exploration (send a random candidate sometimes, log the propensity) and
#    optionally inverse-propensity weighting (IPS) at training time.
# =====================================================================================
K_CAND = 6


def _logged_offers(rng, n, epsilon):
    """Simulate a dispatch log. Each request has K candidate drivers; the logging policy sends the
    nearest one (prob 1-eps) or a uniformly random one (prob eps). Returns only the SENT offers."""
    hour = rng.choice(24, size=n, p=_HOUR_P).astype(float) + rng.random(n)
    scarcity = rng.gamma(2.0, 0.9, n)[:, None]                     # per-request supply scarcity (km scale)
    dist = np.sort(rng.exponential(1.0, (n, K_CAND)) * scarcity + 0.1, axis=1)
    surge = np.clip(1 + 0.45 * np.maximum(dist[:, 0] - 1.2, 0), 1, 2.5)   # scarce supply -> surge
    latent, hist = sample_drivers(rng, (n, K_CAND))
    traffic = 1 + 0.2 * rng.random((n, K_CAND)) + 0.6 * rush(hour)[:, None] * rng.random((n, K_CAND))
    fare = 6 + 1.8 * rng.gamma(2.0, 2.0, n)
    explore = rng.random(n) < epsilon
    pick = np.where(explore, rng.integers(0, K_CAND, n), 0)        # column 0 is the nearest
    prop = np.where(pick == 0, (1 - epsilon) + epsilon / K_CAND, epsilon / K_CAND)   # logged propensity
    i = np.arange(n)
    d, tr, h, la = dist[i, pick], traffic[i, pick], hist[i, pick], latent[i, pick]
    y = (rng.random(n) < true_accept_prob(true_eta_mean(d, tr), la, surge, fare, hour)).astype(int)
    return np.c_[d, tr, hour, h, surge, fare], y, prop


def _target_pairs(rng, n):
    """What a batched matcher must score: ALL candidate pairs, including far drivers at surge 1.0."""
    X, y, _ = _logged_offers(rng, n, 1.0)
    return X, y


@lru_cache(maxsize=32)
def _bias_experiment(eps_pct: int, ips: bool):
    eps = eps_pct / 100.0
    rng = np.random.default_rng(321)
    Xte, yte = _target_pairs(np.random.default_rng(99), 4000)
    Xb, yb, pb = _logged_offers(rng, 4000, eps)
    w = None
    if ips and eps > 0:
        w = np.minimum(1.0 / pb, 30.0); w = w / w.mean()          # clipped + self-normalised IPS
    m = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.15, random_state=0).fit(Xb, yb, sample_weight=w)
    p = np.clip(m.predict_proba(Xte)[:, 1], 1e-4, 1 - 1e-4)
    edges = [0, 1, 2, 3, 4.5, 6.5, 99]; buckets = []
    for a, b in zip(edges[:-1], edges[1:]):
        mt = (Xte[:, 0] >= a) & (Xte[:, 0] < b); ml = (Xb[:, 0] >= a) & (Xb[:, 0] < b)
        if mt.sum() < 20: continue
        buckets.append({"distance_km": f"{a}-{b if b < 99 else '+'}", "train_rows": int(ml.sum()),
                        "train_mean_surge": float(Xb[ml, 4].mean()) if ml.any() else 0.0,
                        "target_mean_surge": float(Xte[mt, 4].mean()),
                        "predicted": float(p[mt].mean()), "actual": float(yte[mt].mean()),
                        "error": float(p[mt].mean() - yte[mt].mean())})
    far = Xte[:, 0] >= 3
    return {
        "epsilon": eps, "ips_weighting": bool(ips and eps > 0), "train_rows": int(len(yb)),
        "explored_rows": int((pb < 0.5).sum()) if eps > 0 else 0,
        "train_share_far (>3km)": float((Xb[:, 0] >= 3).mean()), "target_share_far (>3km)": float(far.mean()),
        "corr(distance, surge) in train": float(np.corrcoef(Xb[:, 0], Xb[:, 4])[0, 1]),
        "corr(distance, surge) in target": float(np.corrcoef(Xte[:, 0], Xte[:, 4])[0, 1]),
        "auc": float(roc_auc_score(yte, p)), "logloss": float(log_loss(yte, p)),
        "far_pairs_predicted": float(p[far].mean()), "far_pairs_actual": float(yte[far].mean()),
        "far_pairs_overprediction": float(p[far].mean() - yte[far].mean()),
        "by_distance": buckets,
    }


class BiasReq(BaseModel):
    epsilon: float = Field(0.1, ge=0.0, le=0.5, description="share of offers sent to a random candidate")
    ips: bool = True


@router.post("/pitfall/selection_bias")
def selection_bias(q: BiasReq):
    """Wrong way (train on greedy-policy log) vs right way (exploration [+ IPS]); both graded on all candidate pairs."""
    eps_pct = int(round(q.epsilon * 100))
    wrong = _bias_experiment(0, False)
    right = _bias_experiment(eps_pct, q.ips) if eps_pct > 0 else wrong
    oracle = _bias_experiment(100, False)
    return {
        "wrong (greedy log, no exploration)": wrong,
        "right (exploration)": right,
        "upper_bound (fully randomised log)": {k: oracle[k] for k in ["auc", "logloss", "far_pairs_overprediction"]},
        "exploration_cost": f"{eps_pct}% of offers go to a non-nearest driver (longer ETA on those trips)",
        "explanation": "Graded on ALL candidate pairs (what the matcher scores), not on the logged distribution.",
    }
