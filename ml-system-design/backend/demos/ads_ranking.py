"""Ads Ranking for the News Feed - interactive demo backend.

Interview concepts illustrated (search for the "# ==" section banners):
  1. Synthetic impression log with HIDDEN ground truth (true pCTR at position 1, examination
     probability per position, pCVR with delayed conversions, "hide ad" negative feedback).
  2. Feature hashing of sparse id features + crosses (hand-written hashing trick).
  3. Three pCTR model families + the Facebook hybrid:
       lr_hash  - logistic regression on hashed crosses (numpy Adagrad; online-learnable)
       gbdt     - sklearn HistGradientBoosting (learns crosses itself, batch only)
       fm       - factorization machine written in numpy (generalises to unseen crosses)
       gbdt_lr  - GBDT leaf indices fed as sparse features into the hashed LR (He et al. 2014)
  4. Metrics: log-loss, AUC, Normalized Entropy (NE), calibration ratio, reliability curves,
     Platt / isotonic calibration.
  5. Auction: rank by eCPM = bid * pCTR (+ user-value terms), GSP pricing, vs bid-only / pCTR-only.
  6. Budget pacing (throttling vs bid modification).
  7. Pitfalls: position bias, miscalibration + winner's curse, negative downsampling without
     correction, delayed conversion labels.
"""
import threading
import time

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

META = {
    "slug": "ads-ranking",
    "title": "Ads Ranking for the News Feed",
    "tagline": "pCTR models, calibration, eCPM auctions with GSP pricing, pacing and position bias",
    "icon": "\U0001F4E2",
    "order": 4,
}
router = APIRouter()

# ====================================================================================
# 1. SYNTHETIC WORLD  (concept: logged impressions are biased samples of a hidden truth)
# ====================================================================================
SEED = 7
N = 40000
N_ADS = 60
DAYS = 14                      # train: days 0-9, validation (calibration): 10-11, test: 12-13
EXAM = np.array([1.0, 0.72, 0.52, 0.38, 0.28])   # P(user examines slot k) - the position bias
AGES = ["18-24", "25-34", "35-44", "45-54", "55+"]
COUNTRIES = ["US", "IN", "BR", "DE"]
CATS = ["gaming", "fashion", "travel", "finance", "fitness", "food", "autos", "tech"]
DEVICES = ["mobile", "desktop"]
HOURS = ["morning", "midday", "evening", "night"]
CREATIVES = ["image", "video", "carousel"]
FIELDS = ["age", "country", "interest", "device", "hour", "ad", "ad_cat", "creative", "advertiser"]
CROSSES = [("interest", "ad_cat"), ("age", "ad_cat"), ("device", "creative"), ("interest", "ad"),
           ("age", "ad"), ("hour", "ad_cat"), ("country", "ad_cat")]
CARD = {"age": 5, "country": 4, "interest": 8, "device": 2, "hour": 4, "ad": N_ADS, "ad_cat": 8,
        "creative": 3, "advertiser": 20, "pos": 5}

_rng = np.random.default_rng(SEED)
AD = {
    "cat": _rng.integers(0, 8, N_ADS),
    "creative": _rng.integers(0, 3, N_ADS),
    "advertiser": _rng.integers(0, 20, N_ADS),
    "q": _rng.normal(0, 0.45, N_ADS),                    # hidden creative quality
    "clickbait": _rng.random(N_ADS) < 0.2,               # high CTR *and* high hide-rate
    "bid": np.round(np.exp(_rng.normal(0, 0.5, N_ADS)), 2),   # CPC bid in $
    "cvr_q": _rng.normal(0, 0.5, N_ADS),
    "delay": _rng.uniform(0.5, 4.0, N_ADS),              # mean click->conversion delay (days)
}
AD_NAMES = [f"ad{i:02d}-{CATS[AD['cat'][i]]}" + ("*" if AD["clickbait"][i] else "") for i in range(N_ADS)]
W_AGE_CAT = _rng.normal(0, 0.4, (5, 8))
W_DEV_CRE = _rng.normal(0, 0.3, (2, 3))
W_HOUR_CAT = _rng.normal(0, 0.2, (4, 8))
W_COUNTRY = _rng.normal(0, 0.2, 4)
W_CVR_CAT = _rng.normal(0, 0.3, 8)


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def with_ad(cols, ad):
    """Attach ad-side columns for an array of ad ids."""
    c = dict(cols)
    c["ad"] = ad
    c["ad_cat"] = AD["cat"][ad]
    c["creative"] = AD["creative"][ad]
    c["advertiser"] = AD["advertiser"][ad]
    return c


def true_p1(c, day=DAYS - 1):
    """Hidden truth: P(click | examined) == CTR the ad would get in slot 1."""
    match = (c["interest"] == c["ad_cat"]).astype(float)
    z = (-3.1 + 1.3 * match + AD["q"][c["ad"]] + 0.8 * AD["clickbait"][c["ad"]]
         + W_AGE_CAT[c["age"], c["ad_cat"]] + W_DEV_CRE[c["device"], c["creative"]]
         + W_HOUR_CAT[c["hour"], c["ad_cat"]] + W_COUNTRY[c["country"]]
         - 0.3 * (np.asarray(day) >= 10))      # market shift on day 10: models trained on days 0-9 over-predict
    return sig(z)


def true_hide1(c):
    """Hidden truth: P(user hides / reports the ad | examined). Clickbait ads are hidden more."""
    match = (c["interest"] == c["ad_cat"]).astype(float)
    return sig(-3.6 + 1.8 * AD["clickbait"][c["ad"]] - 0.9 * match)


def true_cvr(c):
    match = (c["interest"] == c["ad_cat"]).astype(float)
    return sig(-1.2 + AD["cvr_q"][c["ad"]] + 0.6 * match + W_CVR_CAT[c["ad_cat"]])


def _generate():
    r = np.random.default_rng(SEED + 1)
    c = {"age": r.integers(0, 5, N), "country": r.choice(4, N, p=[.4, .3, .2, .1]),
         "interest": r.integers(0, 8, N), "device": r.choice(2, N, p=[.75, .25]),
         "hour": r.integers(0, 4, N)}
    t = np.sort(r.uniform(0, DAYS, N))
    # Logging policy = the OLD production system: higher bidders are shown more often, half of the
    # traffic is interest-targeted, and high bidders get the TOP slots (-> position is confounded
    # with the ad, which is exactly what makes naive training on clicks biased).
    pop = AD["bid"] / AD["bid"].sum()
    ad = r.choice(N_ADS, N, p=pop)
    targeted = r.random(N) < 0.5
    for k in range(8):
        m = targeted & (c["interest"] == k)
        cand = np.where(AD["cat"] == k)[0]
        if len(cand):
            ad[m] = r.choice(cand, m.sum(), p=AD["bid"][cand] / AD["bid"][cand].sum())
    c = with_ad(c, ad)
    bid_rank = np.argsort(np.argsort(-AD["bid"])) / (N_ADS - 1)          # 0 = highest bidder
    pos = np.clip(np.rint(1 + 4 * bid_rank[ad] + r.normal(0, 1.2, N)), 1, 5).astype(int)   # noise = the only randomisation
    c["pos"] = pos - 1
    day = t.astype(int)
    p1 = true_p1(c, day)
    click = (r.random(N) < p1 * EXAM[pos - 1]).astype(float)
    hide = (r.random(N) < true_hide1(c) * EXAM[pos - 1]).astype(float)
    conv = (click * (r.random(N) < true_cvr(c))).astype(float)
    delay = r.exponential(AD["delay"][ad])
    return c, t, day, p1, click, hide, conv, delay


COLS, T, DAY, P1, CLICK, HIDE, CONV, DELAY = _generate()
TR, VA, TE = DAY < 10, (DAY >= 10) & (DAY < 12), DAY >= 12


def sub(c, m):
    return {k: v[m] for k, v in c.items()}


# ====================================================================================
# 2. FEATURE HASHING  (concept: hashing trick - fixed memory for unbounded id/cross vocab)
# ====================================================================================
def _mix64(x):
    """splitmix64 finaliser, vectorised. uint64 arithmetic wraps, which is what we want."""
    x = (x + np.uint64(0x9E3779B97F4A7C15))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def raw_keys(c, use_pos=True, leaves=None):
    """One integer key per (field,value) and per (cross, value_a, value_b) token -> (n, K) uint64."""
    ks = [np.uint64(i * 1000003) + c[f].astype(np.uint64) for i, f in enumerate(FIELDS)]
    ks += [np.uint64((100 + j) * 1000003) + (c[a] * 1009 + c[b]).astype(np.uint64)
           for j, (a, b) in enumerate(CROSSES)]
    if use_pos:      # position enters ONLY as an additive unigram ("shallow tower"), never crossed
        ks.append(np.uint64(99 * 1000003) + c["pos"].astype(np.uint64))
    if leaves is not None:   # GBDT+LR: every tree contributes the id of the leaf the row fell into
        ks += [np.uint64((200 + j) * 1000003) + leaves[:, j].astype(np.uint64) for j in range(leaves.shape[1])]
    return np.stack(ks, axis=1)


def hashed(c, bits, use_pos=True, leaves=None):
    with np.errstate(over="ignore"):
        h = _mix64(raw_keys(c, use_pos, leaves))
    return (h & np.uint64((1 << bits) - 1)).astype(np.int64)


def train_lr(idx, y, nb, epochs=6, lr=0.05, l2=1e-6, sw=None, seed=0):
    """Logistic regression over hashed sparse features, minibatch Adagrad (same per-coordinate
    learning-rate idea as FTRL-Proximal used in production CTR systems; trivially online)."""
    n, K = idx.shape
    r = np.random.default_rng(seed)
    w, G = np.zeros(nb), np.full(nb, 1e-8)
    ybar = float(np.average(y, weights=sw)) if sw is not None else float(y.mean())
    b, Gb = float(logit(ybar)), 1e-8
    for _ in range(epochs):
        perm = r.permutation(n)
        for s in range(0, n, 512):
            bi = perm[s:s + 512]
            ib = idx[bi]
            g = sig(w[ib].sum(1) + b) - y[bi]
            if sw is not None:
                g = g * sw[bi]
            grad = np.bincount(ib.ravel(), weights=np.repeat(g, K), minlength=nb) / len(bi) + l2 * w
            G += grad ** 2
            w -= lr * grad / np.sqrt(G)
            gb = g.mean()
            Gb += gb ** 2
            b -= lr * gb / np.sqrt(Gb)
    return w, b


def lr_predict(wb, idx):
    return sig(wb[0][idx].sum(1) + wb[1])


# ====================================================================================
# 3. OTHER MODEL FAMILIES
# ====================================================================================
OH_FIELDS = FIELDS + ["pos"]
OH_OFF = np.cumsum([0] + [CARD[f] for f in OH_FIELDS])


def onehot(c):
    X = np.zeros((len(c["ad"]), OH_OFF[-1]), dtype=np.float32)
    rows = np.arange(X.shape[0])
    for i, f in enumerate(OH_FIELDS):
        X[rows, OH_OFF[i] + c[f]] = 1.0
    return X


def ordinal(c):
    return np.stack([c[f] for f in OH_FIELDS], axis=1).astype(np.float64)


def train_fm(X, y, k=8, epochs=12, lr=0.05, l2=1e-5, seed=0):
    """Factorization machine (Rendle 2010):  z = b + sum_i w_i x_i + sum_{i<j} <v_i, v_j> x_i x_j.
    Pairwise term computed in O(k * nnz) via 0.5 * ((sum_i v_i x_i)^2 - sum_i v_i^2 x_i^2)."""
    r = np.random.default_rng(seed)
    n, d = X.shape
    w, V, b = np.zeros(d), r.normal(0, 0.05, (d, k)), float(logit(y.mean()))
    Gw, GV, Gb = np.full(d, 1e-8), np.full((d, k), 1e-8), 1e-8
    for _ in range(epochs):
        perm = r.permutation(n)
        for s in range(0, n, 512):
            xb, yb = X[perm[s:s + 512]], y[perm[s:s + 512]]
            S = xb @ V
            z = b + xb @ w + 0.5 * ((S ** 2).sum(1) - (xb @ (V ** 2)).sum(1))
            g = (sig(z) - yb) / len(yb)
            xg = xb.T @ g
            gw = xg + l2 * w
            gV = xb.T @ (g[:, None] * S) - xg[:, None] * V + l2 * V     # x is binary so x^2 = x
            Gw += gw ** 2; w -= lr * gw / np.sqrt(Gw)
            GV += gV ** 2; V -= lr * gV / np.sqrt(GV)
            gb = g.sum(); Gb += gb ** 2; b -= lr * gb / np.sqrt(Gb)
    return w, V, b


def fm_predict(m, X):
    w, V, b = m
    S = X @ V
    return sig(b + X @ w + 0.5 * ((S ** 2).sum(1) - (X @ (V ** 2)).sum(1)))


# ====================================================================================
# 4. METRICS  (concept: AUC measures ranking only; an auction needs CALIBRATED probabilities)
# ====================================================================================
def entropy(p):
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def metrics(y, p):
    """NE = logloss / entropy(base CTR). NE < 1 means better than predicting the average CTR;
    unlike raw log-loss it is comparable across datasets with different base rates."""
    ll = logloss(y, p)
    return {"logloss": round(ll, 5), "ne": round(ll / float(entropy(y.mean())), 4),
            "auc": round(float(roc_auc_score(y, p)), 4),
            "calibration": round(float(p.sum() / max(y.sum(), 1)), 3)}


def reliability(y, p, bins=10):
    order = np.argsort(p)
    out = []
    for ch in np.array_split(order, bins):
        out.append({"pred": round(float(p[ch].mean()), 5), "actual": round(float(y[ch].mean()), 5), "n": int(len(ch))})
    return out


def fit_calibrator(method, p, y):
    if method == "platt":      # 2 parameters: sigmoid(a * logit(p) + b)
        m = LogisticRegression(C=1e4).fit(logit(p)[:, None], y)
        return lambda q: m.predict_proba(logit(q)[:, None])[:, 1]
    if method == "isotonic":   # non-parametric monotone map; needs more data, can overfit tails
        m = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
        return lambda q: m.predict(q)
    return lambda q: q


# ====================================================================================
# TRAIN EVERYTHING ONCE (small data -> a couple of seconds)
# ====================================================================================
BITS = 14
M = {}        # name -> dict(predict=fn(cols) -> pCTR, train_ms=...)
_lock = threading.Lock()
_cache = {}


def _fit_all():
    ctr, cva, cte = sub(COLS, TR), sub(COLS, VA), sub(COLS, TE)
    y = CLICK[TR]

    t0 = time.perf_counter()
    lr = train_lr(hashed(ctr, BITS), y, 1 << BITS)
    M["lr_hash"] = {"label": "LR + hashed crosses", "train_ms": (time.perf_counter() - t0) * 1e3,
                    "predict": lambda c: lr_predict(lr, hashed(c, BITS))}

    t0 = time.perf_counter()
    gb = HistGradientBoostingClassifier(max_iter=40, learning_rate=0.15, max_leaf_nodes=15, early_stopping=False,
                                        categorical_features=[True] * 9 + [False], random_state=0).fit(ordinal(ctr), y)
    M["gbdt"] = {"label": "GBDT (HistGradientBoosting)", "train_ms": (time.perf_counter() - t0) * 1e3,
                 "predict": lambda c: gb.predict_proba(ordinal(c))[:, 1]}

    t0 = time.perf_counter()
    fm = train_fm(onehot(ctr), y)
    M["fm"] = {"label": "Factorization machine (numpy)", "train_ms": (time.perf_counter() - t0) * 1e3,
               "predict": lambda c: fm_predict(fm, onehot(c))}

    # Facebook hybrid: small GBDT = learned feature transform (leaf ids), LR on top = the part
    # that can be refreshed online every few minutes while trees are retrained daily.
    t0 = time.perf_counter()
    sm = np.random.default_rng(1).choice(len(y), 8000, replace=False)
    trees = GradientBoostingClassifier(n_estimators=25, max_depth=3, subsample=0.7, random_state=0)
    trees.fit(ordinal(ctr)[sm][:, :9], y[sm])                      # trees do not see position
    leaves = lambda c: trees.apply(ordinal(c)[:, :9])[:, :, 0].astype(np.int64)
    hy = train_lr(hashed(ctr, BITS, leaves=leaves(ctr)), y, 1 << BITS)
    M["gbdt_lr"] = {"label": "GBDT leaves -> LR (FB hybrid)", "train_ms": (time.perf_counter() - t0) * 1e3,
                    "predict": lambda c: lr_predict(hy, hashed(c, BITS, leaves=leaves(c)))}

    # auxiliary heads used by the auction: P(hide) and an LR that never saw position (pitfall demo)
    hd = train_lr(hashed(ctr, BITS), HIDE[TR], 1 << BITS, seed=1)
    M["_hide"] = {"predict": lambda c: lr_predict(hd, hashed(c, BITS))}
    nop = train_lr(hashed(ctr, BITS, use_pos=False), y, 1 << BITS)
    M["_lr_nopos"] = {"predict": lambda c: lr_predict(nop, hashed(c, BITS, use_pos=False))}

    for name in ("lr_hash", "gbdt", "fm", "gbdt_lr", "_lr_nopos"):
        m = M[name]
        m["p_va"], m["p_te"] = m["predict"](cva), m["predict"](cte)
        m["metrics"] = metrics(CLICK[TE], m["p_te"])
        # production serving path = model -> calibration layer refit on the freshest labelled days
        m["cal"] = fit_calibrator("platt", m["p_va"], CLICK[VA])
        five = sub(cte, np.arange(len(cte["ad"])) < 500)
        t0 = time.perf_counter(); m["predict"](five)
        m["latency_ms_500"] = (time.perf_counter() - t0) * 1e3


_fit_all()
PUBLIC = ["lr_hash", "gbdt", "fm", "gbdt_lr"]


def pos1(c):
    """Serving-time trick: position is unknown before the auction, so score every candidate as
    if it were in slot 1. Position was a feature at training time, so the model has attributed
    the click-through drop of lower slots to the position weight, not to the ad."""
    c = dict(c)
    c["pos"] = np.zeros(len(c["ad"]), dtype=np.int64)
    return c


def serve(name, c):
    """Calibrated, position-normalised pCTR - what the auction consumes."""
    m = M[name]
    return m["cal"](m["predict"](pos1(c)) if name != "_lr_nopos" else m["predict"](c))


def model_or_404(name):
    if name not in PUBLIC:
        raise HTTPException(400, f"model must be one of {PUBLIC}")
    return M[name]


# ====================================================================================
# 5. AUCTION  (concept: eCPM ranking + generalized second price; total value objective)
# ====================================================================================
def run_auctions(pctr, phide, bids, p1, hide1, rule="ecpm", w_click=0.0, w_hide=0.0, slots=3):
    """Vectorised over A auctions x C candidates.
      score_i = bid_i * pCTR_i  +  w_click * pCTR_i  -  w_hide * pHide_i        ($ per impression)
    GSP price per click = smallest bid that would still keep the slot:
      price_i = (score_next - w_click * pCTR_i + w_hide * pHide_i) / pCTR_i     (<= bid_i)
    Outcomes are computed with the HIDDEN true CTR / hide probabilities and the examination curve.
    A slot is left empty when the best remaining total value is <= 0 (ad load is an output)."""
    if rule == "ecpm":
        score = bids * pctr + w_click * pctr - w_hide * phide
    elif rule == "bid":
        score = bids.astype(float).copy()
    else:
        score = pctr.copy()
    order = np.argsort(-score, axis=1, kind="stable")
    top, nxt = order[:, :slots], order[:, 1:slots + 1]
    g = lambda A, ix: np.take_along_axis(A, ix, axis=1)
    shown = g(score, top) > 0
    s_next = np.maximum(g(score, nxt), 0)
    pc, b = g(pctr, top), g(bids, top)
    if rule == "ecpm":
        price = (s_next - w_click * pc + w_hide * g(phide, top)) / pc
    elif rule == "bid":
        price = g(bids, nxt)
    else:
        price = b            # ranking ignores bids -> no competitive price exists; first-price
    price = np.minimum(np.maximum(price, 0.01), b)
    ex = EXAM[:slots]
    clicks = g(p1, top) * ex * shown
    hides = g(hide1, top) * ex * shown
    return {"top": top, "price": price, "shown": shown, "clicks": clicks, "hides": hides,
            "revenue": price * clicks, "welfare": b * clicks, "pred_clicks": pc * ex * shown}


def summarize(res):
    A = res["top"].shape[0]
    imps = max(float(res["shown"].sum()), 1.0)
    return {"revenue_per_1k_auctions": round(float(res["revenue"].sum()) / A * 1000, 2),
            "advertiser_value_per_1k": round(float(res["welfare"].sum()) / A * 1000, 2),
            "ctr_pct": round(float(res["clicks"].sum()) / imps * 100, 3),
            "hides_per_1k_imps": round(float(res["hides"].sum()) / imps * 1000, 2),
            "ads_per_auction": round(imps / A, 3)}


def _auction_batch():
    """1500 simulated auctions: contexts drawn from the test days, 8 random eligible ads each."""
    if "batch" not in _cache:
        r = np.random.default_rng(SEED + 5)
        A, C = 1500, 8
        rows = r.choice(np.where(TE)[0], A)
        ads = np.stack([r.choice(N_ADS, C, replace=False) for _ in range(A)])
        c = {k: np.repeat(COLS[k][rows], C) for k in ("age", "country", "interest", "device", "hour")}
        c = pos1(with_ad(c, ads.ravel()))
        sh = (A, C)
        _cache["batch"] = {"cols": c, "shape": sh, "ads": ads, "bids": AD["bid"][ads],
                           "p1": true_p1(c).reshape(sh), "hide1": true_hide1(c).reshape(sh),
                           "phide": M["_hide"]["predict"](c).reshape(sh)}
    return _cache["batch"]


def _batch_pctr(model):
    key = "pctr_" + model
    if key not in _cache:
        b = _auction_batch()
        _cache[key] = serve(model, b["cols"]).reshape(b["shape"])
    return _cache[key]


# ====================================================================================
# ENDPOINTS
# ====================================================================================
@router.get("/overview")
def overview():
    """Dataset stats + model comparison (log-loss vs AUC vs NE vs calibration)."""
    keys = raw_keys(sub(COLS, TR))
    with np.errstate(over="ignore"):
        hk = _mix64(keys)
    uniq = len(np.unique(keys))
    by_pos = []
    for k in range(5):
        m = COLS["pos"] == k
        by_pos.append({"position": k + 1, "impressions": int(m.sum()), "logged_ctr": round(float(CLICK[m].mean()), 4),
                       "true_ctr_if_slot1": round(float(P1[m].mean()), 4), "examination_prob": float(EXAM[k]),
                       "avg_bid_of_ads_shown": round(float(AD["bid"][COLS["ad"][m]].mean()), 2)})
    return {
        "dataset": {"impressions": N, "ads": N_ADS, "days": DAYS, "ctr": round(float(CLICK.mean()), 4),
                    "cvr_given_click": round(float(CONV.sum() / CLICK.sum()), 4),
                    "hide_rate": round(float(HIDE.mean()), 4),
                    "split": {"train_days_0_9": int(TR.sum()), "valid_days_10_11": int(VA.sum()), "test_days_12_13": int(TE.sum())},
                    "base_entropy_test": round(float(entropy(CLICK[TE].mean())), 4)},
        "features": {"fields": FIELDS + ["pos"], "crosses": [f"{a} x {b}" for a, b in CROSSES],
                     "hash_bits": BITS, "hash_buckets": 1 << BITS, "distinct_tokens": uniq,
                     "buckets_used": int(len(np.unique(hk & np.uint64((1 << BITS) - 1))))},
        "models": [{"model": n, "label": M[n]["label"], **M[n]["metrics"],
                    "train_ms": round(M[n]["train_ms"], 1), "score_500_candidates_ms": round(M[n]["latency_ms_500"], 2)}
                   for n in PUBLIC],
        "by_position": by_pos,
        "vocab": {"ages": AGES, "countries": COUNTRIES, "interests": CATS, "devices": DEVICES, "hours": HOURS},
    }


class HashReq(BaseModel):
    bits: int = Field(12, ge=4, le=18)


@router.post("/hashing")
def hashing(q: HashReq):
    """Hashing trick knob: fewer buckets -> more collisions -> worse NE; memory is 2^bits weights."""
    ctr, cte = sub(COLS, TR), sub(COLS, TE)
    out = []
    keys = raw_keys(ctr)
    uniq = len(np.unique(keys))
    for bits in sorted({6, 8, 10, 12, 14, 16, q.bits}):
        wb = train_lr(hashed(ctr, bits), CLICK[TR], 1 << bits, epochs=4)
        m = metrics(CLICK[TE], lr_predict(wb, hashed(cte, bits)))
        used = len(np.unique(hashed(ctr, bits)))
        out.append({"bits": bits, "buckets": 1 << bits, "distinct_tokens": uniq,
                    "collision_rate": round(1 - used / uniq, 3), "ne": m["ne"], "auc": m["auc"],
                    "memory_kb": round((1 << bits) * 8 / 1024, 1), "selected": bits == q.bits})
    return {"rows": out}


class CalReq(BaseModel):
    model: str = "lr_hash"
    method: str = "isotonic"      # none | platt | isotonic


@router.post("/calibration")
def calibration(q: CalReq):
    """Reliability curve before/after a calibration layer fit on the most recent (validation) days."""
    m = model_or_404(q.model)
    if q.method not in ("none", "platt", "isotonic"):
        raise HTTPException(400, "method must be none|platt|isotonic")
    cal = fit_calibrator(q.method, m["p_va"], CLICK[VA])
    p_raw, p_cal, y = m["p_te"], np.clip(cal(m["p_te"]), 1e-6, 1 - 1e-6), CLICK[TE]
    return {"model": q.model, "method": q.method,
            "before": {**metrics(y, p_raw), "curve": reliability(y, p_raw)},
            "after": {**metrics(y, p_cal), "curve": reliability(y, p_cal)},
            "note": "AUC barely moves (monotone map keeps the ranking); log-loss / NE / calibration do."}


@router.get("/auction/sample")
def auction_sample(seed: int = 0):
    """5 candidate ads (with their default bids) and a user context to seed the simulator."""
    r = np.random.default_rng(1000 + seed)
    interest = int(r.integers(0, 8))
    ads = r.choice(N_ADS, 5, replace=False)
    same = np.where(AD["cat"] == interest)[0]
    if len(same):
        ads[0] = r.choice(same)
        ads = np.unique(ads)
        while len(ads) < 5:
            ads = np.unique(np.append(ads, r.integers(0, N_ADS)))
    return {"user": {"age": int(r.integers(0, 5)), "country": int(r.integers(0, 4)), "interest": interest,
                     "device": int(r.integers(0, 2)), "hour": int(r.integers(0, 4))},
            "ads": [{"ad_id": int(a), "name": AD_NAMES[a], "bid": float(AD["bid"][a])} for a in ads]}


class AdBid(BaseModel):
    ad_id: int = Field(..., ge=0, lt=N_ADS)
    bid: float = Field(..., gt=0, le=100)


class UserCtx(BaseModel):
    age: int = Field(1, ge=0, lt=5)
    country: int = Field(0, ge=0, lt=4)
    interest: int = Field(0, ge=0, lt=8)
    device: int = Field(0, ge=0, lt=2)
    hour: int = Field(2, ge=0, lt=4)


class AuctionReq(BaseModel):
    model: str = "lr_hash"
    rule: str = "ecpm"            # ecpm | bid | pctr
    user: UserCtx = UserCtx()
    ads: list[AdBid]
    w_click: float = Field(0.0, ge=0, le=10)
    w_hide: float = Field(0.0, ge=0, le=20)
    slots: int = Field(3, ge=1, le=4)


@router.post("/auction")
def auction(q: AuctionReq):
    """One auction, fully explained: pCTR (slot-1 normalised), score, slot, GSP price, outcomes."""
    m = model_or_404(q.model)
    if q.rule not in ("ecpm", "bid", "pctr"):
        raise HTTPException(400, "rule must be ecpm|bid|pctr")
    if not 2 <= len(q.ads) <= 8 or len({a.ad_id for a in q.ads}) != len(q.ads):
        raise HTTPException(400, "need 2-8 distinct ads")
    slots = min(q.slots, len(q.ads) - 1)
    n = len(q.ads)
    ad = np.array([a.ad_id for a in q.ads])
    c = {k: np.full(n, getattr(q.user, k), dtype=np.int64) for k in ("age", "country", "interest", "device", "hour")}
    c = pos1(with_ad(c, ad))
    t0 = time.perf_counter()
    pctr, phide = serve(q.model, c)[None, :], M["_hide"]["predict"](c)[None, :]
    score_ms = (time.perf_counter() - t0) * 1e3
    bids = np.array([[a.bid for a in q.ads]])
    p1, h1 = true_p1(c)[None, :], true_hide1(c)[None, :]

    def one(rule):
        return run_auctions(pctr, phide, bids, p1, h1, rule, q.w_click, q.w_hide, slots)

    res = one(q.rule)
    score = (bids * pctr + q.w_click * pctr - q.w_hide * phide)[0]
    rows = []
    for i in range(n):
        w = np.where(res["top"][0] == i)[0]
        s = int(w[0]) if len(w) and res["shown"][0, w[0]] else None
        rows.append({"ad_id": int(ad[i]), "name": AD_NAMES[ad[i]], "bid": float(bids[0, i]),
                     "pctr": round(float(pctr[0, i]), 4), "true_ctr": round(float(p1[0, i]), 4),
                     "phide": round(float(phide[0, i]), 4), "true_hide": round(float(h1[0, i]), 4),
                     "ecpm": round(float(bids[0, i] * pctr[0, i] * 1000), 2),
                     "total_value_x1000": round(float(score[i] * 1000), 2),
                     "slot": s + 1 if s is not None else None,
                     "price_per_click": round(float(res["price"][0, s]), 3) if s is not None else None,
                     "exp_revenue_x1000": round(float(res["revenue"][0, s] * 1000), 2) if s is not None else 0.0})
    rows.sort(key=lambda r_: (r_["slot"] is None, r_["slot"] or 0, -r_["total_value_x1000"]))
    by_rule = []
    for rule in ("ecpm", "bid", "pctr"):
        r_ = one(rule)
        by_rule.append({"rule": rule, "winners": " > ".join(AD_NAMES[ad[j]] for k, j in enumerate(r_["top"][0]) if r_["shown"][0, k]) or "(no ad shown)",
                        "exp_revenue_x1000": round(float(r_["revenue"].sum() * 1000), 2),
                        "exp_clicks_x1000": round(float(r_["clicks"].sum() * 1000), 1),
                        "exp_hides_x1000": round(float(r_["hides"].sum() * 1000), 1),
                        "advertiser_value_x1000": round(float(r_["welfare"].sum() * 1000), 2)})
    return {"rows": rows, "by_rule": by_rule, "slots": slots, "scoring_ms": round(score_ms, 3),
            "formula": "score = bid*pCTR + w_click*pCTR - w_hide*pHide ; GSP price = (next score - w_click*pCTR + w_hide*pHide) / pCTR"}


class TradeReq(BaseModel):
    model: str = "lr_hash"
    w_click: float = Field(0.0, ge=0, le=10)
    w_hide: float = Field(0.5, ge=0, le=20)


@router.post("/tradeoff")
def tradeoff(q: TradeReq):
    """Multi-objective ranking: sweep the user-experience weights over 1500 simulated auctions."""
    model_or_404(q.model)
    b = _auction_batch()
    pctr = _batch_pctr(q.model)
    run = lambda wc, wh, rule="ecpm": summarize(run_auctions(pctr, b["phide"], b["bids"], b["p1"], b["hide1"], rule, wc, wh))
    return {"selected": run(q.w_click, q.w_hide),
            "baselines": [{"objective": lab, **run(0, 0, rule)} for lab, rule in
                          (("bid only", "bid"), ("pCTR only", "pctr"), ("eCPM = bid*pCTR", "ecpm"))],
            "sweep_hide": [{"w_hide": wh, **run(q.w_click, wh)} for wh in (0, .1, .2, .35, .5, .75, 1, 1.5, 2, 3)],
            "sweep_click": [{"w_click": wc, **run(wc, q.w_hide)} for wc in (0, .1, .25, .5, 1, 1.5, 2, 3)],
            "auctions": b["shape"][0]}


class PaceReq(BaseModel):
    ad_id: int = Field(0, ge=0, lt=N_ADS)
    bid: float = Field(1.5, gt=0, le=50)
    budget: float = Field(20.0, gt=0, le=10000)


@router.post("/pacing")
def pacing(q: PaceReq):
    """Budget pacing over one simulated day of 3000 single-slot second-price auctions.
    none      - bid full value until the budget is gone (burns out early, misses cheap evening traffic)
    throttle  - probabilistic pacing: enter each auction with probability x
    bid       - bid modification: bid * x  (x<1 = shading -> wins only the cheapest impressions)
    x is steered by a multiplicative controller toward a linear spend target."""
    r = np.random.default_rng(SEED + 9)
    Tn = 3000
    rows = r.choice(np.where(TE)[0], Tn)
    c = {k: COLS[k][rows] for k in ("age", "country", "interest", "device", "hour")}
    c = pos1(with_ad(c, np.full(Tn, q.ad_id)))
    pctr, p1 = serve("lr_hash", c), true_p1(c)
    # highest competing eCPM: lognormal, more expensive early in the day
    comp = np.exp(r.normal(np.log(0.05), 0.6, Tn)) * np.linspace(1.6, 0.6, Tn)
    coin = r.random(Tn)
    out = {}
    for mode in ("none", "throttle", "bid"):
        x, spend, clicks, wins, exhausted, series = 1.0, 0.0, 0.0, 0, None, []
        for t in range(Tn):
            if t % 50 == 0:
                series.append(round(spend / q.budget, 4))
                if mode != "none" and t:
                    x = float(np.clip(x * (0.85 if spend > q.budget * t / Tn else 1.15), 0.02, 1.0))
            if spend >= q.budget:
                exhausted = exhausted if exhausted is not None else t / Tn
                continue
            enter = coin[t] < x if mode == "throttle" else True
            mult = x if mode == "bid" else 1.0
            if enter and mult * q.bid * pctr[t] > comp[t]:
                spend += comp[t]            # second price: pay the competing eCPM (in expectation)
                clicks += p1[t]
                wins += 1
        series.append(round(spend / q.budget, 4))
        out[mode] = {"spend": round(spend, 2), "impressions": wins, "exp_clicks": round(clicks, 1),
                     "cost_per_click": round(spend / max(clicks, 1e-9), 3),
                     "budget_exhausted_at_pct_of_day": None if exhausted is None else round(exhausted * 100, 1),
                     "series": series}
    return {"ad": AD_NAMES[q.ad_id], "budget": q.budget, "bid": q.bid, "auctions": Tn,
            "x": [round(i * 50 / Tn * 100, 1) for i in range(len(out["none"]["series"]))], "modes": out}


# ====================================================================================
# 7. PITFALLS
# ====================================================================================
@router.get("/pitfall/position-bias")
def pitfall_position(fix: bool = True):
    """Position bias. Logged CTR mixes ad quality with slot visibility, and the old system gave
    high bidders the top slots. wrong: train without position -> incumbents look better than they are.
    right: position as a training feature, fixed to slot 1 at serving."""
    te = sub(COLS, TE)
    truth = P1[TE]
    variants = {"no position feature (wrong)": serve("_lr_nopos", te),
                "position feature, serve with pos=1 (right)": serve("lr_hash", te)}
    bid = AD["bid"][te["ad"]]
    lo, hi = np.quantile(AD["bid"], [1 / 3, 2 / 3])
    groups = [("high bidders (logged in top slots)", bid >= hi), ("mid bidders", (bid > lo) & (bid < hi)),
              ("low bidders (logged in bottom slots)", bid <= lo)]
    rows = []
    for gname, gm in groups:
        row = {"ad_group": gname, "avg_logged_position": round(float(te["pos"][gm].mean() + 1), 2),
               "true_slot1_ctr": round(float(truth[gm].mean()), 4)}
        for vname, p in variants.items():
            row[vname.split(" (")[0] + " - pred/true"] = round(float(p[gm].sum() / truth[gm].sum()), 3)
        rows.append(row)
    # learnt position curve: same impressions scored at each slot, normalised to slot 1
    curve = []
    base = M["lr_hash"]["predict"](pos1(te)).mean()
    for k in range(5):
        c = dict(te); c["pos"] = np.full(len(te["ad"]), k)
        curve.append({"position": k + 1, "true_examination": float(EXAM[k]),
                      "learned_by_model": round(float(M["lr_hash"]["predict"](c).mean() / base), 3)})
    # auction consequence: allocative efficiency vs an oracle that knows the true slot-1 CTR
    b = _auction_batch()
    oracle = run_auctions(b["p1"], b["phide"], b["bids"], b["p1"], b["hide1"])["welfare"].sum()
    eff = {}
    for vname, mname in (("no position feature (wrong)", "_lr_nopos"), ("position feature, serve with pos=1 (right)", "lr_hash")):
        p = serve(mname, b["cols"]).reshape(b["shape"])
        res = run_auctions(p, b["phide"], b["bids"], b["p1"], b["hide1"])
        low = (b["bids"] <= lo)
        eff[vname] = {"advertiser_value_vs_oracle": round(float(res["welfare"].sum() / oracle), 4),
                      "low_bidder_share_of_slot1": round(float(np.take_along_axis(low, res["top"][:, :1], 1).mean()), 3),
                      "pred/true over all candidates": round(float(p.sum() / b["p1"].sum()), 3)}
    oracle_low = float(np.take_along_axis(b["bids"] <= lo, run_auctions(b["p1"], b["phide"], b["bids"], b["p1"], b["hide1"])["top"][:, :1], 1).mean())
    return {"fix": fix, "showing": "position feature, serve with pos=1 (right)" if fix else "no position feature (wrong)",
            "by_ad_group": rows, "position_curve": curve, "auction": eff, "oracle_low_bidder_share_of_slot1": round(oracle_low, 3),
            "explain": "Without a position feature the model predicts logged CTR = true CTR x examination of the slots the ad USED to get. "
                       "Incumbent high bidders are under-discounted relative to low bidders, so the new ranker inherits the old ranker's decisions (feedback loop)."}


class MiscalReq(BaseModel):
    model: str = "lr_hash"
    overconfidence: float = Field(1.6, ge=0.3, le=4)    # logit temperature: >1 = too extreme
    noise_sd: float = Field(0.4, ge=0, le=2)            # per-ad persistent error on the logit
    calibrate: bool = True


@router.post("/pitfall/miscalibration")
def pitfall_miscal(q: MiscalReq):
    """Miscalibrated pCTR breaks the auction. The auction takes an argmax over noisy estimates, so
    winners are selected for POSITIVE error: predicted CTR of winners > realised (winner's curse).
    A global calibration layer repairs the systematic part, not per-ad noise."""
    m = model_or_404(q.model)
    b = _auction_batch()
    base = _batch_pctr(q.model)
    e_ad = np.random.default_rng(SEED + 11).normal(0, q.noise_sd, N_ADS)
    p_va = m["cal"](m["p_va"])
    mu = float(logit(p_va).mean())
    distort = lambda p, ads: sig(mu + q.overconfidence * (logit(p) - mu) + e_ad[ads])
    p_bad = distort(base, b["ads"])
    variants = [("calibrated model", base), ("distorted (overconfident + per-ad noise)", p_bad)]
    if q.calibrate:
        cal = fit_calibrator("isotonic", distort(p_va, COLS["ad"][VA]), CLICK[VA])
        variants.append(("distorted + isotonic calibration", np.clip(cal(p_bad.ravel()), 1e-6, 1).reshape(base.shape)))
    oracle = run_auctions(b["p1"], b["phide"], b["bids"], b["p1"], b["hide1"])
    rows = []
    for name, p in variants:
        r_ = run_auctions(p, b["phide"], b["bids"], b["p1"], b["hide1"])
        rows.append({"pCTR used by auction": name,
                     "calibration all candidates (pred/true)": round(float(p.sum() / b["p1"].sum()), 3),
                     "calibration WINNERS (pred/true)": round(float(r_["pred_clicks"].sum() / r_["clicks"].sum()), 3),
                     "advertiser value vs oracle": round(float(r_["welfare"].sum() / oracle["welfare"].sum()), 4),
                     "revenue per 1k auctions": round(float(r_["revenue"].sum()) / base.shape[0] * 1000, 2),
                     "clicks per 1k auctions": round(float(r_["clicks"].sum()) / base.shape[0] * 1000, 1)})
    return {"rows": rows, "oracle_revenue_per_1k": round(float(oracle["revenue"].sum()) / base.shape[0] * 1000, 2),
            "explain": "Winners' pred/true > all-candidates pred/true is the winner's curse. Under impression-billed "
                       "(oCPM) pricing the advertiser pays bid*pCTR per impression, so that ratio is exactly the overcharge."}


class DownReq(BaseModel):
    rate: float = Field(0.1, gt=0.005, le=1)     # keep-probability for negatives
    correct: bool = True


@router.post("/pitfall/downsampling")
def pitfall_downsampling(q: DownReq):
    """Negative downsampling. Keeping negatives with prob w multiplies the odds by 1/w.
    Correction (He et al. 2014):  q = p / (p + (1 - p) / w)    <=> subtract log(1/w) from the logit."""
    r = np.random.default_rng(SEED + 13)
    tr = np.where(TR)[0]
    keep = tr[(CLICK[tr] == 1) | (r.random(len(tr)) < q.rate)]
    cs = sub(COLS, keep)
    wb = train_lr(hashed(cs, BITS), CLICK[keep], 1 << BITS, epochs=8)
    p = lr_predict(wb, hashed(sub(COLS, TE), BITS))
    pc = p / (p + (1 - p) / q.rate)
    y = CLICK[TE]
    used = pc if q.correct else p
    return {"rate": q.rate, "correct": q.correct,
            "train_rows_full": int(len(tr)), "train_rows_kept": int(len(keep)),
            "train_ctr_after_sampling": round(float(CLICK[keep].mean()), 4), "true_ctr": round(float(CLICK[tr].mean()), 4),
            "uncorrected": {**metrics(y, p), "mean_pctr": round(float(p.mean()), 4), "curve": reliability(y, p)},
            "corrected": {**metrics(y, pc), "mean_pctr": round(float(pc.mean()), 4), "curve": reliability(y, pc)},
            "full_data_reference": M["lr_hash"]["metrics"],
            "served": {"mean_pctr": round(float(used.mean()), 4), "actual_ctr": round(float(y.mean()), 4),
                       "advertiser_overcharge_factor_if_billed_on_pctr": round(float(used.sum() / y.sum()), 2)},
            "formula": "q = p / (p + (1 - p) / w)"}


class DelayReq(BaseModel):
    window_days: float = Field(3.0, ge=0.5, le=7)
    fix: str = "reweight"         # naive | wait | reweight


@router.post("/pitfall/delayed-conversions")
def pitfall_delayed(q: DelayReq):
    """Delayed conversion labels. At training time (day 12) a click with no conversion YET is not
    a true negative. naive: label it 0 -> pCVR biased low, worst for slow-converting advertisers.
    wait: only train on clicks older than the attribution window -> unbiased but stale.
    reweight: use all clicks, weight observed positives by 1 / P(delay <= elapsed) (delayed-feedback
    / importance-weighting idea, Chapelle 2014)."""
    if q.fix not in ("naive", "wait", "reweight"):
        raise HTTPException(400, "fix must be naive|wait|reweight")
    cutoff = 12.0
    ck = np.where(CLICK == 1)[0]
    trn, tst = ck[T[ck] < cutoff], ck[T[ck] >= cutoff]
    elapsed = cutoff - T[trn]
    seen = (CONV[trn] == 1) & (DELAY[trn] <= elapsed)          # conversions visible at the cutoff

    def X(ix):
        c = sub(COLS, ix)
        A = np.zeros((len(ix), N_ADS + 8 + 1))
        A[np.arange(len(ix)), c["ad"]] = 1
        A[np.arange(len(ix)), N_ADS + c["ad_cat"]] = 1
        A[:, -1] = c["interest"] == c["ad_cat"]
        return A

    Xtr, Xte, yte = X(trn), X(tst), CONV[tst]
    cvr_te = true_cvr(sub(COLS, tst))      # hidden truth -> low-noise calibration measurement
    # per-ad delay scale estimated from conversions already observed (shrunk to the global mean)
    ads = COLS["ad"][trn]
    mat = seen & (elapsed >= 6)        # only old clicks: their observed delays are barely truncated
    gmean = float(DELAY[trn][mat].mean())
    cnt = np.bincount(ads[mat], minlength=N_ADS)
    tau = (np.bincount(ads[mat], weights=DELAY[trn][mat], minlength=N_ADS) + 3 * gmean) / (cnt + 3)
    F = np.clip(1 - np.exp(-elapsed / tau[ads]), 0.05, 1)       # P(conversion already observed)

    def fit(kind):
        if kind == "oracle":
            Xa, ya, wa = Xtr, CONV[trn], None
        elif kind == "naive":
            Xa, ya, wa = Xtr, seen.astype(float), None
        elif kind == "wait":
            old = elapsed >= q.window_days
            Xa, ya, wa = Xtr[old], seen[old].astype(float), None
        else:   # every click appears as a positive with weight seen/F and a negative with the rest
            wp = seen / F
            Xa = np.vstack([Xtr, Xtr]); ya = np.r_[np.ones(len(trn)), np.zeros(len(trn))]
            wa = np.r_[wp, np.clip(1 - wp, 0, None)]
            k = wa > 0
            Xa, ya, wa = Xa[k], ya[k], wa[k]
        mdl = LogisticRegression(C=0.5, max_iter=300).fit(Xa, ya, sample_weight=wa)
        p = mdl.predict_proba(Xte)[:, 1]
        slow = AD["delay"][COLS["ad"][tst]] >= 2.5
        return {"method": kind, "train_clicks": int(len(ya) if kind != "reweight" else len(trn)),
                "label_cvr_in_training": round(float(np.average(ya, weights=wa)), 4),
                "pred/true CVR (all)": round(float(p.sum() / cvr_te.sum()), 3),
                "pred/true CVR (slow-converting ads)": round(float(p[slow].sum() / cvr_te[slow].sum()), 3),
                "logloss": round(logloss(yte, p), 4),
                "freshest_training_click_age_days": round(float(q.window_days if kind == "wait" else 0.0), 1)}

    rows = [fit(k) for k in ("naive", "wait", "reweight", "oracle")]
    d = DELAY[ck][CONV[ck] == 1]
    cdf = [{"days": x, "share_of_conversions_arrived": round(float((d <= x).mean()), 3)} for x in (0.5, 1, 2, 3, 5, 7, 10)]
    return {"fix": q.fix, "window_days": q.window_days, "true_cvr_test": round(float(yte.mean()), 4),
            "test_clicks": int(len(tst)), "rows": rows, "delay_cdf": cdf,
            "explain": "naive under-predicts CVR (ratio < 1) -> conversion-optimised bids are too low -> slow advertisers under-deliver."}
