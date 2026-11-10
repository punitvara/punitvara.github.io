"""Ranking Experiment (A/B testing) lab.

Interview prompt: "You've built a new ranking model for the feed. Design the experiment
to decide whether to launch it; how do you evaluate ranking changes online and offline?"

Everything here is a *real* simulation + *real* statistics written with numpy/stdlib:

  world        : sessions with K candidates, each with a true click-relevance; a ranker is
                 "true relevance + Gaussian noise" (lower noise = better model); users click
                 through a position-based model  P(click | item, pos) = exam[pos] * rel[item].
  /ab_test     : user-randomised A/B, Welch t-test, two-proportion z-test, bootstrap CI,
                 and the ratio-metric pitfall (session-level SE vs delta method).
  /power       : sample-size / power / MDE calculator, verified by Monte Carlo.
  /srm         : sample-ratio-mismatch chi-square, with a logging bug toggle.
  /cuped       : variance reduction with a pre-experiment covariate (+ leaky covariate pitfall).
  /novelty     : novelty / primacy effects by day, cohort split, asymptote fit.
  /peeking     : repeated significance testing on A/A tests, alpha-spending fixes.
  /multiple_testing : Bonferroni / Holm / Benjamini-Hochberg.
  /interference: SUTVA violation, user vs cluster randomisation.
  /interleaving: team-draft interleaving vs A/B sensitivity.
  /offline_eval: naive click replay vs position-debiased IPS, and IPS / clipped IPS / SNIPS.
"""
import math
from functools import lru_cache
from statistics import NormalDist

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

META = {
    "slug": "ranking-experiment",
    "title": "Ranking Experiment (A/B Testing)",
    "tagline": "Power, SRM, CUPED, peeking, interference, interleaving and IPS: decide whether a new ranker ships.",
    "icon": "🧬",
    "order": 5,
}
router = APIRouter()

# ----------------------------------------------------------------------------------------
# World model (concept: position-based click model; a ranker is a noisy view of relevance)
# ----------------------------------------------------------------------------------------
K = 10                                            # candidates shown per session
EXAM = 1.0 / (1.0 + np.arange(K)) ** 0.8          # P(user examines position k): position bias
CONTROL_NOISE = 0.15                              # production ranker's score noise
DEFAULT_TREAT_NOISE = 0.12                        # the "new model" (less noise = better ranking)
_ND = NormalDist()


def _p2(z: float) -> float:
    """Two-sided p-value of a z statistic."""
    return math.erfc(abs(float(z)) / math.sqrt(2.0))


_p2v = np.vectorize(_p2, otypes=[float])


def _zq(p: float) -> float:
    return _ND.inv_cdf(p)


def _chi2_p_1df(x: float) -> float:
    """Survival function of chi-square with 1 dof (chi2_1 = z^2)."""
    return math.erfc(math.sqrt(max(x, 0.0) / 2.0))


def _clean(o):
    """numpy -> plain JSON; NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return round(f, 6) if math.isfinite(f) else None
    return o


def _hist(v, bins=30):
    c, e = np.histogram(np.asarray(v, float), bins=bins)
    return {"centers": (e[:-1] + e[1:]) / 2, "counts": c, "lo": e[0], "hi": e[-1]}


def _population(rng, n, sigma_u=0.5):
    """Heterogeneous users: click propensity multiplier, session rate, platform segment."""
    mult = rng.lognormal(-sigma_u ** 2 / 2, sigma_u, n)
    rate = 3.0 * rng.lognormal(-0.125, 0.5, n)
    platform = rng.choice(3, n, p=[0.35, 0.45, 0.20])      # 0 ios, 1 android, 2 web
    return mult, rate, platform


PLATFORMS = ["ios", "android", "web"]


def _run_period(rng, mult, nsess, noise):
    """Simulate every session of every user. `noise` = ranker noise, scalar or per-user array.
    Returns per-user clicks, sessions, expected clicks, and sum of squared session clicks."""
    n = len(mult)
    uid = np.repeat(np.arange(n), nsess)
    rel = np.clip(rng.beta(1.2, 6.0, (len(uid), K)) * mult[uid][:, None], 0, 0.95)
    nz = noise[uid][:, None] if isinstance(noise, np.ndarray) else noise
    score = rel + nz * rng.standard_normal(rel.shape)       # the ranker
    ranked = np.take_along_axis(rel, np.argsort(-score, axis=1), 1)
    pc = ranked * EXAM                                      # position-based click model
    sc = (rng.random(pc.shape) < pc).sum(1).astype(float)
    return {
        "clicks": np.bincount(uid, sc, n),
        "sessions": nsess.astype(float),
        "exp": np.bincount(uid, pc.sum(1), n),
        "sumsq": np.bincount(uid, sc ** 2, n),
    }


def _true_effect(seed, mult, nsess, treat_noise):
    """Ground truth via common random numbers: same users, same sessions, both rankers."""
    c = _run_period(np.random.default_rng(seed), mult, nsess, CONTROL_NOISE)["exp"].mean()
    t = _run_period(np.random.default_rng(seed), mult, nsess, treat_noise)["exp"].mean()
    return float(c), float(t)


def _welch(a, b, alpha=0.05):
    """Welch t-test (normal approx, n is large) of mean(b) - mean(a), + delta-method relative CI."""
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    z = (mb - ma) / se if se > 0 else 0.0
    zc = _zq(1 - alpha / 2)
    rel = mb / ma - 1
    se_rel = math.sqrt(vb + (mb / ma) ** 2 * va) / ma       # delta method for a ratio of means
    return {
        "mean_control": ma, "mean_treatment": mb, "diff": mb - ma, "se": se, "z": z, "p": _p2(z),
        "ci": [mb - ma - zc * se, mb - ma + zc * se],
        "rel_lift": rel, "rel_ci": [rel - zc * se_rel, rel + zc * se_rel],
        "significant": _p2(z) < alpha,
    }


@lru_cache(maxsize=64)
def _ranker_quality(noise: float):
    """Offline NDCG@K (true labels) and expected online clicks/session of a ranker."""
    rng = np.random.default_rng(7)
    rel = rng.beta(1.2, 6.0, (20000, K))
    order = np.argsort(-(rel + noise * rng.standard_normal(rel.shape)), axis=1)
    ranked = np.take_along_axis(rel, order, 1)
    disc = 1 / np.log2(2 + np.arange(K))
    ndcg = ((ranked * disc).sum(1) / (-np.sort(-rel, axis=1) * disc).sum(1)).mean()
    return float((ranked * EXAM).sum(1).mean()), float(ndcg)


# ----------------------------------------------------------------------------------------
# GET /overview
# ----------------------------------------------------------------------------------------
@router.get("/overview")
def overview():
    c_cps, c_ndcg = _ranker_quality(CONTROL_NOISE)
    grid = []
    for nz in [0.0, 0.05, 0.09, 0.12, 0.15, 0.2, 0.3, 0.5]:
        cps, ndcg = _ranker_quality(nz)
        grid.append({"ranker_noise": nz, "offline_ndcg": ndcg, "online_clicks_per_session": cps,
                     "ndcg_gain_vs_control": ndcg / c_ndcg - 1, "online_lift_vs_control": cps / c_cps - 1})
    return _clean({
        "world": {
            "candidates_per_session": K,
            "click_model": "P(click | item, pos) = exam[pos] * relevance[item]",
            "examination_by_position": EXAM,
            "control_ranker_noise": CONTROL_NOISE,
            "default_treatment_noise": DEFAULT_TREAT_NOISE,
            "control_clicks_per_session": c_cps,
            "control_ndcg": c_ndcg,
        },
        "ranker_grid": grid,
        "experiments": [
            {"endpoint": "/ab_test", "concept": "t-test, z-test, bootstrap, ratio metrics / delta method"},
            {"endpoint": "/power", "concept": "sample size, power, MDE"},
            {"endpoint": "/srm", "concept": "sample ratio mismatch (pitfall toggle: logging bug)"},
            {"endpoint": "/cuped", "concept": "variance reduction (pitfall toggle: post-treatment covariate)"},
            {"endpoint": "/novelty", "concept": "novelty / primacy effects, new-user cohort"},
            {"endpoint": "/peeking", "concept": "optional stopping vs alpha spending"},
            {"endpoint": "/multiple_testing", "concept": "Bonferroni, Holm, Benjamini-Hochberg"},
            {"endpoint": "/interference", "concept": "SUTVA, cluster randomisation"},
            {"endpoint": "/interleaving", "concept": "team-draft interleaving sensitivity"},
            {"endpoint": "/offline_eval", "concept": "position bias, IPS, clipped IPS, SNIPS"},
        ],
    })


# ----------------------------------------------------------------------------------------
# POST /ab_test  (concepts: randomisation unit = user; Welch t / two-proportion z / bootstrap;
#                 ratio metric whose analysis unit (session) != randomisation unit (user))
# ----------------------------------------------------------------------------------------
class ABReq(BaseModel):
    n_per_arm: int = Field(4000, ge=200, le=20000)
    treat_noise: float = Field(DEFAULT_TREAT_NOISE, ge=0.0, le=0.6)
    alpha: float = Field(0.05, gt=0.0, lt=0.5)
    n_boot: int = Field(1000, ge=100, le=2000)
    seed: int = 0


def _ratio_tests(y, s, sumsq, T, alpha):
    """CTR = sum(clicks)/sum(sessions). Naive SE treats sessions as i.i.d.; the delta method
    respects that sessions of one user are correlated (users were randomised, not sessions)."""
    out = {}
    stats = []
    for arm in (0, 1):
        m = T == arm
        Y, S, n = y[m], s[m], int(m.sum())
        r = Y.sum() / S.sum()
        var_naive = (sumsq[m].sum() / S.sum() - r ** 2) / S.sum()
        c = np.cov(Y, S)
        var_delta = (c[0, 0] - 2 * r * c[0, 1] + r ** 2 * c[1, 1]) / (n * S.mean() ** 2)
        stats.append((r, var_naive, var_delta))
    d = stats[1][0] - stats[0][0]
    zc = _zq(1 - alpha / 2)
    for name, j in (("naive_session_level", 1), ("delta_method", 2)):
        se = math.sqrt(stats[0][j] + stats[1][j])
        out[name] = {"diff": d, "se": se, "z": d / se, "p": _p2(d / se), "ci": [d - zc * se, d + zc * se]}
    out["ctr_control"], out["ctr_treatment"] = stats[0][0], stats[1][0]
    out["se_underestimated_by"] = out["delta_method"]["se"] / out["naive_session_level"]["se"]
    return out


@lru_cache(maxsize=1)
def _aa_check():
    """400 A/A tests carved out of one 40k-user control population: false-positive rate of the
    naive session-level test vs the delta method on the clicks/session ratio metric."""
    rng = np.random.default_rng(11)
    n, chunk = 40000, 2000
    mult, rate, _ = _population(rng, n)
    nsess = 1 + rng.poisson(rate)
    r = _run_period(rng, mult, nsess, CONTROL_NOISE)
    y, s, q = r["clicks"], r["sessions"], r["sumsq"]
    rej_naive = rej_delta = tot = 0
    for _ in range(40):
        p = rng.permutation(n)
        Y, S, Q = (a[p].reshape(-1, chunk) for a in (y, s, q))
        R = Y.sum(1) / S.sum(1)
        vn = (Q.sum(1) / S.sum(1) - R ** 2) / S.sum(1)
        yc, sc = Y - Y.mean(1, keepdims=True), S - S.mean(1, keepdims=True)
        vy, vs, cys = (yc ** 2).sum(1) / (chunk - 1), (sc ** 2).sum(1) / (chunk - 1), (yc * sc).sum(1) / (chunk - 1)
        vd = (vy - 2 * R * cys + R ** 2 * vs) / (chunk * S.mean(1) ** 2)
        d = R[0::2] - R[1::2]
        rej_naive += int((np.abs(d) / np.sqrt(vn[0::2] + vn[1::2]) > 1.96).sum())
        rej_delta += int((np.abs(d) / np.sqrt(vd[0::2] + vd[1::2]) > 1.96).sum())
        tot += len(d)
    return {"n_aa_tests": tot, "users_per_arm": chunk, "fpr_naive_session_level": rej_naive / tot,
            "fpr_delta_method": rej_delta / tot, "nominal": 0.05}


@router.post("/ab_test")
def ab_test(q: ABReq):
    rng = np.random.default_rng(q.seed)
    n = 2 * q.n_per_arm
    mult, rate, _ = _population(rng, n)
    nsess = 1 + rng.poisson(rate)
    T = (rng.permutation(n) < q.n_per_arm).astype(int)            # user-level randomisation
    r = _run_period(rng, mult, nsess, np.where(T == 1, q.treat_noise, CONTROL_NOISE))
    y, s = r["clicks"], r["sessions"]
    a, b = y[T == 0], y[T == 1]

    tc, tt = _true_effect(q.seed + 99991, mult, nsess, q.treat_noise)
    welch = _welch(a, b, q.alpha)

    # two-proportion z-test on "user had >= 2 clicks" (a binary conversion-style metric)
    xa, xb = int((a >= 2).sum()), int((b >= 2).sum())
    pa, pb, pp = xa / len(a), xb / len(b), (xa + xb) / n
    se0 = math.sqrt(pp * (1 - pp) * (1 / len(a) + 1 / len(b)))
    se1 = math.sqrt(pa * (1 - pa) / len(a) + pb * (1 - pb) / len(b))
    zc = _zq(1 - q.alpha / 2)
    zprop = (pb - pa) / se0
    prop = {"metric": "share of users with >= 2 clicks", "p_control": pa, "p_treatment": pb, "diff": pb - pa,
            "z": zprop, "p": _p2(zprop), "ci": [pb - pa - zc * se1, pb - pa + zc * se1], "rel_lift": pb / pa - 1}

    # non-parametric bootstrap over users (the randomisation unit)
    B = int(max(100, min(q.n_boot, 4_000_000 // q.n_per_arm)))
    ba = a[rng.integers(0, len(a), (B, len(a)))].mean(1)
    bb = b[rng.integers(0, len(b), (B, len(b)))].mean(1)
    lifts = bb / ba - 1
    boot = {"n_boot": B, "rel_ci": np.quantile(lifts, [q.alpha / 2, 1 - q.alpha / 2]),
            "p_two_sided": float(min(1.0, 2 * min((lifts <= 0).mean(), (lifts >= 0).mean()))),
            "hist": _hist(lifts * 100, 30)}

    return _clean({
        "truth": {"clicks_per_user_control": tc, "clicks_per_user_treatment": tt, "true_rel_lift": tt / tc - 1},
        "arms": {"n_control": len(a), "n_treatment": len(b), "sessions_control": s[T == 0].sum(),
                 "sessions_treatment": s[T == 1].sum()},
        "welch_t_clicks_per_user": welch,
        "two_proportion_z": prop,
        "bootstrap": boot,
        "ratio_metric_ctr": _ratio_tests(y, s, r["sumsq"], T, q.alpha),
        "aa_check": _aa_check(),
    })


# ----------------------------------------------------------------------------------------
# POST /power  (concepts: n ~ 16 sigma^2 / delta^2; MDE; power; Monte Carlo sanity check)
# ----------------------------------------------------------------------------------------
class PowerReq(BaseModel):
    baseline: float = Field(0.10, gt=0.001, lt=0.95)
    mde_rel: float = Field(0.02, gt=0.001, le=1.0)
    alpha: float = Field(0.05, gt=0.0005, lt=0.5)
    power: float = Field(0.80, gt=0.5, lt=0.999)
    daily_users: int = Field(200000, ge=100)
    var_reduction: float = Field(0.0, ge=0.0, le=0.9)      # e.g. CUPED rho^2
    seed: int = 0


def _prop_power(p1, p2, n, alpha):
    pbar = (p1 + p2) / 2
    za = _zq(1 - alpha / 2)
    return _ND.cdf((abs(p2 - p1) * math.sqrt(n) - za * math.sqrt(2 * pbar * (1 - pbar)))
                   / math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))


def _prop_n(p1, p2, alpha, power):
    pbar = (p1 + p2) / 2
    za, zb = _zq(1 - alpha / 2), _zq(power)
    return (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2


@router.post("/power")
def power(q: PowerReq):
    p1 = q.baseline
    p2 = min(0.999, p1 * (1 + q.mde_rel))
    vr = 1 - q.var_reduction
    n = math.ceil(_prop_n(p1, p2, q.alpha, q.power) * vr)
    ns = np.unique(np.round(np.geomspace(max(50, n / 20), n * 3, 40)).astype(int))
    power_curve = [_prop_power(p1, p2, x / vr, q.alpha) for x in ns]
    # MDE as a function of n (bisection on the power formula)
    mde_curve = []
    for x in ns:
        lo, hi = 1e-5, 5.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if _prop_power(p1, min(0.999, p1 * (1 + mid)), x / vr, q.alpha) < q.power:
                lo = mid
            else:
                hi = mid
        mde_curve.append(hi * 100)
    # Monte Carlo verification with real binomial draws at the recommended n (no var. reduction)
    rng = np.random.default_rng(q.seed)
    n_mc, sims = math.ceil(n / vr), 4000

    def reject_rate(pt):
        xa, xb = rng.binomial(n_mc, p1, sims), rng.binomial(n_mc, pt, sims)
        pp = (xa + xb) / (2 * n_mc)
        z = (xb - xa) / n_mc / np.sqrt(np.maximum(2 * pp * (1 - pp) / n_mc, 1e-18))
        return float((np.abs(z) > _zq(1 - q.alpha / 2)).mean())

    scen = []
    for f in (0.5, 1.0, 2.0):
        m = q.mde_rel * f
        nn = math.ceil(_prop_n(p1, min(0.999, p1 * (1 + m)), q.alpha, q.power) * vr)
        scen.append({"mde_rel_pct": m * 100, "n_per_arm": nn, "days_at_50_50": 2 * nn / q.daily_users})
    return _clean({
        "p_control": p1, "p_treatment": p2, "abs_delta": p2 - p1,
        "n_per_arm": n, "n_total": 2 * n, "days_needed": 2 * n / q.daily_users,
        "rule_of_thumb_n": 16 * p1 * (1 - p1) / (p2 - p1) ** 2 * vr,
        "power_curve": {"n": ns, "power": power_curve},
        "mde_curve": {"n": ns, "mde_rel_pct": mde_curve},
        "monte_carlo": {"sims": sims, "n_per_arm": n_mc, "empirical_power": reject_rate(p2),
                        "empirical_false_positive_rate": reject_rate(p1)},
        "scenarios": scen,
    })


# ----------------------------------------------------------------------------------------
# POST /srm  (PITFALL: sample ratio mismatch. A logging bug drops low-engagement treatment
#             users -> a fake "win" in an A/A test. Chi-square on counts catches it.)
# ----------------------------------------------------------------------------------------
class SRMReq(BaseModel):
    n_total: int = Field(40000, ge=2000, le=60000)
    inject_bug: bool = False
    drop_rate: float = Field(0.10, ge=0.0, le=0.5)
    seed: int = 0


def _srm_test(nc, nt):
    e = (nc + nt) / 2
    chi2 = (nc - e) ** 2 / e + (nt - e) ** 2 / e if e > 0 else 0.0
    return chi2, _chi2_p_1df(chi2)


@router.post("/srm")
def srm(q: SRMReq):
    rng = np.random.default_rng(q.seed)
    n = q.n_total
    mult, rate, plat = _population(rng, n)
    nsess = 1 + rng.poisson(rate)
    T = rng.random(n) < 0.5                                   # designed 50/50 split
    y = _run_period(rng, mult, nsess, CONTROL_NOISE)["clicks"]   # A/A: both arms get the SAME ranker
    logged = np.ones(n, bool)
    if q.inject_bug:
        # Treatment client on android times out on slow devices (= low-engagement users) and
        # their exposure event is never logged, so they vanish from the analysis table.
        low = y <= np.median(y)
        pdrop = np.where(T & (plat == 1) & low, min(1.0, 2 * q.drop_rate), 0.0)
        logged = rng.random(n) >= pdrop
    nc, nt = int((logged & ~T).sum()), int((logged & T).sum())
    chi2, p = _srm_test(nc, nt)
    seg = []
    for k, name in enumerate(PLATFORMS):
        c, t = int((logged & ~T & (plat == k)).sum()), int((logged & T & (plat == k)).sum())
        x2, pv = _srm_test(c, t)
        seg.append({"platform": name, "control": c, "treatment": t, "treatment_share": t / (c + t),
                    "chi2": x2, "p_value": pv, "srm": pv < 0.001})
    observed = _welch(y[logged & ~T], y[logged & T])
    itt = _welch(y[~T], y[T])                                 # everyone who was *assigned*
    return _clean({
        "designed_split": "50 / 50", "bug_injected": q.inject_bug,
        "control": nc, "treatment": nt, "treatment_share": nt / (nc + nt),
        "missing_treatment_users": int((~logged).sum()),
        "chi2": chi2, "p_value": p, "srm_detected": p < 0.001, "srm_threshold": 0.001,
        "by_platform": seg,
        "true_effect": 0.0,
        "observed_analysis": {k: observed[k] for k in ("rel_lift", "rel_ci", "p", "significant")},
        "intention_to_treat_analysis": {k: itt[k] for k in ("rel_lift", "rel_ci", "p", "significant")},
    })


# ----------------------------------------------------------------------------------------
# POST /cuped  (concept: Y_cv = Y - theta (X - mean X), theta = cov(X,Y)/var(X);
#               variance shrinks by (1 - rho^2). PITFALL: X must be PRE-treatment.)
# ----------------------------------------------------------------------------------------
class CupedReq(BaseModel):
    n_per_arm: int = Field(3000, ge=200, le=15000)
    treat_noise: float = Field(DEFAULT_TREAT_NOISE, ge=0.0, le=0.6)
    user_heterogeneity: float = Field(0.5, ge=0.0, le=1.2)
    leaky_covariate: bool = False
    alpha: float = Field(0.05, gt=0.0, lt=0.5)
    seed: int = 0


@router.post("/cuped")
def cuped(q: CupedReq):
    rng = np.random.default_rng(q.seed)
    n = 2 * q.n_per_arm
    mult, rate, _ = _population(rng, n, q.user_heterogeneity)
    T = (rng.permutation(n) < q.n_per_arm).astype(int)
    noise = np.where(T == 1, q.treat_noise, CONTROL_NOISE)
    pre = _run_period(rng, mult, 1 + rng.poisson(rate), CONTROL_NOISE)["clicks"]   # before the experiment
    s1, s2 = 1 + rng.poisson(rate / 2), 1 + rng.poisson(rate / 2)
    y1 = _run_period(rng, mult, s1, noise)["clicks"]                                # experiment week 1
    y2 = _run_period(rng, mult, s2, noise)["clicks"]                                # experiment week 2
    y = y1 + y2
    x = y1 if q.leaky_covariate else pre        # leaky = measured AFTER assignment, moved by treatment
    xc = x - np.where(T == 1, x[T == 1].mean(), x[T == 0].mean())
    yc = y - np.where(T == 1, y[T == 1].mean(), y[T == 0].mean())
    theta = float((xc * yc).sum() / (xc ** 2).sum())
    rho = float((xc * yc).sum() / math.sqrt((xc ** 2).sum() * (yc ** 2).sum()))
    y_cv = y - theta * (x - x.mean())
    raw = _welch(y[T == 0], y[T == 1], q.alpha)
    adj = _welch(y_cv[T == 0], y_cv[T == 1], q.alpha)
    tc, tt = _true_effect(q.seed + 99991, mult, s1 + s2, q.treat_noise)
    true_diff = tt - tc
    idx = rng.choice(n, 300, replace=False)

    def pack(w):
        return {"diff": w["diff"], "se": w["se"], "ci": w["ci"], "p": w["p"], "significant": w["significant"],
                "ci_width": w["ci"][1] - w["ci"][0], "covers_truth": w["ci"][0] <= true_diff <= w["ci"][1]}

    return _clean({
        "covariate": "clicks in experiment week 1 (POST-treatment: wrong)" if q.leaky_covariate
                     else "clicks in the pre-experiment period (correct)",
        "true_diff": true_diff, "theta": theta, "rho": rho,
        "variance_reduction": 1 - (adj["se"] / raw["se"]) ** 2, "theory_variance_reduction": rho ** 2,
        "equivalent_traffic_multiplier": (raw["se"] / adj["se"]) ** 2,
        "covariate_imbalance": float(x[T == 1].mean() - x[T == 0].mean()),
        "raw": pack(raw), "cuped": pack(adj),
        "scatter": {"x": x[idx], "y": y[idx], "arm": T[idx]},
    })


# ----------------------------------------------------------------------------------------
# POST /novelty  (concept: novelty / primacy. Short-run lift != long-run lift. Diagnose with
#                 the daily lift curve and the new-user cohort, who have no "old UI" habit.)
# ----------------------------------------------------------------------------------------
class NoveltyReq(BaseModel):
    n_per_arm_per_day: int = Field(3000, ge=300, le=10000)
    days: int = Field(28, ge=7, le=42)
    long_term_lift: float = Field(0.01, ge=-0.05, le=0.1)
    novelty_amp: float = Field(0.06, ge=-0.1, le=0.2)       # negative = primacy (change aversion)
    half_life_days: float = Field(4.0, ge=0.5, le=20.0)
    seed: int = 0


def _rel(a, b):
    w = _welch(a, b)
    return w["rel_lift"], w["rel_ci"][0], w["rel_ci"][1]


@router.post("/novelty")
def novelty(q: NoveltyReq):
    rng = np.random.default_rng(q.seed)
    D, n = q.days, q.n_per_arm_per_day
    tau = q.half_life_days / math.log(2)
    d = np.arange(D)
    lift_exist = q.long_term_lift + q.novelty_amp * np.exp(-d / tau)
    new_share = 0.2
    base_c, base_t = 3 * rng.gamma(2.0, 0.5, (D, n)), 3 * rng.gamma(2.0, 0.5, (D, n))
    new_c, new_t = rng.random((D, n)) < new_share, rng.random((D, n)) < new_share
    yc = rng.poisson(base_c).astype(float)
    # existing users feel the novelty; users who joined during the test only get the real effect
    yt = rng.poisson(base_t * (1 + np.where(new_t, q.long_term_lift, lift_exist[:, None]))).astype(float)

    daily = []
    for i in range(D):
        a = _rel(yc[i], yt[i])
        e = _rel(yc[i][~new_c[i]], yt[i][~new_t[i]])
        nw = _rel(yc[i][new_c[i]], yt[i][new_t[i]])
        daily.append({"day": i + 1, "lift": a[0], "lo": a[1], "hi": a[2], "existing": e[0], "new": nw[0],
                      "true": (1 - new_share) * lift_exist[i] + new_share * q.long_term_lift})
    # Fit lift_d = L + A exp(-d / tau): grid over tau, weighted least squares for (L, A)
    yv = np.array([r["lift"] for r in daily])
    w = 1 / np.array([(r["hi"] - r["lo"]) / 3.92 for r in daily]) ** 2
    best = None
    for tg in np.geomspace(0.5, 40, 60):
        X = np.c_[np.ones(D), np.exp(-d / tg)]
        cov = np.linalg.inv(X.T @ (X * w[:, None]))
        beta = cov @ (X.T @ (w * yv))
        sse = float((w * (yv - X @ beta) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, tg, beta, cov)
    _, tg, beta, cov = best
    fit_curve = beta[0] + beta[1] * np.exp(-d / tg)
    se_L = math.sqrt(cov[0, 0])

    def pooled(lo, hi, mask_new=None):
        a, b = yc[lo:hi], yt[lo:hi]
        if mask_new is not None:
            a, b = a[new_c[lo:hi] == mask_new], b[new_t[lo:hi] == mask_new]
        r = _rel(a.ravel(), b.ravel())
        return {"lift": r[0], "ci": [r[1], r[2]]}

    return _clean({
        "true_long_term_lift": q.long_term_lift,
        "daily": daily,
        "fit": {"asymptote": beta[0], "asymptote_ci": [beta[0] - 1.96 * se_L, beta[0] + 1.96 * se_L],
                "amplitude": beta[1], "half_life_days": tg * math.log(2), "curve": fit_curve},
        "estimates": [
            {"estimate": "week 1 only (ship-it-early)", **pooled(0, 7)},
            {"estimate": f"all {D} days pooled", **pooled(0, D)},
            {"estimate": "last 7 days", **pooled(D - 7, D)},
            {"estimate": "new users only (all days)", **pooled(0, D, True)},
            {"estimate": "fitted asymptote L", "lift": beta[0], "ci": [beta[0] - 1.96 * se_L, beta[0] + 1.96 * se_L]},
        ],
    })


# ----------------------------------------------------------------------------------------
# POST /peeking  (PITFALL: checking p < 0.05 every day and stopping at the first "win".
#                 Fix: group-sequential boundaries from an alpha-spending function.)
# ----------------------------------------------------------------------------------------
class PeekReq(BaseModel):
    n_sims: int = Field(1000, ge=200, le=5000)
    n_looks: int = Field(20, ge=2, le=40)
    n_per_look: int = Field(500, ge=50, le=20000)
    base_rate: float = Field(0.10, gt=0.005, lt=0.9)
    true_lift: float = Field(0.0, ge=-0.5, le=0.5)       # 0 => A/A tests
    alpha: float = Field(0.05, gt=0.001, lt=0.3)
    seed: int = 0


def _spend(kind, t, alpha):
    """Lan-DeMets alpha-spending functions: cumulative type-I error allowed by information time t."""
    if kind == "obf":       # O'Brien-Fleming-like: almost nothing early, nearly all at the end
        return 2 * (1 - _ND.cdf(_zq(1 - alpha / 2) / math.sqrt(t)))
    return alpha * math.log(1 + (math.e - 1) * t)       # Pocock-like: spend evenly


@lru_cache(maxsize=32)
def _boundaries(kind: str, n_looks: int, alpha: float):
    """Critical |z| at each look so that the cumulative false-positive probability follows the
    spending function. Solved by Monte Carlo on 200k null Brownian paths (deterministic seed)."""
    M = 200_000 if n_looks <= 25 else 120_000
    rng = np.random.default_rng(2024)
    Z = np.abs(np.cumsum(rng.standard_normal((M, n_looks), dtype=np.float32), axis=1)
               / np.sqrt(np.arange(1, n_looks + 1, dtype=np.float32)))
    alive = np.ones(M, bool)
    out, prev = [], 0.0
    for k in range(n_looks):
        cum = _spend(kind, (k + 1) / n_looks, alpha)
        need = (cum - prev) * M                         # number of null paths allowed to reject now
        prev = cum
        zk = Z[alive, k]
        if need < 5:                                    # too little alpha for MC: analytic, conservative
            c = min(6.0, _zq(1 - min(cum, 0.5) / 2))
        else:
            c = float(np.quantile(zk, 1 - min(1.0, need / len(zk))))
        out.append(c)
        alive &= Z[:, k] <= c
    return tuple(out)


@router.post("/peeking")
def peeking(q: PeekReq):
    rng = np.random.default_rng(q.seed)
    N, Kl, m = q.n_sims, q.n_looks, q.n_per_look
    pa, pb = q.base_rate, min(0.999, q.base_rate * (1 + q.true_lift))
    ca = np.cumsum(rng.binomial(m, pa, (N, Kl)), axis=1)      # real Bernoulli conversions, arriving in batches
    cb = np.cumsum(rng.binomial(m, pb, (N, Kl)), axis=1)
    nk = m * np.arange(1, Kl + 1)
    pp = (ca + cb) / (2 * nk)
    z = (cb - ca) / nk / np.sqrt(np.maximum(2 * pp * (1 - pp) / nk, 1e-18))
    az = np.abs(z)
    zfix = _zq(1 - q.alpha / 2)
    inf = np.full(Kl, np.inf)
    fixed = inf.copy()
    fixed[-1] = zfix
    methods = {
        "fixed horizon (look once at the end)": fixed,
        "naive peeking (p < alpha at every look)": np.full(Kl, zfix),
        "Bonferroni over looks (alpha / K)": np.full(Kl, _zq(1 - q.alpha / (2 * Kl))),
        "alpha spending: Pocock-like": np.array(_boundaries("pocock", Kl, q.alpha)),
        "alpha spending: O'Brien-Fleming-like": np.array(_boundaries("obf", Kl, q.alpha)),
    }
    res, curves, bounds = [], {}, {}
    for name, b in methods.items():
        cross = az > b[None, :]
        rejected = cross.any(1)
        first = np.where(rejected, cross.argmax(1) + 1, Kl)
        # among rejections, how often did it "win" in the wrong direction / what lift was reported
        zstop = z[np.arange(N), first - 1]
        est = ((cb - ca) / nk / np.maximum(ca / nk, 1e-9))[np.arange(N), first - 1]
        res.append({"method": name, "reject_rate": rejected.mean(),
                    "avg_sample_fraction": first.mean() / Kl,
                    "mean_reported_lift_when_stopped_pct": float(est[rejected].mean() * 100) if rejected.any() else None,
                    "wrong_sign_rate": float(((zstop < 0) & rejected).mean()) if q.true_lift > 0 else None})
        curves[name] = np.maximum.accumulate(cross, axis=1).mean(0)
        bounds[name] = np.where(np.isfinite(b), b, np.nan)
    naive_cross = (az > zfix).any(1)
    fooled = np.where(naive_cross & (az[:, -1] <= zfix))[0][:3]
    others = np.where(~naive_cross)[0][:3]
    paths = [{"sim": int(i), "crossed_naive": bool(naive_cross[i]), "z": z[i]} for i in list(fooled) + list(others)]
    return _clean({
        "mode": "A/A (no true effect): reject_rate = false positive rate" if q.true_lift == 0
                else "A/B with a true effect: reject_rate = power",
        "nominal_alpha": q.alpha, "looks": np.arange(1, Kl + 1), "n_per_arm_at_end": m * Kl,
        "methods": res, "cumulative_reject_curves": curves, "boundaries": bounds, "sample_paths": paths,
        "mc_se_of_rate": math.sqrt(q.alpha * (1 - q.alpha) / N),
    })


# ----------------------------------------------------------------------------------------
# POST /multiple_testing  (concept: 20 metrics at alpha=0.05 => 64% chance of a false "win".
#                          FWER control: Bonferroni, Holm. FDR control: Benjamini-Hochberg.)
# ----------------------------------------------------------------------------------------
class MTReq(BaseModel):
    n_metrics: int = Field(20, ge=2, le=100)
    n_true: int = Field(3, ge=0, le=100)
    effect_z: float = Field(3.0, ge=0.0, le=6.0)     # true effect in standard-error units
    alpha: float = Field(0.05, gt=0.001, lt=0.3)
    n_sims: int = Field(1000, ge=100, le=5000)
    seed: int = 0


def _mt_decisions(p, alpha):
    """p: (sims, m). Returns dict of boolean reject arrays in the ORIGINAL metric order."""
    S, m = p.shape
    order = np.argsort(p, axis=1)
    ps = np.take_along_axis(p, order, 1)
    i = np.arange(m)
    holm_s = np.cumprod(ps <= alpha / (m - i), axis=1).astype(bool)          # step-down: stop at first failure
    ok = ps <= (i + 1) * alpha / m
    bh_s = np.flip(np.maximum.accumulate(np.flip(ok, 1), axis=1), 1)         # step-up: largest k that passes
    back = np.argsort(order, axis=1)
    return {
        "uncorrected": p <= alpha,
        "Bonferroni": p <= alpha / m,
        "Holm": np.take_along_axis(holm_s, back, 1),
        "Benjamini-Hochberg": np.take_along_axis(bh_s, back, 1),
    }


@router.post("/multiple_testing")
def multiple_testing(q: MTReq):
    rng = np.random.default_rng(q.seed)
    m, k = q.n_metrics, min(q.n_true, q.n_metrics)
    mu = np.zeros(m)
    mu[:k] = q.effect_z
    # z = (mean_T - mean_C) / SE is exactly N(effect/SE, 1) for large n: sample the sufficient statistic
    z = rng.standard_normal((q.n_sims, m)) + mu
    p = _p2v(z)
    dec = _mt_decisions(p, q.alpha)
    is_true = mu > 0
    summ = []
    for name, r in dec.items():
        fp, tp = (r & ~is_true).sum(1), (r & is_true).sum(1)
        summ.append({"method": name, "controls": {"uncorrected": "nothing", "Bonferroni": "FWER", "Holm": "FWER",
                                                  "Benjamini-Hochberg": "FDR"}[name],
                     "FWER (P any false win)": (fp > 0).mean(),
                     "FDR (false / all wins)": (fp / np.maximum(fp + tp, 1)).mean(),
                     "power (true effects found)": tp.sum() / max(1, k * q.n_sims) if k else None,
                     "avg false wins per experiment": fp.mean()})
    o = np.argsort(p[0])
    one = [{"rank": j + 1, "metric": f"metric_{i + 1:02d}", "truly_moved": bool(is_true[i]), "p_value": p[0, i],
            "bonferroni_thr": q.alpha / m, "holm_thr": q.alpha / (m - j), "bh_thr": (j + 1) * q.alpha / m,
            **{name: bool(r[0, i]) for name, r in dec.items()}} for j, i in enumerate(o)]
    return _clean({
        "n_metrics": m, "n_truly_moved": k, "alpha": q.alpha,
        "analytic_p_any_false_win_uncorrected": 1 - (1 - q.alpha) ** (m - k),
        "summary": summ, "one_experiment": one,
    })


# ----------------------------------------------------------------------------------------
# POST /interference  (concept: SUTVA violation. My outcome depends on my friends' treatment.
#                      User randomisation leaks treatment into control; cluster (graph-cluster)
#                      randomisation recovers the total effect at a variance cost.)
# ----------------------------------------------------------------------------------------
class NetReq(BaseModel):
    n_clusters: int = Field(100, ge=20, le=200)
    cluster_size: int = Field(20, ge=5, le=50)
    direct_effect: float = Field(0.5, ge=-2.0, le=2.0)
    spillover: float = Field(0.4, ge=-2.0, le=2.0)
    within_cluster_edge_frac: float = Field(0.9, ge=0.0, le=1.0)
    n_reps: int = Field(300, ge=50, le=500)
    seed: int = 0


@router.post("/interference")
def interference(q: NetReq):
    rng = np.random.default_rng(q.seed)
    C, m, deg = q.n_clusters, q.cluster_size, 8
    n = C * m
    reps = int(max(50, min(q.n_reps, 6_000_000 // (n * deg))))
    cl = np.repeat(np.arange(C), m)
    # social graph: each user has 8 neighbours, a fraction inside their own community
    inside = rng.random((n, deg)) < q.within_cluster_edge_frac
    nb_in = cl[:, None] * m + rng.integers(0, m, (n, deg))
    nbrs = np.where(inside, nb_in, rng.integers(0, n, (n, deg)))
    base = 5 + rng.normal(0, 0.6, C)[cl] + rng.normal(0, 1.0, n)        # community + user baseline
    tte = q.direct_effect + q.spillover                                 # everyone treated vs nobody

    def outcomes(T):                                                    # T: (reps, n) 0/1
        frac = T[:, nbrs].mean(2)                                       # share of my neighbours treated
        return base + q.direct_effect * T + q.spillover * frac + rng.normal(0, 1.0, T.shape)

    def diff_se(y, T):
        n1 = T.sum(1)
        n0 = T.shape[1] - n1
        m1, m0 = (y * T).sum(1) / n1, (y * (1 - T)).sum(1) / n0
        v1 = ((y - m1[:, None]) ** 2 * T).sum(1) / (n1 - 1)
        v0 = ((y - m0[:, None]) ** 2 * (1 - T)).sum(1) / (n0 - 1)
        return m1 - m0, np.sqrt(v1 / n1 + v0 / n0)

    Tu = (np.argsort(rng.random((reps, n)), axis=1) < n // 2).astype(float)        # user-level coin flips
    Tc_cl = (np.argsort(rng.random((reps, C)), axis=1) < C // 2).astype(float)     # cluster-level coin flips
    Tc = np.repeat(Tc_cl, m, axis=1)
    yu, yc = outcomes(Tu), outcomes(Tc)
    est_u, se_u = diff_se(yu, Tu)
    est_c_naive, se_c_naive = diff_se(yc, Tc)                           # wrong: treats users as independent
    est_c, se_c = diff_se(yc.reshape(reps, C, m).mean(2), Tc_cl)        # right: analyse at the cluster level

    def row(name, est, se):
        return {"design": name, "mean_estimate": est.mean(), "bias": est.mean() - tte, "sd": est.std(ddof=1),
                "rmse": math.sqrt(((est - tte) ** 2).mean()), "avg_reported_se": se.mean(),
                "ci_coverage_of_true_effect": (np.abs(est - tte) <= 1.96 * se).mean()}

    lo = float(min(est_u.min(), est_c.min(), tte)) - 0.02
    hi = float(max(est_u.max(), est_c.max(), tte)) + 0.02
    bins = np.linspace(lo, hi, 36)
    return _clean({
        "n_users": n, "n_reps": reps, "true_total_effect": tte, "direct_effect": q.direct_effect,
        "spillover": q.spillover,
        "expected_user_rand_estimate": q.direct_effect,
        "expected_cluster_rand_estimate": q.direct_effect + q.spillover * float(
            (cl[nbrs] == cl[:, None]).mean()),
        "designs": [
            row("user randomisation", est_u, se_u),
            row("cluster randomisation, cluster-level SE", est_c, se_c),
            row("cluster randomisation, naive user-level SE", est_c_naive, se_c_naive),
        ],
        "hist": {"centers": (bins[:-1] + bins[1:]) / 2, "user": np.histogram(est_u, bins)[0],
                 "cluster": np.histogram(est_c, bins)[0]},
    })


# ----------------------------------------------------------------------------------------
# POST /interleaving  (concept: team-draft interleaving. Each user sees a blend of both rankers
#                      and acts as their own control => 10-100x fewer sessions than A/B.)
# ----------------------------------------------------------------------------------------
class ILReq(BaseModel):
    n_sessions: int = Field(2000, ge=200, le=4000)
    treat_noise: float = Field(DEFAULT_TREAT_NOISE, ge=0.0, le=0.6)
    n_reps: int = Field(150, ge=50, le=300)
    seed: int = 0


def _team_draft(rng, A, B):
    """Team-draft interleaving (Radlinski et al. 2008), vectorised over sessions.
    A, B: (S, K) item ids in each ranker's order. Each round the team with fewer picks (coin flip
    on ties) adds its best not-yet-picked item. Returns the shown list and each slot's team."""
    S, Kk = A.shape
    rows = np.arange(S)
    taken = np.zeros((S, Kk), bool)
    out = np.empty((S, Kk), np.int64)
    team = np.empty((S, Kk), np.int8)
    ca, cb = np.zeros(S, np.int16), np.zeros(S, np.int16)
    for pos in range(Kk):
        pick_a = (ca < cb) | ((ca == cb) & (rng.random(S) < 0.5))
        R = np.where(pick_a[:, None], A, B)
        idx = np.argmin(np.take_along_axis(taken, R, 1), axis=1)       # first item not yet shown
        item = R[rows, idx]
        taken[rows, item] = True
        out[:, pos] = item
        team[:, pos] = np.where(pick_a, 0, 1)
        ca += pick_a
        cb += ~pick_a
    return out, team


@router.post("/interleaving")
def interleaving(q: ILReq):
    rng = np.random.default_rng(q.seed)
    reps = int(max(50, min(q.n_reps, 360_000 // q.n_sessions)))
    nmax = q.n_sessions - q.n_sessions % 2
    S = reps * nmax
    rel = rng.beta(1.2, 6.0, (S, K)).astype(np.float32)
    A = np.argsort(-(rel + CONTROL_NOISE * rng.standard_normal((S, K), dtype=np.float32)), axis=1)
    B = np.argsort(-(rel + q.treat_noise * rng.standard_normal((S, K), dtype=np.float32)), axis=1)
    exam = EXAM.astype(np.float32)

    # --- interleaved experiment: every session sees the blended list
    shown, team = _team_draft(rng, A, B)
    clk = rng.random((S, K), dtype=np.float32) < np.take_along_axis(rel, shown, 1) * exam
    cA, cB = (clk & (team == 0)).sum(1), (clk & (team == 1)).sum(1)
    winB, winA = (cB > cA).reshape(reps, nmax), (cA > cB).reshape(reps, nmax)
    # --- classic A/B: half the sessions get ranker A, half get ranker B
    ya = (rng.random((S, K), dtype=np.float32) < np.take_along_axis(rel, A, 1) * exam).sum(1).reshape(reps, nmax)
    yb = (rng.random((S, K), dtype=np.float32) < np.take_along_axis(rel, B, 1) * exam).sum(1).reshape(reps, nmax)

    grid = [g for g in (50, 100, 200, 400, 800, 1200, 2000, 3000, 4000) if g < nmax] + [nmax]
    rows = []
    for g in grid:
        wb, wa = winB[:, :g].sum(1), winA[:, :g].sum(1)
        z_il = (wb - wa) / np.sqrt(np.maximum(wb + wa, 1))            # sign test, ties dropped
        h = g // 2
        a, b = ya[:, :h].astype(float), yb[:, h:2 * h].astype(float)
        z_ab = (b.mean(1) - a.mean(1)) / np.sqrt(a.var(1, ddof=1) / h + b.var(1, ddof=1) / h)
        rows.append({"sessions": g, "interleaving_power": (np.abs(z_il) > 1.96).mean(),
                     "ab_power": (np.abs(z_ab) > 1.96).mean(),
                     "interleaving_mean_z": z_il.mean(), "ab_mean_z": z_ab.mean()})
    # analytic sessions-for-80%-power from the measured effect sizes
    zz = (1.96 + 0.8416) ** 2
    pw = winB.mean() / max(winB.mean() + winA.mean(), 1e-9)           # P(B wins | not a tie)
    decisive = winB.mean() + winA.mean()
    n_il = zz / (4 * (pw - 0.5) ** 2) / decisive if abs(pw - 0.5) > 1e-6 else None
    dlt = yb.mean() - ya.mean()
    n_ab = 2 * zz * (ya.var() + yb.var()) / dlt ** 2 if abs(dlt) > 1e-9 else None
    ex = 0
    inv_a, inv_b = np.argsort(A[ex]), np.argsort(B[ex])
    example = [{"slot": i + 1, "item": int(shown[ex, i]), "team": "A (control)" if team[ex, i] == 0 else "B (new)",
                "rank_in_A": int(inv_a[shown[ex, i]]) + 1, "rank_in_B": int(inv_b[shown[ex, i]]) + 1,
                "true_relevance": float(rel[ex, shown[ex, i]]), "clicked": bool(clk[ex, i])} for i in range(K)]
    return _clean({
        "n_reps": reps, "is_aa": q.treat_noise == CONTROL_NOISE,
        "true_ab_lift": _ranker_quality(q.treat_noise)[0] / _ranker_quality(CONTROL_NOISE)[0] - 1,
        "interleaving": {"b_win_rate": winB.mean(), "a_win_rate": winA.mean(), "tie_rate": 1 - decisive,
                         "p_b_wins_given_decisive": pw},
        "ab": {"clicks_per_session_A": ya.mean(), "clicks_per_session_B": yb.mean(), "observed_lift": yb.mean() / ya.mean() - 1},
        "power_by_sessions": rows,
        "sessions_for_80pct_power": {"interleaving": n_il, "ab_test": n_ab,
                                     "sensitivity_ratio": (n_ab / n_il) if n_il and n_ab else None},
        "example_session": example,
    })


# ----------------------------------------------------------------------------------------
# POST /offline_eval  (concepts: logged clicks are biased by where the OLD ranker put items.
#   Part 1: naive click replay vs position-debiased IPS (propensities from shuffled traffic).
#   Part 2: off-policy evaluation of a top-slot policy: naive match, IPS, clipped IPS, SNIPS.)
# ----------------------------------------------------------------------------------------
class OPEReq(BaseModel):
    n_logged: int = Field(20000, ge=2000, le=60000)
    explore_frac: float = Field(0.05, ge=0.0, le=0.5)     # share of traffic with uniformly shuffled order
    temperature: float = Field(0.10, ge=0.02, le=1.0)     # logging-policy softmax temperature (part 2)
    clip: float = Field(10.0, ge=1.0, le=200.0)
    seed: int = 0


@router.post("/offline_eval")
def offline_eval(q: OPEReq):
    rng = np.random.default_rng(q.seed)
    S = q.n_logged
    rows = np.arange(S)[:, None]
    # ---------- Part 1: ranking evaluation under position bias ----------
    rel = rng.beta(1.2, 6.0, (S, K))
    log_order = np.argsort(-(rel + CONTROL_NOISE * rng.standard_normal((S, K))), axis=1)
    explore = rng.random(S) < q.explore_frac
    log_order[explore] = np.argsort(rng.random((int(explore.sum()), K)), axis=1)    # randomised slice
    log_pos = np.argsort(log_order, axis=1)                    # item -> position it was shown at
    click = rng.random((S, K)) < rel * EXAM[log_pos]           # position-biased feedback (item-indexed)

    # propensity estimation: on shuffled traffic CTR(pos) is proportional to exam[pos]
    clicks_by_pos = np.take_along_axis(click, log_order, 1)
    if explore.sum() >= 200:
        ctr = clicks_by_pos[explore].mean(0)
        prop_src = "randomised traffic (correct)"
    else:   # pitfall: CTR-by-position on ranked traffic confounds position with relevance
        ctr = clicks_by_pos.mean(0)
        prop_src = "observational CTR by position (confounded with relevance: wrong)"
    prop = np.maximum(ctr / max(ctr[0], 1e-9), 0.02)

    def evaluate(order):
        new_pos = np.argsort(order, axis=1)
        e_new = EXAM[new_pos]
        truth = (rel * e_new).sum(1).mean()                              # needs the simulator's hidden truth
        naive = (click * e_new).sum(1).mean()                            # clicks treated as relevance labels
        ips = (click * e_new / prop[log_pos]).sum(1).mean()              # each click re-weighted by 1/P(examined)
        return truth, naive, ips

    base = evaluate(log_order)
    cands = [("logging policy itself", None, base)]
    for nz in (0.0, 0.05, 0.10, 0.15, 0.30, 0.60):
        order = np.argsort(-(rel + nz * rng.standard_normal((S, K))), axis=1)
        label = f"ranker noise {nz:.2f}" + (" (oracle)" if nz == 0 else " (same quality as logger)" if nz == CONTROL_NOISE else "")
        cands.append((label, nz, evaluate(order)))
    ranking = [{"candidate": name, "true_clicks_per_session": t, "true_lift_pct": (t / base[0] - 1) * 100,
                "naive_replay_lift_pct": (nv / base[1] - 1) * 100, "ips_lift_pct": (ip / base[2] - 1) * 100,
                "naive_says": "better" if nv > base[1] * 1.0001 else "worse" if nv < base[1] * 0.9999 else "-",
                "truth_says": "better" if t > base[0] * 1.0001 else "worse" if t < base[0] * 0.9999 else "-"}
               for name, nz, (t, nv, ip) in cands]

    # ---------- Part 2: off-policy evaluation of the top slot (contextual bandit) ----------
    R, n = 100, 4000
    relb = rng.beta(1.2, 6.0, (R * n, K))
    s_log = (relb + CONTROL_NOISE * rng.standard_normal(relb.shape)) / q.temperature
    pi0 = np.exp(s_log - s_log.max(1, keepdims=True))
    pi0 /= pi0.sum(1, keepdims=True)                                     # stochastic logging policy
    a = (rng.random((R * n, 1)) > np.cumsum(pi0, axis=1)).sum(1).clip(0, K - 1)
    ar = np.arange(R * n)
    reward = (rng.random(R * n) < relb[ar, a]).astype(float)
    tgt = np.argmax(relb + DEFAULT_TREAT_NOISE * rng.standard_normal(relb.shape), axis=1)   # new deterministic policy
    truth = float(relb[ar, tgt].mean())
    match = (a == tgt).astype(float)
    w = match / pi0[ar, a]                                               # importance weight pi_new / pi_log
    wr, mr, rr = w.reshape(R, n), match.reshape(R, n), reward.reshape(R, n)
    wc = np.minimum(wr, q.clip)
    ests = {
        "naive (avg reward where logger agreed)": (mr * rr).sum(1) / np.maximum(mr.sum(1), 1),
        "IPS": (wr * rr).mean(1),
        f"clipped IPS (w <= {q.clip:g})": (wc * rr).mean(1),
        "SNIPS (self-normalised)": (wr * rr).sum(1) / np.maximum(wr.sum(1), 1e-9),
    }
    ope = [{"estimator": k, "mean": v.mean(), "bias": v.mean() - truth, "sd": v.std(ddof=1),
            "rmse": math.sqrt(((v - truth) ** 2).mean())} for k, v in ests.items()]
    ess = (wr.sum(1) ** 2 / np.maximum((wr ** 2).sum(1), 1e-9)).mean()
    lo = min(float(v.min()) for v in ests.values())
    hi = max(float(v.max()) for v in ests.values())
    bins = np.linspace(min(lo, truth) - 1e-3, max(hi, truth) + 1e-3, 41)
    return _clean({
        "position_bias": {
            "n_logged_sessions": S, "n_randomised_sessions": int(explore.sum()), "propensity_source": prop_src,
            "true_examination": EXAM, "estimated_propensity": prop,
            "logged_ctr_by_position": clicks_by_pos.mean(0),
            "candidates": ranking,
        },
        "off_policy": {
            "replications": R, "logs_per_replication": n, "true_value_of_new_policy": truth,
            "logging_policy_value": float(reward.mean()),
            "logger_agreement_rate": float(match.mean()), "max_weight": float(w.max()),
            "effective_sample_size": float(ess), "estimators": ope,
            "hist": {"centers": (bins[:-1] + bins[1:]) / 2,
                     **{k.split(" (")[0]: np.histogram(v, bins)[0] for k, v in ests.items()}},
        },
    })
