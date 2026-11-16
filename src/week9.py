"""Week 9 — naive result (Project 1). Every choice comes from lock note v2.

Run (repo root, from the cache, no network):
    python -m src.week9                  full run: tau 10 x seeds 0,1,2; sensitivity tau 5, 15 (seed 0)
    python -m src.week9 --quick          small smoke test (few trees, one C value)
    python -m src.week9 --figures-only   redraw the figures from reports/week9_metrics.csv

Outputs
    reports/week9_splits.csv      group -> fold per seed (reused in week 11)
    reports/week9_hparams.csv     hyperparameters chosen inside the training folds
    reports/week9_metrics.csv     every metric: model, tau, seed, fold, set, partition, weighting
    reports/week9_results.json    summary used by the report
    figures/week9/*.png           figures cited by the report and slides

Sets scored per split (model trained on the FAFB training types of that split):
    within         held-out FAFB types (within-sex)
    within_shared  held-out FAFB types that have a MaleCNS match
    cross          MaleCNS neurons of the held-out groups (cross-sex)
    cross_em       cross, decisions re-weighted by EM-estimated test priors (no test labels)
    cross_costrule / within_costrule
                   diagnostic only: argmin_k sum_j C(j,k) p_j on the naive probabilities
    unmatched      MaleCNS neurons with no FAFB type (scored by every fold model)
    efferent       16 MaleCNS efferent neurons (class never seen in training)
    floor_same / floor_opp
                   within-animal floor: trained on one FAFB side, tested on the held-out
                   types on the same side / the other side
"""
import json
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import config
from eda_shared.eda_helpers import add_provisional_flags, MALECNS_NERVE_CORD
from src import features as F
from src.build import fafb_paths, malecns_paths

warnings.filterwarnings("ignore")
REP, FIG = Path("reports"), Path("figures/week9")
SEEDS, K = (0, 1, 2), 5

CLASSES = ["sensory", "sensory_ascending", "ascending", "optic", "visual_projection",
           "visual_centrifugal", "central", "descending", "motor", "endocrine"]
FLOW = {"sensory": "afferent", "sensory_ascending": "afferent", "ascending": "afferent",
        "optic": "intrinsic", "visual_projection": "intrinsic", "visual_centrifugal": "intrinsic",
        "central": "intrinsic", "descending": "efferent", "motor": "efferent",
        "endocrine": "efferent", "efferent": "efferent"}
REGIONS = ["optic_lobe", "sensory_periphery", "motor_periphery", "higher_order"]
DIMS = ["isomorphic", "dimorphic", "sex_specific"]


# ------------------------------------------------------------------ cost matrices
def cost_matrix(kind, rows=CLASSES + ["efferent"]):
    """C[j, k]: cost of predicting CLASSES[k] when the truth is rows[j] (lock note)."""
    big = {"flow125": 5.0, "flow1210": 10.0, "unit": 1.0}[kind]
    C = np.zeros((len(rows), len(CLASSES)))
    for j, t in enumerate(rows):
        for k, p in enumerate(CLASSES):
            if t == p:
                continue
            if kind == "unit":
                C[j, k] = 1.0
            elif FLOW[t] == FLOW[p]:
                C[j, k] = 1.0
            elif "intrinsic" in (FLOW[t], FLOW[p]):
                C[j, k] = 2.0
            else:
                C[j, k] = big
    return C


COSTS = {k: cost_matrix(k) for k in ("flow125", "unit", "flow1210")}
C_MAIN = COSTS["flow125"]


# ------------------------------------------------------------------ data
class UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _dimorphism(s):
    s = s.astype("string").str.lower()
    out = pd.Series("isomorphic", index=s.index, dtype="string")
    out[s.str.contains("dimorphic", na=False)] = "dimorphic"
    out[s.str.contains("specific", na=False)] = "sex_specific"
    potential = s.str.contains("potentially", na=False)
    definite = out.where(~potential)          # headline: definite labels only
    return definite, out


def load_tables(tau):
    """Neurons with labels, groups, weights, partitions and the locked features."""
    f = pd.read_parquet(fafb_paths()["neurons"])
    m = pd.read_parquet(malecns_paths()["neurons"])
    f = add_provisional_flags(f)
    m = add_provisional_flags(m, nerve_cord=MALECNS_NERVE_CORD)
    for d in (f, m):
        d["region4"] = np.where(d["is_optic"], "optic_lobe", d["region_group"].astype(str))
        d["dim"], d["dim_all"] = _dimorphism(d["dimorphism"])
        ct = d["cell_type"].astype("string")
        size = ct.map(ct.value_counts())
        d["w_type"] = (1.0 / size).fillna(1.0).astype(float)
        d["w_neuron"] = 1.0
        d["super_class"] = d["super_class"].astype(str)

    # group key: FAFB cell_type; MaleCNS flywire_type; multi-type lists merge groups
    fct = f["cell_type"].astype("string")
    fafb_types = set(fct.dropna())
    uf = UF()
    for t in fafb_types:
        uf.find(t)
    ftype = m["flywire_type"].astype("string")
    for v in ftype.dropna().unique():
        parts = [x.strip() for x in str(v).split(",") if x.strip() in fafb_types]
        for a in parts[1:]:
            uf.union(parts[0], a)
    f["group"] = [("T:" + uf.find(t)) if pd.notna(t) else f"U:fafb:{i}"
                  for t, i in zip(fct, f["neuron_id"])]

    def mgroup(v):
        if pd.isna(v):
            return None
        parts = [x.strip() for x in str(v).split(",") if x.strip() in fafb_types]
        return ("T:" + uf.find(parts[0])) if parts else None

    m["group"] = [mgroup(v) for v in ftype]
    m["matched"] = m["group"].notna()

    Xf = F.load("fafb", tau).set_index("neuron_id").reindex(f["neuron_id"])
    Xm = F.load("malecns", tau).set_index("neuron_id").reindex(m["neuron_id"])
    cols = list(Xf.columns)
    return f, m, Xf[cols].to_numpy("float32"), Xm[cols].to_numpy("float32"), cols


def assign_folds(groups, sizes, seed, k=K):
    """Greedy size-balanced group folds with a seeded order (GroupKFold with shuffling)."""
    rng = np.random.default_rng(seed)
    g = np.array(groups)
    order = np.lexsort((rng.random(len(g)), -np.asarray(sizes)))
    load_, fold = np.zeros(k), {}
    for i in order:
        j = int(np.argmin(load_))
        fold[g[i]] = j
        load_[j] += sizes[i]
    return fold


# ------------------------------------------------------------------ models
def make_model(name, hp, seed, quick, n_trees=500):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    if name == "majority":
        return Majority()
    if name.startswith("logreg"):
        cw = "balanced" if name == "logreg_balanced" else None
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=hp, class_weight=cw, max_iter=2000,
                                                random_state=seed))
    if name == "rf":
        return RandomForestClassifier(n_estimators=40 if quick else n_trees, max_features="sqrt",
                                      min_samples_leaf=hp, n_jobs=-1, random_state=seed)
    raise ValueError(name)


class Majority:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        c = np.bincount(np.searchsorted(self.classes_, y), minlength=len(self.classes_))
        self.p_ = c / c.sum()
        return self

    def predict_proba(self, X):
        return np.tile(self.p_, (len(X), 1))


def proba_full(model, X):
    """Probabilities over all CLASSES (columns of classes absent in training stay 0)."""
    P = np.zeros((len(X), len(CLASSES)))
    if len(X) == 0:
        return P
    cls = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    P[:, np.asarray(cls)] = model.predict_proba(X)
    return P


GRID = {"logreg": [0.01, 0.1, 1.0, 10.0], "logreg_balanced": [0.01, 0.1, 1.0, 10.0],
        "rf": [1, 5, 20], "majority": [None]}


def tune(name, X, y, groups, seed, quick):
    """Inner 3-fold GroupKFold on the training rows; criterion log-loss (what the library fits)."""
    from sklearn.model_selection import GroupKFold
    grid = GRID[name]
    if quick:
        grid = grid[2:3] if name != "majority" else grid
    if len(grid) == 1:
        return grid[0]
    from joblib import Parallel, delayed
    folds = list(GroupKFold(n_splits=3).split(X, y, groups))

    def one(hp, tr, va):
        mdl = make_model(name, hp, seed, quick, n_trees=100).fit(X[tr], y[tr])   # RF tuned with 100 trees
        P = np.clip(proba_full(mdl, X[va]), 1e-12, 1)
        return -np.mean(np.log(P[np.arange(len(va)), y[va]]))

    jobs = [(hp, tr, va) for hp in grid for tr, va in folds]
    # logistic regression fits are single-threaded: run the grid in parallel;
    # the random forest already uses every core inside one fit
    n_jobs = -1 if name.startswith("logreg") else 1
    ll = Parallel(n_jobs=n_jobs)(delayed(one)(*j) for j in jobs)
    scores = np.array(ll).reshape(len(grid), len(folds)).mean(1)
    return grid[int(np.argmin(scores))]


def em_priors(P, prior_train, iters=200):
    """Saerens et al. 2002: test priors from unlabelled predictions; returns adjusted P."""
    pi = prior_train.copy()
    Q = P
    for _ in range(iters):
        Q = P * (pi / np.clip(prior_train, 1e-12, None))
        Q = Q / Q.sum(1, keepdims=True)
        new = Q.mean(0)
        if np.abs(new - pi).max() < 1e-8:
            break
        pi = new
    return Q, pi


# ------------------------------------------------------------------ metrics
def score_rows(df, yhat, set_name, base, partitions=("region4", "dim")):
    """Metric rows for one scored set. df: neurons with super_class, weights, partitions."""
    from sklearn.metrics import f1_score, balanced_accuracy_score
    rows = []
    truth = df["super_class"].to_numpy()
    rowsC = np.array([(CLASSES + ["efferent"]).index(t) for t in truth]) if len(df) else np.array([], int)
    cuts = [("all", "all", np.ones(len(df), bool))]
    for p in partitions:
        vals = (df[p].astype("string").fillna("<NA>").to_numpy(dtype=object).astype(str)
                if len(df) else np.array([]))
        for v in (REGIONS if p == "region4" else DIMS):
            cuts.append((p, v, vals == v))
    for pname, pval, mask in cuts:
        if mask.sum() == 0:
            continue
        for wname in ("w_type", "w_neuron"):
            w = df[wname].to_numpy()[mask]
            jj, kk = rowsC[mask], yhat[mask]
            r = {**base, "set": set_name, "partition": pname, "cell": pval,
                 "weighting": wname[2:], "n": int(mask.sum())}
            for cn, C in COSTS.items():
                r[f"cost_{cn}"] = float(np.average(C[jj, kk], weights=w))
            yt = truth[mask]
            yp = np.array(CLASSES)[kk]
            r["accuracy"] = float(np.average(yt == yp, weights=w))
            lab = sorted(set(yt) & set(CLASSES))
            if lab:
                r["macro_f1"] = float(f1_score(yt, yp, labels=lab, average="macro", sample_weight=w,
                                               zero_division=0))
                r["balanced_accuracy"] = float(balanced_accuracy_score(yt, yp, sample_weight=w))
            rows.append(r)
    return rows


# ------------------------------------------------------------------ one tau
def run_tau(tau, seeds, models, quick, log, do_floor=True, keep_oof=False):
    t0 = time.time()
    f, m, Xf, Xm, cols = load_tables(tau)
    cidx = {c: i for i, c in enumerate(CLASSES)}
    trainable = f["super_class"].isin(CLASSES).to_numpy()
    yf = f["super_class"].map(cidx).fillna(-1).astype(int).to_numpy()
    m_in = m["super_class"].isin(CLASSES).to_numpy()
    matched = m["matched"].to_numpy() & m_in
    unmatched = (~m["matched"].to_numpy()) & m_in
    efferent = (m["super_class"] == "efferent").to_numpy()
    side = f["side"].astype(str).to_numpy()
    log(f"tau={tau}: FAFB {trainable.sum():,} trainable; MaleCNS matched {matched.sum():,}, "
        f"unmatched {unmatched.sum():,}, efferent {efferent.sum()}; {len(cols)} features")

    gsize = f.loc[trainable, "group"].value_counts()
    rows, hps, splits, oof = [], [], [], {}
    msets = {}
    for seed in seeds:
        fold = assign_folds(gsize.index.to_numpy(), gsize.to_numpy(), seed)
        splits += [{"tau": tau, "seed": seed, "group": g, "fold": k} for g, k in fold.items()]
        ff = f["group"].map(fold).to_numpy()
        mf = m["group"].map(fold).to_numpy()
        for k in range(K):
            tr = trainable & (ff != k)
            te = trainable & (ff == k)
            mte = matched & (mf == k)
            prior_train = np.bincount(yf[tr], minlength=len(CLASSES)) / tr.sum()
            fsh = te & f["group"].isin(set(m.loc[matched, "group"])).to_numpy()
            for name in models:
                base = {"tau": tau, "model": name, "seed": seed, "fold": k}
                hp = tune(name, Xf[tr], yf[tr], f.loc[tr, "group"].to_numpy(), seed, quick)
                mdl = make_model(name, hp, seed, quick).fit(Xf[tr], yf[tr])
                hps.append({**base, "hyperparameter": hp})
                Pw = proba_full(mdl, Xf[te])
                Pc = proba_full(mdl, Xm[mte])
                Pu = proba_full(mdl, Xm[unmatched])
                Pe = proba_full(mdl, Xm[efferent])
                rows += score_rows(f[te], Pw.argmax(1), "within", base)
                rows += score_rows(f[fsh], proba_full(mdl, Xf[fsh]).argmax(1), "within_shared", base)
                rows += score_rows(m[mte], Pc.argmax(1), "cross", base)
                rows += score_rows(m[unmatched], Pu.argmax(1), "unmatched", base)
                rows += score_rows(m[efferent], Pe.argmax(1), "efferent", base, partitions=())
                if name != "majority":
                    Q, pi = em_priors(Pc, prior_train)
                    rows += score_rows(m[mte], Q.argmax(1), "cross_em", base)
                    rows += score_rows(f[te], (Pw @ C_MAIN[:len(CLASSES)]).argmin(1), "within_costrule", base)
                    rows += score_rows(m[mte], (Pc @ C_MAIN[:len(CLASSES)]).argmin(1), "cross_costrule", base)
                    rows += rule_change_rows(m[mte], Pc, base)
                    msets.setdefault(name, []).append({"seed": seed, "fold": k,
                                                       "em_prior": pi.round(4).tolist(),
                                                       "train_prior": prior_train.round(4).tolist()})
                # unmatched reference: within-sex cost of the same super_class x region cells
                rows += unmatched_reference(f[te], Pw.argmax(1), m[unmatched], Pu.argmax(1), base)
                if keep_oof and seed == seeds[0]:
                    store = oof.setdefault(name, {"within": {}, "cross": {}})
                    for idx, P in ((np.where(te)[0], Pw), (np.where(mte)[0], Pc)):
                        tgt = store["within"] if P is Pw else store["cross"]
                        tgt.update({int(i): p for i, p in zip(idx, P)})
                    if name == "logreg":
                        oof.setdefault("efferent_pred", []).extend(
                            np.array(CLASSES)[Pe.argmax(1)].tolist())
                # within-animal floor: train one side, test held-out types on both sides
                if do_floor and name != "majority":
                    for a, b in (("left", "right"), ("right", "left")):
                        tr_s = tr & (side == a)
                        if tr_s.sum() < 50 or len(np.unique(yf[tr_s])) < 2:
                            continue
                        ms = make_model(name, hp, seed, quick).fit(Xf[tr_s], yf[tr_s])
                        for s_name, s_side in (("floor_same", a), ("floor_opp", b)):
                            sel = te & (side == s_side)
                            rows += score_rows(f[sel], proba_full(ms, Xf[sel]).argmax(1),
                                               s_name, {**base, "direction": f"{a}->{b}"})
            done = (seeds.index(seed)) * K + k + 1
            el = time.time() - t0
            log(f"  tau={tau} seed={seed} fold={k} done ({el / 60:.1f} min; "
                f"about {el / done * (len(seeds) * K - done) / 60:.0f} min left for this tau)")
    return pd.DataFrame(rows), pd.DataFrame(hps), pd.DataFrame(splits), oof, msets, f, m


def rule_change_rows(df, P, base):
    """Share of neurons whose decision changes from argmax p to argmin sum_j C(j,k) p_j."""
    out = []
    if len(df) == 0:
        return out
    changed = P.argmax(1) != (P @ C_MAIN[:len(CLASSES)]).argmin(1)
    reg = df["region4"].astype(str).to_numpy()
    for cell in ["all"] + REGIONS:
        mask = np.ones(len(df), bool) if cell == "all" else reg == cell
        if mask.sum() == 0:
            continue
        for wname in ("w_type", "w_neuron"):
            out.append({**base, "set": "rule_change", "partition": "all" if cell == "all" else "region4",
                        "cell": cell, "weighting": wname[2:], "n": int(mask.sum()),
                        "share_changed": float(np.average(changed[mask], weights=df[wname].to_numpy()[mask]))})
    return out


def unmatched_reference(fw, yw, mu, yu, base):
    """Cost of unmatched MaleCNS neurons vs within-sex FAFB cost of the same super_class x region cells."""
    out = []
    if len(mu) == 0 or len(fw) == 0:
        return out
    rowsC = lambda d: np.array([(CLASSES + ["efferent"]).index(t) for t in d["super_class"]])
    cw = C_MAIN[rowsC(fw), yw]
    cu = C_MAIN[rowsC(mu), yu]
    for wname in ("w_type", "w_neuron"):
        a = pd.DataFrame({"sc": fw["super_class"].to_numpy(), "rg": fw["region4"].to_numpy(),
                          "c": cw, "w": fw[wname].to_numpy()})
        a["wc"] = a["c"] * a["w"]
        cell = a.groupby(["sc", "rg"])[["wc", "w"]].sum()
        within_cell = (cell["wc"] / cell["w"])
        b = pd.DataFrame({"sc": mu["super_class"].to_numpy(), "rg": mu["region4"].to_numpy(),
                          "c": cu, "w": mu[wname].to_numpy()})
        b["ref"] = [within_cell.get((s, r), np.nan) for s, r in zip(b["sc"], b["rg"])]
        ok = b["ref"].notna()
        if ok.sum() == 0:
            continue
        out.append({**base, "set": "unmatched_vs_ref", "partition": "all", "cell": "all",
                    "weighting": wname[2:], "n": int(ok.sum()),
                    "cost_flow125": float(np.average(b.loc[ok, "c"], weights=b.loc[ok, "w"])),
                    "reference_within": float(np.average(b.loc[ok, "ref"], weights=b.loc[ok, "w"])),
                    "coverage": float(ok.mean())})
    return out


# ------------------------------------------------------------------ bootstrap over types
def bootstrap_drop(f, m, oof, model="logreg", n_boot=1000, seed=0):
    """95% interval of the type-weighted cost drop (cross - within) per region, resampling groups."""
    if model not in oof:
        return {}
    rng = np.random.default_rng(seed)
    out = {}
    W, Cr = oof[model]["within"], oof[model]["cross"]
    fi = np.array(sorted(W)); mi = np.array(sorted(Cr))
    if len(fi) == 0 or len(mi) == 0:
        return {}
    rc = lambda d: np.array([(CLASSES + ["efferent"]).index(t) for t in d["super_class"]])
    fsub, msub = f.iloc[fi], m.iloc[mi]
    cf = C_MAIN[rc(fsub), np.array([W[i] for i in fi]).argmax(1)]
    cm = C_MAIN[rc(msub), np.array([Cr[i] for i in mi]).argmax(1)]
    groups = pd.Index(sorted(set(fsub["group"]) | set(msub["group"])))
    gf, gm = groups.get_indexer(fsub["group"]), groups.get_indexer(msub["group"])
    for reg in ["all"] + REGIONS:
        sf = np.ones(len(fsub), bool) if reg == "all" else (fsub["region4"] == reg).to_numpy()
        sm = np.ones(len(msub), bool) if reg == "all" else (msub["region4"] == reg).to_numpy()
        agg = lambda g, c, w, s: (np.bincount(g[s], weights=(c * w)[s], minlength=len(groups)),
                                  np.bincount(g[s], weights=w[s], minlength=len(groups)))
        fwc, fw = agg(gf, cf, fsub["w_type"].to_numpy(), sf)
        mwc, mw = agg(gm, cm, msub["w_type"].to_numpy(), sm)
        if fw.sum() == 0 or mw.sum() == 0:
            continue
        draws = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=n_boot)
        d = draws @ mwc / np.clip(draws @ mw, 1e-12, None) - draws @ fwc / np.clip(draws @ fw, 1e-12, None)
        out[reg] = {"drop": float(mwc.sum() / mw.sum() - fwc.sum() / fw.sum()),
                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out


# ------------------------------------------------------------------ summaries and figures
def summarise(M, metric="cost_flow125", weighting="type"):
    d = M[(M["weighting"] == weighting)]
    g = d.groupby(["tau", "model", "set", "partition", "cell"])[metric]
    return g.agg(["mean", "std", "count"]).reset_index()


def drops(M, metric="cost_flow125", weighting="type", partition="region4"):
    """Per split: cross - within, and floor_opp - floor_same, per cell of the partition."""
    d = M[(M["weighting"] == weighting) & (M["partition"].isin([partition, "all"]))]
    key = ["tau", "model", "seed", "fold", "partition", "cell"]
    piv = d.pivot_table(index=key, columns="set", values=metric, aggfunc="mean").reset_index()
    out = piv[key].copy()
    out["drop_cross"] = piv.get("cross") - piv.get("within")
    if "floor_opp" in piv:
        out["floor"] = piv["floor_opp"] - piv["floor_same"]
    if "within_shared" in piv:
        out["drop_cross_shared"] = piv.get("cross") - piv.get("within_shared")
    if "cross_em" in piv:
        out["drop_cross_em"] = piv.get("cross_em") - piv.get("within")
    return out


def figures(M, R, f, m, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    lab = {"optic_lobe": "Optic lobe", "sensory_periphery": "Sensory\nperiphery",
           "motor_periphery": "Motor\nperiphery", "higher_order": "Higher-order", "all": "All"}
    D = drops(M[M["tau"] == 10])

    # 1 headline: drop by region vs floor (logreg, type-weighted cost)
    fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=200)
    x = np.arange(len(REGIONS))
    s = D[(D["model"] == "logreg") & (D["partition"] == "region4")].groupby("cell")
    for off, col, cname, lbl in ((-0.19, "#9bb7d9", "drop_cross", "vs all held-out female types (pre-registered)"),
                                 (0.19, "#2f6db5", "drop_cross_shared", "vs held-out female types with a male match")):
        ax.bar(x + off, s[cname].mean().reindex(REGIONS), 0.36, yerr=s[cname].std().reindex(REGIONS),
               color=col, capsize=2, label=lbl)
    ax.errorbar(x + 0.42, s["floor"].mean().reindex(REGIONS), yerr=s["floor"].std().reindex(REGIONS),
                fmt="D", color="#333", ms=3.5, capsize=2, lw=0.8, label="within-animal floor (FAFB left ↔ right)")
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_xticks(x, [lab[r] for r in REGIONS], fontsize=8)
    ax.set_ylabel("Δ expected cost per type\n(male test − female held-out)", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.set_title("Naive logistic regression: no region gets worse on the male brain (τ = 10, mean ± SD, 15 splits)",
                 fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig1_headline_drop_by_region.png"); plt.close(fig)

    # 2 models: within vs cross, type- and neuron-weighted cost
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=200, sharey=True)
    for ax, wt in zip(axs, ("type", "neuron")):
        S = summarise(M[M["tau"] == 10], weighting=wt)
        S = S[(S["partition"] == "all")]
        mdls = [x_ for x_ in ("majority", "logreg", "logreg_balanced", "rf") if x_ in set(S["model"])]
        for i, st in enumerate(("within", "cross")):
            v = S[S["set"] == st].set_index("model").reindex(mdls)
            ax.bar(np.arange(len(mdls)) + (i - 0.5) * 0.38, v["mean"], 0.36, yerr=v["std"], capsize=2,
                   color=("#9bb7d9", "#2f6db5")[i], label=("female held-out", "male test")[i])
        ax.set_xticks(np.arange(len(mdls)), [x_.replace("_", "\n") for x_ in mdls], fontsize=7)
        ax.set_title(f"{wt}-weighted", fontsize=8)
    axs[0].set_ylabel("expected cost per neuron", fontsize=8); axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig2_models_cost.png"); plt.close(fig)

    # 3 decision rule: cost saved by argmin sum_j C(j,k) p_j over argmax p, same probabilities
    d = M[(M["tau"] == 10) & (M["weighting"] == "type") & M["partition"].isin(["all", "region4"])
          & M["set"].isin(["cross", "cross_costrule"])]
    pv = d.pivot_table(index=["model", "seed", "fold", "cell"], columns="set", values="cost_flow125").reset_index()
    pv["saved"] = pv["cross"] - pv["cross_costrule"]
    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=200)
    cells = ["all"] + REGIONS
    for i, (mdl, col, lbl) in enumerate((("logreg", "#2f6db5", "logistic regression"),
                                         ("logreg_balanced", "#cc4125", "logistic regression, class_weight='balanced'"))):
        g = pv[pv["model"] == mdl].groupby("cell")["saved"]
        ax.bar(np.arange(5) + (i - 0.5) * 0.38, g.mean().reindex(cells), 0.36, yerr=g.std().reindex(cells),
               capsize=2, color=col, label=lbl)
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_xticks(np.arange(5), [lab[r] for r in cells], fontsize=8)
    ax.set_ylabel("cost saved per type by the\ncost-minimising decision", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Male test set: same probabilities, argmin Σ C(j,k) p_j instead of argmax p (τ = 10)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig3_decision_rule.png"); plt.close(fig)

    # 4 dimorphism
    Dd = drops(M[M["tau"] == 10], partition="dim")
    fig, ax = plt.subplots(figsize=(5.2, 2.8), dpi=200)
    s = Dd[(Dd["model"] == "logreg") & (Dd["partition"] == "dim")].groupby("cell")["drop_cross"]
    ax.bar(np.arange(3), s.mean().reindex(DIMS), yerr=s.std().reindex(DIMS), color="#2f6db5", capsize=2)
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_xticks(np.arange(3), ["isomorphic", "sexually\ndimorphic", "sex-specific\n(matched label)"], fontsize=8)
    ax.set_ylabel("Δ expected cost per type", fontsize=8)
    ax.set_title("Cross-sex drop by dimorphism label (logreg, τ = 10)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "fig4_drop_by_dimorphism.png"); plt.close(fig)

    # 5 class mix: FAFB vs MaleCNS brain part
    fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=200)
    a = f["super_class"].value_counts(normalize=True).reindex(CLASSES).fillna(0)
    b = m["super_class"].value_counts(normalize=True).reindex(CLASSES).fillna(0)
    ax.bar(np.arange(10) - 0.2, a, 0.38, color="#c27ba0", label="FAFB (female)")
    ax.bar(np.arange(10) + 0.2, b, 0.38, color="#2f6db5", label="MaleCNS brain part")
    ax.set_xticks(np.arange(10), [c.replace("_", "\n") for c in CLASSES], fontsize=6.5)
    ax.set_ylabel("share of neurons", fontsize=8); ax.legend(fontsize=7, frameon=False)
    ax.set_title("Class mix differs between the datasets (prior shift)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "fig5_class_mix.png"); plt.close(fig)
    log(f"figures -> {FIG}")


def eda_counts(f, m):
    """Numbers behind the prior-shift explanation (no model involved)."""
    out = {}
    for k, d in (("fafb", f), ("malecns", m)):
        ct = d["cell_type"].astype("string")
        out[k] = {"neurons": int(len(d)),
                  "sensory": int((d["super_class"] == "sensory").sum()),
                  "R1-6": int((ct == "R1-6").sum()),
                  "region_LA": int((d["neuropil"] == "LA").sum()),
                  "super_class_share": d["super_class"].value_counts(normalize=True).round(4).to_dict()}
    return out


# ------------------------------------------------------------------ main
def main(quick=False):
    t0 = time.time()
    log_lines = []

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log_lines.append(s)

    import sklearn
    log(f"seeds {SEEDS}; folds {K}; sklearn {sklearn.__version__}; numpy {np.__version__}; "
        f"pandas {pd.__version__}; config {config.CONFIG_VERSION}")
    models = ["majority", "logreg", "logreg_balanced", "rf"]
    M, H, S, oof, msets, f, m = run_tau(10, SEEDS, models, quick, log, keep_oof=True)
    REP.mkdir(exist_ok=True)
    M.to_csv(REP / "week9_metrics_tau10.csv", index=False)      # checkpoint: main result is safe
    allM, allH, allS = [M], [H], [S]
    if not quick and "--no-sensitivity" not in sys.argv:
        for t in (5, 15):            # sensitivity, seed 0 (5 splits), naive LR and RF
            try:
                M2, H2, S2, _, _, _, _ = run_tau(t, (0,), ["majority", "logreg", "rf"], quick, log,
                                                 do_floor=False)
                allM.append(M2); allH.append(H2); allS.append(S2)
            except Exception as ex:                              # keep the main result
                log(f"sensitivity tau={t} failed: {ex!r}")
    M = pd.concat(allM, ignore_index=True)
    REP.mkdir(exist_ok=True)
    M.to_csv(REP / "week9_metrics.csv", index=False)
    pd.concat(allH).to_csv(REP / "week9_hparams.csv", index=False)
    pd.concat(allS).to_csv(REP / "week9_splits.csv", index=False)

    R = {"config_version": config.CONFIG_VERSION, "seeds": list(SEEDS), "folds": K,
         "classes": CLASSES, "quick": quick}
    for wt in ("type", "neuron"):
        for metric in ("cost_flow125", "cost_unit", "cost_flow1210", "accuracy", "macro_f1",
                       "balanced_accuracy"):
            Sx = summarise(M, metric, wt)
            Sx = Sx[Sx["partition"] == "all"]
            R.setdefault("overall", {}).setdefault(wt, {})[metric] = {
                f"{r.tau}|{r.model}|{r.set}": [round(r.mean, 4), round(r.std, 4) if r.std == r.std else None]
                for r in Sx.itertuples()}
    for part in ("region4", "dim"):
        D = drops(M[M["tau"] == 10], partition=part)
        cols = [c for c in ("drop_cross", "floor", "drop_cross_shared", "drop_cross_em") if c in D]
        g = D[D["partition"].isin([part, "all"])].groupby(["model", "cell"])[cols]
        R[f"drops_{part}"] = {}
        tab = g.agg(["mean", "std"])
        for (mdl, cell), row in tab.iterrows():
            R[f"drops_{part}"][f"{mdl}|{cell}"] = {c: [round(float(row[(c, "mean")]), 4),
                                                       round(float(row[(c, "std")]), 4)] for c in cols}
    U = M[M["set"] == "unmatched_vs_ref"]
    R["unmatched_vs_ref"] = {f"{mdl}|{wt}": {"cost": round(float(d["cost_flow125"].mean()), 4),
                                             "cost_sd": round(float(d["cost_flow125"].std()), 4),
                                             "reference_within": round(float(d["reference_within"].mean()), 4),
                                             "n": int(d["n"].iloc[0]), "coverage": round(float(d["coverage"].mean()), 3)}
                             for (mdl, wt), d in U.groupby(["model", "weighting"])}
    RC = M[M["set"] == "rule_change"]
    R["rule_change_share"] = {f"{r.tau}|{r.model}|{r.weighting}|{r.cell}": [round(r.mean, 4), round(r.std, 4)]
                              for r in RC.groupby(["tau", "model", "weighting", "cell"])["share_changed"]
                              .agg(["mean", "std"]).reset_index().itertuples()}
    R["efferent_predictions_logreg"] = pd.Series(oof.get("efferent_pred", [])).value_counts().to_dict()
    R["em_priors_logreg_first_split"] = msets.get("logreg", [{}])[0]
    R["bootstrap_drop_logreg_seed0"] = bootstrap_drop(f, m, oof, "logreg", n_boot=200 if quick else 1000)
    R["eda_counts"] = eda_counts(f, m)
    R["hparams_counts"] = pd.concat(allH).groupby(["tau", "model"])["hyperparameter"] \
        .agg(lambda s: s.astype(str).value_counts().to_dict()).to_dict() if len(allH) else {}
    R["hparams_counts"] = {f"{k[0]}|{k[1]}": v for k, v in R["hparams_counts"].items()}
    R["cells"] = cell_table(M)
    R["runtime_min"] = round((time.time() - t0) / 60, 1)
    (REP / "week9_results.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    figures(M, R, f, m, log)
    (REP / "week9_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"done in {R['runtime_min']} min -> reports/week9_results.json")
    return R


def cell_table(M, sets=("within", "within_shared", "cross", "unmatched"), tau=10):
    """Type- and neuron-weighted cost, accuracy and n per set x partition cell (mean, SD over splits)."""
    d = M[(M["tau"] == tau) & M["set"].isin(sets)]
    g = d.groupby(["model", "set", "weighting", "partition", "cell"])
    t = g[["cost_flow125", "accuracy"]].agg(["mean", "std"])
    t["n"] = g["n"].mean()
    return {"|".join(map(str, k)): {"cost": [round(r[("cost_flow125", "mean")], 4), round(r[("cost_flow125", "std")], 4)],
                                    "accuracy": [round(r[("accuracy", "mean")], 4), round(r[("accuracy", "std")], 4)],
                                    "n": round(float(r["n"].iloc[0] if hasattr(r["n"], "iloc") else r["n"]), 1)}
            for k, r in t.iterrows()}


def figures_only():
    """Redraw figures/week9 from the saved metrics, without refitting any model."""
    M = pd.read_csv(REP / "week9_metrics.csv")
    R = json.loads((REP / "week9_results.json").read_text())
    f, m, *_ = load_tables(10)
    R["cells"] = cell_table(M)
    (REP / "week9_results.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    figures(M, R, f, m, print)


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        figures_only()
    else:
        main(quick="--quick" in sys.argv)
