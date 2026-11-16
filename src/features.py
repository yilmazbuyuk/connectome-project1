"""Week 9 feature table (lock note v2, item Features). Connectivity only, one row per neuron.

  n_in_all, n_out_all   all-synapse totals, same scope in both datasets, log1p:
                        FAFB    per_neuron_neuropil_count_post / _pre_783 (every partner)
                        MaleCNS neuPrint upstream / downstream (connections) over the
                                brain-part primary ROIs, divided by SCALE (1.484, B6)
  in_deg, out_deg       partners over edges with syn_count >= tau, log1p
  has_input, has_output 1 if in_deg / out_deg > 0
  fin_<np>, fout_<np>   share of the neuron's input / output synapses in neuropil <np>,
                        over edges whose summed syn_count >= tau (D8: sum, then threshold).
                        Neuropils = shared table base names, L and R merged
                        (eda_shared canonical_neuropil), plus one 'unassigned' column for
                        UNASGD (FAFB) and <unspecified> (MaleCNS). Zero input -> all 0.

Thresholds (lock note, Cross-dataset): MaleCNS tau = round(1.484 x FAFB tau).
Run:  python -m src.features      builds and caches tau 10, 5, 15 for both datasets.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import config
from eda_shared.eda_helpers import (load_neuropil_table, canonical_neuropil,
                                    MALECNS_NERVE_CORD)

SCALE = {"fafb": 1.0, "malecns": 1.484}
FAFB_TAUS = (10, 5, 15)


def malecns_tau(tau):
    return int(round(SCALE["malecns"] * tau))


def neuropil_columns():
    t = load_neuropil_table()
    return list(t.index) + ["unassigned"]


def _base_names(values, dataset, table):
    """Map raw neuropil labels to base names; 'unknown' -> 'unassigned'."""
    u = pd.Series(pd.unique(pd.Series(values).astype("string").dropna()), dtype="string")
    nc = MALECNS_NERVE_CORD if dataset == "malecns" else frozenset()
    base = canonical_neuropil(u, table, nc).replace({"unknown": "unassigned"})
    return dict(zip(u, base))


def _fractions(er, key, ids, cols):
    """Share of syn per (key neuron, base neuropil); rows = ids, columns = cols."""
    g = er.groupby([key, "base"], observed=True, sort=False)["syn_count"].sum()
    tot = g.groupby(level=0).transform("sum")
    frac = (g / tot).astype("float32").unstack("base", fill_value=0.0)
    return frac.reindex(index=ids, columns=cols, fill_value=0.0).fillna(0.0)


def totals(dataset, ids):
    """(n_in_all, n_out_all) raw totals, before scaling, indexed by neuron id."""
    if dataset == "fafb":
        from src import fafb
        out = {}
        for name, path in (("in", fafb.COUNT_POST_FILE), ("out", fafb.COUNT_PRE_FILE)):
            df = pd.read_feather(path)
            idc = next(c for c in ("post_pt_root_id", "pre_pt_root_id", "root_id", "neuron_id")
                       if c in df.columns)
            cc = next(c for c in ("count", "syn_count") if c in df.columns)
            out[name] = df.groupby(idc)[cc].sum()
        return (out["in"].reindex(ids, fill_value=0), out["out"].reindex(ids, fill_value=0))
    from src.build import malecns_paths
    roi = pd.read_parquet(malecns_paths()["roi_raw"])
    roi = roi[roi["primary"]]
    table = load_neuropil_table()
    m = _base_names(roi["roi"], "malecns", table)
    in_brain = roi["roi"].astype("string").map(m).ne("nerve_cord")
    r = roi[in_brain.fillna(False).to_numpy()]
    up = r.groupby("bodyId")["upstream"].sum()
    down = r.groupby("bodyId")["downstream"].sum()
    return up.reindex(ids, fill_value=0), down.reindex(ids, fill_value=0)


def build(dataset, tau_fafb, log=print):
    """Feature table for one dataset at the locked threshold (FAFB tau; MaleCNS scaled)."""
    from src.build import fafb_paths, malecns_paths
    p = fafb_paths() if dataset == "fafb" else malecns_paths()
    tau = tau_fafb if dataset == "fafb" else malecns_tau(tau_fafb)
    n = pd.read_parquet(p["neurons"], columns=["neuron_id"])
    ids = pd.Index(n["neuron_id"].astype("int64"), name="neuron_id")
    e = pd.read_parquet(p["edges"], columns=["pre_id", "post_id", "syn_count"])
    e = e[e["syn_count"] >= tau]
    er = pd.read_parquet(p["edgeroi"], columns=["pre_id", "post_id", "neuropil", "syn_count"])
    er = er.merge(e[["pre_id", "post_id"]], on=["pre_id", "post_id"], how="inner")
    table = load_neuropil_table()
    m = _base_names(er["neuropil"], dataset, table)
    er["base"] = er["neuropil"].astype("string").map(m).fillna("unassigned")
    er = er[er["base"] != "nerve_cord"]
    cols = neuropil_columns()
    fin = _fractions(er, "post_id", ids, cols).add_prefix("fin_")
    fout = _fractions(er, "pre_id", ids, cols).add_prefix("fout_")
    indeg = e.groupby("post_id").size().reindex(ids, fill_value=0)
    outdeg = e.groupby("pre_id").size().reindex(ids, fill_value=0)
    tin, tout = totals(dataset, ids)
    s = SCALE[dataset]
    f = pd.DataFrame({
        "n_in_all": np.log1p(tin.to_numpy() / s),
        "n_out_all": np.log1p(tout.to_numpy() / s),
        "in_deg": np.log1p(indeg.to_numpy()),
        "out_deg": np.log1p(outdeg.to_numpy()),
        "has_input": (indeg.to_numpy() > 0).astype("float32"),
        "has_output": (outdeg.to_numpy() > 0).astype("float32"),
    }, index=ids).astype("float32")
    f = pd.concat([f, fin, fout], axis=1)
    f.columns = [str(c) for c in f.columns]
    log(f"features {dataset} tau={tau}: {f.shape[0]:,} neurons x {f.shape[1]} columns; "
        f"no input {int((indeg == 0).sum()):,}, no output {int((outdeg == 0).sum()):,}")
    return f.reset_index()


def feature_path(dataset, tau_fafb):
    tau = tau_fafb if dataset == "fafb" else malecns_tau(tau_fafb)
    return config.CACHE_DIR / f"{dataset}_features_week9_tau{tau}.parquet"


def load(dataset, tau_fafb, force=False, log=print):
    path = feature_path(dataset, tau_fafb)
    if path.exists() and not force:
        return pd.read_parquet(path)
    f = build(dataset, tau_fafb, log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(path, index=False)
    return f


if __name__ == "__main__":
    force = "--force" in sys.argv
    for t in FAFB_TAUS:
        for d in ("fafb", "malecns"):
            load(d, t, force=force)
