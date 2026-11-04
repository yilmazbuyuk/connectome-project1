"""Önbelleği Data Conventions + Week 4 Shared Data Corrections (eda_shared v0.2.0) ile kurar.

Çalıştır (repo kökünden):
  python -m src.build fafb         dişi beyin
  python -m src.build malecns      erkek beyin (neuPrint token + syn-partners dosyası gerekir)
  python -m src.build step0        iki önbellekten Step 0 raporu: reports/step0.md
  --force                          önbelleği yeniden üret
Günlük: reports/build_log_<hedef>.txt
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import config
from src.cache import cached
from src import validate as V

LOG = []
REP = Path("reports")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def _save_json(name, obj):
    REP.mkdir(exist_ok=True)
    (REP / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=int), encoding="utf-8")


# ------------------------------------------------------------------ FAFB
FAFB_REL = str(config.FAFB_MATERIALIZATION)
FAFB_ANN = "ann" + config.FAFB_ANNOTATIONS_TAG.lstrip("v")
FAFB_SRC = config.FAFB_SYNAPSE_TABLE


def fafb_paths():
    return {
        "ann_raw": config.CACHE_DIR / f"fafb_annotations_raw_{FAFB_REL}_{FAFB_ANN}.parquet",
        "edgeroi": config.cache_path("fafb", "edgeroi", FAFB_REL, FAFB_SRC, min_syn=1),
        "edges": config.cache_path("fafb", "edges", FAFB_REL, FAFB_SRC, min_syn=1),
        # v0.2.0: yeni ad, eski (139,262 satırlık) tablo yanlışlıkla okunmasın
        "neurons": config.cache_path("fafb", "neurons", FAFB_REL, FAFB_ANN + "-proofread"),
    }


def build_fafb(force=False):
    from src import fafb
    p = fafb_paths()
    ann_raw = cached(p["ann_raw"], fafb.fetch_annotations_raw, log=log)
    roi = cached(p["edgeroi"], lambda: fafb.build_edgeroi(log=log), V.validate_edgeroi, force, log=log)
    e = cached(p["edges"], lambda: fafb.build_edges(roi, log=log), V.validate_edges, force, log=log)
    rep = {}

    def _neurons():
        ids = np.load(fafb.PROOFREAD_FILE)
        return fafb.build_neurons(ann_raw, ids, fafb.read_counts(), log=log, report=rep)

    n = cached(p["neurons"], _neurons, V.validate_neurons, force, log=log)
    if rep:
        _save_json("fafb_neuron_report.json", rep)
    V.validate_edges(e, n)
    V.validate_edgeroi(roi, n)
    log("Doğrulama: fafb neurons/edges/edgeroi tüm kontroller geçti (References dahil)")
    return n, e, roi


# ------------------------------------------------------------------ MaleCNS
M_REL = "v1.0"


def malecns_paths():
    c = config.CACHE_DIR
    conf = config.MALECNS_SYN_CONFIDENCE
    return {
        "neurons_raw": c / f"malecns_neuprint_neurons_raw_{M_REL}_traced.parquet",
        "roi_raw": c / f"malecns_neuprint_roicounts_raw_{M_REL}_traced.parquet",
        "neurons_cns": config.cache_path("malecns", "neurons", M_REL, "traced-cns"),
        "edgeroi_cns": config.cache_path("malecns", "edgeroi", M_REL, "traced-cns", min_syn=1, conf=conf),
        "neurons": config.cache_path("malecns", "neurons", M_REL, "brain"),
        "edges": config.cache_path("malecns", "edges", M_REL, "brain", min_syn=1, conf=conf),
        "edgeroi": config.cache_path("malecns", "edgeroi", M_REL, "brain", min_syn=1, conf=conf),
    }


def build_malecns(force=False):
    from src import malecns as M
    p = malecns_paths()
    if p["neurons_raw"].exists() and p["roi_raw"].exists() and not force:
        raw, roi = pd.read_parquet(p["neurons_raw"]), pd.read_parquet(p["roi_raw"])
        log(f"[cache] neuPrint ham tabloları: {len(raw):,} nöron, {len(roi):,} ROI satırı")
    else:
        raw, roi = M.fetch_neuprint(log=log)
        p["neurons_raw"].parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(p["neurons_raw"], index=False)
        roi.to_parquet(p["roi_raw"], index=False)

    n_cns = cached(p["neurons_cns"], lambda: M.build_neurons(raw, roi, log=log),
                   V.validate_neurons, force, log=log)

    if p["edgeroi_cns"].exists() and not force:
        er = pd.read_parquet(p["edgeroi_cns"])
        er["neuropil"] = er["neuropil"].astype("category")
        log(f"[cache] {p['edgeroi_cns'].name}: {len(er):,} satır")
        self_pairs = json.loads((REP / "malecns_edgeroi_self.json").read_text())["self_pairs"] \
            if (REP / "malecns_edgeroi_self.json").exists() else None
    else:
        er, self_pairs = M.build_edgeroi(n_cns["neuron_id"], log=log)
        V.validate_edgeroi(er, n_cns)
        er.to_parquet(p["edgeroi_cns"], index=False)
        _save_json("malecns_edgeroi_self.json", {"self_pairs": self_pairs})
        log(f"[build] {p['edgeroi_cns'].name}: {len(er):,} satır")

    if all(p[k].exists() for k in ("neurons", "edges", "edgeroi")) and not force:
        n_b, e_b = pd.read_parquet(p["neurons"]), pd.read_parquet(p["edges"])
        er_b = pd.read_parquet(p["edgeroi"])
        log(f"[cache] brain part: {len(n_b):,} nöron, {len(e_b):,} kenar")
    else:
        n_b, e_b, er_b, rep = M.brain_part(n_cns, er, M.roi_counts(roi), log=log)
        rep["self_pairs_dropped"] = self_pairs
        V.validate_neurons(n_b)
        V.validate_edges(e_b, n_b)
        V.validate_edgeroi(er_b, n_b)
        for k in ("neurons", "edges", "edgeroi"):
            {"neurons": n_b, "edges": e_b, "edgeroi": er_b}[k].to_parquet(p[k], index=False)
        _save_json("malecns_brain_report.json", rep)
    del er
    V.validate_neurons(n_b)
    V.validate_edges(e_b, n_b)
    V.check_plausibility(n_b, e_b)
    log("Doğrulama: malecns brain neurons/edges tüm kontroller geçti (References, Plausibility dahil)")
    return n_b, e_b


# ------------------------------------------------------------------ Step 0
def step0():
    """Week 4 'Step 0, again' tablosu: reports/step0.md ve reports/step0.json."""
    from eda_shared.corrections import nt_correction_summary
    from eda_shared.eda_helpers import add_provisional_flags, MALECNS_NERVE_CORD
    out, md = {}, ["# Step 0 — corrected tables (eda_shared v0.2.0, config v0.2.0)", ""]

    fp = fafb_paths()
    n = pd.read_parquet(fp["neurons"])
    e = pd.read_parquet(fp["edges"], columns=["syn_count"])
    nrep = json.loads((REP / "fafb_neuron_report.json").read_text())
    sc = n["super_class"].value_counts()
    sc = {k: int(v) for k, v in sc.items() if v > 0}
    reg = n["region"].value_counts()
    reg.rename("neurons").rename_axis("region").to_csv(REP / "step0_fafb_region_counts.csv")
    fl = add_provisional_flags(n)
    out["fafb"] = {
        "neurons": len(n), **{f"set_{k}": v for k, v in nrep["neuron_set"].items()},
        "super_class": sc,
        "edges_tau1": int(len(e)), "edges_ge5": int((e["syn_count"] >= 5).sum()),
        "region_ties": nrep["region_ties"], "region_unknown": nrep["region_unknown"],
        "self_connections_dropped": 0,
        "central_population": int(((~fl["is_optic"]) & (fl["neuropil"] != "unknown")).sum()),
        "dopamine_before": int((n["nt_type"] == "dopamine").sum()),
        "dopamine_after": int((n["nt_type_corrected"] == "dopamine").sum()),
    }
    nts = nt_correction_summary(n)
    md += ["## FAFB", "",
           f"- Neurons: **{len(n):,}** = {nrep['neuron_set']['annotation_rows'] - nrep['neuron_set']['dropped_not_in_release']:,} kept "
           f"+ {nrep['neuron_set']['added_unknown']} added; {nrep['neuron_set']['dropped_not_in_release']} dropped",
           "- super_class: " + "; ".join(f"{k} {v:,}" for k, v in sc.items()),
           f"- Edges: τ=1 {out['fafb']['edges_tau1']:,}; ≥5 **{out['fafb']['edges_ge5']:,}**",
           f"- Region: {len(reg)} values (reports/step0_fafb_region_counts.csv); ties {nrep['region_ties']:,}; "
           f"unknown {nrep['region_unknown']:,}",
           "- Self-connections dropped: 0",
           f"- Central population (is_optic False, known region): {out['fafb']['central_population']:,}",
           f"- Transmitter correction: " + "; ".join(f"{r.nt_correction_source} {r.rows:,} rows, {r.changed:,} changed"
                                                    for r in nts.itertuples()),
           f"- Dopamine: {out['fafb']['dopamine_before']:,} → {out['fafb']['dopamine_after']:,}", ""]

    mp = malecns_paths()
    if mp["neurons"].exists():
        m = pd.read_parquet(mp["neurons"])
        me = pd.read_parquet(mp["edges"], columns=["syn_count"])
        mrep = json.loads((REP / "malecns_brain_report.json").read_text())
        mf = add_provisional_flags(m, nerve_cord=MALECNS_NERVE_CORD)
        msc = {k: int(v) for k, v in m["super_class"].value_counts().items() if v > 0}
        out["malecns"] = {**mrep, "super_class": msc,
                          "central_population": int(((~mf["is_optic"]) & (mf["neuropil"] != "unknown")).sum())}
        mts = nt_correction_summary(m)
        md += ["## MaleCNS (brain part)", "",
               f"- Neurons kept: **{mrep['neurons_kept']:,}** of {mrep['neurons_in']:,} Traced; dropped "
               f"{mrep['neurons_dropped']:,}; ascending kept {mrep['ascending_kept']:,}; "
               f"sensory_ascending kept {mrep['sensory_ascending_kept']:,}",
               f"- Brain edges: τ=1 {mrep['edges_tau1']:,}; ≥5 **{mrep['edges_ge5']:,}**",
               f"- Synapses: brain {mrep['edgeroi_syn_brain']:,}; nerve cord {mrep['edgeroi_syn_nerve_cord']:,}; "
               f"no location {mrep['edgeroi_syn_no_location']:,}",
               f"- Region ties {mrep['region_ties']:,}; region changed by the brain rule {mrep['region_changed']:,}",
               f"- Self-connections dropped: {mrep.get('self_pairs_dropped')}",
               f"- Central population (is_optic False, known region): {out['malecns']['central_population']:,}",
               "- super_class: " + "; ".join(f"{k} {v:,}" for k, v in msc.items()),
               f"- Transmitter correction: " + "; ".join(f"{r.nt_correction_source} {r.rows:,} rows, {r.changed:,} changed"
                                                        for r in mts.itertuples()), ""]
    else:
        md += ["## MaleCNS", "", "- not built yet (python -m src.build malecns)", ""]
    _save_json("step0.json", out)
    (REP / "step0.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return out


if __name__ == "__main__":
    args = sys.argv[1:] or ["fafb"]
    force = "--force" in args
    which = [a for a in args if a != "--force"]
    if "fafb" in which:
        build_fafb(force)
    if "malecns" in which:
        build_malecns(force)
    if "step0" in which:
        step0()
    REP.mkdir(exist_ok=True)
    (REP / f"build_log_{'_'.join(which)}.txt").write_text("\n".join(LOG), encoding="utf-8")
