"""FAFB (dişi beyin): 783 + flywire_annotations v3.2.0 + Codex Buhmann bağlantıları.

v0.2.0 (Week 4 Shared Data Corrections), README'deki sırayla:
  map_super_class -> align_to_proofread -> region_from_counts -> add_nt_type_corrected
"""
from pathlib import Path
import numpy as np
import pandas as pd
import config
from src.common import recode, apply_schema
from eda_shared.corrections import (map_super_class, align_to_proofread, fafb_neuropil_counts,
                                    region_from_counts, add_nt_type_corrected)

DATASET, RELEASE = "fafb", str(config.FAFB_MATERIALIZATION)
ANN_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
           f"{config.FAFB_ANNOTATIONS_TAG}/supplemental_files/Supplemental_file1_neuron_annotations.tsv")
BUHMANN_FILE = Path("data/connections_buhmann_no_threshold.csv.gz")
PRINCETON_FILE = Path("data/connections_princeton_no_threshold.csv.gz")  # yalnızca duyarlılık
PROOFREAD_FILE = Path("data/raw/proofread_root_ids_783.npy")
COUNT_PRE_FILE = Path("data/raw/per_neuron_neuropil_count_pre_783.feather")
COUNT_POST_FILE = Path("data/raw/per_neuron_neuropil_count_post_783.feather")

SIDE_MAP = {"left": "left", "right": "right", "center": "center", "na": "unknown"}
NT_MAP = {v: v for v in ["acetylcholine", "glutamate", "gaba", "dopamine", "serotonin", "octopamine"]}

# Nöron setini kurarken kullanılan şema: nt düzeltme kolonları en sonda eklenir,
# çünkü align_to_proofread 'unknown' seviyesi olmayan kolonları kabul etmez (README).
_PRE_SCHEMA = {k: v for k, v in config.NEURON_SCHEMA.items()
               if k not in ("nt_type_corrected", "nt_correction_source")}


def fetch_annotations_raw():
    return pd.read_csv(ANN_URL, sep="\t", low_memory=False, dtype={"root_id": "int64"})


def build_edgeroi(path=BUHMANN_FILE, log=print):
    raw = pd.read_csv(path, usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"],
                      dtype={"pre_root_id": "int64", "post_root_id": "int64",
                             "syn_count": "int32", "neuropil": "category"})
    r = raw.rename(columns={"pre_root_id": "pre_id", "post_root_id": "post_id"})
    r = (r.groupby(["pre_id", "post_id", "neuropil"], observed=True, sort=False)["syn_count"]
          .sum().reset_index())
    self_mask = r["pre_id"] == r["post_id"]
    log(f"edgeroi: {len(raw):,} satır okundu; öz-bağlantı satırı atıldı: {int(self_mask.sum()):,}")
    r = r[~self_mask]
    r["dataset"], r["release"], r["min_syn_fetched"] = DATASET, RELEASE, config.FETCH_MIN_SYN
    return apply_schema(r, config.EDGEROI_SCHEMA)


def build_edges(edgeroi, log=print):
    """D8: önce nöropiller üzerinden topla, sonra eşik (eşik bellekte uygulanır)."""
    e = edgeroi.groupby(["pre_id", "post_id"], sort=False)["syn_count"].sum().reset_index()
    log(f"edges: {len(edgeroi):,} nöropil satırı -> {len(e):,} çift; öz-bağlantı: 0 (edgeroi'de atıldı)")
    e["dataset"], e["release"], e["min_syn_fetched"] = DATASET, RELEASE, config.FETCH_MIN_SYN
    return apply_schema(e, config.EDGE_SCHEMA)


def read_counts():
    pre = pd.read_feather(COUNT_PRE_FILE)
    post = pd.read_feather(COUNT_POST_FILE)
    return fafb_neuropil_counts(pre, post)


def build_neurons(ann, proofread_ids, counts, log=print, report=None):
    """report: dict verilirse Step 0 sayıları içine yazılır."""
    a = ann
    sc = map_super_class(a["super_class"], "fafb")                       # correction 2
    n = pd.DataFrame({
        "neuron_id": a["root_id"].astype("int64"),
        "dataset": DATASET,
        "release": RELEASE,
        "cell_type": a["cell_type"],
        "cell_class": a["cell_class"],
        "cell_sub_class": a["cell_sub_class"],
        "super_class": sc["super_class"].array,
        "side": recode(a["side"], SIDE_MAP, "side"),
        "region": "unknown",                                              # aşağıda hesaplanıyor
        "nt_type": recode(a["top_nt"], NT_MAP, "nt_type"),
        "nt_confidence": a["top_nt_conf"],
        "n_pre": pd.NA,
        "n_post": pd.NA,
        "flywire_type": pd.NA,
        "manc_type": pd.NA,
        "hemibrain_type": a["hemibrain_type"],
        "dimorphism": a["dimorphism"],
        "status": a["status"],
        "hemilineage": a["ito_lee_hemilineage"],
        "nt_known": a["known_nt"],
        "super_class_source": sc["super_class_source"].array,
    })
    n.loc[n["nt_type"] == "unknown", "nt_confidence"] = np.nan
    n = apply_schema(n, _PRE_SCHEMA)

    n, rep = align_to_proofread(n, proofread_ids)                        # correction 3
    log(f"neurons: etiket satırı {rep['annotation_rows']:,}; release dışı atılan "
        f"{rep['dropped_not_in_release']:,}; etiketsiz eklenen {rep['added_unknown']:,}; "
        f"toplam {rep['total']:,}")

    reg = region_from_counts(n["neuron_id"], counts)                     # correction 6
    reg = reg.set_index("neuron_id")
    n["region"] = n["neuron_id"].map(reg["region"]).astype("string")
    ties, unknown = int(reg["tied"].sum()), int((reg["region"] == "unknown").sum())
    log(f"region (FlyWire sayım dosyaları, tüm sinapslar): eşitlik {ties:,}; unknown {unknown:,}")

    n = add_nt_type_corrected(n)                                         # correction 1
    n = apply_schema(n, config.NEURON_SCHEMA)
    if report is not None:
        report.update({"neuron_set": rep, "region_ties": ties, "region_unknown": unknown})
    return n
