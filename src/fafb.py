"""FAFB (dişi beyin): 783 + flywire_annotations v3.2.0 + Codex Buhmann bağlantıları."""
from pathlib import Path
import pandas as pd
import config
from src.common import recode, apply_schema, top_region

DATASET, RELEASE = "fafb", str(config.FAFB_MATERIALIZATION)
ANN_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
           f"{config.FAFB_ANNOTATIONS_TAG}/supplemental_files/Supplemental_file1_neuron_annotations.tsv")
BUHMANN_FILE = Path("data/connections_buhmann_no_threshold.csv.gz")
PRINCETON_FILE = Path("data/connections_princeton_no_threshold.csv.gz")  # yalnızca duyarlılık

SIDE_MAP = {"left": "left", "right": "right", "center": "center", "na": "unknown"}
NT_MAP = {v: v for v in ["acetylcholine", "glutamate", "gaba", "dopamine", "serotonin", "octopamine"]}
SUPER_CLASS_MAP = {v: v for v in ["sensory", "ascending", "descending", "motor", "endocrine",
                                  "visual_projection", "visual_centrifugal", "optic", "central"]}
# GEÇİCİ: config.SUPER_CLASS_VALUES içinde 'sensory_ascending' yok, ama FAFB'de 612 nöron
# (MaleCNS'te de var). Gruba bildirildi; karar gelene kadar 'ascending' sayılıyor.
SUPER_CLASS_MAP["sensory_ascending"] = "ascending"


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


def build_neurons(ann, edgeroi, log=print):
    a = ann.copy()
    # Kenarlarda olup etiket tablosunda olmayan kimlikler: unknown etiketlerle eklenir
    ids = pd.Index(pd.unique(pd.concat([edgeroi["pre_id"], edgeroi["post_id"]])))
    extra = ids.difference(pd.Index(a["root_id"]))
    log(f"neurons: etiket tablosu {len(a):,}; kenarlarda olup tabloda olmayan kimlik: {len(extra):,}")
    if len(extra):
        a = pd.concat([a, pd.DataFrame({"root_id": extra.astype("int64")})], ignore_index=True)
    region = top_region(edgeroi, log=log)
    n = pd.DataFrame({
        "neuron_id": a["root_id"].astype("int64"),
        "dataset": DATASET,
        "release": RELEASE,
        "cell_type": a["cell_type"],
        "cell_class": a["cell_class"],
        "cell_sub_class": a["cell_sub_class"],
        "super_class": recode(a["super_class"], SUPER_CLASS_MAP, "super_class"),
        "side": recode(a["side"], SIDE_MAP, "side"),
        "region": a["root_id"].map(region).fillna("unknown"),
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
    })
    n.loc[n["nt_type"] == "unknown", "nt_confidence"] = pd.NA
    n = apply_schema(n, config.NEURON_SCHEMA)
    log(f"neurons: {len(n):,} satır; region=unknown: {int((n.region == 'unknown').sum()):,}")
    return n
