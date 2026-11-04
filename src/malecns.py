"""MaleCNS (erkek beyin + sinir kordonu), male-cns:v1.0, yalnızca Traced, sinaps güveni >= 0.5.

v0.2.0 (Week 4 Shared Data Corrections), README'deki sırayla:
  map_super_class -> add_nt_type_corrected -> malecns_brain_part (region_from_counts dahil)

Girdiler
  neuPrint  fetch_neurons(Traced): nöron tablosu + bölge (ROI) başına upstream/downstream,
            fetch_primary_rois(): birincil ROI listesi. Token ortam değişkeninden okunur:
            NEUPRINT_APPLICATION_CREDENTIALS (asla dosyaya / not defterine yazılmaz).
  Toplu     syn-partners-male-cns-v1.0-minconf-0.5.feather (Janelia, 6.8 GB): her sinaps
            çifti ve postsinapsın birincil nöropili (primary_post). Buradan nöropil başına
            edgeroi (1 sinaps, conf >= 0.5, Traced–Traced) kurulur.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import config
from src.common import recode, apply_schema
from eda_shared.corrections import (map_super_class, add_nt_type_corrected,
                                    malecns_neuropil_counts, malecns_brain_part)

DATASET, RELEASE = "malecns", "v1.0"
# Büyük dosya başka bir diske taşındıysa yolu ortam değişkeniyle verilir: MALECNS_SYN_PARTNERS
SYN_PARTNERS_FILE = Path(os.environ.get("MALECNS_SYN_PARTNERS",
                                        "data/raw/syn-partners-male-cns-v1.0-minconf-0.5.feather"))

SIDE_MAP = {"L": "left", "R": "right", "M": "center", "unknown": "unknown"}
NT_MAP = {v: v for v in ["acetylcholine", "glutamate", "gaba", "dopamine", "serotonin",
                         "octopamine", "histamine"]}
NT_MAP.update({"unclear": "unknown", "unknown": "unknown"})

NEURON_COLS = ["bodyId", "type", "class", "subclass", "superclass", "somaSide", "rootSide",
               "predictedNt", "predictedNtConfidence", "pre", "post", "flywireType", "mancType",
               "hemibrainType", "dimorphism", "status", "itoleeHl"]

_PRE_SCHEMA = {k: v for k, v in config.NEURON_SCHEMA.items()
               if k not in ("nt_type_corrected", "nt_correction_source")}


# ------------------------------------------------------------------ neuPrint
def _client():
    from neuprint import Client
    # token=None -> neuprint-python NEUPRINT_APPLICATION_CREDENTIALS ortam değişkenini okur
    return Client("neuprint.janelia.org", dataset=config.MALECNS_DATASET)


def fetch_neuprint(log=print):
    """(neurons_raw, roi_raw, primary_rois) — Traced nöronlar, ROI tablosu, birincil ROI listesi."""
    from neuprint import fetch_neurons, fetch_primary_rois, NeuronCriteria as NC
    c = _client()
    log("neuPrint: Traced nöronlar ve ROI sayımları çekiliyor (birkaç dakika sürebilir)...")
    neurons, roi = fetch_neurons(NC(status=config.MALECNS_STATUS), client=c)
    primary = list(fetch_primary_rois(client=c))
    keep = [col for col in NEURON_COLS if col in neurons.columns]
    neurons = neurons[keep].copy()
    roi = roi[["bodyId", "roi", "pre", "post", "upstream", "downstream"]].copy()
    roi["primary"] = roi["roi"].isin(primary)
    log(f"neuPrint: {len(neurons):,} nöron, {len(roi):,} ROI satırı, {len(primary)} birincil ROI")
    return neurons, roi


# ------------------------------------------------------------------ edgeroi
N_BUCKETS = int(os.environ.get("EDGEROI_BUCKETS", "64"))
BATCH_ROWS = int(os.environ.get("EDGEROI_BATCH_ROWS", "2000000"))


def _partial_counts(batch, ids, conf, n_buckets):
    """Bir veri parçası: güven + Traced süzgeci, parça içinde (pre, post, nöropil) sayımı, kova no."""
    import pyarrow as pa
    import pyarrow.compute as pc
    keep = pc.and_(pc.and_(pc.greater_equal(batch["conf_pre"], conf),
                           pc.greater_equal(batch["conf_post"], conf)),
                   pc.and_(pc.is_in(batch["body_pre"], value_set=ids),
                           pc.is_in(batch["body_post"], value_set=ids)))
    b = batch.filter(keep)
    if b.num_rows == 0:
        return None
    npil = pc.fill_null(pc.cast(b["primary_post"], pa.string()), "<unspecified>")
    t = pa.table({"pre_id": b["body_pre"], "post_id": b["body_post"], "neuropil": npil})
    g = t.group_by(["pre_id", "post_id", "neuropil"]).aggregate([("pre_id", "count")])
    g = g.select(["pre_id", "post_id", "neuropil", "pre_id_count"]).rename_columns(
        ["pre_id", "post_id", "neuropil", "n"])                      # sütun sırası sürüme bağlı
    bucket = (g["pre_id"].to_numpy() % n_buckets).astype(np.int16)
    return g.append_column("b", pa.array(bucket))


def build_edgeroi(traced_ids, path=SYN_PARTNERS_FILE, log=print, n_buckets=N_BUCKETS):
    """syn-partners dosyasından nöropil başına edgeroi (1 sinaps, conf >= 0.5, Traced–Traced).

    Bellek sınırlı iki geçiş (6.8 GB dosya hiçbir zaman bütün olarak belleğe girmez):
      1) dosya parça parça okunur; her parça süzülür, kendi içinde sayılır ve pre_id'ye göre
         n_buckets kovaya bölünüp diske (DUCKDB_TMP, yoksa data/tmp) yazılır;
      2) her kova ayrı ayrı toplanır (aynı pre_id hep aynı kovada, toplam doğru) ve birleştirilir.
    """
    import shutil
    import pyarrow as pa
    import pyarrow.dataset as ds
    conf = config.MALECNS_SYN_CONFIDENCE
    work = Path(os.environ.get("DUCKDB_TMP") or "data/tmp") / "edgeroi_buckets"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    log(f"edgeroi: geçici kova klasörü {work}")
    ids = pa.array(np.unique(np.asarray(traced_ids, dtype=np.int64)))
    syn = ds.dataset(str(path), format="feather")
    cols = ["body_pre", "body_post", "conf_pre", "conf_post", "primary_post"]
    total = syn.count_rows()
    log(f"edgeroi: 1. geçiş — {total:,} sinaps satırı parça parça okunuyor (10-40 dakika)...")
    schema = pa.schema([("pre_id", pa.int64()), ("post_id", pa.int64()), ("neuropil", pa.string()),
                        ("n", pa.int64()), ("b", pa.int16())])
    state = {"rows": 0, "step": 0}

    def batches():
        for batch in syn.to_batches(columns=cols, batch_size=BATCH_ROWS):
            state["rows"] += batch.num_rows
            if total and state["rows"] * 20 // total > state["step"]:
                state["step"] = state["rows"] * 20 // total
                log(f"  %{state['step'] * 5:3d}  ({state['rows']:,} / {total:,})")
            g = _partial_counts(batch, ids, conf, n_buckets)
            if g is not None:
                yield from g.cast(schema).to_batches()

    ds.write_dataset(batches(), work, schema=schema, format="parquet",
                     partitioning=ds.partitioning(pa.schema([("b", pa.int16())]), flavor="hive"),
                     existing_data_behavior="overwrite_or_ignore",
                     max_rows_per_group=1_000_000)
    log(f"edgeroi: 2. geçiş — {n_buckets} kova toplanıyor...")
    parts = []
    for d in sorted(work.glob("b=*")):
        t = ds.dataset(d, format="parquet").to_table(columns=["pre_id", "post_id", "neuropil", "n"])
        t = t.group_by(["pre_id", "post_id", "neuropil"]).aggregate([("n", "sum")])
        t = t.select(["pre_id", "post_id", "neuropil", "n_sum"]).rename_columns(
            ["pre_id", "post_id", "neuropil", "syn_count"])
        parts.append(t.set_column(2, "neuropil", t["neuropil"].dictionary_encode()))
    tbl = pa.concat_tables(parts).unify_dictionaries()
    del parts
    shutil.rmtree(work, ignore_errors=True)
    r = tbl.to_pandas()
    del tbl
    r["syn_count"] = r["syn_count"].astype("int32")
    r["pre_id"] = r["pre_id"].astype("int64")
    r["post_id"] = r["post_id"].astype("int64")
    self_mask = r["pre_id"] == r["post_id"]
    n_self_pairs = r.loc[self_mask, ["pre_id"]].drop_duplicates().shape[0]
    log(f"edgeroi: {len(r):,} satır; öz-bağlantı satırı {int(self_mask.sum()):,} "
        f"({n_self_pairs:,} nöron kendine bağlı) atıldı")
    r = r[~self_mask].reset_index(drop=True)
    r["dataset"], r["release"], r["min_syn_fetched"] = DATASET, RELEASE, config.FETCH_MIN_SYN
    out = pd.DataFrame({c: r[c] for c in config.EDGEROI_SCHEMA})
    for col, (dtype, _) in config.EDGEROI_SCHEMA.items():
        if col != "neuropil":                 # neuropil kategorik kalır (eda_shared README: bellek)
            out[col] = out[col].astype(dtype)
    return out, n_self_pairs


# ------------------------------------------------------------------ neurons
def build_neurons(raw, roi, log=print):
    """Ortak şemadaki MaleCNS nöron tablosu (henüz kırpılmamış, region = CNS genelinde en büyük ROI)."""
    r = raw.copy()
    sc = map_super_class(r["superclass"], "malecns")                     # correction 2
    side_raw = r["somaSide"].where(r["somaSide"].notna(), r["rootSide"])
    p = roi[roi["primary"]]
    tot = p.assign(c=p["upstream"].fillna(0) + p["downstream"].fillna(0))
    top = (tot.sort_values(["bodyId", "c", "roi"], ascending=[True, False, True])
              .drop_duplicates("bodyId").set_index("bodyId")["roi"])
    n = pd.DataFrame({
        "neuron_id": r["bodyId"].astype("int64"),
        "dataset": DATASET,
        "release": RELEASE,
        "cell_type": r["type"],
        "cell_class": r["class"],
        "cell_sub_class": r["subclass"],
        "super_class": sc["super_class"].array,
        "side": recode(side_raw, SIDE_MAP, "side"),
        "region": r["bodyId"].map(top).fillna("unknown"),               # brain_part yeniden hesaplar
        "nt_type": recode(r["predictedNt"], NT_MAP, "nt_type"),
        "nt_confidence": r["predictedNtConfidence"],
        "n_pre": r["pre"],
        "n_post": r["post"],
        "flywire_type": r["flywireType"],
        "manc_type": r["mancType"],
        "hemibrain_type": r["hemibrainType"],
        "dimorphism": r["dimorphism"],
        "status": r["status"],
        "hemilineage": r["itoleeHl"],
        "nt_known": pd.NA,
        "super_class_source": sc["super_class_source"].array,
    })
    n.loc[n["nt_type"] == "unknown", "nt_confidence"] = np.nan
    n = apply_schema(n, _PRE_SCHEMA)
    n = add_nt_type_corrected(n)                                         # correction 1
    n = apply_schema(n, config.NEURON_SCHEMA)
    log(f"neurons (tüm CNS, Traced): {len(n):,}; super_class: "
        f"{n.super_class.value_counts().to_dict()}")
    return n


def roi_counts(roi):
    """Bölge kuralı girdisi: yalnızca birincil ROI'ler, upstream + downstream (README)."""
    return malecns_neuropil_counts(roi[roi["primary"]])


def brain_part(neurons, edgeroi, counts, log=print):
    """correction 4: sinaps konumuna göre beyin kısmı. (neurons_b, edges_b, edgeroi_b, report)."""
    n_b, e_b, er_b, rep = malecns_brain_part(neurons, edgeroi, counts)
    n_b = apply_schema(n_b, config.NEURON_SCHEMA)
    e_b = apply_schema(e_b, config.EDGE_SCHEMA)
    log("brain part: " + ", ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}"
                                   for k, v in rep.items()))
    return n_b, e_b, er_b, rep
