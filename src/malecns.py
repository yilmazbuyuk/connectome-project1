"""MaleCNS (erkek beyin + sinir kordonu), male-cns:v1.0, yalnızca Traced.

GEÇİCİ (interim) sürüm: 2. haftada neuPrint'ten çekilen ham nöron tablosundan kurulur.
roiInfo bu dosyada yok, bu yüzden region = "unknown". Toplu indirme (bulk) gelince
region roiInfo'dan hesaplanacak ve dosya adındaki "-interim" kalkacak.
"""
from pathlib import Path
import pandas as pd
import config
from src.common import recode, apply_schema

DATASET, RELEASE = "malecns", "v1.0"
RAW_FILE = Path("data/raw/malecns_neurons_raw_male-cns_v1.0.parquet")  # 2. hafta neuPrint dışa aktarımı

SIDE_MAP = {"L": "left", "R": "right", "M": "center", "unknown": "unknown"}
NT_MAP = {v: v for v in ["acetylcholine", "glutamate", "gaba", "dopamine", "serotonin",
                         "octopamine", "histamine"]}
NT_MAP.update({"unclear": "unknown", "unknown": "unknown"})
SUPER_CLASS_MAP = {
    "ol_intrinsic": "optic", "cb_intrinsic": "central",
    "visual_projection": "visual_projection", "visual_centrifugal": "visual_centrifugal",
    "ascending_neuron": "ascending", "descending_neuron": "descending",
    "ol_sensory": "sensory", "cb_sensory": "sensory", "vnc_sensory": "sensory",
    "vnc_motor": "motor", "cb_motor": "motor",
    "endocrine": "endocrine", "cb_endocrine": "endocrine", "vnc_endocrine": "endocrine",
    # GEÇİCİ eşlemeler: MaleCNS'e özgü seviyeler config'te yok (Data Conventions: "plus MaleCNS-only
    # levels"). Gruba bildirildi; karar gelene kadar en yakın FAFB seviyesine eşleniyor.
    "sensory_ascending": "ascending", "sensory_ascending_tbc": "ascending",   # FAFB'deki gibi
    "efferent_ascending": "ascending",
    "sensory_descending": "descending", "efferent_descending": "descending",
    "vnc_efferent": "motor", "cb_efferent": "motor",
    "cb_sensory_tbc": "sensory", "vnc_sensory_tbc": "sensory",
    "visual_projection_tbc": "visual_projection",
    # Sinir kordonu ara nöronları ve sınıflanamayanlar: beyin karşılaştırmasının dışında / bilinmiyor
    "vnc_intrinsic": "unknown", "vnc_tbc": "unknown", "ENS": "unknown",
}


def read_raw(path=RAW_FILE):
    return pd.read_parquet(path)


def build_neurons_interim(raw, log=print):
    r = raw[raw["status"] == config.MALECNS_STATUS].copy()
    log(f"malecns: ham tablo {len(raw):,}; Traced {len(r):,}")
    side_raw = r["somaSide"].where(r["somaSide"].notna(), r["rootSide"])
    log("malecns superclass (ham -> sayılar): " + str(r["superclass"].value_counts(dropna=False).to_dict()))
    n = pd.DataFrame({
        "neuron_id": r["bodyId"].astype("int64"),
        "dataset": DATASET,
        "release": RELEASE,
        "cell_type": r["type"],
        "cell_class": r["class"],
        "cell_sub_class": r["subclass"],
        "super_class": recode(r["superclass"], SUPER_CLASS_MAP, "super_class"),
        "side": recode(side_raw, SIDE_MAP, "side"),
        "region": "unknown",                      # GEÇİCİ: roiInfo toplu indirmeyle gelecek
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
    })
    n.loc[n["nt_type"] == "unknown", "nt_confidence"] = pd.NA
    n = apply_schema(n, config.NEURON_SCHEMA)
    log(f"malecns neurons (interim): {len(n):,} satır; side: {n.side.value_counts().to_dict()}; "
        f"super_class unknown: {int((n.super_class == 'unknown').sum()):,}")
    return n
