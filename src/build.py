"""Önbelleği Data Conventions'a göre kurar.   Çalıştır:  python -m src.build fafb
Çıktı ekrana ve reports/build_log.txt dosyasına yazılır."""
import sys
from pathlib import Path
import pandas as pd
import config
from src.cache import cached
from src import validate as V

LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def build_fafb(force=False):
    from src import fafb
    rel, ann = str(config.FAFB_MATERIALIZATION), "ann" + config.FAFB_ANNOTATIONS_TAG.lstrip("v")
    src = config.FAFB_SYNAPSE_TABLE
    ann_raw = cached(config.CACHE_DIR / f"fafb_annotations_raw_{rel}_{ann}.parquet",
                     fafb.fetch_annotations_raw, log=log)
    roi = cached(config.cache_path("fafb", "edgeroi", rel, src, min_syn=1),
                 lambda: fafb.build_edgeroi(log=log), V.validate_edgeroi, force, log=log)
    e = cached(config.cache_path("fafb", "edges", rel, src, min_syn=1),
               lambda: fafb.build_edges(roi, log=log), V.validate_edges, force, log=log)
    n = cached(config.cache_path("fafb", "neurons", rel, ann),
               lambda: fafb.build_neurons(ann_raw, roi, log=log), V.validate_neurons, force, log=log)
    V.validate_edges(e, n); V.validate_edgeroi(roi, n)
    log("Doğrulama: fafb neurons/edges/edgeroi tüm kontroller geçti (References dahil)")
    # Adım 0 sayım kontrolü (beklenen: nöron 139,248 + eksik kimlikler; >=5 kenar 2,700,513)
    log(f"ADIM0 fafb nöron: {len(n):,}")
    for t in (1, 5, 10):
        log(f"ADIM0 fafb kenar >= {t}: {int((e.syn_count >= t).sum()):,}")
    log(f"ADIM0 fafb toplam sinaps (>=1): {int(e.syn_count.sum()):,}")
    return n, e, roi, ann_raw


def build_malecns_interim(force=False):
    from src import malecns
    n = cached(config.cache_path("malecns", "neurons", malecns.RELEASE, "traced-interim"),
               lambda: malecns.build_neurons_interim(malecns.read_raw(), log=log),
               V.validate_neurons, force, log=log)
    log(f"ADIM0 malecns nöron (Traced, interim): {len(n):,}")
    return n


if __name__ == "__main__":
    which = sys.argv[1:] or ["fafb"]   # python -m src.build malecns  (geçici nöron tablosu)
    force = "--force" in which
    if "fafb" in which:
        build_fafb(force)
    if "malecns" in which:
        build_malecns_interim(force)
    Path("reports").mkdir(exist_ok=True)
    name = "build_log.txt" if "fafb" in which else "build_log_malecns.txt"
    Path("reports", name).write_text("\n".join(LOG), encoding="utf-8")
