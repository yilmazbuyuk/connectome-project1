"""Gerekli dış dosyaları bir kez indirir (data/raw/ altına; git'e girmez).

Çalıştır:  python -m src.download fafb       (≈ 250 MB)
           python -m src.download malecns    (≈ 6.8 GB, syn-partners)
"""
import sys
import urllib.request
from pathlib import Path

RAW = Path("data/raw")

ZENODO = "https://zenodo.org/records/10676866/files/{}?download=1"       # FlyWire 783 data release
FAFB_FILES = ["proofread_root_ids_783.npy",
              "per_neuron_neuropil_count_pre_783.feather",
              "per_neuron_neuropil_count_post_783.feather"]

MALECNS_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/{}"
MALECNS_FILES = ["syn-partners-male-cns-v1.0-minconf-0.5.feather"]


def fetch(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[var] {dest.name} ({dest.stat().st_size / 1e6:,.0f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[indiriliyor] {dest.name}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, step = 0, 0
        while True:
            chunk = r.read(8 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and done * 20 // total > step:
                step = done * 20 // total
                print(f"  %{step * 5:3d}  ({done / 1e6:,.0f} / {total / 1e6:,.0f} MB)", flush=True)
    tmp.replace(dest)
    print(f"[tamam] {dest.name} ({dest.stat().st_size / 1e6:,.0f} MB)")
    return dest


def fafb():
    return {name: fetch(ZENODO.format(name), RAW / name) for name in FAFB_FILES}


def malecns():
    return {name: fetch(MALECNS_BASE.format(name), RAW / name) for name in MALECNS_FILES}


if __name__ == "__main__":
    which = sys.argv[1:] or ["fafb"]
    if "fafb" in which:
        fafb()
    if "malecns" in which:
        malecns()
