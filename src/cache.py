"""Önbellek yardımcısı (Data Conventions, 'Caching').

Dosya varsa okur; yoksa üretir, doğrular, yazar. Yeniden üretmenin tek yolu force_refresh=True.
"""
from pathlib import Path
import time
import pandas as pd


def cached(path, fetch_fn, validate_fn=None, force_refresh=False, log=print):
    path = Path(path)
    if path.exists() and not force_refresh:
        t0 = time.time()
        df = pd.read_parquet(path)
        log(f"[cache] {path.name}: {len(df):,} satır diskten ({time.time() - t0:.1f} sn)")
        return df
    t0 = time.time()
    df = fetch_fn()
    if validate_fn is not None:
        validate_fn(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log(f"[build] {path.name}: {len(df):,} satır üretildi ({time.time() - t0:.1f} sn, "
        f"{path.stat().st_size / 1e6:.1f} MB)")
    return df
