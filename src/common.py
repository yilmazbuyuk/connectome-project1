"""İki veri seti için ortak dönüştürme yardımcıları.

v0.2.0: bölge (region) artık burada değil, eda_shared.corrections.region_from_counts ile
hesaplanıyor (Week 4 correction 6). Yerel bölge fonksiyonu silindi.
"""
import pandas as pd


def recode(series, mapping, column):
    """Data Conventions'taki recode: eşlenmemiş değer hata verir, boş -> unknown."""
    out = series.map(mapping)
    unmapped = series[out.isna() & series.notna()].unique()
    if len(unmapped):
        raise ValueError(f"{column}: unmapped values {list(unmapped)[:10]}")
    return out.fillna("unknown")


def apply_schema(df, schema):
    """Kolonları şemadaki sırayla ve tiple döndürür."""
    out = pd.DataFrame(index=df.index)
    for col, (dtype, _) in schema.items():
        out[col] = df[col].astype(dtype)
    return out.reset_index(drop=True)
