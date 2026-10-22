"""İki veri seti için ortak dönüştürme yardımcıları."""
import pandas as pd
import config


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


def top_region(edgeroi, log=print):
    """Her nöron için girdi+çıktı sinapslarının en çok olduğu nöropil (tüm sinapslar, τ=1).

    Eşitlik kuralı 3. haftada kararlaştırılacak; şimdilik belirleyici olsun diye
    eşitlikte alfabetik olarak ilk nöropil seçilir ve eşitlik sayısı loglanır.
    """
    a = edgeroi.groupby(["pre_id", "neuropil"], observed=True)["syn_count"].sum()
    b = edgeroi.groupby(["post_id", "neuropil"], observed=True)["syn_count"].sum()
    a.index.names = b.index.names = ["neuron_id", "neuropil"]
    tot = a.add(b, fill_value=0).rename("syn").reset_index()
    tot = tot.sort_values(["neuron_id", "syn", "neuropil"], ascending=[True, False, True])
    first = tot.groupby("neuron_id").head(2).copy()
    first["rank"] = first.groupby("neuron_id").cumcount()
    w = first.pivot(index="neuron_id", columns="rank", values="syn")
    ties = int((w.get(1) == w[0]).sum()) if 1 in w else 0
    top = tot.groupby("neuron_id").head(1).set_index("neuron_id")["neuropil"]
    log(f"region: {len(top):,} nörona atandı; ilk iki nöropilde tam eşitlik: {ties:,} "
        f"(alfabetik seçildi; kural 3. haftada); UNASGD atanan: {int((top == 'UNASGD').sum()):,}")
    return top
