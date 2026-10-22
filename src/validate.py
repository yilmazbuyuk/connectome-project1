"""Data Conventions 'Validation': her kontrol hatalı kolonu adıyla belirterek durur."""
import pandas as pd
import config


class ValidationError(ValueError):
    pass


def _fail(table, check, msg):
    raise ValidationError(f"[{table}] {check}: {msg}")


def check_schema(df, schema, table):
    # Columns
    expected = list(schema)
    if list(df.columns) != expected:
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        _fail(table, "Columns", f"eksik={missing} fazla={extra} (sıra da şemayla aynı olmalı)")
    # Dtypes
    for col, (dtype, _) in schema.items():
        want = pd.api.types.pandas_dtype(dtype) if isinstance(dtype, str) else dtype
        if df[col].dtype != want:
            _fail(table, "Dtypes", f"{col}: {df[col].dtype} (beklenen {want})")
    # Nulls
    for col, (_, nullable) in schema.items():
        if not nullable and df[col].isna().any():
            _fail(table, "Nulls", f"{col}: {int(df[col].isna().sum()):,} boş değer")


def check_keys(df, keys, table):
    dup = df.duplicated(keys)
    if dup.any():
        _fail(table, "Keys", f"{list(keys)} {int(dup.sum()):,} kez tekrar ediyor")


def check_vocabulary(df, table):
    # Kategori kolonlarında listede olmayan değer NaN olurdu; Nulls kontrolü bunu yakalar.
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype) and df[col].isna().any():
            _fail(table, "Vocabulary", f"{col}: listede olmayan değer(ler) var")


def validate_neurons(n):
    check_schema(n, config.NEURON_SCHEMA, "neurons")
    check_keys(n, ["neuron_id", "dataset"], "neurons")
    check_vocabulary(n, "neurons")


def validate_edges(e, neurons=None, table="edges", schema=None, keys=("pre_id", "post_id", "dataset")):
    check_schema(e, schema or config.EDGE_SCHEMA, table)
    check_keys(e, list(keys), table)
    if (e["syn_count"] < e["min_syn_fetched"]).any():
        _fail(table, "Threshold", "syn_count < min_syn_fetched olan satır var")
    if (e["pre_id"] == e["post_id"]).any():
        _fail(table, "Self", "öz-bağlantı kalmış")
    if neurons is not None:
        ids = pd.Index(neurons["neuron_id"].unique())
        for col in ("pre_id", "post_id"):
            bad = ~e[col].isin(ids)
            if bad.any():
                _fail(table, "References", f"{col}: {int(bad.sum()):,} kimlik nöron tablosunda yok")


def validate_edgeroi(r, neurons=None):
    validate_edges(r, neurons, table="edgeroi", schema=config.EDGEROI_SCHEMA,
                   keys=("pre_id", "post_id", "neuropil", "dataset"))


def check_plausibility(neurons, edges):
    """Bir nöronun kenar tablosundaki girdi sinaps toplamı n_post'u aşmamalı (n_post boşsa atlanır)."""
    tot = edges.groupby("post_id")["syn_count"].sum()
    n = neurons.set_index("neuron_id")["n_post"].dropna()
    common = tot.index.intersection(n.index)
    over = tot[common] > n[common].astype("int64")
    if over.any():
        _fail("edges", "Plausibility", f"{int(over.sum()):,} nöronda girdi toplamı n_post'u aşıyor")
    return len(common)


def assert_interchangeable(a, b):
    assert list(a.columns) == list(b.columns), "column sets differ"
    assert a.dtypes.equals(b.dtypes), "dtypes differ"
