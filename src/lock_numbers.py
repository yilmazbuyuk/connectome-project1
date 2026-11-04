"""Lock note için EDA sayıları (düzeltilmiş tablolar, model çıktısı yok).

Çalıştır:  python -m src.lock_numbers      ->  reports/lock_numbers.json
  - eşik başına kenar, girdisiz / çıktısız / yalıtılmış nöron (FAFB ve MaleCNS beyin kısmı)
  - ölçek farkı: kenar başına sinaps, nöron başına kenar (B2) ve eşleşen tip çiftlerinde
    sinaps oranı (B6: MaleCNS flywire_type <-> FAFB cell_type, tek eşleşmeli tipler)
  - flywire_type'ı virgüllü (birden çok FAFB tipine eşlenen) MaleCNS nöronları
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.build import fafb_paths, malecns_paths

TAUS = (5, 8, 10, 15, 31, 46)        # FAFB 5/10/31 ve ≈1.5 ile ölçeklenmiş karşılıkları


def per_tau(n, e):
    ids = n["neuron_id"]
    res = {}
    for t in TAUS:
        x = e[e["syn_count"] >= t]
        has_in, has_out = ids.isin(x["post_id"]), ids.isin(x["pre_id"])
        indeg = x.groupby("post_id").size().reindex(ids, fill_value=0)
        res[t] = {"edges": int(len(x)),
                  "no_input": int((~has_in).sum()), "no_output": int((~has_out).sum()),
                  "isolated": int((~has_in & ~has_out).sum()),
                  "median_in_degree": float(indeg.median())}
    return res


def type_flow(n, e, type_col):
    """Tip çifti başına sinaps / presinaptik tipteki nöron sayısı (τ = 1)."""
    t = n.set_index("neuron_id")[type_col]
    size = n[type_col].value_counts()
    d = pd.DataFrame({"a": e["pre_id"].map(t).to_numpy(), "b": e["post_id"].map(t).to_numpy(),
                      "w": e["syn_count"].to_numpy()}).dropna()
    g = d.groupby(["a", "b"], observed=True)["w"].sum()
    return g / g.index.get_level_values("a").map(size).to_numpy()


def main():
    out = {}
    fp, mp = fafb_paths(), malecns_paths()
    cols = ["neuron_id", "cell_type", "flywire_type", "super_class"]
    f = pd.read_parquet(fp["neurons"], columns=cols)
    m = pd.read_parquet(mp["neurons"], columns=cols)
    fe = pd.read_parquet(fp["edges"], columns=["pre_id", "post_id", "syn_count"])
    me = pd.read_parquet(mp["edges"], columns=["pre_id", "post_id", "syn_count"])

    for k, n, e in (("fafb", f, fe), ("malecns", m, me)):
        out[k] = {"neurons": int(len(n)), "per_tau": per_tau(n, e),
                  "syn_per_edge_tau1": round(float(e["syn_count"].mean()), 3),
                  "edges_per_neuron_tau1": round(len(e) / len(n), 2)}
        print(k, json.dumps(out[k]["per_tau"], indent=0))
    out["B2_syn_per_edge_ratio"] = round(out["malecns"]["syn_per_edge_tau1"] / out["fafb"]["syn_per_edge_tau1"], 3)
    out["B2_edges_per_neuron_ratio"] = round(out["malecns"]["edges_per_neuron_tau1"] / out["fafb"]["edges_per_neuron_tau1"], 3)

    ft = m["flywire_type"].astype("string")
    multi = ft.str.contains(",", na=False)
    out["malecns_flywire_type_multi"] = int(multi.sum())
    out["malecns_flywire_type_missing"] = int(ft.isna().sum())
    m1 = m.assign(ftype=ft.where(~multi))
    ff = type_flow(f.assign(ftype=f["cell_type"].astype("string")), fe, "ftype")
    mm = type_flow(m1, me, "ftype")
    j = pd.concat([ff.rename("fafb"), mm.rename("malecns")], axis=1, join="inner")
    strong = j[(j["fafb"] >= 10) & (j["malecns"] >= 10)]      # zayıf çiftlerin gürültüsü dışarıda
    out["B6_matched_type_pairs"] = int(len(strong))
    out["B6_ratio_median"] = round(float((strong["malecns"] / strong["fafb"]).median()), 3)
    out["B6_ratio_total"] = round(float(strong["malecns"].sum() / strong["fafb"].sum()), 3)
    out["B6_ratio_iqr"] = [round(float(q), 3) for q in (strong["malecns"] / strong["fafb"]).quantile([.25, .75])]

    for k, n in (("fafb", f), ("malecns", m)):
        out[f"{k}_super_class"] = {s: int(v) for s, v in n["super_class"].value_counts().items() if v}

    Path("reports").mkdir(exist_ok=True)
    Path("reports/lock_numbers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print({k: v for k, v in out.items() if k.startswith(("B2", "B6", "malecns_flywire"))})
    print("yazıldı: reports/lock_numbers.json")


if __name__ == "__main__":
    main()
