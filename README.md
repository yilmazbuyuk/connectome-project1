# Connectome course — Project 1 (Yılmaz BÜYÜK, 263031030001)

**Project 1:** where connectivity-based cell typing fails. A cost-sensitive classifier is trained on the female brain (FlyWire FAFB) and tested on the male brain (MaleCNS). The errors are then studied by brain region and by sexual dimorphism.

This repository holds the shared week 2 data pipeline (Week 2 Data Conventions Final) and **Workstream A — Labels and cell identity** of the week 3 EDA.

## Layout

```
config.py                 Data Conventions config, unchanged
eda_shared/               shared EDA files v0.1.1-eda, unchanged (pytest eda_shared/tests: 15 passed)
src/                      fetch, cache and validator code
  cache.py  validate.py  common.py  fafb.py  malecns.py  build.py
notebooks/
  00_integration.ipynb    Step 0: cache check, the eight validators, counts, source value counts
  eda_A.ipynb             Workstream A (A1–A7), regenerates every figure from the cache, no network calls
figures/eda/              A*_<dataset>_<release>_<population>_min1.png
reports/
  eda_A.md                one-page summary (one line per item + one surprise)
  eda_A_numbers.json      every number quoted in eda_A.md
  A4_multi_superclass_types.csv
  build_log.txt           FAFB build log
  build_log_malecns.txt   MaleCNS build log
data/                     not in git (see below)
```

## Sources and versions (D6)

| Dataset | Release | Neurons | Edges |
|---|---|---|---|
| FAFB (female brain) | materialisation 783, annotations `flywire_annotations` v3.2.0 | 139,248 + 14 ids that appear in edges but not in annotations (added as `unknown`) = 139,262 | Codex Buhmann per-neuropil table, no threshold (`data/connections_buhmann_no_threshold.csv.gz`) |
| MaleCNS (male brain + nerve cord) | male-cns:v1.0, status Traced | 165,122 (144,757 in the brain, interim rule) | not used by Workstream A |

**MaleCNS interim.** The MaleCNS neuron table is built from the week 2 neuPrint export (`data/raw/malecns_neurons_raw_male-cns_v1.0.parquet`). That export has no `roiInfo`, so `region` is `unknown`. In `eda_A.ipynb`, brain vs nerve cord and optic vs central come from the raw `superclass` (`vnc_*` = nerve cord, `ol_*` = optic). Every malecns figure caption says so. When the bulk download is in place, `region` will come from `roiInfo` and the cache name will drop `-interim`. Until then, A2 and A7 plot 1 have no malecns panel.

## How to run

```
python -m venv .venv && .venv\Scripts\activate
pip install pandas pyarrow matplotlib jupyter nbconvert pytest
pytest eda_shared/tests                      # 15 passed
python -m src.build fafb                     # FAFB cache + validation; log in reports/build_log.txt
python -m src.build malecns                  # MaleCNS interim neuron cache; reports/build_log_malecns.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/00_integration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/eda_A.ipynb
```

The first `fafb` build downloads the annotation TSV once from GitHub. Every later run reads `data/cache/` only. Tokens are never stored in the repository.

## Step 0 counts (FAFB, all eight validators pass)

| | Count |
|---|---|
| neurons | 139,262 |
| self-connections | 0 |
| edges ≥ 1 | 15,091,983 |
| edges ≥ 5 | 2,700,513 (expected value) |
| edges ≥ 10 | 1,066,822 |
| edges ≥ 31 | 165,352 |

## Decisions and notes reported to the group

- `sensory_ascending` (612 FAFB neurons) is not in `config.SUPER_CLASS_VALUES`. It is mapped to `ascending` for now (interim, in `src/fafb.py`).
- MaleCNS-only superclass levels are not in the config yet. Interim mapping in `src/malecns.py`: `sensory_/efferent_ascending` → ascending, `sensory_/efferent_descending` → descending, `*_efferent` → motor, `*_tbc` → their base class, `vnc_intrinsic`/`vnc_tbc`/`ENS` → unknown.
- `region` = the neuropil with the most synapses (in + out, τ = 1). Ties are broken alphabetically (319 FAFB neurons) until the week 3 rule.
- Neuropil review (Project 1 duty): only change proposed is LAL → motor_periphery; GNG is kept as motor_periphery but flagged as mixed (A2 confirms: sensory 35%, ascending 28%, central 24%, descending 11%).
- Disclosure: the D18 pairing proposal in `reports/eda_A.md` suits Project 1.
