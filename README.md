# Connectome course — Project 1 (Yılmaz BÜYÜK, 263031030001)

**Project 1: where connectivity-based cell typing fails.** A cost-sensitive classifier is trained on the female brain (FlyWire FAFB) and tested on the male brain (MaleCNS). The errors are then studied by brain region and by sexual dimorphism.

This repository holds the shared week 2 data pipeline (Week 2 Data Conventions Final), Workstream A — Labels and cell identity of the week 3 EDA, and the week 4 Shared Data Corrections (eda_shared v0.2.0). All tables and numbers below are on the corrected tables.

## Layout

```
config.py                 Data Conventions config v0.2.0, unchanged
eda_shared/               shared files v0.2.0, unchanged (pytest eda_shared/tests: 49 passed)
src/
  cache.py  validate.py  common.py   cache helper, the validators, schema helpers
  download.py             one-off download of the bulk files into data/raw/
  fafb.py  malecns.py     fetch + corrections, in the README order of eda_shared
  build.py                python -m src.build fafb | malecns | step0
  lock_numbers.py         threshold / scale numbers for the lock note
notebooks/
  00_integration.ipynb    Step 0: cache check, validators, counts
  eda_A.ipynb             Workstream A (A1–A7) on the corrected tables, no network calls
figures/eda/              A*_<dataset>_<release>_<population>_min1.png (corrected tables)
figures/eda/v0.1.1/       the same figures before the corrections, kept so the change is visible
reports/
  step0.md, step0.json    Step 0 on the corrected tables
  step0_fafb_region_counts.csv
  eda_A.md                one-page summary (one line per item + one surprise)
  eda_A_numbers.json      every number quoted in eda_A.md
  lock_numbers.json       edges / isolated neurons per threshold, B2 and B6 scale ratios
  fafb_neuron_report.json, malecns_brain_report.json, malecns_edgeroi_self.json
  build_log_*.txt         build logs
data/                     not in git (see below)
```

## Sources and versions (D6)

| Dataset | Release | Neurons | Connections |
| --- | --- | --- | --- |
| FAFB (female brain) | materialisation 783, flywire_annotations v3.2.0 | 139,255 (proofread set, correction 3) | Codex Buhmann per-neuropil table, no threshold |
| MaleCNS (male brain + nerve cord) | male-cns:v1.0, status Traced, synapse confidence ≥ 0.5 | 165,122 Traced; 144,414 in the brain part (correction 4) | syn-partners-male-cns-v1.0-minconf-0.5.feather (Janelia bulk file), aggregated per primary ROI |

Extra files (week 4), downloaded by `python -m src.download fafb malecns`:

- Zenodo 10676866: `proofread_root_ids_783.npy`, `per_neuron_neuropil_count_pre_783.feather`, `per_neuron_neuropil_count_post_783.feather` (FAFB region rule, correction 6).
- Janelia flat connectome: `syn-partners-male-cns-v1.0-minconf-0.5.feather` (6.8 GB, MaleCNS per-neuropil connections).
- neuPrint (token from the environment variable `NEUPRINT_APPLICATION_CREDENTIALS`): Traced neurons, per-ROI upstream/downstream counts, primary ROI list (MaleCNS region rule).

## How to run

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install pandas pyarrow duckdb neuprint-python matplotlib jupyter nbconvert pytest
pytest eda_shared/tests                      # 49 passed
python -m src.download fafb malecns          # once; files go to data/raw/
python -m src.build fafb                     # FAFB tables + validation
python -m src.build malecns                  # MaleCNS tables, brain part + validation
python -m src.build step0                    # reports/step0.md
python -m src.lock_numbers                   # reports/lock_numbers.json
jupyter nbconvert --to notebook --execute --inplace notebooks/00_integration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/eda_A.ipynb
```

Environment variables (optional unless stated): `NEUPRINT_APPLICATION_CREDENTIALS` (required for the first MaleCNS build), `MALECNS_SYN_PARTNERS` (path to the syn-partners file if it is not in data/raw/), `DUCKDB_TMP` (folder for temporary files of the MaleCNS aggregation), `EDGEROI_BATCH_ROWS` (smaller value = less memory). The MaleCNS aggregation reads the 6.8 GB file in batches and never loads it whole; on this machine it takes about 5 minutes. Tokens are never stored in the repository.

## Step 0 — corrected tables (eda_shared v0.2.0)

| Quantity | FAFB | MaleCNS (brain part) |
| --- | --- | --- |
| Neurons | **139,255** = 139,241 kept + 14 added; 7 dropped | **144,414** of 165,122 Traced; 20,708 dropped; ascending kept 1,841; sensory_ascending kept 534 |
| Edges τ = 1 | 15,091,983 | 21,706,010 |
| Edges ≥ 5 | **2,700,513** | **5,143,528** |
| Synapses by location | — | brain 99,982,458; nerve cord 24,041,864; no location 251 |
| Region | 78 values; ties 232; unknown 281 | ties 76; changed by the brain rule 1,657 |
| Self-connections dropped | 0 | 101 |
| Central population (is_optic False, known region) | 45,203 | 45,642 |
| Transmitter correction | known_nt 77,169 rows, 18,513 changed; dopamine 5,909 → 638 | kc_override 4,064 rows, 4,064 changed |

super_class — FAFB: optic 77,537; central 32,383; sensory 16,904; visual_projection 8,038; ascending 1,750; descending 1,303; sensory_ascending 612; visual_centrifugal 524; motor 110; endocrine 80; unknown 14.
MaleCNS brain part: optic 89,388; central 32,160; visual_projection 9,202; sensory 8,918; ascending 1,841; descending 1,312; visual_centrifugal 563; sensory_ascending 534; unknown 302; motor 107; endocrine 71; efferent 16.

Every FAFB reference number of the corrections document is matched exactly, and MaleCNS self-connections (101) match the reference.

## Threshold and scale numbers (reports/lock_numbers.json)

| τ | FAFB edges | FAFB isolated | MaleCNS edges | MaleCNS isolated |
| --- | --- | --- | --- | --- |
| 5 | 2,700,513 | 5,074 | 5,143,528 | 601 |
| 10 | 1,066,822 | 14,810 | 2,190,021 | 2,309 |
| 15 | 584,895 | 31,188 | 1,244,247 | 8,639 |
| 31 | 165,352 | 83,547 | 391,456 | 50,139 |

Scale: MaleCNS has 1.28× the synapses per edge and 1.39× the edges per neuron (B2); matched type pairs carry 1.484× the synapses (B6, median over 119,406 pairs; Scheffer 2025 preprint: 1.49).

## Notes reported to the group

- The interim mappings of weeks 2–3 (sensory_ascending → ascending, MaleCNS-only levels, the local region function, the edgeroi-based region) are removed; eda_shared v0.2.0 replaces them.
- Neuropil review (Project 1 duty): only change proposed is LAL → motor_periphery; GNG is kept as motor_periphery but flagged as mixed (A2: sensory 34%, central 23%, ascending 19%, descending 13%).
- FAFB sensory_ascending types are almost never matched in MaleCNS (2.3% of neurons, A6), and MaleCNS has 8,918 brain sensory neurons against 16,904 in FAFB. Both are reported to the group as possible naming / scope differences.
- Disclosure: the D18 pairing proposal in reports/eda_A.md suits Project 1.
- Project 1 lock note: submitted on UZEM; no model has been run before its approval.

## Week 9 — naive result (tag `week9`)

Lock note v2 is in force with no deviation (`reports/lock_note_v2.pdf`): target `super_class` (10 levels), whole cell types held out (5 folds × seeds 0, 1, 2; `reports/week9_splits.csv`), FAFB τ = 10 (sensitivity 5 and 15, seed 0), MaleCNS τ = round(1.484 τ) and synapse totals ÷ 1.484, features = all-synapse totals, degrees and neuropil shares (L/R merged), type-weighted metrics as headline, flow cost matrix 1/2/5, within-animal floor FAFB left ↔ right, unadjusted prior with an EM check.

```powershell
pip install -r requirements.txt               # pinned versions
python -m src.features                        # feature tables, tau 10 / 5 / 15 (cache)
python -m src.week9                           # naive models, metrics, figures (1-3 h)
jupyter nbconvert --to notebook --execute --inplace notebooks/week9_naive.ipynb
```

Outputs: `reports/week9_results.json`, `reports/week9_metrics.csv`, `reports/week9_hparams.csv`, `reports/week9_splits.csv`, `reports/week9_log.txt`, `figures/week9/fig*.png`. Seeds are fixed and printed in the log. Everything runs from `data/cache/`, with no network access.
