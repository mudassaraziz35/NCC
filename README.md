# Sexism and the eco-gender gap in climate attitudes across 30 European countries

Replication materials for the manuscript *"Sexism and climate scepticism are fused in
Europe's most gender-equal countries"* (prepared for *Nature Climate Change*).

Using European Social Survey Round 11 (2023–24; 30 countries; ~48,700 adults) — the first
cross-national probability survey to measure sexism, gendered self-identity **and** climate
attitudes in the same interview — this project shows that sexist attitudes predict lower
climate worry, weaker attribution of climate change to human activity and less felt
responsibility to act; that sexism and gendered self-identity account for a large share of
the gender gap in climate concern; and that the sexism–climate link **steepens with national
gender equality, not with fossil-fuel dependence** — evidence for a cultural-backlash rather
than a material reading of "petro-masculinity".

Every number and figure in the manuscript is produced by the code here, injected
programmatically from the generated result files (no hand-transcribed statistics).

---

## Repository layout

```
.
├── code/                     analysis pipeline (Python) + optional manuscript builders (Node)
│   ├── _config.py            resolves data/results paths (env-overridable)
│   ├── 01_prep.py … 11_summary.py
│   ├── run                   Code Ocean entry point / master script
│   └── manuscript_src/       ncc_paper.js, cover_letter.js (optional, requires Node)
├── data/
│   ├── README.md             how to obtain the ESS microdata (free, registration required)
│   ├── country_moderators.json          country-level moderators (compiled here; see below)
│   └── SI_country_moderators_sources.csv per-value provenance
├── results/                  generated outputs (committed reference copies)
│   ├── *.json, coef_*.csv, twostep_*.csv
│   ├── figures/*.png
│   ├── REPLICATION_SUMMARY.md  legible confirmation of headline numbers  (generated)
│   └── key_results.csv         flat table of the same                    (generated)
├── manuscript/               ESS11_NCC_manuscript.pdf + submission figures
├── environment/Dockerfile    Code Ocean reproducible environment
├── requirements.txt          pinned Python dependencies (Python 3.11.15)
└── run_all.sh                local convenience wrapper
```

## Data access (read this first)

The one input you must supply is the **ESS Round 11 integrated file**. ESS microdata are free
but require registration and **may not be redistributed**, so they are *not* included here.

1. Register (free) and download the ESS11 integrated file, **edition 4.2**, from the ESS Data
   Portal: https://ess.sikt.no  (DOI: https://doi.org/10.21338/ess11e04_2).
2. Export/convert it to CSV and place it at **`data/ESS11.csv`**.

That single file is all the pipeline needs; every other input (the country-level moderators)
is already in `data/`. See `data/README.md` for details, including the required data citation.

## Quick start (local)

```bash
pip install -r requirements.txt      # Python 3.11
# place the ESS file at data/ESS11.csv  (see above)
bash run_all.sh                      # runs code/run: steps 01 → 11
```

Outputs land in `results/` (JSON, CSV, `figures/*.png`) with a legible
`results/REPLICATION_SUMMARY.md` you can check against the manuscript. A full run is a few
minutes (the H3 bootstrap and H5 random-slope models dominate).

## Running on Code Ocean

The repository is already in Code Ocean capsule shape: code in `code/`, the `code/run` entry
point, the environment in `environment/Dockerfile`, and outputs written to `../results`.

1. Create a capsule and import this repository (GitHub → Code Ocean), or upload the zip.
2. Upload `ESS11.csv` to the capsule's **Data** section so it mounts at `../data/ESS11.csv`
   (keeping it out of the public code, consistent with ESS terms).
3. Click **Reproducible Run**. `code/run` executes the whole pipeline; `results/` is captured.

`_config.py` honours `DATA_DIR` / `RESULTS_DIR`, which Code Ocean sets to `/data` and
`/results`; no code changes are needed.

## Pipeline

| Step | Script | Produces |
|------|--------|----------|
| 01 | `01_prep.py` | recodes, sexism index, weights → `analysis.parquet`, `prep_report.json` |
| 02 | `02_h1h2.py` | H1/H2 multilevel models (blocks M1–M3) → `h1h2_results.json`, `coef_*.csv` |
| 03 | `03_h3_decomp.py` | H3 gender-gap decomposition, 1,000× bootstrap → `h3_results.json` |
| 04 | `04_h4_interactions.py` | H4 interactions (sexism×gender, sexism×centrality) → `h4_results.json` |
| 05 | `05_robustness.py` | weighting, ordinal/logit, exclusions, two-step meta → `robustness_results.json`, `twostep_*.csv` |
| 06 | `06_figures.py` | Figures 1–4 (main + Extended Data specification plot) |
| 07 | `07_bundle.py` | assembles `results_bundle.json` (everything the manuscript cites) |
| 08 | `08_h5.py` | H5 random-effects meta-regressions → `h5_results.json` |
| 09 | `09_onestep.py` | H5 one-step random-slope validation → `h5_onestep.json` |
| 10 | `10_fig5.py` | Figure 4 — contextual moderation (backlash vs material) |
| 11 | `11_summary.py` | `REPLICATION_SUMMARY.md`, `key_results.csv` |

Steps 02–07 depend on 01; 08–10 depend on 05 (two-step slopes) and the moderators; 11 depends
on 07–09. The `run` script executes them in order.

## Hypotheses and headline results

- **H1** Sexism → lower worry (β ≈ −0.08), weaker human attribution (≈ −0.10) and less
  responsibility (≈ −0.07) per s.d., net of sociodemographics; all clear the pre-committed
  |β| ≥ 0.05 benchmark.
- **H2** Robust to left–right ideology (6–14% attenuation; for responsibility sexism > ideology).
- **H3** Sexism + gendered self-identity explain ~40% of the worry gap and ~68% of the
  responsibility gap; within that, self-ascribed femininity carries most of it, sexism ~7–8%.
- **H4** Roughly twice as strong among men for worry/responsibility; gender-symmetric for
  attribution; strongest among men for whom gender identity is *least* central (a reversal we
  report rather than explain away).
- **H5** The slope steepens with the EIGE Gender Equality Index (γ ≈ −0.06 per s.d.; up to 48%
  of between-country variance) and **not** with fossil-fuel dependence (null / opposite sign).

The exact figures a fresh run should reproduce are in `results/REPLICATION_SUMMARY.md`.

## Optional: rebuilding the manuscript and cover letter

The Word documents are generated from the result files by Node scripts in
`code/manuscript_src/` (require Node ≥ 18 and `npm install docx`). They are **not** part of the
`run` pipeline — the scientific replication is Python-only. See that folder for usage.

## Citing

- **Paper:** Aziz, M. Sexism and climate scepticism are fused in Europe's most gender-equal
  countries (in preparation, 2026).
- **Data:** European Social Survey European Research Infrastructure (ESS ERIC) (2025) *ESS11 –
  integrated file, edition 4.2* [Data set]. Sikt. https://doi.org/10.21338/ess11e04_2
- **This repository:** see `CITATION.cff`.

## Licensing

- **Code** (`code/`, `run_all.sh`): MIT — see `LICENSE`.
- **Compiled country-level data** (`data/country_moderators.json`, `SI_*.csv`): values are
  aggregate national indicators drawn from EIGE, Eurostat, the Energy Institute (via Our World
  in Data), the World Bank and UNDP; each value's source is in the SI file. Redistributed here
  for reproducibility under those sources' terms.
- **ESS microdata:** governed by ESS ERIC terms — **not** included; download per `data/README.md`.
