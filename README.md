# ORBITAL: Space Debris Collision-Risk Platform

Machine learning + Apache Spark project that predicts satellite conjunction (collision) risk from Space-Track orbital data, with a black-and-white mission-control dashboard.

Everything runs **today in demo mode** (deterministic rules engine + seeded synthetic objects, flagged `DEMO DATA` in the UI). Model training is left to you: drop the dataset in, run one command, restart the API.

```
orbital/
├── README.md
├── run.sh                      # starts the API (which also serves the frontend)
├── frontend/
│   ├── index.html              # Tailwind CDN + design tokens, all sections
│   └── app.js                  # fetches /api/*, draws globe, charts, predictor
└── backend/
    ├── requirements.txt        # API + tests
    ├── requirements-train.txt  # PySpark etc. (training only)
    ├── app/                    # main.py (routes), risk.py (rules + model hook), demo.py
    ├── spark/
    │   ├── download_data.py    # Space-Track downloader
    │   ├── train.py            # full Spark pipeline  <-- YOU run this
    │   ├── benchmark.py        # Spark vs scikit-learn scalability timing
    │   └── make_sample_data.py # synthetic Space-Track-format data, for smoke tests only
    ├── tests/test_api.py
    ├── data/raw/               # <-- PUT DATASET HERE (gp.json, satcat.json, cdm.json)
    ├── data/processed/         # generated: objects.json, stats.json, pairs.parquet
    └── models/                 # generated: gbt_pipeline/, model_meta.json, metrics.json
```

## 1. Run the demo (no dataset needed)

Python 3.10+.

```bash
cd orbital/backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd ..
uvicorn backend.app.main:app --reload --port 8000      # or ./run.sh
```

Open http://localhost:8000. The API serves the frontend, so there is no CORS or build step. (If you open `frontend/index.html` straight from disk, it calls `http://localhost:8000`.)

Run the API tests: `python -m pytest backend/tests -q`

## 2. Where to put the dataset

Put three JSON files in **`backend/data/raw/`**:

| File | Space-Track class | Purpose |
|---|---|---|
| `gp.json` | `gp` | Current orbital elements (features, clustering) |
| `satcat.json` | `satcat` | Object type, country (metadata) |
| `cdm.json` | `cdm_public` | Conjunction events and `PC` (labels) |

Download them automatically (free account at space-track.org; approval can take a day or two):

```bash
pip install -r backend/requirements-train.txt
export SPACETRACK_USER="you@email.com"
export SPACETRACK_PASS="your-password"
python backend/spark/download_data.py                  # 365 days of CDMs; --cdm-days 730 for more
```

Rate limit is roughly 30 requests/min and 300/hour; the script sleeps and retries. Space-Track terms forbid redistributing raw data, which is why `data/raw` is git-ignored.

## 3. Train (you run this)

Needs Java 11 or 17 (`java -version`) and `pip install -r backend/requirements-train.txt`.

```bash
python backend/spark/train.py                  # defaults
python backend/spark/train.py --tune-k         # choose KMeans k (4..9) by silhouette
python backend/spark/train.py --drop-features regime_1,regime_2,shell_density_max   # ablation
python backend/spark/train.py --driver-memory 8g --cores 8
```

Pipeline: load and cast JSON, join SATCAT, `alt=(apoapsis+periapsis)/2`, 50 km shells and shell density, KMeans regimes (standardised log-altitude, inclination, eccentricity), build conjunction pairs from CDM `SAT_1_ID`/`SAT_2_ID`, label `1` if `PC > 1e-4`, **time-based split** at the 80th percentile of CDM creation time, class weights, train Logistic Regression / Random Forest / GBT, evaluate PR-AUC, recall and F1 of class 1 and the confusion matrix on the test split, save the best pipeline.

Outputs: `backend/models/gbt_pipeline/`, `models/model_meta.json`, `models/metrics.json`, `data/processed/objects.json`, `stats.json`, `pairs.parquet`.

**Restart the API** afterwards. The `DEMO DATA` badge disappears and every section shows your real numbers.

### Use the trained model in the predictor

```bash
USE_SPARK_MODEL=1 ./run.sh
```

Things to know:

- The classifier outputs **P(PC > 1e-4)**, not a Pc value. In model mode the card shows that probability as a percentage; in rules mode it shows a Pc.
- `PC` and `MIN_RNG` are never used as features (leakage). So by default the model ignores the *miss distance* slider, and the card says so. To include it, set `INCLUDE_MISS_DISTANCE = True` in `train.py` and retrain (scores will be higher partly because miss distance is circular with Pc).
- The four sliders map to model features in `FEATURE_MAP` at the top of `backend/app/risk.py` (`rel_vel` -> `v_rel_est`, `inc_div` -> `inc_diff`, `bstar` -> `bstar_diff` x 1e-4). All other features use training medians. Adapt this if you change `FEATURES`.
- Check the **unit of `MIN_RNG`** in your `cdm.json` and set `MIN_RNG_TO_KM` in `train.py` (default `0.001`, i.e. metres).

### Smoke-test the training script without real data

```bash
python backend/spark/make_sample_data.py
python backend/spark/train.py --data-dir backend/data/sample --out-dir backend/data/sample_out
```

This writes synthetic data to `data/sample` and outputs to `data/sample_out`, never touching `raw` or `models`. Do not report anything trained on it.

### Scalability study (for the paper)

```bash
python backend/spark/benchmark.py --cores 1 2 4 8 --write
```

Times Spark GBT training at `local[N]` against single-node scikit-learn on `pairs.parquet` (or synthetic rows if you have not trained yet) and merges the result into `models/metrics.json` under `"scalability"`. The Scalability section appears only when that key exists. `local[N]` on one machine only simulates a cluster; repeat on a real multi-node cluster for a stronger result.

## 4. API

Base path `/api`, all JSON.

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | `{"status":"ok"}` |
| GET | `/api/stats` | `{tracked, debris, high_risk, pr_auc, demo}` |
| GET | `/api/objects` | array of `{norad, name, type, alt, inc, ecc, regime, raan, phase, spd}` |
| GET | `/api/metrics` | `{demo, models, importance, confusion, scalability?}` |
| POST | `/api/predict` | `{pc, prob, score, level, chip, action, high_risk, miss_m, rel_vel, engine, note?}` |

`/api/predict` body: `{"miss_km": 0.34, "rel_vel": 14.2, "inc_div": 2.4, "bstar": 1.84}` (ranges 0.01-10, 0.5-16, 0-15, 0.1-5). `pc` is `null` and `prob` is set when the Spark model answers.

### Demo rules engine

```
pc = exp(-1.8 * miss_km) * (rel_vel / 14.0) * 1e-3
pc *= 1 + min(inc_div, 15) / 30
pc *= 1 + min(bstar, 5) / 10
pc  = clamp(pc, 1e-6, 4.2e-3);  score = clamp((log10(pc) + 6) / 3, 0, 1)
```

`pc >= 1e-4` HIGH, `pc >= 2.5e-5` MODERATE, else LOW. This is a demo proxy, **not physics**. Demo numbers (48,905 objects, PR-AUC 0.942, the confusion matrix, 700 objects) are placeholders and must never be reported as results.

## 5. Evaluation plan and limitations (for the paper)

- Metrics: PR-AUC (primary), recall, F1, confusion matrix. Baselines: Logistic Regression and a miss-distance threshold rule (added automatically when `MIN_RNG` exists).
- Ablations: `--drop-features`. Optional novelty: propagate TLEs with `sgp4` in a `pandas_udf` and build your own labels.
- CDM updates are de-duplicated to the latest per (pair, TCA day) so one event is not spread across train and test (`--no-dedupe` to disable).
- Object features come from the **current** GP snapshot, not the elements at CDM time, which adds noise for older CDMs.
- TLE accuracy is kilometre-level and public CDM data is limited: frame results as screening / risk-ranking, not precise collision prediction.
- `v_rel_est` is a heuristic from orbital speeds and plane angle, not a true encounter velocity.

## 6. Tech stack and license

Python 3.10+, FastAPI, Uvicorn, Pydantic, PySpark MLlib, Requests; HTML, Tailwind CDN, vanilla JS, Canvas/SVG. Choose a license for your code (MIT is common). Space-Track and ESA data remain under their own terms.
