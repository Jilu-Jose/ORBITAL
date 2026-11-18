#!/usr/bin/env python3
"""Optional scalability timing: Spark GBT training time vs cores, against single-node scikit-learn.

    python backend/spark/benchmark.py                       # uses data/processed/pairs.parquet if train.py has run
    python backend/spark/benchmark.py --rows 500000         # otherwise synthetic data
    python backend/spark/benchmark.py --cores 1 2 4 8 --write

Prints JSON for the "scalability" key of models/metrics.json. With --write it merges it into
models/metrics.json (if that file exists) or saves data/processed/scalability.json (the API
picks that up too). On one machine, local[N] only simulates a cluster; for a paper, repeat on a
real multi-node cluster.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure PySpark uses the current Python environment and its bundled matching JARs
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
FEATURES = ["inc_diff", "alt_diff", "ecc_diff", "bstar_diff", "raan_diff",
            "v_rel_est", "alt_mean", "shell_density_max", "regime_1", "regime_2"]


def load_frame(rows: int) -> pd.DataFrame:
    parquet = BACKEND / "data" / "processed" / "pairs.parquet"
    if parquet.exists():
        df = pd.read_parquet(parquet)
        feats = [c for c in df.columns if c not in ("created_ts", "label")]
        print(f"using {len(df):,} real rows from {parquet} ({len(feats)} features)")
        return df[feats + ["label"]].fillna(0.0)
    print(f"no pairs.parquet; generating {rows:,} synthetic rows")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(rows, len(FEATURES)))
    logit = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.7 * X[:, 5] * X[:, 2] + rng.normal(scale=0.5, size=rows) - 3.0
    y = (rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(float)
    df = pd.DataFrame(X, columns=FEATURES)
    df["label"] = y
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--rows", type=int, default=400_000)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    from pyspark.ml.classification import GBTClassifier
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import SparkSession
    from sklearn.ensemble import HistGradientBoostingClassifier

    pdf = load_frame(args.rows)
    feats = [c for c in pdf.columns if c != "label"]
    tmp = Path(tempfile.mkdtemp(prefix="orbital_bench_")) / "data.parquet"
    pdf.to_parquet(tmp)

    # single-node scikit-learn baseline (measured once, repeated across the cores axis)
    log_ = lambda m: print(f"[bench] {m}", flush=True)  # noqa: E731
    log_("scikit-learn HistGradientBoosting ...")
    t0 = time.time()
    HistGradientBoostingClassifier(max_iter=args.iters, max_depth=args.depth, random_state=0).fit(pdf[feats].values, pdf["label"].values)
    sk = round(time.time() - t0, 1)
    log_(f"  {sk}s")

    spark_times = []
    for c in args.cores:
        spark = (SparkSession.builder.master(f"local[{c}]").appName(f"bench-{c}")
                 .config("spark.ui.enabled", "false").config("spark.driver.memory", "3g").getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        df = VectorAssembler(inputCols=feats, outputCol="features").transform(spark.read.parquet(str(tmp))).select("features", "label").cache()
        df.count()  # materialise the cache so only training is timed
        t0 = time.time()
        GBTClassifier(maxIter=args.iters, maxDepth=args.depth, seed=0).fit(df)
        secs = round(time.time() - t0, 1)
        log_(f"Spark local[{c}]: {secs}s")
        spark_times.append(secs)
        spark.stop()

    out = {"cores": args.cores, "spark_s": spark_times, "sklearn_s": [sk] * len(args.cores)}
    print(json.dumps(out, indent=2))

    if args.write:
        metrics = BACKEND / "models" / "metrics.json"
        if metrics.exists():
            m = json.loads(metrics.read_text())
            m["scalability"] = out
            metrics.write_text(json.dumps(m, indent=2))
            print(f"merged into {metrics}")
        else:
            p = BACKEND / "data" / "processed" / "scalability.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(out, indent=2))
            print(f"metrics.json not found yet; saved {p} (train.py will not overwrite it; the API merges it)")


if __name__ == "__main__":
    main()
