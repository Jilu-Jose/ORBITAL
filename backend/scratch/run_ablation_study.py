"""
Run the 5 ablation configurations from Table 7 using PySpark MLlib on pairs.parquet:
  - Full (d=10)
  - A1: without cluster indices (drop regime_1, regime_2) (d=8)
  - A2: without shell density (drop shell_density_max) (d=9)
  - A3: without both (drop regime_1, regime_2, shell_density_max) (d=7)
  - A4: element differences only (inc_diff, alt_diff, ecc_diff, bstar_diff, raan_diff) (d=5)

Evaluates on the exact same 80th-percentile time split with Random Forest (numTrees=100, maxDepth=6).
Reports both:
  1) Single run (seed=42)
  2) Multi-seed (seeds=[42, 123, 456, 789, 2026]) to compute mean ± s.d.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np

# Spark env fix
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

BACKEND = Path(__file__).resolve().parents[1]
PAIRS_PATH = BACKEND / "data" / "processed" / "pairs.parquet"

CONFIGS = {
    "Full feature set (d = 10)": [
        "inc_diff", "alt_diff", "ecc_diff", "bstar_diff", "raan_diff",
        "v_rel_est", "alt_mean", "shell_density_max", "regime_1", "regime_2"
    ],
    "A1: without cluster indices": [
        "inc_diff", "alt_diff", "ecc_diff", "bstar_diff", "raan_diff",
        "v_rel_est", "alt_mean", "shell_density_max"
    ],
    "A2: without shell density": [
        "inc_diff", "alt_diff", "ecc_diff", "bstar_diff", "raan_diff",
        "v_rel_est", "alt_mean", "regime_1", "regime_2"
    ],
    "A3: without both": [
        "inc_diff", "alt_diff", "ecc_diff", "bstar_diff", "raan_diff",
        "v_rel_est", "alt_mean"
    ],
    "A4: element differences only": [
        "inc_diff", "alt_diff", "ecc_diff", "bstar_diff", "raan_diff"
    ],
}

spark = (
    SparkSession.builder
    .master("local[4]")
    .appName("ablation_study")
    .config("spark.driver.memory", "3g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# Load pairs
pairs = spark.read.parquet(str(PAIRS_PATH))

# Time split (80th percentile)
pairs = pairs.withColumn("created_num", F.col("created_ts").cast("double"))
cut = pairs.approxQuantile("created_num", [0.8], 0.001)[0]
train_raw = pairs.filter(F.col("created_num") <= cut)
test_raw = pairs.filter(F.col("created_num") > cut)

n_pos = train_raw.filter("label = 1").count()
n_neg = train_raw.filter("label = 0").count()
total = n_pos + n_neg
w_pos, w_neg = total / (2.0 * n_pos), total / (2.0 * n_neg)

train = train_raw.withColumn("w", F.when(F.col("label") == 1, w_pos).otherwise(w_neg)).cache()
test = test_raw.cache()

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

SEEDS = [42, 123, 456, 789, 2026]
results = {}

print("Running ablation experiments...")
for name, feats in CONFIGS.items():
    print(f"\n--- {name} (d={len(feats)}) ---")
    asm = VectorAssembler(inputCols=feats, outputCol="features_raw", handleInvalid="keep")
    train_vec = asm.transform(train).cache()
    test_vec = asm.transform(test).cache()

    scores_list = []
    for s in SEEDS:
        rf = RandomForestClassifier(
            featuresCol="features_raw", labelCol="label", weightCol="w",
            numTrees=100, maxDepth=6, seed=s
        )
        model = rf.fit(train_vec)
        pred = model.transform(test_vec)
        pr_auc = evaluator.evaluate(pred)
        scores_list.append(pr_auc)
        print(f"  seed={s}: PR-AUC = {pr_auc:.4f}")

    results[name] = {
        "d": len(feats),
        "seed_42": scores_list[0],
        "all_seeds": scores_list,
        "mean": float(np.mean(scores_list)),
        "std": float(np.std(scores_list, ddof=1)),
    }

spark.stop()

# Print formatted summary table
full_mean = results["Full feature set (d = 10)"]["mean"]
full_42 = results["Full feature set (d = 10)"]["seed_42"]

print("\n" + "="*80)
print("TABLE 7 RESULTS SUMMARY")
print("="*80)
print(f"{'Configuration':<35} | {'PR-AUC (seed 42)':<16} | {'PR-AUC (mean ± s.d.)':<22} | {'Δ vs. full (mean)':<16}")
print("-" * 95)

for name, r in results.items():
    delta_mean = r["mean"] - full_mean
    delta_sign = "+" if delta_mean >= 0 else ""
    delta_str = f"{delta_sign}{delta_mean:.4f}" if name != "Full feature set (d = 10)" else "—"
    
    mean_sd_str = f"{r['mean']:.4f} ± {r['std']:.4f}"
    print(f"{name:<35} | {r['seed_42']:<16.4f} | {mean_sd_str:<22} | {delta_str:<16}")

# Save JSON
out_json = BACKEND / "scratch" / "table7_ablation_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved raw results to: {out_json}")
