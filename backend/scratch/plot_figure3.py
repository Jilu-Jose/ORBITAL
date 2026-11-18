"""
Figure 3 generator for ORBITAL research paper.
Produces:
  - Figure 3a: Silhouette coefficient vs K  (line chart, K=4..9)
  - Figure 3b: Inclination vs log-altitude scatter coloured by K=4 cluster

Runs a lightweight KMeans sweep on the same gp.json used for training.
No full pipeline is re-run.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

# Fix Spark env (same as train.py)
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler

# ── Paths ──────────────────────────────────────────────────────────────────
BACKEND  = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND / "data" / "raw"
OUT_DIR  = Path(__file__).resolve().parent
FIG3A    = OUT_DIR / "figure3a_silhouette_vs_k.png"
FIG3B    = OUT_DIR / "figure3b_inc_logalt_scatter.png"
META     = BACKEND / "models" / "model_meta.json"

EARTH_R  = 6378.137
BEST_K   = 4

# ── Spark session (minimal) ────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .master("local[2]")
    .appName("figure3")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print("[fig3] Spark ready")

# ── Load & prep objects (mirror of train.py build_objects) ─────────────────
gp = spark.read.option("multiLine", True).json(str(DATA_DIR / "gp.json"))
from pyspark.sql import Window
# Dedupe on the raw column names BEFORE aliasing
w  = Window.partitionBy("NORAD_CAT_ID").orderBy(F.col("EPOCH").desc())

obj = (
    gp.withColumn("_rn", F.row_number().over(w))
    .filter("_rn = 1")
    .drop("_rn")
    .select(
        F.col("NORAD_CAT_ID").cast("int").alias("norad"),
        F.col("INCLINATION").cast("double").alias("inc"),
        F.col("ECCENTRICITY").cast("double").alias("ecc"),
        F.col("APOAPSIS").cast("double").alias("apo"),
        F.col("PERIAPSIS").cast("double").alias("peri"),
        F.col("SEMIMAJOR_AXIS").cast("double").alias("sma"),
    )
    .withColumn("alt", (F.col("apo") + F.col("peri")) / 2)
    .dropna(subset=["norad", "inc", "ecc", "alt"])
    .filter((F.col("alt") > 150) & (F.col("alt") < 60000) & (F.col("ecc") < 1.0))
    .withColumn("log_alt", F.log10("alt"))
)
print(f"[fig3] Objects loaded: {obj.count():,}")

# Build scaled features for clustering
asm    = VectorAssembler(inputCols=["log_alt", "inc", "ecc"], outputCol="raw")
scaler = StandardScaler(inputCol="raw", outputCol="scaled", withMean=True, withStd=True)
prep   = Pipeline(stages=[asm, scaler]).fit(obj)
vec    = prep.transform(obj).cache()

evaluator = ClusteringEvaluator(featuresCol="scaled", predictionCol="regime",
                                 metricName="silhouette")

# ── Silhouette sweep K=4..9 ────────────────────────────────────────────────
print("[fig3] Running silhouette sweep K=4..9 ...")
sil_by_k = {}
for kk in range(4, 10):
    m  = KMeans(k=kk, seed=42, featuresCol="scaled", predictionCol="regime").fit(vec)
    s  = evaluator.evaluate(m.transform(vec))
    sil_by_k[kk] = round(s, 4)
    print(f"  k={kk}  silhouette={s:.4f}")

# ── Save to model_meta.json ────────────────────────────────────────────────
if META.exists():
    with open(META) as f:
        meta = json.load(f)
    meta["cluster"]["silhouette_by_k"] = sil_by_k
    with open(META, "w") as f:
        json.dump(meta, f, indent=2)
    print("[fig3] model_meta.json updated with silhouette_by_k")

# ── Cluster assignments for scatter (best K=4) ─────────────────────────────
print(f"[fig3] Computing cluster assignments for K={BEST_K} ...")
km_best  = KMeans(k=BEST_K, seed=42, featuresCol="scaled", predictionCol="regime").fit(vec)
clustered = (
    km_best.transform(vec)
    .select("inc", "log_alt", "regime")
    .sample(fraction=0.15, seed=42)   # ~5000 pts for a readable scatter
    .toPandas()
)
print(f"[fig3] Scatter sample: {len(clustered):,} points")
spark.stop()

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3a — Silhouette vs K
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 3a ...")

ks   = list(sil_by_k.keys())
sils = list(sil_by_k.values())
best_k_actual = max(sil_by_k, key=sil_by_k.get)

fig3a, ax = plt.subplots(figsize=(5.5, 3.8))

ax.plot(ks, sils, color="#1565C0", linewidth=2.2, marker="o",
        markersize=7, markerfacecolor="#1565C0", label="Silhouette coefficient")

# Highlight best K
best_s = sil_by_k[best_k_actual]
ax.scatter([best_k_actual], [best_s],
           color="#E53935", zorder=6, s=100, marker="*",
           label=f"Best K={best_k_actual} (s={best_s:.4f})")

for k, s in zip(ks, sils):
    offset = (5, 6) if k != best_k_actual else (5, -14)
    ax.annotate(f"{s:.4f}", xy=(k, s), xytext=offset,
                textcoords="offset points", fontsize=8.0,
                color="#1565C0" if k != best_k_actual else "#E53935")

ax.set_xlabel("Number of clusters K", fontsize=12)
ax.set_ylabel("Mean silhouette coefficient", fontsize=12)
ax.set_title("(a) Silhouette coefficient vs K", fontsize=12,
             fontweight="bold", loc="left")
ax.set_xticks(ks)
ax.set_xlim(3.5, 9.5)
ax.set_ylim(min(sils) * 0.98, max(sils) * 1.02)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.35, color="gray")
ax.legend(fontsize=9, loc="lower right", framealpha=0.9, edgecolor="#cccccc")

fig3a.tight_layout()
fig3a.savefig(FIG3A, dpi=300, bbox_inches="tight")
plt.close(fig3a)
print(f"  Saved → {FIG3A}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3b — Inclination vs log-altitude scatter coloured by cluster
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 3b ...")

CLUSTER_COLORS = ["#1565C0", "#E53935", "#2E7D32", "#F57F17"]
CLUSTER_NAMES  = [
    "Cluster 0 — LEO polar",
    "Cluster 1 — LEO equatorial",
    "Cluster 2 — MEO/GEO",
    "Cluster 3 — HEO/Sun-sync",
]

fig3b, ax2 = plt.subplots(figsize=(6.0, 4.4))

for cid in sorted(clustered["regime"].unique()):
    sub = clustered[clustered["regime"] == cid]
    label = CLUSTER_NAMES[cid] if cid < len(CLUSTER_NAMES) else f"Cluster {cid}"
    ax2.scatter(sub["log_alt"], sub["inc"],
                c=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                s=4, alpha=0.45, linewidths=0, label=label)

# Altitude tick labels (convert log10(alt) → km)
log_ticks = [2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0, 4.5]
valid_ticks = [t for t in log_ticks
               if clustered["log_alt"].min() - 0.05 <= t <= clustered["log_alt"].max() + 0.05]
ax2.set_xticks(valid_ticks)
ax2.set_xticklabels([f"{10**t:,.0f}" for t in valid_ticks], fontsize=9)

ax2.set_xlabel("Altitude (km, log scale)", fontsize=12)
ax2.set_ylabel("Inclination (°)", fontsize=12)
ax2.set_title("(b) Inclination vs altitude coloured by K-means cluster",
              fontsize=11, fontweight="bold", loc="left")

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.tick_params(labelsize=10)
ax2.grid(linestyle="--", alpha=0.25, color="gray")

legend = ax2.legend(fontsize=8.5, loc="upper left",
                    framealpha=0.92, edgecolor="#cccccc",
                    markerscale=3.0)

fig3b.tight_layout()
fig3b.savefig(FIG3B, dpi=300, bbox_inches="tight")
plt.close(fig3b)
print(f"  Saved → {FIG3B}")

print("\nDone. Both Figure 3 panels written.")
