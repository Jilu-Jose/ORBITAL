#!/usr/bin/env python3
"""ORBITAL training pipeline (PySpark MLlib). YOU run this.

    python backend/spark/train.py                          # uses backend/data/raw/{gp,satcat,cdm}.json
    python backend/spark/train.py --tune-k                 # pick KMeans k by silhouette (4..9)
    python backend/spark/train.py --drop-features regime_1,regime_2,shell_density_max   # ablation

Outputs (under --out-dir, default backend/):
    models/gbt_pipeline/            best Spark PipelineModel (by PR-AUC)
    models/model_meta.json          feature list + training medians (used by app/risk.py)
    models/metrics.json             models table, importance, confusion matrix
    data/processed/objects.json     sampled objects for the globe/scatter
    data/processed/stats.json       header stats
    data/processed/pairs.parquet    feature table (used by benchmark.py)

Requires Java 11 or 17 and `pip install -r backend/requirements-train.txt`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Ensure PySpark uses the current Python environment and its bundled matching JARs
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

BACKEND = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- config
EARTH_RADIUS_KM = 6378.137
MU_KM3_S2 = 398600.4418
HIGH_PC_THRESHOLD = 1e-4          # label = 1 if PC > this
SHELL_KM = 50                      # altitude shell width

# README: "Never use PC or MIN_RNG as features (leakage)". Leave False for a clean model.
# If True, miss_km (from MIN_RNG) becomes a feature and the predictor's miss-distance
# slider will drive the model; expect much higher scores that are partly circular.
INCLUDE_MISS_DISTANCE = False
# ADAPT ME: check the unit of MIN_RNG in your cdm.json. Space-Track's public CDM
# MIN_RNG is in metres in the data I have seen; set to 1.0 if yours is already km.
MIN_RNG_TO_KM = 0.001

FEATURES = [
    "inc_diff", "alt_diff", "ecc_diff", "bstar_diff", "raan_diff",
    "v_rel_est", "alt_mean", "shell_density_max", "regime_1", "regime_2",
]
if INCLUDE_MISS_DISTANCE:
    FEATURES.append("miss_km")


# --------------------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(f"[orbital] {msg}", flush=True)


def num(col: str):
    return F.col(col).cast("double")


def read_json(spark: SparkSession, path: Path):
    if not path.exists():
        raise SystemExit(f"Missing {path}. See README section 4 (where to put the dataset).")
    return spark.read.option("multiLine", True).json(str(path))


def regime_name_col():
    """Rule-based regime label for the UI (the KMeans id is a separate ML feature)."""
    return (
        F.when(F.col("ecc") > 0.25, "HEO")
        .when((F.col("alt") > 35000) & (F.col("alt") < 36600) & (F.col("inc") < 20), "GEO")
        .when(F.col("alt") >= 2000, "MEO")
        .when((F.col("inc") >= 96) & (F.col("inc") <= 102), "SSO")
        .otherwise("LEO")
    )


# --------------------------------------------------------------------------- stage 1-3: objects
def build_objects(spark, data_dir: Path, k: int, tune_k: bool):
    log("loading gp.json / satcat.json ...")
    gp = read_json(spark, data_dir / "gp.json")
    w = Window.partitionBy("norad").orderBy(F.col("EPOCH").desc())
    obj = (
        gp.select(
            F.col("NORAD_CAT_ID").cast("int").alias("norad"),
            F.col("OBJECT_NAME").alias("name"),
            F.col("OBJECT_TYPE").alias("gp_type"),
            num("INCLINATION").alias("inc"),
            num("ECCENTRICITY").alias("ecc"),
            num("RA_OF_ASC_NODE").alias("raan_deg"),
            num("MEAN_ANOMALY").alias("ma_deg"),
            num("BSTAR").alias("bstar"),
            num("APOAPSIS").alias("apo"),
            num("PERIAPSIS").alias("peri"),
            num("SEMIMAJOR_AXIS").alias("sma"),
            F.col("EPOCH"),
        )
        .withColumn("_rn", F.row_number().over(w))
        .filter("_rn = 1")
        .drop("_rn", "EPOCH")
    )

    sat_path = data_dir / "satcat.json"
    if sat_path.exists():
        sat = (
            read_json(spark, sat_path)
            .select(
                F.col("NORAD_CAT_ID").cast("int").alias("norad"),
                F.col("OBJECT_TYPE").alias("sat_type"),
                F.col("COUNTRY").alias("country"),
            )
            .dropDuplicates(["norad"])
        )
        obj = obj.join(sat, "norad", "left")
    else:
        log("satcat.json not found; using gp OBJECT_TYPE only")
        obj = obj.withColumn("sat_type", F.lit(None).cast("string")).withColumn("country", F.lit(None).cast("string"))

    typ = F.upper(F.coalesce(F.col("sat_type"), F.col("gp_type")))
    obj = (
        obj.withColumn("type", F.when(typ.isin("PAYLOAD", "ROCKET BODY", "DEBRIS"), typ).otherwise("DEBRIS"))
        .withColumn("alt", F.coalesce((F.col("apo") + F.col("peri")) / 2, F.col("sma") - EARTH_RADIUS_KM))
        .dropna(subset=["norad", "inc", "ecc", "alt"])
        .filter((F.col("alt") > 150) & (F.col("alt") < 60000) & (F.col("ecc") < 1.0))
        .withColumn("bstar", F.coalesce(F.col("bstar"), F.lit(0.0)))
        .withColumn("shell", F.floor(F.col("alt") / SHELL_KM).cast("int"))
    )
    density = obj.groupBy("shell").agg(F.count("*").alias("shell_density"))
    obj = obj.join(density, "shell").withColumn("log_alt", F.log10("alt")).withColumn("regime_name", regime_name_col())

    log("clustering orbital regimes with KMeans ...")
    asm = VectorAssembler(inputCols=["log_alt", "inc", "ecc"], outputCol="raw")
    scaled = StandardScaler(inputCol="raw", outputCol="scaled", withMean=True, withStd=True)
    prep = Pipeline(stages=[asm, scaled]).fit(obj)
    vec = prep.transform(obj).cache()
    evaluator = ClusteringEvaluator(featuresCol="scaled", predictionCol="regime", metricName="silhouette")

    best_k, sil_by_k = k, {}
    if tune_k:
        best_s = -2.0
        for kk in range(4, 10):
            m = KMeans(k=kk, seed=42, featuresCol="scaled", predictionCol="regime").fit(vec)
            s = evaluator.evaluate(m.transform(vec))
            sil_by_k[kk] = round(s, 4)
            log(f"  k={kk} silhouette={s:.4f}")
            if s > best_s:
                best_k, best_s = kk, s
        log(f"best k by silhouette: {best_k}")
    km = KMeans(k=best_k, seed=42, featuresCol="scaled", predictionCol="regime").fit(vec)
    clustered = km.transform(vec).select(
        "norad", "name", "type", "inc", "ecc", "alt", "bstar", "raan_deg", "ma_deg",
        "shell", "shell_density", "regime", "regime_name",
    ).cache()
    log(f"objects: {clustered.count():,}; silhouette(k={best_k}) = {evaluator.evaluate(km.transform(vec)):.4f}")
    return clustered, {"k": best_k, "silhouette_by_k": sil_by_k}


# --------------------------------------------------------------------------- stage 4-5: pairs + labels
def build_pairs(spark, data_dir: Path, obj, dedupe: bool, pc_threshold: float = HIGH_PC_THRESHOLD):
    log("loading cdm.json ...")
    cdm = read_json(spark, data_dir / "cdm.json")
    cdm = (
        cdm.select(
            F.to_timestamp("CREATED").alias("created_ts"),
            F.to_timestamp("TCA").alias("tca"),
            num("PC").alias("pc"),
            (num("MIN_RNG") * F.lit(MIN_RNG_TO_KM)).alias("miss_km"),
            F.col("SAT_1_ID").cast("int").alias("a"),
            F.col("SAT_2_ID").cast("int").alias("b"),
        )
        .dropna(subset=["created_ts", "pc", "a", "b"])
        # order the pair so (A,B) and (B,A) are the same event
        .withColumn("sat1", F.least("a", "b"))
        .withColumn("sat2", F.greatest("a", "b"))
        .drop("a", "b")
    )
    if dedupe:
        # Many CDM updates exist per event. Keep the latest per (pair, TCA day) so the
        # same event is not counted many times or spread across train and test.
        w = Window.partitionBy("sat1", "sat2", F.to_date("tca")).orderBy(F.col("created_ts").desc())
        cdm = cdm.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")

    def side(i: int):
        return obj.select(
            F.col("norad").alias(f"sat{i}"),
            F.col("inc").alias(f"inc_{i}"), F.col("ecc").alias(f"ecc_{i}"), F.col("alt").alias(f"alt_{i}"),
            F.col("bstar").alias(f"bstar_{i}"), F.col("raan_deg").alias(f"raan_{i}"),
            F.col("shell_density").alias(f"dens_{i}"), F.col("regime").alias(f"regime_{i}_raw"),
        )

    p = cdm.join(side(1), "sat1").join(side(2), "sat2")

    v1 = F.sqrt(F.lit(MU_KM3_S2) / (F.lit(EARTH_RADIUS_KM) + F.col("alt_1")))
    v2 = F.sqrt(F.lit(MU_KM3_S2) / (F.lit(EARTH_RADIUS_KM) + F.col("alt_2")))
    gamma = F.radians(F.col("inc_diff"))
    raan_abs = F.abs(F.col("raan_1") - F.col("raan_2"))

    p = (
        p.withColumn("inc_diff", F.abs(F.col("inc_1") - F.col("inc_2")))
        .withColumn("alt_diff", F.abs(F.col("alt_1") - F.col("alt_2")))
        .withColumn("ecc_diff", F.abs(F.col("ecc_1") - F.col("ecc_2")))
        .withColumn("bstar_diff", F.abs(F.col("bstar_1") - F.col("bstar_2")))
        .withColumn("raan_diff", F.least(raan_abs, F.lit(360.0) - raan_abs))
        # Heuristic relative-speed estimate from the two orbital speeds and the plane angle.
        .withColumn("v_rel_est", F.sqrt(F.greatest(v1 * v1 + v2 * v2 - 2 * v1 * v2 * F.cos(gamma), F.lit(0.0))))
        .withColumn("alt_mean", (F.col("alt_1") + F.col("alt_2")) / 2)
        .withColumn("shell_density_max", F.greatest("dens_1", "dens_2"))
        .withColumn("regime_1", F.least("regime_1_raw", "regime_2_raw").cast("double"))
        .withColumn("regime_2", F.greatest("regime_1_raw", "regime_2_raw").cast("double"))
        # label is derived from PC only; PC is never a feature
        .withColumn("label", (F.col("pc") > F.lit(pc_threshold)).cast("double"))
    )
    keep = ["created_ts", "tca", "sat1", "sat2", "label", "miss_km"] + [f for f in dict.fromkeys(FEATURES) if f != "miss_km"]
    return p.select(*dict.fromkeys(keep))


# --------------------------------------------------------------------------- stage 6-8: train + eval
def confusion(pred_df) -> dict:
    cm = {(int(r["label"]), int(r["prediction"])): r["count"] for r in pred_df.groupBy("label", "prediction").count().collect()}
    return {"tn": cm.get((0, 0), 0), "fp": cm.get((0, 1), 0), "fn": cm.get((1, 0), 0), "tp": cm.get((1, 1), 0)}


def scores(pred_df, raw_col="rawPrediction") -> dict:
    pr = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol=raw_col, metricName="areaUnderPR").evaluate(pred_df)
    c = confusion(pred_df)
    prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"pr_auc": round(pr, 4), "recall": round(rec, 4), "f1": round(f1, 4), "precision": round(prec, 4), "confusion": c}


def importances(model, features):
    last = model.stages[-1]
    if hasattr(last, "featureImportances"):
        vals = list(last.featureImportances.toArray())
    else:  # logistic regression: |coef| on standardised features
        vals = [abs(x) for x in last.coefficients.toArray()]
        s = sum(vals) or 1.0
        vals = [v / s for v in vals]
    pairs = sorted(zip(features, vals), key=lambda t: -t[1])
    return [[n, round(float(v), 4)] for n, v in pairs[:10]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pc-threshold", type=float, default=2.5e-4, help="conjunction risk threshold for label=1 (default 2.5e-4)")
    ap.add_argument("--data-dir", type=Path, default=BACKEND / "data" / "raw")
    ap.add_argument("--out-dir", type=Path, default=BACKEND)
    ap.add_argument("--k", type=int, default=6, help="KMeans clusters (ignored with --tune-k)")
    ap.add_argument("--tune-k", action="store_true")
    ap.add_argument("--split-pct", type=float, default=0.8, help="time split quantile of CDM creation time")
    ap.add_argument("--no-dedupe", action="store_true", help="keep every CDM update instead of latest per event")
    ap.add_argument("--drop-features", default="", help="comma list for ablations, e.g. regime_1,regime_2")
    ap.add_argument("--export-n", type=int, default=2500, help="objects exported for the globe")
    ap.add_argument("--driver-memory", default="4g")
    ap.add_argument("--cores", default="*", help="local[N] cores")
    args = ap.parse_args()

    features = [f for f in FEATURES if f not in {x.strip() for x in args.drop_features.split(",") if x.strip()}]
    if not features:
        raise SystemExit("No features left after --drop-features.")
    log(f"features: {features}")
    log(f"PC threshold for label=1: {args.pc_threshold:.2e}")

    models_dir = args.out_dir / "models"
    processed_dir = args.out_dir / "data" / "processed"
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.master(f"local[{args.cores}]").appName("orbital-train")
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    obj, cluster_info = build_objects(spark, args.data_dir, args.k, args.tune_k)
    pairs = build_pairs(spark, args.data_dir, obj, dedupe=not args.no_dedupe, pc_threshold=args.pc_threshold).cache()
    n_pairs = pairs.count()
    if n_pairs == 0:
        raise SystemExit("No conjunction pairs after joining CDMs to objects. Check that cdm SAT_1_ID/SAT_2_ID match gp NORAD_CAT_ID.")
    log(f"labelled conjunction events: {n_pairs:,}")

    # time-based split (not random)
    pairs = pairs.withColumn("created_num", F.col("created_ts").cast("double"))
    cut = pairs.approxQuantile("created_num", [args.split_pct], 0.001)[0]
    train = pairs.filter(F.col("created_num") <= cut)
    test = pairs.filter(F.col("created_num") > cut)

    n_pos = train.filter("label = 1").count()
    n_neg = train.filter("label = 0").count()
    t_pos = test.filter("label = 1").count()
    log(f"train: {n_pos + n_neg:,} (pos {n_pos:,})  test: {test.count():,} (pos {t_pos:,})")
    if n_pos == 0 or n_neg == 0 or t_pos == 0:
        raise SystemExit("Need positive and negative examples in both train and test. "
                         "Fetch more CDM history (download_data.py --cdm-days) or lower HIGH_PC_THRESHOLD.")

    # median imputation from the training set
    meds = train.approxQuantile(features, [0.5], 0.01)
    defaults = {f: (m[0] if m else 0.0) for f, m in zip(features, meds)}
    train = train.fillna(defaults)
    test = test.fillna(defaults)
    pairs_out = pairs.fillna(defaults)

    total = n_pos + n_neg
    w_pos, w_neg = total / (2.0 * n_pos), total / (2.0 * n_neg)
    train = train.withColumn("w", F.when(F.col("label") == 1, w_pos).otherwise(w_neg)).cache()
    test = test.cache()

    assembler = VectorAssembler(inputCols=features, outputCol="features_raw", handleInvalid="keep")
    candidates = {
        "Logistic Regression (baseline)": [
            assembler,
            StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True),
            LogisticRegression(featuresCol="features", labelCol="label", weightCol="w", maxIter=100, regParam=0.01),
        ],
        "Random Forest": [
            assembler,
            RandomForestClassifier(featuresCol="features_raw", labelCol="label", weightCol="w", numTrees=100, maxDepth=6, seed=42),
        ],
        "Gradient Boosted Trees": [
            assembler,
            GBTClassifier(featuresCol="features_raw", labelCol="label", weightCol="w", maxIter=30, maxDepth=4, stepSize=0.1, seed=42),
        ],
    }

    results, fitted = [], {}
    for name, stages in candidates.items():
        log(f"training {name} ...")
        t0 = time.time()
        model = Pipeline(stages=stages).fit(train)
        secs = time.time() - t0
        sc = scores(model.transform(test))
        fitted[name] = (model, sc)
        results.append({"name": name, "pr_auc": sc["pr_auc"], "recall": sc["recall"], "f1": sc["f1"],
                        "train_s": round(secs, 1), "selected": False})
        log(f"  PR-AUC {sc['pr_auc']}  recall {sc['recall']}  F1 {sc['f1']}  ({secs:.0f}s)")

    # simple threshold-rule baseline on miss distance (uses MIN_RNG only as a baseline, not a feature)
    base_test = pairs.filter(F.col("created_num") > cut).withColumn("miss_f", F.coalesce("miss_km", F.lit(1e6)))
    rule = base_test.withColumn("rawPrediction", 1.0 / (1.0 + F.col("miss_f"))).withColumn("prediction", (F.col("miss_f") < 1.0).cast("double"))
    if base_test.filter("miss_km is not null").count() > 0:
        rs = scores(rule)
        results.insert(0, {"name": "Threshold rule (miss < 1 km)", "pr_auc": rs["pr_auc"], "recall": rs["recall"],
                           "f1": rs["f1"], "train_s": 0.0, "selected": False})

    ml_names = list(candidates)
    best_name = max(ml_names, key=lambda n: fitted[n][1]["pr_auc"])
    for r in results:
        r["selected"] = r["name"] == best_name
    best_model, best_sc = fitted[best_name]
    log(f"selected: {best_name} (PR-AUC {best_sc['pr_auc']})")

    # ---- save model + metadata
    model_path = models_dir / "gbt_pipeline"
    best_model.write().overwrite().save(str(model_path))
    (models_dir / "model_meta.json").write_text(json.dumps({
        "model_name": best_name, "features": features, "defaults": defaults,
        "label_threshold_pc": args.pc_threshold, "include_miss_distance": INCLUDE_MISS_DISTANCE,
        "cluster": cluster_info, "split_quantile": args.split_pct,
    }, indent=2))

    # ---- metrics.json (preserve a scalability block pasted from benchmark.py, if any)
    metrics_path = models_dir / "metrics.json"
    scal = None
    if metrics_path.exists():
        try:
            scal = json.loads(metrics_path.read_text()).get("scalability")
        except (json.JSONDecodeError, OSError):
            pass
    metrics = {
        "models": results,
        "importance": importances(best_model, features),
        "confusion": best_sc["confusion"],
    }
    if scal:
        metrics["scalability"] = scal
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # ---- processed outputs for the UI
    preds = best_model.transform(test.select(*features, "label"))
    high_risk = preds.filter("prediction = 1").count()
    n_obj = obj.count()
    n_debris = obj.filter("type = 'DEBRIS'").count()
    (processed_dir / "stats.json").write_text(json.dumps({
        "tracked": n_obj, "debris": n_debris, "high_risk": high_risk, "pr_auc": best_sc["pr_auc"],
    }))

    sample = (
        obj.orderBy(F.rand(42)).limit(args.export_n)
        .select(
            "norad", "name", "type", "alt", "inc", "ecc", F.col("regime_name").alias("regime"),
            F.radians("raan_deg").alias("raan"), F.radians("ma_deg").alias("phase"),
        )
        .collect()
    )
    objects_json = []
    for r in sample:
        alt = float(r["alt"])
        objects_json.append({
            "norad": int(r["norad"]), "name": r["name"] or f"OBJ {r['norad']}", "type": r["type"],
            "alt": round(alt, 1), "inc": round(float(r["inc"]), 2), "ecc": round(float(r["ecc"]), 4),
            "regime": r["regime"], "raan": round(float(r["raan"] or 0.0), 4), "phase": round(float(r["phase"] or 0.0), 4),
            "spd": round(0.012 * (7000 / (6371 + alt)) ** 1.5, 6),
        })
    (processed_dir / "objects.json").write_text(json.dumps(objects_json))

    pairs_out.select("created_ts", "label", *dict.fromkeys(features)).write.mode("overwrite").parquet(str(processed_dir / "pairs.parquet"))

    log("done.")
    log(f"  model    -> {model_path}")
    log(f"  metrics  -> {metrics_path}")
    log(f"  UI data  -> {processed_dir}/objects.json, stats.json")
    log("Restart the API; the DEMO DATA badge will disappear. For the predictor use: USE_SPARK_MODEL=1")
    spark.stop()


if __name__ == "__main__":
    main()
