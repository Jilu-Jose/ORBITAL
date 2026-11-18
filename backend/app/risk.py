"""Risk scoring for the /api/predict endpoint.

Two engines:

* RULES  - deterministic demo proxy (README section 7). Not physics. Used when
           no trained model exists or USE_SPARK_MODEL is not set.
* SPARK  - the trained Spark pipeline saved by backend/spark/train.py.
           Enabled with USE_SPARK_MODEL=1. Falls back to RULES (with a note)
           if the model cannot be loaded.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
from pathlib import Path

# Ensure PySpark uses the current Python environment and its bundled matching JARs
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BACKEND_DIR / "models" / "gbt_pipeline"
META_PATH = BACKEND_DIR / "models" / "model_meta.json"

HIGH_PC = 1e-4
MOD_PC = 2.5e-5
PC_MIN, PC_MAX = 1e-6, 4.2e-3

# Model decision thresholds on P(PC > 1e-4)
HIGH_PROB = 0.5
MOD_PROB = 0.2

# ---------------------------------------------------------------------------
# ADAPT ME: map the four UI sliders onto columns of FEATURES in spark/train.py.
# A slider is only used if its target column is actually in the trained model's
# feature list (see models/model_meta.json). Every other feature is filled with
# its training-set median (also stored in model_meta.json).
#
#   slider     -> (feature column, multiplier applied to the slider value)
# ---------------------------------------------------------------------------
FEATURE_MAP = {
    "miss_km": ("miss_km", 1.0),      # only present if INCLUDE_MISS_DISTANCE=True in train.py
    "rel_vel": ("v_rel_est", 1.0),    # km/s
    "inc_div": ("inc_diff", 1.0),     # degrees
    "bstar": ("bstar_diff", 1e-4),    # UI shows BSTAR in units of 1e-4
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _score_from_pc(pc: float) -> float:
    return _clamp((math.log10(pc) + 6.0) / 3.0, 0.0, 1.0)


def rule_pc(miss_km: float, rel_vel: float, inc_div: float, bstar: float) -> float:
    pc = math.exp(-1.8 * miss_km) * (rel_vel / 14.0) * 1e-3
    pc *= 1 + min(inc_div, 15) / 30
    pc *= 1 + min(bstar, 5) / 10
    return _clamp(pc, PC_MIN, PC_MAX)


def _text_for(level: str, pc: float | None, prob: float | None) -> tuple[str, str, str, str]:
    """Returns (level_text, chip, action, key)."""
    metric = f"Pc {pc:.2e}" if pc is not None else f"P(high risk) {prob * 100:.0f}%"
    if level == "HIGH":
        return (
            "HIGH CONJUNCTION RISK",
            "CRITICAL EVENT",
            f"RECOMMENDATION: PREPARE MANEUVER ADVISORY. {metric} is above the 1.0e-4 "
            "action threshold. Assess an in-track or radial avoidance burn at least 24 h "
            "before TCA and request updated tracking.",
            "HIGH",
        )
    if level == "MODERATE":
        return (
            "MODERATE CONJUNCTION RISK",
            "ELEVATED",
            f"RECOMMENDATION: MONITOR. {metric}. Request additional tracking and "
            "re-screen as TCA approaches.",
            "MODERATE",
        )
    return (
        "LOW CONJUNCTION RISK",
        "NOMINAL",
        f"RECOMMENDATION: NO ACTION. {metric}. Continue routine screening.",
        "LOW",
    )


def rule_risk(miss_km: float, rel_vel: float, inc_div: float, bstar: float) -> dict:
    pc = rule_pc(miss_km, rel_vel, inc_div, bstar)
    level = "HIGH" if pc >= HIGH_PC else "MODERATE" if pc >= MOD_PC else "LOW"
    level_text, chip, action, _ = _text_for(level, pc, None)
    return {
        "pc": pc,
        "prob": None,
        "score": round(_score_from_pc(pc), 4),
        "level": level_text,
        "chip": chip,
        "action": action,
        "high_risk": level == "HIGH",
        "miss_m": round(miss_km * 1000),
        "rel_vel": rel_vel,
        "engine": "RULES",
    }


# ---------------------------------------------------------------------------
# Trained model hook
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_model_cache: dict = {}


def model_available() -> bool:
    return MODEL_DIR.exists() and META_PATH.exists()


def _load_model():
    with _lock:
        if "model" in _model_cache:
            return _model_cache["spark"], _model_cache["model"], _model_cache["meta"]
        from pyspark.ml import PipelineModel
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.master("local[2]")
            .appName("orbital-serve")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.memory", "1g")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        model = PipelineModel.load(str(MODEL_DIR))
        meta = json.loads(META_PATH.read_text())
        _model_cache.update(spark=spark, model=model, meta=meta)
        return spark, model, meta


def _short_name(name: str) -> str:
    n = name.lower()
    return "LR" if "logistic" in n else "RF" if "random" in n else "GBT" if "gradient" in n else (name.upper() or "MODEL")


def model_risk(miss_km: float, rel_vel: float, inc_div: float, bstar: float) -> dict:
    spark, model, meta = _load_model()
    features: list[str] = meta["features"]
    row = {f: float(meta["defaults"].get(f, 0.0)) for f in features}
    slider_values = {"miss_km": miss_km, "rel_vel": rel_vel, "inc_div": inc_div, "bstar": bstar}
    used, ignored = [], []
    for slider, (col, mult) in FEATURE_MAP.items():
        if col in row:
            row[col] = slider_values[slider] * mult
            used.append(slider)
        else:
            ignored.append(slider)

    df = spark.createDataFrame([tuple(row[f] for f in features)], features)
    out = model.transform(df).select("probability").collect()[0][0]
    prob = float(out[1])

    level = "HIGH" if prob >= HIGH_PROB else "MODERATE" if prob >= MOD_PROB else "LOW"
    level_text, chip, action, _ = _text_for(level, None, prob)
    res = {
        # The classifier predicts P(PC > 1e-4), not a Pc value itself.
        "pc": None,
        "prob": round(prob, 4),
        "score": round(_clamp(prob, 0.0, 1.0), 4),
        "level": level_text,
        "chip": chip,
        "action": action,
        "high_risk": level == "HIGH",
        "miss_m": round(miss_km * 1000),
        "rel_vel": rel_vel,
        "engine": "SPARK " + _short_name(meta.get("model_name", "")),
    }
    if ignored:
        res["note"] = "Not model features (ignored): " + ", ".join(ignored)
    return res


def predict(miss_km: float, rel_vel: float, inc_div: float, bstar: float) -> dict:
    if os.environ.get("USE_SPARK_MODEL") == "1":
        if model_available():
            try:
                return model_risk(miss_km, rel_vel, inc_div, bstar)
            except Exception as exc:  # pragma: no cover - depends on local Spark
                res = rule_risk(miss_km, rel_vel, inc_div, bstar)
                res["note"] = f"Spark model failed ({type(exc).__name__}); using rules engine."
                return res
        res = rule_risk(miss_km, rel_vel, inc_div, bstar)
        res["note"] = "USE_SPARK_MODEL=1 but no trained model found; using rules engine."
        return res
    return rule_risk(miss_km, rel_vel, inc_div, bstar)
