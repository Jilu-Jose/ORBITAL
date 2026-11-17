"""Seeded synthetic data used when no processed data / trained model exists.

EVERYTHING here is a placeholder. The UI shows a DEMO DATA badge whenever
these values are served. Never report them as results.
"""
from __future__ import annotations

import math
import random

EARTH_R = 6371.0

_COUNTS = {"LEO": 380, "SSO": 110, "MEO": 70, "GEO": 100, "HEO": 40}
_TYPE_WEIGHTS = {
    "LEO": (0.30, 0.15, 0.55),
    "SSO": (0.35, 0.15, 0.50),
    "MEO": (0.75, 0.10, 0.15),
    "GEO": (0.80, 0.05, 0.15),
    "HEO": (0.35, 0.25, 0.40),
}
_TYPES = ("PAYLOAD", "ROCKET BODY", "DEBRIS")
_PAYLOAD_NAMES = ["STARLINK", "ONEWEB", "COSMOS", "IRIDIUM", "NOAA", "SENTINEL", "GLOBALSTAR",
                  "INTELSAT", "GOES", "METEOSAT", "GPS BIIR", "GALILEO", "BEIDOU", "MOLNIYA"]
_RB_NAMES = ["CZ-4B R/B", "SL-8 R/B", "FALCON 9 R/B", "ARIANE 5 R/B", "PSLV R/B", "ATLAS 5 CENTAUR R/B"]
_DEB_NAMES = ["COSMOS 2251 DEB", "FENGYUN 1C DEB", "IRIDIUM 33 DEB", "CZ-3 DEB", "SL-16 DEB", "STARLINK DEB"]


def _name(rng: random.Random, typ: str, i: int) -> str:
    if typ == "PAYLOAD":
        return f"{rng.choice(_PAYLOAD_NAMES)}-{rng.randint(1, 5999)}"
    if typ == "ROCKET BODY":
        return rng.choice(_RB_NAMES)
    return rng.choice(_DEB_NAMES)


def _shape(rng: random.Random, regime: str) -> tuple[float, float, float]:
    """Returns (alt_km, inc_deg, ecc) for a regime."""
    if regime == "LEO":
        alt = rng.choice([rng.gauss(550, 60), rng.gauss(800, 90), rng.gauss(1100, 120)])
        alt = min(max(alt, 300), 1800)
        inc = rng.choice([rng.gauss(53, 1.5), rng.gauss(70, 2), rng.gauss(82, 1.5),
                          rng.gauss(87, 1), rng.gauss(28.5, 1)])
        return alt, min(max(inc, 0.5), 110), abs(rng.gauss(0.001, 0.0008))
    if regime == "SSO":
        return min(max(rng.gauss(700, 80), 500), 900), rng.gauss(98.2, 0.5), abs(rng.gauss(0.001, 0.0006))
    if regime == "MEO":
        alt = rng.choice([rng.gauss(19130, 120), rng.gauss(20180, 100), rng.gauss(23220, 120)])
        return alt, rng.choice([rng.gauss(55, 1.2), rng.gauss(56, 0.8), rng.gauss(64.8, 0.6)]), abs(rng.gauss(0.002, 0.001))
    if regime == "GEO":
        return 35786 + rng.gauss(0, 25), abs(rng.gauss(4, 3.5)), abs(rng.gauss(0.0004, 0.0003))
    # HEO: mix of Molniya (inc ~63) and GTO (inc ~ 7-28)
    if rng.random() < 0.5:
        return rng.gauss(26500, 400), rng.gauss(63.4, 0.5), rng.gauss(0.72, 0.02)
    return rng.gauss(18500, 1500), rng.uniform(6, 28), rng.gauss(0.72, 0.03)


def get_objects() -> list[dict]:
    rng = random.Random(42)
    objs: list[dict] = [{
        "norad": 25544, "name": "ISS (ZARYA)", "type": "PAYLOAD", "alt": 420.0, "inc": 51.64,
        "ecc": 0.0006, "regime": "LEO", "raan": 1.2, "phase": 0.4,
        "spd": round(0.012 * (7000 / (EARTH_R + 420)) ** 1.5, 6),
    }]
    for regime, n in _COUNTS.items():
        w = _TYPE_WEIGHTS[regime]
        for i in range(n - (1 if regime == "LEO" else 0)):
            typ = rng.choices(_TYPES, weights=w)[0]
            alt, inc, ecc = _shape(rng, regime)
            objs.append({
                "norad": rng.randint(10000, 59999),
                "name": _name(rng, typ, i),
                "type": typ,
                "alt": round(alt, 1),
                "inc": round(inc, 2),
                "ecc": round(ecc, 4),
                "regime": regime,
                "raan": round(rng.uniform(0, 2 * math.pi), 4),
                "phase": round(rng.uniform(0, 2 * math.pi), 4),
                "spd": round(0.012 * (7000 / (EARTH_R + alt)) ** 1.5, 6),
            })
    rng.shuffle(objs)
    return objs


def get_stats() -> dict:
    return {"tracked": 48905, "debris": 34661, "high_risk": 115, "pr_auc": 0.942, "demo": True}


def get_metrics() -> dict:
    return {
        "demo": True,
        "models": [
            {"name": "Logistic Regression (baseline)", "pr_auc": 0.731, "recall": 0.688, "f1": 0.724, "train_s": 54.2, "selected": False},
            {"name": "Random Forest", "pr_auc": 0.884, "recall": 0.831, "f1": 0.864, "train_s": 142.0, "selected": False},
            {"name": "Gradient Boosted Trees", "pr_auc": 0.942, "recall": 0.912, "f1": 0.931, "train_s": 198.0, "selected": True},
        ],
        "importance": [
            ["v_rel_est", 0.31], ["inc_diff", 0.22], ["alt_diff", 0.14], ["bstar_diff", 0.11],
            ["shell_density_max", 0.08], ["ecc_diff", 0.06], ["raan_diff", 0.04], ["alt_mean", 0.04],
        ],
        "confusion": {"tn": 234812, "fp": 311, "fn": 415, "tp": 4200},
        "scalability": {"cores": [1, 2, 4, 8], "spark_s": [320, 170, 92, 51], "sklearn_s": [300, 300, 300, 300]},
    }
