#!/usr/bin/env python3
"""Generate a SYNTHETIC dataset in Space-Track format, only to smoke-test train.py.

    python backend/spark/make_sample_data.py            # writes backend/data/sample/
    python backend/spark/train.py --data-dir backend/data/sample --out-dir backend/data/sample_out

It writes to data/sample (never data/raw). Do NOT report models trained on this as results.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", type=int, default=4000)
    ap.add_argument("--cdms", type=int, default=60000)
    ap.add_argument("--out", type=Path, default=BACKEND / "data" / "sample")
    args = ap.parse_args()
    rng = random.Random(7)
    args.out.mkdir(parents=True, exist_ok=True)

    gp, satcat = [], []
    for i in range(args.objects):
        norad = 10000 + i
        r = rng.random()
        if r < 0.72:      # LEO
            alt, inc, ecc = rng.choice([rng.gauss(550, 50), rng.gauss(800, 70), rng.gauss(1000, 100)]), rng.choice([53, 70, 82, 98.2, 87]) + rng.gauss(0, 1), abs(rng.gauss(0.002, 0.002))
        elif r < 0.80:    # MEO
            alt, inc, ecc = rng.gauss(20200, 1500), rng.gauss(56, 4), abs(rng.gauss(0.003, 0.002))
        elif r < 0.92:    # GEO
            alt, inc, ecc = 35786 + rng.gauss(0, 30), abs(rng.gauss(4, 3)), abs(rng.gauss(0.0005, 0.0003))
        else:             # HEO
            alt, inc, ecc = rng.gauss(26000, 3000), rng.choice([63.4, 20, 10]) + rng.gauss(0, 1), rng.gauss(0.72, 0.03)
        alt = max(alt, 250)
        a = 6378.137 + alt
        peri, apo = a * (1 - ecc) - 6378.137, a * (1 + ecc) - 6378.137
        typ = rng.choices(["PAYLOAD", "ROCKET BODY", "DEBRIS"], weights=[0.3, 0.1, 0.6])[0]
        gp.append({
            "NORAD_CAT_ID": str(norad), "OBJECT_NAME": f"SYN-{norad}", "OBJECT_TYPE": typ,
            "INCLINATION": f"{abs(inc):.4f}", "ECCENTRICITY": f"{ecc:.7f}",
            "RA_OF_ASC_NODE": f"{rng.uniform(0, 360):.4f}", "ARG_OF_PERICENTER": f"{rng.uniform(0, 360):.4f}",
            "MEAN_ANOMALY": f"{rng.uniform(0, 360):.4f}", "BSTAR": f"{abs(rng.gauss(0.0002, 0.0004)):.8f}" if alt < 2000 else "0.0",
            "SEMIMAJOR_AXIS": f"{a:.3f}", "APOAPSIS": f"{apo:.3f}", "PERIAPSIS": f"{peri:.3f}",
            "EPOCH": "2026-11-10T00:00:00",
        })
        satcat.append({"NORAD_CAT_ID": str(norad), "OBJECT_TYPE": typ, "COUNTRY": rng.choice(["US", "PRC", "CIS", "UK", "IND"]), "LAUNCH": "2015-01-01"})

    leo = [g for g in gp if float(g["PERIAPSIS"]) < 2000]
    by_id = {g["NORAD_CAT_ID"]: g for g in gp}
    t0 = dt.datetime(2026, 1, 1)
    cdm = []
    for k in range(args.cdms):
        a_, b_ = rng.sample(leo, 2) if rng.random() < 0.9 else rng.sample(gp, 2)
        ga, gb = by_id[a_["NORAD_CAT_ID"]], by_id[b_["NORAD_CAT_ID"]]
        alt_d = abs((float(ga["APOAPSIS"]) + float(ga["PERIAPSIS"])) / 2 - (float(gb["APOAPSIS"]) + float(gb["PERIAPSIS"])) / 2)
        inc_d = abs(float(ga["INCLINATION"]) - float(gb["INCLINATION"]))
        # Synthetic ground truth: closer shells + larger inclination difference => higher risk (plus noise)
        miss_km = max(0.01, rng.expovariate(1 / (0.5 + alt_d / 40)))
        logpc = -5.2 - 0.9 * math.log10(1 + alt_d) + 0.05 * inc_d - 0.8 * miss_km + rng.gauss(0, 0.8)
        pc = 10 ** logpc
        created = t0 + dt.timedelta(minutes=k * (330 * 24 * 60 / args.cdms))
        tca = created + dt.timedelta(days=rng.uniform(0.5, 6))
        cdm.append({
            "CDM_ID": str(k), "CREATED": created.strftime("%Y-%m-%d %H:%M:%S"), "TCA": tca.strftime("%Y-%m-%d %H:%M:%S"),
            "MIN_RNG": f"{miss_km * 1000:.1f}", "PC": f"{pc:.6e}",
            "SAT_1_ID": ga["NORAD_CAT_ID"], "SAT_1_NAME": ga["OBJECT_NAME"], "SAT1_OBJECT_TYPE": ga["OBJECT_TYPE"],
            "SAT_2_ID": gb["NORAD_CAT_ID"], "SAT_2_NAME": gb["OBJECT_NAME"], "SAT2_OBJECT_TYPE": gb["OBJECT_TYPE"],
        })

    (args.out / "gp.json").write_text(json.dumps(gp))
    (args.out / "satcat.json").write_text(json.dumps(satcat))
    (args.out / "cdm.json").write_text(json.dumps(cdm))
    pos = sum(float(c["PC"]) > 1e-4 for c in cdm)
    print(f"Wrote {args.out}: {len(gp)} objects, {len(cdm)} CDMs, {pos} positives ({pos / len(cdm):.1%})")


if __name__ == "__main__":
    main()
