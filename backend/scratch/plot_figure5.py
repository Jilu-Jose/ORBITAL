"""
Figure 5 generator for ORBITAL research paper.
Produces:
  - Figure 5: Line chart — Training runtime vs number of Spark cores
              Solid: Apache Spark  |  Dashed: single-node scikit-learn
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
METRICS    = ROOT / "models" / "metrics.json"
OUT_DIR    = Path(__file__).resolve().parent
FIG5_OUT   = OUT_DIR / "figure5_scalability.png"

# ── Load data ──────────────────────────────────────────────────────────────
with open(METRICS) as f:
    metrics = json.load(f)

sc = metrics["scalability"]
cores    = sc["cores"]          # [1, 2, 4, 6]
spark_s  = sc["spark_s"]        # [14.9, 9.7, 9.0, 8.2]
sklearn_s = sc["sklearn_s"]     # [0.4, 0.4, 0.4, 0.4]

# ── Figure ─────────────────────────────────────────────────────────────────
print("Generating Figure 5 …")

fig, ax = plt.subplots(figsize=(6.0, 4.2))

SPARK_COLOR   = "#1565C0"   # deep blue
SKLEARN_COLOR = "#B71C1C"   # deep red

# --- Spark (solid) --------------------------------------------------------
ax.plot(cores, spark_s,
        color=SPARK_COLOR, linewidth=2.2, linestyle="-",
        marker="o", markersize=7, markerfacecolor=SPARK_COLOR,
        label="Apache Spark (distributed)")

# Annotate each Spark point
for x, y in zip(cores, spark_s):
    ax.annotate(f"{y:.1f}s",
                xy=(x, y), xytext=(4, 6),
                textcoords="offset points",
                fontsize=8.5, color=SPARK_COLOR)

# --- scikit-learn (dashed) ------------------------------------------------
ax.plot(cores, sklearn_s,
        color=SKLEARN_COLOR, linewidth=2.0, linestyle="--",
        marker="s", markersize=6, markerfacecolor=SKLEARN_COLOR,
        label="scikit-learn (single-node)")

# Annotate sklearn (only once, at the right end — it's flat)
ax.annotate(f"{sklearn_s[0]:.1f}s",
            xy=(cores[-1], sklearn_s[-1]), xytext=(4, -14),
            textcoords="offset points",
            fontsize=8.5, color=SKLEARN_COLOR)

# --- Ideal-speedup reference (optional dashed grey) -----------------------
# Spark speedup from 1→6 cores: 14.9 / 8.2 ≈ 1.82×
ideal = [spark_s[0] / c for c in cores]
ax.plot(cores, ideal,
        color="#9E9E9E", linewidth=1.0, linestyle=":",
        label="Ideal linear speedup (reference)")

# --- Formatting ------------------------------------------------------------
ax.set_xlabel("Number of Spark cores", fontsize=12)
ax.set_ylabel("Training wall-clock time (s)", fontsize=12)
ax.set_title("Fig. 5  Training Runtime vs. Spark Cores", fontsize=12,
             fontweight="bold")

ax.set_xticks(cores)
ax.set_xlim(0.5, 6.5)
ax.set_ylim(0, max(spark_s) * 1.22)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.0f}s"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.35, color="gray")

ax.legend(fontsize=9, loc="upper right",
          framealpha=0.92, edgecolor="#cccccc")

# Small speedup annotation box
speedup = spark_s[0] / spark_s[-1]
ax.text(0.03, 0.08,
        f"Spark speedup 1→6 cores: ×{speedup:.2f}",
        transform=ax.transAxes, fontsize=8.5,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.8))

fig.tight_layout()
fig.savefig(FIG5_OUT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {FIG5_OUT}")
print("Done.")
