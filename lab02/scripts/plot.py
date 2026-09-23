"""Графики lab02 по results/bench.csv: speedup и efficiency от числа потоков."""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

with open(ROOT / "results" / "bench.csv", newline="") as f:
    rows = list(csv.DictReader(f))

sizes = sorted({int(r["n"]) for r in rows})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
for n in sizes:
    sel = sorted((r for r in rows if int(r["n"]) == n), key=lambda r: int(r["threads"]))
    t = np.array([int(r["threads"]) for r in sel])
    tmin = np.array([float(r["t_min_ms"]) for r in sel])
    base = tmin[t == 1][0]              # опора: 1 поток
    speedup = base / tmin
    ax1.plot(t, speedup, "o-", label=f"N={n}")
    ax2.plot(t, speedup / t, "s-", label=f"N={n}")

tmax = max(int(r["threads"]) for r in rows)
ax1.plot([1, tmax], [1, tmax], ":", color="gray", label="идеально линейно")
ax1.set_xlabel("потоки")
ax1.set_ylabel("speedup = t_min(1) / t_min(T)")
ax1.set_title("Масштабируемость")
ax1.grid(alpha=0.4)
ax1.legend()
ax2.set_xlabel("потоки")
ax2.set_ylabel("efficiency = speedup / T")
ax2.set_title("Эффективность")
ax2.grid(alpha=0.4)
ax2.legend()
fig.tight_layout()
out = ROOT / "results" / "bench.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")