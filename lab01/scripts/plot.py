"""Графики по results/bench.csv: время (log-log, с опорной кривой ~N^3) и GFLOPS."""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

with open(ROOT / "results" / "bench.csv", newline="") as f:
    rows = list(csv.DictReader(f))
n = np.array([int(r["n"]) for r in rows])
t_min = np.array([float(r["t_min_ms"]) for r in rows])
t_mean = np.array([float(r["t_mean_ms"]) for r in rows])
gflops = np.array([float(r["gflops"]) for r in rows])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

nn = np.linspace(n.min(), n.max(), 100)
k = t_min[-1] / n[-1] ** 3
ax1.loglog(n, t_min, "o-", label="t_min")
ax1.loglog(n, t_mean, "s--", label="t_mean")
ax1.loglog(nn, k * nn ** 3, ":", color="gray", label="опорная кривая ~ N^3")
ax1.set_xlabel("N")
ax1.set_ylabel("время, мс")
ax1.set_title("Время умножения от размера матрицы")
ax1.grid(True, which="both", alpha=0.4)
ax1.legend()

ax2.plot(n, gflops, "o-")
ax2.set_xlabel("N")
ax2.set_ylabel("GFLOPS")
ax2.set_title("Достигнутая производительность")
ax2.grid(True, alpha=0.4)

fig.tight_layout()
out = ROOT / "results" / "bench.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")