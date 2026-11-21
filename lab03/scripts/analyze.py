"""CSV -> статистика, ускорение, эффективность, графики и фрагмент отчёта."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--source", choices=["auto", "local", "cluster"], default="auto",
                        help="origin of measurements; auto uses metadata if available")
    args = parser.parse_args()
    directory = args.csv.resolve().parent
    groups = defaultdict(list)
    with args.csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n, p, nodes = (int(row[k]) for k in ("n", "p", "nodes"))
            total, compute = (float(row[k]) for k in ("parallel_s", "compute_s"))
            if min(n, p, nodes) <= 0 or not (math.isfinite(total) and math.isfinite(compute)) or total <= 0 or compute < 0:
                raise ValueError("invalid measurement in CSV")
            groups[n, p].append((total, compute, nodes))
    if not groups:
        parser.error("CSV has no measurements")
    summary = {}
    for (n, p), rows in sorted(groups.items()):
        times = [r[0] for r in rows]
        nodes = sorted(set(r[2] for r in rows))
        if len(nodes) != 1:
            raise ValueError(f"N={n}, P={p}: do not mix runs with different node counts")
        summary[n, p] = dict(n=n, p=p, nodes=nodes[0], repeats=len(times),
            min_s=min(times), max_s=max(times), median_s=stats.median(times), mean_s=stats.mean(times),
            std_s=stats.stdev(times) if len(times) > 1 else 0.0,
            compute_median_s=stats.median(r[1] for r in rows))
    missing = []
    for (n, p), s in summary.items():
        baseline = summary.get((n, 1))
        s["speedup"] = baseline["median_s"] / s["median_s"] if baseline else math.nan
        s["efficiency"] = s["speedup"] / p
        s["gflops"] = 2.0 * n ** 3 / s["median_s"] / 1e9
        if not baseline:
            missing.append(n)
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(summary.values())).keys()))
        writer.writeheader()
        writer.writerows(summary.values())

    plt.rcParams.update({"figure.dpi": 180, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    sizes = sorted({n for n, p in groups})
    procs = sorted({p for n, p in groups})
    plots = []

    def save(name, caption, xlabel, ylabel, curves, ideal=False, percent=False, loglog=False):
        fig, ax = plt.subplots(figsize=(7.3, 4.7))
        for label, points in curves:
            points = [q for q in points if math.isfinite(q[1])]
            if points:
                if len(points[0]) == 4:
                    ax.errorbar([q[0] for q in points], [q[1] for q in points],
                                yerr=[[q[1] - q[2] for q in points], [q[3] - q[1] for q in points]],
                                fmt="o-", capsize=3, label=label, markersize=4, elinewidth=1)
                else:
                    ax.plot([q[0] for q in points], [q[1] for q in points], "o-", label=label, markersize=4)
        if ideal:
            ax.plot(procs, procs, "--", color="0.5", label="Ideal S(P)=P")
        if percent:
            ax.axhline(100, color="0.5", linestyle="--", label="100%")
        if ax.lines:
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, "P=1 baseline required", transform=ax.transAxes, ha="center")
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")
            from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter
            ax.xaxis.set_major_locator(FixedLocator(sizes))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(directory / name)
        fig.savefig((directory / name).with_suffix(".svg"))
        plt.close(fig)
        plots.append((caption, name))

    save("time_vs_size.png", "Время от размера матриц", "Matrix size N", "Parallel time, s (median)",
         [(f"P={p}", [(n, summary[n, p]["median_s"], summary[n, p]["min_s"], summary[n, p]["max_s"])
                      for n in sizes if (n, p) in summary]) for p in procs])
    save("time_vs_size_log.png", "Время от размера матриц в логарифмических шкалах", "Matrix size N (log scale)",
         "Parallel time, s (log scale)",
         [(f"P={p}", [(n, summary[n, p]["median_s"], summary[n, p]["min_s"], summary[n, p]["max_s"])
                      for n in sizes if (n, p) in summary]) for p in procs], loglog=True)
    save("gflops_vs_size.png", "Производительность от размера матриц", "Matrix size N", "GFLOPS (parallel median)",
         [(f"P={p}", [(n, summary[n, p]["gflops"]) for n in sizes if (n, p) in summary]) for p in procs])
    save("parallel_vs_compute.png", "Полное и вычислительное время", "Matrix size N", "Time, s (median of rank maxima)",
         [(f"P={p}, {label}", [(n, summary[n, p][key]) for n in sizes if (n, p) in summary])
          for p in procs for label, key in [("parallel", "median_s"), ("compute", "compute_median_s")]])
    if len(procs) > 1:
        save("time_vs_processes.png", "Время от числа процессов", "MPI processes P", "Parallel time, s (median)",
             [(f"N={n}", [(p, summary[n, p]["median_s"], summary[n, p]["min_s"], summary[n, p]["max_s"])
                          for p in procs if (n, p) in summary]) for n in sizes])
    valid_sizes = [n for n in sizes if (n, 1) in summary]
    scaling_sizes = [n for n in valid_sizes if any((n, p) in summary for p in procs if p > 1)]
    if scaling_sizes:
        save("speedup.png", "Ускорение", "MPI processes P", "Speedup S(P) = T(1) / T(P)",
             [(f"N={n}", [(p, summary[n, p]["speedup"]) for p in procs if (n, p) in summary]) for n in scaling_sizes], ideal=True)
        save("efficiency.png", "Эффективность", "MPI processes P", "Efficiency E(P), %",
             [(f"N={n}", [(p, summary[n, p]["efficiency"] * 100) for p in procs if (n, p) in summary]) for n in scaling_sizes], percent=True)

if __name__ == "__main__":
    main()
