"""Построение графиков по CSV серии экспериментов CUDA."""

import argparse
import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#2166a3", "#b35013", "#007c68", "#865393"]
MARKERS = ["o", "s", "^", "D"]
REQUIRED_COLUMNS = {"n", "block_x", "block_y", "kernel_min_ms", "total_min_ms"}


@dataclass(frozen=True)
class Timing:
    kernel_ms: float
    total_ms: float


def read_timings(path: Path) -> dict[tuple[int, int], dict[int, Timing]]:
    groups: dict[tuple[int, int], dict[int, Timing]] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing CSV columns: {', '.join(sorted(missing))}")
        for row in reader:
            try:
                n = int(row["n"])
                block = (int(row["block_x"]), int(row["block_y"]))
                timing = Timing(float(row["kernel_min_ms"]), float(row["total_min_ms"]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid value at line {reader.line_num}") from error

            if min(n, *block) <= 0 or not all(
                isfinite(ms) and ms > 0 for ms in (timing.kernel_ms, timing.total_ms)
            ):
                raise ValueError(f"expected positive sizes and times at line {reader.line_num}")

            series = groups.setdefault(block, {})
            if n in series:
                raise ValueError(f"duplicate combination N={n}, block={block}")
            series[n] = timing
    if not groups:
        raise ValueError("CSV has no measurements")
    return groups


def block_label(block: tuple[int, int]) -> str:
    return f"{block[0]}×{block[1]}"


def plot_timings(groups, sizes, path: Path) -> None:
    fig, (kernel_ax, total_ax) = plt.subplots(
        1, 2, sharey=True, figsize=(11.2, 4.4), layout="constrained"
    )
    for index, (block, measurements) in enumerate(sorted(groups.items())):
        points = sorted(measurements.items())
        ns = [n for n, _ in points]
        style = {
            "label": block_label(block),
            "color": COLORS[index % len(COLORS)],
            "marker": MARKERS[index % len(MARKERS)],
            "linewidth": 1.6,
            "markersize": 5,
        }
        kernel_ax.loglog(ns, [timing.kernel_ms for _, timing in points], **style)
        total_ax.loglog(ns, [timing.total_ms for _, timing in points], **style)

    for ax, title in ((kernel_ax, "Ядро CUDA"), (total_ax, "Полный цикл с копированием")):
        ax.set(title=title, xlabel="Размер матрицы N")
        ax.set_xticks(sizes, labels=[str(n) for n in sizes])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(which="major", color="#dce3e9")
        ax.set_axisbelow(True)
    kernel_ax.set_ylabel("Минимальное время, мс")
    handles, labels = kernel_ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Размер блока", loc="outside upper center",
               ncols=min(4, len(groups)), frameon=False)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_block_comparison(groups, sizes, path: Path) -> None:
    blocks = sorted(groups)
    best = {n: min(series[n].kernel_ms for series in groups.values() if n in series)
            for n in sizes}
    differences = np.full((len(blocks), len(sizes)), np.nan)
    for i, block in enumerate(blocks):
        for j, n in enumerate(sizes):
            if n in groups[block]:
                differences[i, j] = 100 * (groups[block][n].kernel_ms / best[n] - 1)

    fig, ax = plt.subplots(figsize=(max(7.8, 1.2 * len(sizes)), 2.2 + 0.48 * len(blocks)),
                           layout="constrained")
    upper = max(1.0, float(np.nanmax(differences)))
    image = ax.imshow(differences, cmap=matplotlib.colormaps["Blues"].with_extremes(bad="#eef1f4"),
                      aspect="auto", vmin=0, vmax=upper)
    ax.set_title("Превышение времени ядра над минимумом при том же N", pad=14)
    ax.set_xlabel("Размер матрицы N")
    ax.set_ylabel("Размер блока")
    ax.set_xticks(range(len(sizes)), labels=[str(n) for n in sizes])
    ax.set_yticks(range(len(blocks)), labels=[block_label(block) for block in blocks])
    ax.tick_params(length=0)
    for (i, j), value in np.ndenumerate(differences):
        label = f"{value:.1f}%".replace(".", ",") if isfinite(value) else "—"
        color = "white" if value > 0.57 * upper else "#17304a"
        ax.text(j, i, label, ha="center", va="center", color=color, fontweight="medium")
    percent = FuncFormatter(lambda value, _: f"{value:g}%".replace(".", ","))
    fig.colorbar(image, ax=ax, shrink=0.86, format=percent)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "bench.csv")
    parser.add_argument("--out-dir", type=Path, help="по умолчанию каталог входного CSV")
    args = parser.parse_args()

    try:
        groups = read_timings(args.csv)
        sizes = sorted({n for measurements in groups.values() for n in measurements})
        out_dir = args.out_dir if args.out_dir is not None else args.csv.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        timings_path = out_dir / "timings.png"
        comparison_path = out_dir / "block_comparison.png"
        plot_timings(groups, sizes, timings_path)
        plot_block_comparison(groups, sizes, comparison_path)
        print(f"saved {timings_path}")
        print(f"saved {comparison_path}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"plot: {error}\n")


if __name__ == "__main__":
    main()
