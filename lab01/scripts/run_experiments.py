"""Прогон экспериментов для всех размеров матриц.

Для каждого N: генерирует данные (если нет), запускает C++-бинарник с повторами,
верифицирует результат через numpy, дописывает строку в results/bench.csv.
В конце: собирает markdown-таблицу (results/bench.md), подставляет её в README.md
между маркерами BENCH-TABLE и перерисовывает график (scripts/plot.py).
"""
import argparse
import csv
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
BEGIN, END = "<!-- BENCH-TABLE-BEGIN -->", "<!-- BENCH-TABLE-END -->"


def run(cmd) -> None:
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def markdown_table(rows) -> str:
    lines = ["| N | t_min, ms | t_mean, ms | GFLOPS (по t_min) |",
             "|---:|---:|---:|---:|"]
    lines += [f"| {r['n']} | {float(r['t_min_ms']):.3f} | {float(r['t_mean_ms']):.3f} "
              f"| {float(r['gflops']):.2f} |" for r in rows]
    return "\n".join(lines)


def update_readme(md: str) -> None:
    p = ROOT / "README.md"
    if not p.exists():
        return
    text = p.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print("README.md: маркеры таблицы не найдены, пропускаю обновление")
        return
    i = text.index(BEGIN) + len(BEGIN)
    j = text.index(END, i)
    p.write_text(text[:i] + "\n" + md + "\n" + text[j:], encoding="utf-8")
    print("README.md: таблица результатов обновлена")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bin", type=pathlib.Path, default=ROOT / "build" / "matrix_mul")
    p.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800, 1200, 1600, 2000])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--no-readme", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    data, results = ROOT / "data", ROOT / "results"
    results.mkdir(exist_ok=True)
    csv_path = results / "bench.csv"
    if csv_path.exists():
        csv_path.unlink()  # начинаем новую сессию измерений

    for n in args.sizes:
        a, b, c = data / f"A_{n}.txt", data / f"B_{n}.txt", results / f"C_{n}.txt"
        if not (a.exists() and b.exists()):
            run([PY, ROOT / "scripts" / "generate_matrices.py", "--sizes", n, "--dir", data])
        run([args.bin, "--a", a, "--b", b, "--out", c,
             "--repeat", args.repeats, "--csv", csv_path])
        run([PY, ROOT / "scripts" / "verify.py", "--a", a, "--b", b, "--c", c])

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    md = markdown_table(rows)
    (results / "bench.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md + "\n")
    if not args.no_readme:
        update_readme(md)
    if not args.no_plot:
        run([PY, ROOT / "scripts" / "plot.py"])


if __name__ == "__main__":
    main()