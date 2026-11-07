"""Запуск CUDA-серии с верификацией и сохранением результатов в bench.csv."""

import argparse
import csv
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

from verify import load, verify

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN = ROOT / "build" / (
    "Release/matrix_mul_cuda.exe" if os.name == "nt" else "matrix_mul_cuda"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800, 1200, 1600, 2000])
    parser.add_argument("--blocks", nargs="+", default=["8x8", "16x16", "32x8", "32x32"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    binary = args.bin.resolve()
    if not binary.is_file():
        parser.error(f"executable not found: {binary}")
    if args.repeats < 1 or any(n < 1 for n in args.sizes):
        parser.error("--sizes and --repeats must be positive")
    if any(not re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", block) for block in args.blocks):
        parser.error("--blocks must contain values of the form XxY, e.g. 16x16")

    try:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".bench-", dir=args.results_dir) as temp:
            work = Path(temp)
            result_path = work / "C.txt"
            raw_csv = work / "timings.csv"
            errors = []

            for n in args.sizes:
                a = args.data_dir / f"A_{n}.txt"
                b = args.data_dir / f"B_{n}.txt"
                if not a.is_file() or not b.is_file():
                    subprocess.run(
                        [sys.executable, ROOT / "scripts" / "generate_matrices.py",
                         "--sizes", str(n), "--dir", args.data_dir],
                        check=True,
                    )
                matrix_a, matrix_b = load(a), load(b)
                if matrix_a.shape != (n, n) or matrix_b.shape != (n, n):
                    raise ValueError(f"{a} and {b} must both have shape ({n}, {n})")
                reference = matrix_a @ matrix_b

                for block in args.blocks:
                    print(f"N={n}, block={block}", flush=True)
                    subprocess.run(
                        [binary, "--a", a, "--b", b, "--out", result_path,
                         "--block", block, "--repeat", str(args.repeats), "--csv", raw_csv],
                        check=True,
                    )
                    errors.append(verify(load(result_path), reference))

            final_csv = args.results_dir / "bench.csv"
            checked_csv = work / "bench.csv"
            with raw_csv.open(newline="", encoding="utf-8") as source, \
                    checked_csv.open("w", newline="", encoding="utf-8") as target:
                reader = csv.DictReader(source)
                writer = csv.DictWriter(target, [*reader.fieldnames, "verify_rel_error"])
                writer.writeheader()
                for row, error in zip(reader, errors, strict=True):
                    writer.writerow({**row, "verify_rel_error": f"{error:.9g}"})
            checked_csv.replace(final_csv)
        print(f"saved {final_csv}")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"benchmark: {error}\n")


if __name__ == "__main__":
    main()
