"""Короткая функциональная проверка: P>N, остаток строк, ошибки входных данных."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from matrix_io import load, check


def save(path, matrix):
    np.savetxt(path, matrix, header=f"{matrix.shape[0]} {matrix.shape[1]}", comments="", fmt="%.17g")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=Path("build/matrix_mul_mpi"))
    parser.add_argument("--launcher-arg", action="append", default=[])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    exe = str(args.exe.resolve())
    records = []
    with tempfile.TemporaryDirectory(prefix="lab03_check_") as temporary:
        folder = Path(temporary)
        for n, p, kind in [(1, 1, "scalar"), (1, 4, "empty_blocks"),
                           (5, 3, "uneven"), (17, 4, "random"),
                           (5, 4, "identity"), (5, 4, "zero")]:
            rng = np.random.default_rng([42, n])
            a = rng.normal(size=(n, n))
            b = rng.normal(size=(n, n))
            if kind == "identity":
                b = np.eye(n)
            if kind == "zero":
                b = np.zeros((n, n))
            save(folder / "A.txt", a)
            save(folder / "B.txt", b)
            csv_path = folder / f"{kind}.csv"
            command = ["mpirun"] + args.launcher_arg + ["-np", str(p), exe,
                "--a", str(folder / "A.txt"), "--b", str(folder / "B.txt"),
                "--out", str(folder / "C.txt"), "--csv", str(csv_path), "--repeat", "2", "--warmup", "1"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            record = dict(case=kind, n=n, p=p, **check(a @ b, load(folder / "C.txt")))
            with csv_path.open() as f:
                rows = list(csv.DictReader(f))
            record["timing_rows"] = len(rows)
            if not record["ok"] or len(rows) != 2:
                raise AssertionError(record)
            records.append(record)
        for kind, broken in [("bad_dimensions", "2 3\n1 2 3\n4 5 6\n"),
                             ("truncated", "5 5\n1 2\n"),
                             ("extra_values", "1 1\n2 3\n")]:
            (folder / "A.txt").write_text(broken)
            command = ["mpirun"] + args.launcher_arg + ["-np", "3", exe,
                "--a", str(folder / "A.txt"), "--b", str(folder / "B.txt"),
                "--out", str(folder / "C.txt")]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                raise AssertionError(f"{kind}: expected nonzero exit")
            records.append(dict(case=kind, expected_failure=True, exit_code=result.returncode, ok=True))
        save(folder / "A.txt", np.eye(5))
        save(folder / "B.txt", np.eye(5))
        command = ["mpirun"] + args.launcher_arg + ["-np", "2", exe,
            "--a", str(folder / "A.txt"), "--b", str(folder / "B.txt"),
            "--out", str(folder / "C.txt"), "--expected-procs", "4"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 or "MPI process count differs" not in result.stderr:
            raise AssertionError("incorrect process count must be rejected explicitly")
        records.append(dict(case="wrong_process_count", expected_failure=True, exit_code=result.returncode, ok=True))
    output = json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    print(output)
    if args.json:
        args.json.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
