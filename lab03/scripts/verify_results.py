"""Проверка сохранённых матриц после запуска run_sizes.slurm или --skip-verify."""
import argparse
import json
from pathlib import Path
import re
from matrix_io import load, check


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    a = p.parse_args()
    files = sorted(a.directory.glob("C_*_p*.txt")) + sorted((a.directory / "matrices").glob("C_*_p*.txt"))
    if not files:
        p.error("no C_N_pP.txt files found")
    cache_n, reference = None, None
    records = []
    for path in sorted(files, key=lambda f: tuple(map(int, re.fullmatch(r"C_(\d+)_p(\d+)\.txt", f.name).groups()))):
        n, procs = map(int, re.fullmatch(r"C_(\d+)_p(\d+)\.txt", path.name).groups())
        if n != cache_n:
            left = load(a.data_dir / f"A_{n}.txt")
            right = load(a.data_dir / f"B_{n}.txt")
            if left.shape != (n, n) or right.shape != (n, n):
                raise ValueError(f"input dimensions differ from N={n}")
            reference = left @ right
            cache_n = n
        record = dict(n=n, p=procs, **check(reference, load(path)))
        record["status"] = "passed" if record["ok"] else "failed"
        records.append(record)
        print(json.dumps(record), flush=True)
    (a.directory / "verification.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    raise SystemExit(0 if all(r["ok"] for r in records) else 1)


if __name__ == "__main__":
    main()
