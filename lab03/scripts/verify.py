"""Проверка C = A @ B независимой реализацией NumPy/BLAS."""
import argparse
import json
from pathlib import Path
from matrix_io import load, check


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--c", required=True)
    p.add_argument("--json", type=Path)
    p.add_argument("--rtol", type=float, default=1e-7)
    p.add_argument("--atol", type=float, default=1e-9)
    a = p.parse_args()
    if a.rtol < 0 or a.atol < 0:
        p.error("tolerances must be nonnegative")
    result = check(load(a.a) @ load(a.b), load(a.c), a.rtol, a.atol)
    print(json.dumps(result, ensure_ascii=False))
    if a.json:
        a.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
