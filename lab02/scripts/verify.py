"""Автоматическая верификация: сравнивает файл результата C++-программы с numpy (A @ B).

    python3 scripts/verify.py --a data/A_400.txt --b data/B_400.txt --c results/C_400.txt

Код возврата 0 — совпадает, 1 — расхождение. numpy считает через BLAS, т.е. это
независимая от нашей реализация: порядок сложений другой, поэтому допускаем
малую погрешность (--tol, относительная).
"""
import argparse
import sys

import numpy as np


def load(path: str) -> np.ndarray:
    with open(path) as f:
        rows, cols = map(int, f.readline().split())
        vals = np.array(f.read().split(), dtype=np.float64)
    if vals.size != rows * cols:
        sys.exit(f"verify: {path}: expected {rows * cols} values, got {vals.size}")
    return vals.reshape(rows, cols)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--c", required=True)
    p.add_argument("--tol", type=float, default=1e-7)
    args = p.parse_args()

    a, b, c = load(args.a), load(args.b), load(args.c)
    ref = a @ b
    diff = float(np.max(np.abs(c - ref)))
    scale = max(1.0, float(np.max(np.abs(ref))))
    ok = diff <= args.tol * scale
    print(f"verify: n={c.shape[0]}, max|C - A*B| = {diff:.3e}, "
          f"rel = {diff / scale:.3e}, tol = {args.tol:g} -> {'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()