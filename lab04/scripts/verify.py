"""Сравнение результата CUDA с NumPy по критерию из лабораторной № 1."""

import argparse
import math
from pathlib import Path

import numpy as np

DEFAULT_TOL = 1e-7


def load(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as file:
        rows, cols = map(int, file.readline().split())
        if rows < 1 or cols < 1:
            raise ValueError(f"{path}: dimensions must be positive")
        values = np.array(file.read().split(), dtype=np.float64)
    if values.size != rows * cols:
        raise ValueError(f"{path}: expected {rows * cols} values, got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: matrix values must be finite")
    return values.reshape(rows, cols)


def verify(result: np.ndarray, reference: np.ndarray, tol: float = DEFAULT_TOL) -> float:
    if result.shape != reference.shape:
        raise ValueError(f"expected result shape {reference.shape}, got {result.shape}")
    if not np.isfinite(reference).all():
        raise ValueError("NumPy reference contains non-finite values")

    max_error = float(np.max(np.abs(result - reference)))
    scale = max(1.0, float(np.max(np.abs(reference))))
    relative_error = max_error / scale
    ok = relative_error <= tol
    print(f"verify: max|C - A*B| = {max_error:.3e}, rel = {relative_error:.3e}, "
          f"tol = {tol:g} -> {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise ValueError("CUDA result differs from NumPy")
    return relative_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    args = parser.parse_args()
    if not math.isfinite(args.tol) or args.tol <= 0:
        parser.error("--tol must be positive and finite")

    try:
        reference = load(args.a) @ load(args.b)
        verify(load(args.c), reference, args.tol)
    except (OSError, ValueError) as error:
        parser.exit(1, f"verify: {error}\n")


if __name__ == "__main__":
    main()
