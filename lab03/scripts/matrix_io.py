"""Общий формат первой лабораторной: rows cols, затем строки double."""
import numpy as np


def load(path):
    with open(path, encoding="utf-8") as f:
        header = f.readline().split()
        if len(header) != 2:
            raise ValueError(f"{path}: expected rows cols")
        rows, cols = map(int, header)
        if rows <= 0 or cols <= 0:
            raise ValueError(f"{path}: dimensions must be positive")
        values = np.loadtxt(f, dtype=np.float64, ndmin=2)
    if values.shape != (rows, cols) or not np.isfinite(values).all():
        raise ValueError(f"{path}: wrong shape or non-finite values")
    return values


def check(reference, actual, rtol=1e-7, atol=1e-9):
    if actual.shape != reference.shape:
        raise ValueError(f"result shape {actual.shape} != {reference.shape}")
    error = np.abs(actual - reference)
    ok = bool(np.isfinite(reference).all() and np.isfinite(actual).all()
              and np.all(error <= atol + rtol * np.abs(reference)))
    return {"ok": ok, "max_abs_error": float(np.max(error)),
            "scaled_max_error": float(np.max(error) / max(1.0, float(np.max(np.abs(reference))))),
            "rtol": rtol, "atol": atol}
