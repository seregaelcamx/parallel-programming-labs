"""Генератор л/р 1 с тем же default_rng([seed, N]) и форматом %.9g."""
import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", nargs="+", type=int, default=[200, 400, 800, 1200, 1600, 2000])
    p.add_argument("--dir", type=Path, default=Path("data"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.seed < 0 or any(n < 1 or n * n > 2147483647 for n in a.sizes):
        p.error("seed must be nonnegative; 0 < N*N <= INT_MAX")
    a.dir.mkdir(parents=True, exist_ok=True)
    for n in a.sizes:
        paths = [a.dir / f"{label}_{n}.txt" for label in ("A", "B")]
        if all(path.is_file() for path in paths) and not a.force:
            print(f"N={n}: existing files retained", flush=True)
            continue
        if any(path.exists() for path in paths) and not a.force:
            raise SystemExit(f"Incomplete pair for N={n}; use --force to regenerate both")
        rng = np.random.default_rng([a.seed, n])
        for path in paths:
            values = rng.uniform(-1.0, 1.0, (n, n))
            with path.open("w", encoding="utf-8", newline="\n") as f:
                f.write(f"{n} {n}\n")
                np.savetxt(f, values, fmt="%.9g")
        print(f"N={n}: generated {paths[0]} and {paths[1]}", flush=True)


if __name__ == "__main__":
    main()
