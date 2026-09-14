"""Генерация пар квадратных матриц заданных размеров (детерминированная, по seed)."""
import argparse
import pathlib

import numpy as np


def save(m: np.ndarray, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{m.shape[0]} {m.shape[1]}\n")
        np.savetxt(f, m, fmt="%.9g")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 800, 1200, 1600, 2000])
    p.add_argument("--dir", type=pathlib.Path,
                   default=pathlib.Path(__file__).resolve().parents[1] / "data")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    for n in args.sizes:
        rng = np.random.default_rng([args.seed, n])
        a = rng.uniform(-1.0, 1.0, (n, n))
        b = rng.uniform(-1.0, 1.0, (n, n))
        save(a, args.dir / f"A_{n}.txt")
        save(b, args.dir / f"B_{n}.txt")
        print(f"generated {args.dir / f'A_{n}.txt'}, {args.dir / f'B_{n}.txt'} (n={n})")


if __name__ == "__main__":
    main()