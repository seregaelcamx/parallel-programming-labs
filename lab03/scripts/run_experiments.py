"""Серия MPI-запусков. Каждый (N,P) — отдельный MPI job step, шаги идут последовательно."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys

# Не даём NumPy/BLAS создавать дополнительные потоки внутри выделения Slurm.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[200, 400, 800, 1200, 1600, 2000])
    parser.add_argument("--procs", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--exe", type=Path, default=Path("build/matrix_mul_mpi"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--launcher", choices=["mpirun", "srun"], default="mpirun")
    parser.add_argument("--mpi", default="", help="Slurm MPI plugin; empty means site default")
    parser.add_argument("--launcher-arg", action="append", default=[], help="repeat option; use --launcher-arg=--flag")
    parser.add_argument("--skip-verify", action="store_true", help="save C for later verification")
    parser.add_argument("--discard-matrices", action="store_true", help="delete C only after successful NumPy check")
    args = parser.parse_args()
    if min(args.sizes + args.procs) <= 0 or args.repeat < 1 or args.warmup < 0:
        parser.error("N, P, repeats must be positive; warmup >= 0")
    if len(set(args.sizes)) != len(args.sizes) or len(set(args.procs)) != len(args.procs):
        parser.error("duplicate sizes or process counts")
    if args.skip_verify and args.discard_matrices:
        parser.error("cannot discard unverified matrices")
    if args.launcher == "srun":
        if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_NTASKS"):
            parser.error("srun mode must run inside a Slurm allocation with --ntasks")
        if max(args.procs) > int(os.environ["SLURM_NTASKS"]):
            parser.error("process count exceeds SLURM_NTASKS; increase sbatch --ntasks")
    if not shutil.which(args.launcher):
        parser.error(f"launcher not found: {args.launcher}")
    exe = args.exe.resolve()
    if not exe.is_file() or not os.access(exe, os.X_OK):
        parser.error(f"compile executable first: {exe}")
    data = args.data_dir.resolve()
    for n in args.sizes:
        for label in ("A", "B"):
            if not (data / f"{label}_{n}.txt").is_file():
                parser.error(f"missing data/{label}_{n}.txt; run generate_matrices.py")
    if not args.skip_verify:
        try:
            from matrix_io import load, check
        except ImportError:
            parser.error("NumPy required; install it, or use --skip-verify and check saved results later")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    label = "cluster_" + os.environ["SLURM_JOB_ID"] if os.environ.get("SLURM_JOB_ID") else "local_" + stamp
    out = (args.out_dir or Path("results") / label).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "matrices").mkdir()
    (out / "logs").mkdir()
    metadata = {"created_utc": stamp, "status": "running", "sizes": args.sizes,
                "procs": args.procs, "repeat": args.repeat, "warmup": args.warmup,
                "launcher": args.launcher, "mpi_plugin": args.mpi or "site default",
                "platform": platform.platform(), "python": sys.version,
                "exe": str(exe), "data_dir": str(data), "verification": not args.skip_verify,
                "slurm": {key: os.environ[key] for key in
                    ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NTASKS", "SLURM_JOB_NUM_NODES",
                     "SLURM_CPUS_PER_TASK", "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT") if key in os.environ},
                "commands": []}
    for name, command in (("launcher_version", [args.launcher, "--version"]),
                          ("compiler_version", ["mpicxx", "--version"])):
        try:
            metadata[name] = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            metadata[name] = "unavailable"
    try:
        for n in args.sizes:
            reference = None
            if not args.skip_verify:
                a, b = load(data / f"A_{n}.txt"), load(data / f"B_{n}.txt")
                if a.shape != (n, n) or b.shape != (n, n):
                    raise ValueError(f"input dimensions differ from requested N={n}")
                reference = a @ b
                del a, b
            for p in args.procs:
                c = out / "matrices" / f"C_{n}_p{p}.txt"
                if args.launcher == "srun":
                    nodes = min(p, int(os.environ.get("SLURM_JOB_NUM_NODES", "1")))
                    command = ["srun", f"--ntasks={p}", "--cpus-per-task=1", f"--nodes=1-{nodes}",
                               "--cpu-bind=cores", "--kill-on-bad-exit=1"]
                    if args.mpi:
                        command += [f"--mpi={args.mpi}"]
                else:
                    command = ["mpirun", "-np", str(p)]
                command += args.launcher_arg
                command += [str(exe), str(n), "--data-dir", str(data), "--out", str(c),
                            "--csv", str(out / "bench.csv"), "--repeat", str(args.repeat),
                            "--warmup", str(args.warmup), "--expected-procs", str(p)]
                print(shlex.join(command), flush=True)
                metadata["commands"].append(command)
                (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                with (out / "logs" / f"n{n}_p{p}.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                record = {"n": n, "p": p, "status": "skipped"}
                if reference is not None:
                    record.update(check(reference, load(c)))
                    record["status"] = "passed" if record["ok"] else "failed"
                with (out / "verification.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"N={n} P={p}: verification {record['status']}", flush=True)
                if record["status"] == "failed":
                    raise RuntimeError(f"verification failed: N={n}, P={p}")
                if args.discard_matrices:
                    c.unlink()
        metadata["status"] = "completed"
    except BaseException as e:
        metadata["status"] = "failed"
        metadata["error"] = str(e)
        raise
    finally:
        (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    main()
