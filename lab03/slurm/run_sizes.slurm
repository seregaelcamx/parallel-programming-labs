#!/bin/bash
#SBATCH --job-name=lab03_sizes
#SBATCH --time=00:15:00
#SBATCH --partition=batch
#SBATCH --constraint=mem32
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --output=lab03_sizes_%j.out
#SBATCH --error=lab03_sizes_%j.err
##SBATCH --account=YOUR_ACCOUNT

# Короткий вариант по образцу со скриншота: фиксированное P, разные N.
# Перед sbatch сгенерируйте data/A_N.txt, data/B_N.txt для всех размеров.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
# module load <ваш модуль MPI>
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT="results/sizes_${SLURM_JOB_ID}"
mkdir -p "$OUT"
MPI_OPTIONS=()
if [[ -n "${MPI_TYPE:-}" ]]; then MPI_OPTIONS+=("--mpi=$MPI_TYPE"); fi

for N in 200 400 800 1200 1600 2000; do
    srun "${MPI_OPTIONS[@]}" --ntasks="$SLURM_NTASKS" --cpus-per-task=1 \
        --cpu-bind=cores --kill-on-bad-exit=1 \
        ./build/matrix_mul_mpi "$N" \
        --out "$OUT/C_${N}_p${SLURM_NTASKS}.txt" \
        --csv "$OUT/bench.csv" --repeat 5 --warmup 1 --expected-procs "$SLURM_NTASKS"
done

# После завершения можно выполнить проверку и графики на своём ПК:
# python3 scripts/verify_results.py "$OUT" --data-dir data
# python3 scripts/analyze.py "$OUT/bench.csv"
