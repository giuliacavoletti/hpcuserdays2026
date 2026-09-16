#!/bin/bash
#SBATCH --job-name=hens_scaling
#SBATCH --account=e-ben-2026b09-120
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=hens_scaling_%j.out

cd /e/project1/e-ben-2026b09-120/

export FAST_SCRATCH="/e/scratch/e-ben-2026b09-120/$USER/earth2_fast"
export EARTH2STUDIO_CACHE="$FAST_SCRATCH/earth2_cache"

rm -f scaling_results_hens_strong.csv scaling_results_hens_weak.csv

run_in_container() {
    apptainer exec --nv \
        --bind /e/project1/e-ben-2026b09-120:/e/project1/e-ben-2026b09-120 \
        --bind /e/scratch/e-ben-2026b09-120:/e/scratch/e-ben-2026b09-120 \
        containers/earth2_py311.sif bash -c "$1"
}

COMMON_ENV="source /e/project1/e-ben-2026b09-120/earth2_env/bin/activate && export FAST_SCRATCH=$FAST_SCRATCH && export EARTH2STUDIO_CACHE=$EARTH2STUDIO_CACHE && export OMP_NUM_THREADS=16 && export TORCHDYNAMO_DISABLE=1 && export MASTER_ADDR=127.0.0.1 && export TORCH_DISTRIBUTED_IPV6=0 && export GLOO_SOCKET_IFNAME=lo && export NCCL_SOCKET_IFNAME=lo && export TP_SOCKET_IFNAME=lo"

echo "=== PERFORMING WARMUP PASS ==="
run_in_container "$COMMON_ENV && python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 hens_scaling.py --mode strong --size 1 > /dev/null 2>&1"
rm -f scaling_results_hens_strong.csv scaling_results_hens_weak.csv

echo "=== STARTING STRONG SCALING (24 total members) ==="
for gpus in 1 2 4; do
    echo "--- Test on $gpus GPU (Strong) ---"
    cmd="$COMMON_ENV && python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=$gpus hens_scaling.py --mode strong --size 24"
    run_in_container "$cmd"
done

echo "=== STARTING WEAK SCALING (6 members per GPU) ==="
for gpus in 1 2 4; do
    echo "--- Test on $gpus GPU (Weak) ---"
    cmd="$COMMON_ENV && python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=$gpus hens_scaling.py --mode weak --size 6"
    run_in_container "$cmd"
done

echo "Benchmark run complete! Output CSVs generated."