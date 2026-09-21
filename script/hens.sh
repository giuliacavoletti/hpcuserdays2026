#!/bin/bash
#SBATCH --job-name=hens_4gpu_test
#SBATCH --account=e-ben-2026b09-120
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --output=/e/project1/e-ben-2026b09-120/hens_test_%j.out

cd /e/project1/e-ben-2026b09-120/

export FAST_SCRATCH="/e/scratch/e-ben-2026b09-120/$USER/earth2_fast"
export EARTH2STUDIO_CACHE="$FAST_SCRATCH/earth2_cache"
mkdir -p "$FAST_SCRATCH/outputs" outputs

# Pulisce vecchi CSV e store Zarr per evitare collisioni di scrittura
rm -f scaling_results_hens_strong.csv scaling_results_hens_weak.csv
rm -rf outputs/*.zarr "$FAST_SCRATCH/outputs"/*.zarr

COMMON_ENV="source /e/project1/e-ben-2026b09-120/earth2_env/bin/activate && \
            export FAST_SCRATCH=$FAST_SCRATCH && \
            export EARTH2STUDIO_CACHE=$EARTH2STUDIO_CACHE && \
            export OMP_NUM_THREADS=16 && \
            export TORCHDYNAMO_DISABLE=1 && \
            export MASTER_ADDR=127.0.0.1 && \
            export TORCH_DISTRIBUTED_IPV6=0 && \
            export GLOO_SOCKET_IFNAME=lo && \
            export NCCL_SOCKET_IFNAME=lo && \
            export TP_SOCKET_IFNAME=lo"

run_in_container() {
    apptainer exec --nv \
        --bind /e/project1/e-ben-2026b09-120:/e/project1/e-ben-2026b09-120 \
        --bind /e/scratch/e-ben-2026b09-120:/e/scratch/e-ben-2026b09-120 \
        containers/earth2_py311.sif bash -c "$COMMON_ENV && $1"
}

echo "=== TEST VELOCE 4 GPU: WARMUP (1 member su 4 GPU) ==="
run_in_container "python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 hens_scaling.py --mode strong --size 4"
rm -f scaling_results_hens_strong.csv scaling_results_hens_weak.csv

echo "=== TEST STRONG SCALING SU 4 GPU (24 members) ==="
run_in_container "python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 hens_scaling.py --mode strong --size 24"

echo "=== TEST WEAK SCALING SU 4 GPU (6 members/GPU = 24 totali) ==="
run_in_container "python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 hens_scaling.py --mode weak --size 6"

echo "=== RISULTATI GENERATI ==="
cat scaling_results_hens_strong.csv 2>/dev/null || echo "Nessun CSV strong trovato"
cat scaling_results_hens_weak.csv 2>/dev/null || echo "Nessun CSV weak trovato"

echo "Test 4 GPU completato!"
EOF
chmod +x /e/project1/e-ben-2026b09-120/test_hens_4gpu.sh