#!/bin/bash
#SBATCH --job-name=earth2_weak_scaling
#SBATCH --account=e-ben-2026b09-120
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=weak_scaling_%j.out

cd /e/project1/e-ben-2026b09-120/

source /e/project1/e-ben-2026b09-120/earth2_env/bin/activate
export EARTH2STUDIO_CACHE="/e/project1/e-ben-2026b09-120/earth2_cache"
export TORCHDYNAMO_DISABLE=1
export MASTER_ADDR="127.0.0.1"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"

rm -f scaling_results_weak.csv

# Wrapper function to run commands inside the Apptainer container
run_in_container() {
    apptainer exec --nv \
        --bind /e/project1/e-ben-2026b09-120:/e/project1/e-ben-2026b09-120 \
        containers/earth2_py311.sif bash -c "$1"
}

echo "=== STARTING WEAK SCALING ==="
for gpus in 1 2 3 4; do
    echo "--- Test on $gpus GPU (Weak) ---"
    cmd="source /e/project1/e-ben-2026b09-120/earth2_env/bin/activate && \
         export MASTER_ADDR=127.0.0.1 && export GLOO_SOCKET_IFNAME=lo && export NCCL_SOCKET_IFNAME=lo && \
         python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=$gpus 08_scaling.py"
    run_in_container "$cmd"
done

echo "Benchmark completed. Generating plot..."

python3 - << 'EOF'
import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

if os.path.exists("scaling_results_weak.csv"):
    df_w = pd.read_csv("scaling_results_weak.csv")

    plt.figure(figsize=(7, 5))
    plt.plot(df_w["gpus"], df_w["time_sec"], marker="s", color="orange", linestyle="-", linewidth=2, label="Measured Time")
    plt.xlabel("Number of GPUs", fontsize=12)
    plt.ylabel("Execution Time (s)", fontsize=12)
    plt.title("Weak Scaling - DLWP Inference", fontsize=14)
    plt.xticks([1, 2, 3, 4])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/weak_scaling.png", dpi=150)
    plt.close()
    print("Saved outputs/weak_scaling.png")
EOF

echo "Done! Check the 'outputs' directory for the weak scaling plot."