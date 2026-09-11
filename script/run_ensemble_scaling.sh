#!/bin/bash
#SBATCH --job-name=earth2_scaling
#SBATCH --account=e-ben-2026b09-120
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=scaling_%j.out

cd /e/project1/e-ben-2026b09-120/

source /e/project1/e-ben-2026b09-120/earth2_env/bin/activate
export EARTH2STUDIO_CACHE="/e/project1/e-ben-2026b09-120/earth2_cache"
export TORCHDYNAMO_DISABLE=1
export MASTER_ADDR="127.0.0.1"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"

rm -f scaling_results_strong.csv scaling_results_weak.csv

# Wrapper function to run commands inside the Apptainer container
run_in_container() {
    apptainer exec --nv \
        --bind /e/project1/e-ben-2026b09-120:/e/project1/e-ben-2026b09-120 \
        containers/earth2_py311.sif bash -c "$1"
}

echo "=== STARTING STRONG SCALING (24 total members) ==="
for gpus in 1 2 3 4; do
    echo "--- Test on $gpus GPU (Strong) ---"
    cmd="source /e/project1/e-ben-2026b09-120/earth2_env/bin/activate && \
         export MASTER_ADDR=127.0.0.1 && export GLOO_SOCKET_IFNAME=lo && export NCCL_SOCKET_IFNAME=lo && \
         python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=$gpus ensemble_scaling.py --mode strong --size 24"
    run_in_container "$cmd"
done

echo "=== STARTING WEAK SCALING (6 members per GPU) ==="
for gpus in 1 2 3 4; do
    echo "--- Test on $gpus GPU (Weak) ---"
    cmd="source /e/project1/e-ben-2026b09-120/earth2_env/bin/activate && \
         export MASTER_ADDR=127.0.0.1 && export GLOO_SOCKET_IFNAME=lo && export NCCL_SOCKET_IFNAME=lo && \
         python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=$gpus ensemble_scaling.py --mode weak --size 6"
    run_in_container "$cmd"
done

echo "Benchmark completed. Generating plots..."

# Automatically generate plots using Python
python3 - << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt

os_plots = "outputs"
os.makedirs(os_plots, exist_ok=True)

# 1. Plot Strong Scaling (Speedup)
if os.path.exists("scaling_results_strong.csv"):
    df_s = pd.read_csv("scaling_results_strong.csv")
    t1 = df_s.loc[df_s["gpus"] == 1, "time_sec"].values[0]
    df_s["speedup"] = t1 / df_s["time_sec"]
    df_s["ideal_speedup"] = df_s["gpus"]

    plt.figure(figsize=(7, 5))
    plt.plot(df_s["gpus"], df_s["speedup"], marker="o", linestyle="-", linewidth=2, label="Measured Speedup")
    plt.plot(df_s["gpus"], df_s["ideal_speedup"], linestyle="--", color="gray", label="Ideal Linear")
    plt.xlabel("Number of GPUs", fontsize=12)
    plt.ylabel("Speedup (T1 / Tp)", fontsize=12)
    plt.title("Strong Scaling - FCN Ensemble", fontsize=14)
    plt.xticks([1, 2, 3, 4])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/strong_scaling.png", dpi=150)
    plt.close()
    print("Saved outputs/strong_scaling.png")

# 2. Plot Weak Scaling (Execution Time)
if os.path.exists("scaling_results_weak.csv"):
    df_w = pd.read_csv("scaling_results_weak.csv")

    plt.figure(figsize=(7, 5))
    plt.plot(df_w["gpus"], df_w["time_sec"], marker="s", color="orange", linestyle="-", linewidth=2, label="Measured Time")
    plt.xlabel("Number of GPUs", fontsize=12)
    plt.ylabel("Execution Time (s)", fontsize=12)
    plt.title("Weak Scaling - FCN Ensemble (6 members/GPU)", fontsize=14)
    plt.xticks([1, 2, 3, 4])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/weak_scaling.png", dpi=150)
    plt.close()
    print("Saved outputs/weak_scaling.png")
EOF

echo "Done! Check the 'outputs' directory for plots and scaling results."