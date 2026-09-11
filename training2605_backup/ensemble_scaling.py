import os
import argparse
import warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
import torch
import torch.distributed as dist

from earth2studio.io import ZarrBackend
from earth2studio.models.px import FCN
from earth2studio.perturbation import SphericalGaussian
from earth2studio.run import ensemble
from earth2_offline import LocalGFSSource, OfflinePackage, CACHE_PATH

# Argument parsing for scaling mode and size
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="strong", choices=["strong", "weak"])
parser.add_argument("--size", type=int, default=24, help="Total ensemble for strong, or per-GPU members for weak")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup Distributed Environment (Compatible with torch.distributed.run)
# ---------------------------------------------------------------------------
if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    is_distributed = True
else:
    rank = 0
    world_size = 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    is_distributed = False

os.makedirs("outputs", exist_ok=True)

if args.mode == "strong":
    total_ensemble = args.size
    rank_ensemble = total_ensemble // world_size
else:
    rank_ensemble = args.size
    total_ensemble = rank_ensemble * world_size

batch_size = rank_ensemble
nsteps = 10

# ---------------------------------------------------------------------------
# Model & Data
# ---------------------------------------------------------------------------
package = OfflinePackage(f"{CACHE_PATH}/fcn")
model = FCN.load_model(package).to(device)

seed = 42 + rank
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

sg = SphericalGaussian(noise_amplitude=0.15)

GRIB_MAP = {
    np.datetime64("2024-01-01T00:00", "ns"): f"{CACHE_PATH}/gfs/gfs.20240101.t00z.pgrb2.0p25.f000",
}
data = LocalGFSSource(GRIB_MAP)

output_path = f"/e/scratch/e-ben-2026b09-120/cavoletti1/earth2_fast/outputs/scale_{args.mode}_gpus{world_size}_rank{rank}.zarr"
io = ZarrBackend(
    file_name=output_path,
    chunks={"ensemble": 1, "time": 1, "lead_time": 1},
    backend_kwargs={"overwrite": True},
)

if rank == 0:
    print(f"[{args.mode.upper()} SCALING] GPUs: {world_size} | Total Members: {total_ensemble} | Members/GPU: {rank_ensemble}")

# Timing
torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()

io = ensemble(
    ["2024-01-01"],
    nsteps,
    rank_ensemble,
    model,
    data,
    io,
    sg,
    batch_size=batch_size,
    output_coords={"variable": np.array(["t2m", "tcwv"])},
    device=device,
)

end_event.record()
torch.cuda.synchronize()

elapsed_time_sec = start_event.elapsed_time(end_event) / 1000.0 # ms to s conversion

if is_distributed:
    dist.barrier()

times_tensor = torch.tensor([elapsed_time_sec], device=device) 
if is_distributed:
    dist.all_reduce(times_tensor, op=dist.ReduceOp.MAX)

max_time = times_tensor.item()

if rank == 0:
    print(f"RESULT -> Mode: {args.mode}, GPUs: {world_size}, Total Members: {total_ensemble}, Time: {max_time:.4f}s")
    
    csv_file = f"scaling_results_{args.mode}.csv"
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a") as f:
        if not file_exists:
            f.write("gpus,total_ensemble,rank_ensemble,time_sec\n")
        f.write(f"{world_size},{total_ensemble},{rank_ensemble},{max_time:.4f}\n")