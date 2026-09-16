import os
import argparse
import warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
import torch
import torch.distributed as dist

from earth2studio.io import ZarrBackend
from earth2studio.models.auto import Package
from earth2studio.models.px import SFNO
from earth2studio.perturbation import CorrelatedSphericalGaussian, HemisphericCentredBredVector
from earth2studio.run import ensemble
from earth2_offline import LocalGFSSource, CACHE_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="strong", choices=["strong", "weak"])
parser.add_argument("--size", type=int, default=24)
args = parser.parse_args()

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

batch_size = 8 
nsteps = 10

# Direct local checkpoint directories on fast scratch
user = os.environ.get("USER", "cavoletti1")
ckpt_root = f"/e/scratch/e-ben-2026b09-120/{user}/earth2_fast/earth2_cache"
my_checkpoint_dir = f"{ckpt_root}/hens_ckpt_{rank % 4}/sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed102"
package = Package(my_checkpoint_dir)
model = SFNO.load_model(package).to(device)

seed = 42 + rank
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Offline GFS Data Setup
GRIB_MAP = {
    np.datetime64("2026-08-01T00:00", "ns"): f"{CACHE_PATH}/gfs/gfs.20260801.t00z.pgrb2.0p25.f000",
}
data = LocalGFSSource(GRIB_MAP)

# HENS Perturbation Setup
noise_amplification = torch.zeros(model.input_coords()["variable"].shape[0], device=device)
index_z500 = list(model.input_coords()["variable"]).index("z500")
noise_amplification[index_z500] = 39.27
noise_amplification = noise_amplification.reshape(1, 1, 1, -1, 1, 1)

seed_perturbation = CorrelatedSphericalGaussian(noise_amplitude=noise_amplification)
perturbation = HemisphericCentredBredVector(
    model, data, seed_perturbation, noise_amplitude=noise_amplification
)

output_path = f"/e/scratch/e-ben-2026b09-120/$USER/earth2_fast/outputs/hens_{args.mode}_gpus{world_size}_rank{rank}.zarr"
io_backend = ZarrBackend(
    file_name=output_path,
    chunks={"ensemble": 1, "time": 1, "lead_time": 1},
    backend_kwargs={"overwrite": True},
)

if rank == 0:
    print(f"[HENS {args.mode.upper()} SCALING] GPUs: {world_size} | Total: {total_ensemble} | Per-GPU: {rank_ensemble}")

# Pure Compute Isolation
if is_distributed:
    dist.barrier()
torch.cuda.synchronize()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

ensemble(
    ["2026-08-01T00:00:00"],
    nsteps,
    rank_ensemble,
    model,
    data,
    io_backend,
    perturbation,
    batch_size=batch_size,
    output_coords={"variable": np.array(["u10m", "v10m"])},
    device=device,
)

end_event.record()
torch.cuda.synchronize()

elapsed_time_sec = start_event.elapsed_time(end_event) / 1000.0

if is_distributed:
    dist.barrier()

times_tensor = torch.tensor([elapsed_time_sec], device=device) 
if is_distributed:
    dist.all_reduce(times_tensor, op=dist.ReduceOp.MAX)

max_time = times_tensor.item()

if rank == 0:
    csv_file = f"scaling_results_hens_{args.mode}.csv"
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a") as f:
        if not file_exists:
            f.write("gpus,total_ensemble,rank_ensemble,time_sec\n")
        f.write(f"{world_size},{total_ensemble},{rank_ensemble},{max_time:.4f}\n")

if is_distributed:
    dist.destroy_process_group()