import os
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

# ---------------------------------------------------------------------------
# Setup PyTorch Distributed Environment
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

# ---------------------------------------------------------------------------
# Model & Offline Data
# ---------------------------------------------------------------------------
package = OfflinePackage(f"{CACHE_PATH}/fcn")
model = FCN.load_model(package).to(device)

sg = SphericalGaussian(noise_amplitude=0.15)

GRIB_MAP = {
    np.datetime64("2024-01-01T00:00", "ns"): f"{CACHE_PATH}/gfs/gfs.20240101.t00z.pgrb2.0p25.f000",
}
data = LocalGFSSource(GRIB_MAP)

# ---------------------------------------------------------------------------
# Workload Partitioning (8 members across 4 GPUs = 2 members per GPU)
# ---------------------------------------------------------------------------
total_ensemble = 8
rank_ensemble = total_ensemble // world_size
nsteps = 10
batch_size = rank_ensemble  # Run local members simultaneously on each GH200

# Each GPU rank writes to its own separate Zarr store
output_path = f"/e/scratch/e-ben-2026b09-120/cavoletti1/earth2_fast/outputs/ensemble_fcn_rank{rank}.zarr"
io = ZarrBackend(
    file_name=output_path,
    chunks={"ensemble": 1, "time": 1, "lead_time": 1},
    backend_kwargs={"overwrite": True},
)

if rank == 0:
    print(f"Running {total_ensemble}-member ensemble across {world_size} GPUs ({rank_ensemble} per GPU)...")

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

if is_distributed:
    dist.barrier()

if rank == 0:
    print("Inference complete on all GPUs!")