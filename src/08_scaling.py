import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["NCCL_DEBUG"] = "OFF"

import warnings
import logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
import torch
from loguru import logger
from physicsnemo.distributed import DistributedManager

from earth2_offline import LocalGFSSource, OfflinePackage, CACHE_PATH
from earth2studio.io import ZarrBackend
from earth2studio.models.px import DLWP
import earth2studio.run as run
import time

# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------
DistributedManager.initialize()
dist = DistributedManager()

logger.info(
    f"Inference runner {dist.rank} of {dist.world_size} with device {dist.device}"
)

timings = {}

# ---------------------------------------------------------------------------
# Model — rank 0 loads first, others wait, then load from local cache
# ---------------------------------------------------------------------------
start_time = time.time()
package = OfflinePackage(f"{CACHE_PATH}/dlwp")
if dist.rank == 0:
    model = DLWP.load_model(package)

if dist.rank != 0:
    model = DLWP.load_model(package)
timings["model_load"] = time.time() - start_time


# ---------------------------------------------------------------------------
# Data source — local GRIB files, no internet needed
# ---------------------------------------------------------------------------
start_time = time.time()
GRIB_MAP = {
    # Jan 1 2024
    np.datetime64("2023-12-31T18:00", "ns"): f"{CACHE_PATH}/gfs/gfs.20231231.t18z.pgrb2.0p25.f000",
    np.datetime64("2024-01-01T00:00", "ns"): f"{CACHE_PATH}/gfs/gfs.20240101.t00z.pgrb2.0p25.f000",
    # Jan 15 2024
    np.datetime64("2024-01-14T18:00", "ns"): f"{CACHE_PATH}/gfs/gfs.20240114.t18z.pgrb2.0p25.f000",
    np.datetime64("2024-01-15T00:00", "ns"): f"{CACHE_PATH}/gfs/gfs.20240115.t00z.pgrb2.0p25.f000",
    # Aug 1 2026
    np.datetime64("2026-07-31T18:00", "ns"): f"{CACHE_PATH}/gfs/gfs.20260731.t18z.pgrb2.0p25.f000",
    np.datetime64("2026-08-01T00:00", "ns"): f"{CACHE_PATH}/gfs/gfs.20260801.t00z.pgrb2.0p25.f000",
    # Aug 15 2026
    np.datetime64("2026-08-14T18:00", "ns"): f"{CACHE_PATH}/gfs/gfs.20260814.t18z.pgrb2.0p25.f000",
    np.datetime64("2026-08-15T00:00", "ns"): f"{CACHE_PATH}/gfs/gfs.20260815.t00z.pgrb2.0p25.f000",
}
data = LocalGFSSource(GRIB_MAP)
timings["data_load"] = time.time() - start_time

# ---------------------------------------------------------------------------
# IO — each GPU writes its own file
# ---------------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
chunks = {"time": 1, "lead_time": 1}
io = ZarrBackend(
    file_name=f"outputs/weak_output_{dist.rank}.zarr",
    chunks=chunks,
    backend_kwargs={"overwrite": True},
)

# ---------------------------------------------------------------------------
# Split the time range among the available GPUs
# ---------------------------------------------------------------------------
all_times = np.array([
    "2024-01-01T00:00:00",
    "2024-01-15T00:00:00",
    "2026-08-01T00:00:00",
    "2026-08-15T00:00:00",
])

# If you use N GPUs, you select the first N dates for weak scaling
times = all_times[:dist.world_size]
time_shard = np.array_split(times, dist.world_size)[dist.rank]

# ---------------------------------------------------------------------------
# Run inference with global timing sync
# ---------------------------------------------------------------------------
nsteps = 20
output_coords = {"variable": np.array(["tcwv"])}

torch.cuda.synchronize()
start_time = time.time()

if len(time_shard) > 0:
    io = run.deterministic(
        time_shard, nsteps, model, data, io,
        output_coords=output_coords,
        device=dist.device,
    )

torch.cuda.synchronize()
inference_time_sec = time.time() - start_time
times_tensor = torch.tensor([inference_time_sec], device=dist.device)

# Synchronize to find the maximum inference time across all active GPUs
torch.distributed.all_reduce(times_tensor, op=torch.distributed.ReduceOp.MAX)
max_inference_time = times_tensor.item()
timings["inference"] = max_inference_time

torch.distributed.barrier()

# Save weak scaling results to CSV
if dist.rank == 0:
    csv_file = "scaling_results_weak.csv"
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a") as f:
        if not file_exists:
            f.write("gpus,total_dates,time_sec\n")
        f.write(f"{dist.world_size},{len(times)},{max_inference_time:.4f}\n")
    print(f"Logged weak scaling result: {dist.world_size} GPUs, {max_inference_time:.4f}s")