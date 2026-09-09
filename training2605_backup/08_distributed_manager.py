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

#torch.distributed.barrier()
if dist.rank != 0:
    model = DLWP.load_model(package)
timings["model_load"] = time.time() - start_time


# ---------------------------------------------------------------------------
# Data source — local GRIB files, no internet needed
# Add more dates here if you download more GRIB files
# ---------------------------------------------------------------------------
start_time = time.time()
GRIB_MAP = {
    # Jan 1 2024
    np.datetime64("2023-12-31T18:00", "ns"): f"{CACHE_PATH}/gfs/gfs.20231231.t18z.pgrb2.0p25.f000",
    np.datetime64("2024-01-01T00:00", "ns"): f"{CACHE_PATH}/gfs/gfs.20240101.t00z.pgrb2.0p25.f000",
    # Jan 15 2024
    np.datetime64("2024-01-14T18:00", "ns"): f"{CACHE_PATH}/gfs/gfs.20240114.t18z.pgrb2.0p25.f000",
    np.datetime64("2024-01-15T00:00", "ns"): f"{CACHE_PATH}/gfs/gfs.20240115.t00z.pgrb2.0p25.f000",
}
data = LocalGFSSource(GRIB_MAP)
timings["data_load"] = time.time() - start_time

# ---------------------------------------------------------------------------
# IO — each GPU writes its own file
# ---------------------------------------------------------------------------
chunks = {"time": 1, "lead_time": 1}
io = ZarrBackend(
    file_name=f"outputs/08_output_{dist.rank}.zarr",
    chunks=chunks,
    backend_kwargs={"overwrite": True},
)

# ---------------------------------------------------------------------------
# Times — add one date per GPU to keep all GPUs busy
# With only 1 date and 2 GPUs, one GPU will sit idle (fine for testing)
# ---------------------------------------------------------------------------
times = np.array([
    "2024-01-01T00:00:00",
    "2024-01-15T00:00:00",  #  uncomment and download GRIB files to use 2 GPUs
])

time_shard = np.array_split(times, dist.world_size)[dist.rank]

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
nsteps = 20
output_coords = {"variable": np.array(["tcwv"])}
start_time = time.time()

if len(time_shard) > 0:
    io = run.deterministic(
        time_shard, nsteps, model, data, io,
        output_coords=output_coords,
        device=dist.device,
    )
    print(io.root.tree())
else:
    logger.info(f"Rank {dist.rank}: no dates assigned, sitting idle.")

torch.distributed.barrier()
timings["inference"] = time.time() - start_time

# ---------------------------------------------------------------------------
# Timing summary
# ---------------------------------------------------------------------------
if dist.rank == 0:
 total = sum(timings.values())
 print(f"\n{'='*45}")
 print(f"  Timing Summary — Rank {dist.rank} / {dist.world_size} (device: {dist.device})")
 print(f"{'='*45}")
 for key, val in timings.items():
    print(f"  {key:<20} {val:>8.2f}s  ({100*val/total:.1f}%)")
 print(f"  {'TOTAL':<20} {total:>8.2f}s")
 print(f"{'='*45}\n")

torch.distributed.barrier()

if dist.rank != 0:
 total = sum(timings.values())
 print(f"\n{'='*45}")
 print(f"  Timing Summary — Rank {dist.rank} / {dist.world_size} (device: {dist.device})")
 print(f"{'='*45}")
 for key, val in timings.items():
    print(f"  {key:<20} {val:>8.2f}s  ({100*val/total:.1f}%)")
 print(f"  {'TOTAL':<20} {total:>8.2f}s")
 print(f"{'='*45}\n")


# ---------------------------------------------------------------------------
# Post processing — only rank 0
# ---------------------------------------------------------------------------
if dist.rank == 0:
    import matplotlib.pyplot as plt
    import xarray as xr
    from earth2studio.utils.time import timearray_to_datetime

    # Only load files from ranks that actually ran
    n_active = min(len(times), dist.world_size)
    paths = [f"outputs/08_output_{i}.zarr" for i in range(n_active)]
    paths = [p for p in paths if os.path.exists(p)]

    if paths:
        ds = xr.open_mfdataset(
            paths, combine="nested", concat_dim="time", engine="zarr"
        )
        print(ds)

        time = timearray_to_datetime(
            ds.coords["time"].values.astype("datetime64[ns]")
        )
        n_times = len(time)
        ncols = min(3, n_times)
        nrows = (n_times + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
        )

        for i in range(n_times):
            ax = axes[i // ncols, i % ncols]
            ax.imshow(
                ds["tcwv"].isel(time=i, lead_time=-1).values,
                cmap="gist_earth", vmin=0, vmax=100,
            )
            ax.set_title(time[i].strftime("%Y-%m-%d"))

        # Hide any unused subplots
        for j in range(n_times, nrows * ncols):
            axes[j // ncols, j % ncols].set_visible(False)

        lead_days = (
            ds.coords["lead_time"].values[-1]
            .astype("timedelta64[ns]")
            .astype("timedelta64[D]")
            .astype(int)
        )
        plt.suptitle(f"TCWV — Forecast Lead Time: {lead_days} days")
        plt.tight_layout()
        plt.savefig("outputs/08_tcwv_distributed_manager.jpg", dpi=150)
        print("Saved outputs/08_tcwv_distributed_manager.jpg")