"""
ensemble_offline.py
-------------------
Offline ensemble forecast using FCN + SphericalGaussian perturbation.
Adapted from earth2studio example 03 for JUPITER cluster (no internet on GPU nodes).

Run on GPU node:
    source /e/project1/e-ben-2026b09-120/earth2_final_env/bin/activate
    export EARTH2STUDIO_CACHE="/e/project1/e-ben-2026b09-120/earth2_cache"
    export TORCHDYNAMO_DISABLE=1
    export LD_LIBRARY_PATH=~/cuda_libs:$LD_LIBRARY_PATH
    python ensemble_offline.py

What to download first (on login node):
    python -c "
    import os
    os.environ['EARTH2STUDIO_CACHE'] = '/e/project1/e-ben-2026b09-120/earth2_cache'
    from earth2studio.models.px import FCN
    FCN.load_model(FCN.load_default_package())
    print('FCN weights downloaded')
    "
"""

import warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed on cluster
import matplotlib.pyplot as plt

from earth2studio.io import ZarrBackend
from earth2studio.models.px import FCN
from earth2studio.perturbation import SphericalGaussian
from earth2studio.run import ensemble
from earth2_offline import LocalGFSSource, OfflinePackage, CACHE_PATH

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------------
# Model — FCN needs only ONE initial timestep (unlike DLWP which needs two)
# ---------------------------------------------------------------------------
package = OfflinePackage(f"{CACHE_PATH}/fcn")
model = FCN.load_model(package)

# ---------------------------------------------------------------------------
# Perturbation method
# Small Gaussian noise added to initial conditions to generate ensemble spread
# noise_amplitude=0.15 is the standard value from the earth2studio examples
# ---------------------------------------------------------------------------
sg = SphericalGaussian(noise_amplitude=0.15)

# ---------------------------------------------------------------------------
# Data source — FCN only needs t0, no t-6h required
# ---------------------------------------------------------------------------
GRIB_MAP = {
    np.datetime64("2024-01-01T00:00", "ns"): f"{CACHE_PATH}/gfs/gfs.20240101.t00z.pgrb2.0p25.f000",
}
data = LocalGFSSource(GRIB_MAP)

# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
io = ZarrBackend(
    file_name="outputs/ensemble_fcn.zarr",
    chunks={"ensemble": 1, "time": 1, "lead_time": 1},
    backend_kwargs={"overwrite": True},
)

# ---------------------------------------------------------------------------
# Run ensemble
# 8 members, 10 steps (60 hours), batch size 4 (runs 2 batches of 4)
# Saving t2m and tcwv only to keep output small
# ---------------------------------------------------------------------------
nsteps    = 10
nensemble = 8
batch_size = 4

print(f"Running {nensemble}-member FCN ensemble for 2024-01-01 ...")
io = ensemble(
    ["2024-01-01"],
    nsteps,
    nensemble,
    model,
    data,
    io,
    sg,
    batch_size=batch_size,
    output_coords={"variable": np.array(["t2m", "tcwv"])},
    device="cuda",
)
print("Inference complete.")

# ---------------------------------------------------------------------------
# Plotting — 3 panels: member 0, member 1, ensemble std deviation
# Uses pre-downloaded cartopy shapefiles (no internet needed)
# ---------------------------------------------------------------------------
import cartopy.crs as ccrs
import cartopy.feature as cfeature

LAND = cfeature.NaturalEarthFeature(
    category="physical", name="land", scale="110m",
    facecolor="#dddddd", edgecolor="none",
)
COASTLINE = cfeature.NaturalEarthFeature(
    category="physical", name="coastline", scale="110m",
    facecolor="none", edgecolor="black",
)

def plot_panel(ax, data, title, cmap, vmin=None, vmax=None):
    im = ax.pcolormesh(
        io["lon"][:], io["lat"][:], data,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto",
    )
    plt.colorbar(im, ax=ax, shrink=0.6, pad=0.04)
    ax.add_feature(LAND, zorder=1)
    ax.add_feature(COASTLINE, linewidth=0.5, zorder=2)
    ax.gridlines(color="#aaaaaa", linewidth=0.3, linestyle="--")
    ax.set_title(title, fontsize=10)

for variable, cmap in [("tcwv", "Blues"), ("t2m", "RdYlBu_r")]:
    for step in [4, 8]:  # 24h and 48h lead times
        lead_h = step * 6
        field   = io[variable][:, 0, step]   # shape: (nensemble, lat, lon)
        std_map = np.std(field, axis=0)
        vmin    = float(np.percentile(field, 2))
        vmax    = float(np.percentile(field, 98))

        plt.close("all")
        proj = ccrs.Robinson()
        fig, axes = plt.subplots(
            1, 3, figsize=(18, 4),
            subplot_kw={"projection": proj},
        )

        plot_panel(axes[0], field[0], f"Member 0 | +{lead_h}h", cmap, vmin, vmax)
        plot_panel(axes[1], field[1], f"Member 1 | +{lead_h}h", cmap, vmin, vmax)
        plot_panel(axes[2], std_map,  f"Std dev (N={nensemble}) | +{lead_h}h", "Oranges")

        fig.suptitle(
            f"FCN ensemble  |  2024-01-01  |  {variable.upper()}  |  Lead: +{lead_h}h",
            fontsize=12, y=1.02,
        )
        plt.tight_layout()
        out = f"outputs/ensemble_{variable}_{lead_h}h.jpg"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")

print("All plots saved to outputs/")