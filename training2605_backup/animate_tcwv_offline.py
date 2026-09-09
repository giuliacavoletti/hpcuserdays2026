"""
animate_tcwv_offline.py
-----------------------
Animation of TCWV across all lead times — no cartopy, fully offline.

Run:
    python animate_tcwv_offline.py

Output: outputs/tcwv_animation.gif
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
ds = xr.open_mfdataset(
    ["outputs/08_output_0.zarr", "outputs/08_output_1.zarr"],
    combine="nested", concat_dim="time", engine="zarr"
)
print(ds)

n_times    = len(ds.time)
n_leads    = len(ds.lead_time)
lead_hours = ds.lead_time.values / np.timedelta64(1, "h")

lons = ds.lon.values
lats = ds.lat.values
LON, LAT = np.meshgrid(lons, lats)

vmin, vmax = 0, 70
cmap = "Blues"
start_labels = [str(ds.time.values[i])[:10] for i in range(n_times)]

# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, n_times, figsize=(22, 6))
fig.patch.set_facecolor("#0d1117")

if n_times == 1:
    axes = [axes]

mesh_list = []
for i, ax in enumerate(axes):
    ax.set_facecolor("#0d1117")

    da = ds["tcwv"].isel(time=i, lead_time=0).values
    mesh = ax.pcolormesh(
        LON, LAT, da,
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto",
    )
    mesh_list.append(mesh)

    cb = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.08,
                      shrink=0.8, aspect=30)
    cb.set_label("Total Column Water Vapour (kg/m²)", color="white", fontsize=9)
    cb.ax.xaxis.set_tick_params(color="white")
    plt.setp(cb.ax.xaxis.get_ticklabels(), color="white")
    cb.outline.set_edgecolor("white")

    ax.set_title(f"Start: {start_labels[i]}", color="white", fontsize=11, pad=8)
    ax.set_xlabel("Longitude", color="white", fontsize=9)
    ax.set_ylabel("Latitude",  color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    # Simple lat/lon grid lines
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(color="#333333", linewidth=0.4, linestyle="--")

title = fig.suptitle(
    "DLWP  ·  TCWV  ·  Lead time: +0 h",
    color="white", fontsize=13, y=1.01,
)
plt.tight_layout()

# ---------------------------------------------------------------------------
# Update function
# ---------------------------------------------------------------------------
def update(frame):
    for i, mesh in enumerate(mesh_list):
        da = ds["tcwv"].isel(time=i, lead_time=frame).values
        mesh.set_array(da.ravel())
    title.set_text(f"DLWP  ·  TCWV  ·  Lead time: +{int(lead_hours[frame])} h")
    return mesh_list

# ---------------------------------------------------------------------------
# Render and save
# ---------------------------------------------------------------------------
print(f"Rendering {n_leads} frames ...")
ani = animation.FuncAnimation(
    fig, update, frames=n_leads, interval=250, blit=True
)

output_path = "outputs/tcwv_animation.gif"
ani.save(output_path, writer="pillow", fps=5, dpi=130,
         savefig_kwargs={"facecolor": "#0d1117"})
print(f"Saved {output_path}")
plt.close()
