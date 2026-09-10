import warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from PIL import Image
import os

Path("outputs").mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load and Combine Data
# ---------------------------------------------------------------------------
paths = ["outputs/08_output_0.zarr", "outputs/08_output_1.zarr"]
paths = [p for p in paths if os.path.exists(p)]

if len(paths) < 2:
    raise FileNotFoundError("Need at least 2 Zarr files for a side-by-side comparison.")

# Combine along the 'time' dimension
ds = xr.open_mfdataset(paths, engine="zarr", concat_dim="time", combine="nested", consolidated=False)

n_leads     = len(ds.lead_time)
lead_hours  = ds.lead_time.values / np.timedelta64(1, "h")
lons, lats  = ds.lon.values, ds.lat.values
LON, LAT    = np.meshgrid(lons, lats)

vmin, vmax  = 0, 70
cmap        = "Blues"
central_lons = np.linspace(0, 360, n_leads, endpoint=False)

# ---------------------------------------------------------------------------
# Pre-load Features
# ---------------------------------------------------------------------------
LAND = cfeature.NaturalEarthFeature(category="physical", name="land", scale="110m",
                                    facecolor="#1a1a2e", edgecolor="none")
COASTLINE = cfeature.NaturalEarthFeature(category="physical", name="coastline", scale="110m",
                                         facecolor="none", edgecolor="white")

# ---------------------------------------------------------------------------
# Dual Rendering Loop
# ---------------------------------------------------------------------------
print(f"Rendering {n_leads} dual-globe frames...")

frames = []
for i in range(n_leads):
    # Wider figure for two globes
    fig = plt.figure(figsize=(16, 8), facecolor="#0d1117")
    
    # Common projection settings
    data_crs = ccrs.PlateCarree()
    ortho = ccrs.Orthographic(central_longitude=central_lons[i], central_latitude=20)

    # Loop through the two start dates
    for t_idx in range(2):
        start_date = str(ds.time.values[t_idx])[:10]
        ax = fig.add_subplot(1, 2, t_idx + 1, projection=ortho)
        ax.set_facecolor("#0a1628")
        ax.set_global()
        ax.add_feature(LAND, zorder=1)

        # Plot Data
        da = ds["tcwv"].isel(time=t_idx, lead_time=i).values
        ax.pcolormesh(LON, LAT, da, transform=data_crs, cmap=cmap, 
                      vmin=vmin, vmax=vmax, shading="auto", zorder=2, alpha=0.85)

        ax.add_feature(COASTLINE, linewidth=0.7, zorder=3)
        ax.set_title(f"Start: {start_date} | +{int(lead_hours[i])}h", color="white", fontsize=14)

    plt.suptitle("Earth-2 DLWP: Side-by-Side TCWV Forecast Comparison", 
                 color="white", fontsize=18, y=0.95)
    
    # Convert to image
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    frames.append(Image.fromarray(buf, mode="RGBA").convert("RGB"))
    plt.close(fig)
    
    if (i+1) % 5 == 0: print(f"   Frame {i+1}/{n_leads} done")

# ---------------------------------------------------------------------------
# Save Result
# ---------------------------------------------------------------------------
output_path = "outputs/tcwv_side_by_side.gif"
frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=150, loop=0)
print(f"Saved side-by-side animation to: {output_path}")