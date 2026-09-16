"""
earth2_offline.py  — v5 (all fixes)
-------------------------------------
Drop this file in your project directory and import from it.

Changes from previous versions:
  - OfflinePackage.get() added for FCN3/makani compatibility
  - _STATIC_MAP extended with 50m and 100m wind entries for FCN3
  - NetCDF4Backend uses mode='w' to overwrite existing files
  - ZarrBackend used automatically for FCN3
  - offline_model() factory handles both DLWP and FCN3

Usage:
    from earth2_offline import offline_model
    from earth2studio.run import deterministic as run

    model, ds, io = offline_model("dlwp", "output.nc")
    run(["2024-01-01"], 10, model, ds, io, device="cuda")

    model, ds, io = offline_model("fcn3", "fcn3_output.zarr")
    run(["2024-01-01"], 10, model, ds, io, device="cuda")
"""

import os
import re
import numpy as np
import xarray as xr
import cfgrib
from pathlib import Path
from datetime import datetime
from earth2studio.models.px import DLWP, FCN3
from earth2studio.io import NetCDF4Backend, ZarrBackend
from earth2studio.lexicon import GFSLexicon

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------
CACHE_PATH = os.environ.get("EARTH2STUDIO_CACHE", "/e/project1/e-ben-2026b09-120/earth2_cache")
os.environ["EARTH2STUDIO_CACHE"] = CACHE_PATH

GFS_LAT = np.linspace(90, -90, 721)
GFS_LON = np.linspace(0, 359.75, 1440)

GRIB_MAP_JAN2024 = {
    np.datetime64("2023-12-31T18:00", "ns"): f"{CACHE_PATH}/gfs/gfs.t18z.pgrb2.0p25.f000",
    np.datetime64("2024-01-01T00:00", "ns"): f"{CACHE_PATH}/gfs/gfs.t00z.pgrb2.0p25.f000",
}


# ---------------------------------------------------------------------------
# GFSLexicon -> cfgrib translation table
# Built from diagnose_grib.py output: NCEP names -> ECMWF/cfgrib names
# ---------------------------------------------------------------------------

_STATIC_MAP: dict[str, dict] = {
    # 2m / surface
    "TMP::2 m above ground":        {"shortName": "2t",    "typeOfLevel": "heightAboveGround", "level": 2},
    "SPFH::2 m above ground":       {"shortName": "2sh",   "typeOfLevel": "heightAboveGround", "level": 2},
    # 10m winds
    "UGRD::10 m above ground":      {"shortName": "10u",   "typeOfLevel": "heightAboveGround", "level": 10},
    "VGRD::10 m above ground":      {"shortName": "10v",   "typeOfLevel": "heightAboveGround", "level": 10},
    # 50m winds (FCN3)
    "UGRD::50 m above ground":      {"shortName": "u",     "typeOfLevel": "heightAboveGround", "level": 50},
    "VGRD::50 m above ground":      {"shortName": "v",     "typeOfLevel": "heightAboveGround", "level": 50},
    # 100m winds (FCN3)
    "UGRD::100 m above ground":     {"shortName": "u",     "typeOfLevel": "heightAboveGround", "level": 100},
    "VGRD::100 m above ground":     {"shortName": "v",     "typeOfLevel": "heightAboveGround", "level": 100},
    # Surface / sea level pressure
    "PRES::surface":                {"shortName": "sp",    "typeOfLevel": "surface"},
    "PRMSL::mean sea level":        {"shortName": "prmsl", "typeOfLevel": "meanSea"},
    # Total column water vapour
    "PWAT::entire atmosphere (considered as a single layer)":
                                    {"shortName": "pwat",  "typeOfLevel": "atmosphereSingleLayer"},
}

# NCEP shortName -> cfgrib shortName for isobaric (pressure-level) variables.
# The regex handles any level number so all pressure levels work automatically.
_ISOBARIC_MAP = {
    "TMP":  "t",
    "UGRD": "u",
    "VGRD": "v",
    "HGT":  "gh",
    "SPFH": "q",
    "RH":   "r",
    "VVEL": "w",
    "ABSV": "absv",
}

def _gfslex_to_filter(gfs_key: str) -> dict:
    """Convert a GFSLexicon key to a cfgrib filter_by_keys dict."""
    if gfs_key in _STATIC_MAP:
        return _STATIC_MAP[gfs_key]
    m = re.match(r"^([A-Z]+)::(\d+) mb$", gfs_key)
    if m:
        ncep_name, level = m.group(1), int(m.group(2))
        cfgrib_name = _ISOBARIC_MAP.get(ncep_name)
        if cfgrib_name:
            return {"shortName": cfgrib_name, "typeOfLevel": "isobaricInhPa", "level": level}
    raise KeyError(
        f"No cfgrib mapping for GFSLexicon key: '{gfs_key}'\n"
        f"Add it to _STATIC_MAP or _ISOBARIC_MAP in earth2_offline.py"
    )


# ---------------------------------------------------------------------------
# OfflinePackage — compatible with DLWP (.resolve) and FCN3/makani (.get)
# ---------------------------------------------------------------------------
class OfflinePackage:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Model cache not found: {self.root_dir}")

    def resolve(self, name: str) -> str:
        """Used by DLWP."""
        return str(self.root_dir / name)

    def get(self, name: str) -> str:
        """Used by FCN3/makani."""
        return str(self.root_dir / name)


# ---------------------------------------------------------------------------
# LocalGFSSource — earth2studio DataSource backed by local GRIB files
# ---------------------------------------------------------------------------
class LocalGFSSource:
    """
    Reads pre-downloaded GFS 0.25 degree GRIB2 files from disk.

    Parameters
    ----------
    grib_map : dict[np.datetime64, str]
        Maps each IC timestamp (datetime64[ns]) to its local GRIB file path.
    """

    def __init__(self, grib_map: dict):
        self.grib_map = grib_map

    def _read_variable(self, grib_path: str, gfs_key: str) -> np.ndarray:
        filter_keys = _gfslex_to_filter(gfs_key)
        datasets = cfgrib.open_datasets(
            grib_path,
            backend_kwargs={
                "filter_by_keys": filter_keys,
                "indexpath": "",
                "errors": "ignore",
            },
        )
        for ds in datasets:
            for var_name in ds.data_vars:
                da = ds[var_name].load()
                arr = np.array(da.values, dtype=np.float32)
                if arr.ndim == 2:
                    return arr
                if arr.ndim == 3 and arr.shape[0] == 1:
                    return arr[0]
        raise RuntimeError(
            f"cfgrib returned no data for '{gfs_key}' (filter={filter_keys})\n"
            f"File: {grib_path}"
        )

    def __call__(self, time, variable):
        if isinstance(time, (datetime, np.datetime64)):
            time = [time]
        times = np.array(time, dtype="datetime64[ns]")
        if isinstance(variable, str):
            variable = [variable]
        variables = list(variable)

        data = np.full(
            (len(times), len(variables), len(GFS_LAT), len(GFS_LON)),
            fill_value=np.nan, dtype=np.float32,
        )

        for t_idx, t in enumerate(times):
            grib_path = self.grib_map.get(t)
            if grib_path is None:
                raise KeyError(
                    f"No GRIB file for time {t}.\n"
                    f"Available: {sorted(str(k) for k in self.grib_map.keys())}"
                )
            for v_idx, var in enumerate(variables):
                gfs_key, modifier = GFSLexicon[var]
                arr = self._read_variable(grib_path, gfs_key)
                if arr.shape != (len(GFS_LAT), len(GFS_LON)):
                    raise ValueError(
                        f"Shape mismatch for '{var}': {arr.shape}, "
                        f"expected ({len(GFS_LAT)}, {len(GFS_LON)})"
                    )
                data[t_idx, v_idx] = modifier(arr)

        return xr.DataArray(
            data=data,
            dims=["time", "variable", "lat", "lon"],
            coords={"time": times, "variable": variables, "lat": GFS_LAT, "lon": GFS_LON},
        )


# ---------------------------------------------------------------------------
# Model registry & convenience factory
# ---------------------------------------------------------------------------
_MODEL_REGISTRY = {
    #  name    class   cache_subdir  io_type
    "dlwp": (DLWP,  "dlwp",  "netcdf4"),
    "fcn3": (FCN3,  "fcn3",  "zarr"),
    # Uncomment as you download more models:
    # "pangu6":  (Pangu6,  "pangu6",  "zarr"),
    # "atlas":   (Atlas,   "atlas",   "zarr"),
}

def offline_model(
    model_name: str,
    output_path: str = None,
    grib_map: dict = None,
    device: str = "cuda",
):
    """
    One-liner setup for an offline earth2studio forecast.

    Parameters
    ----------
    model_name  : str   "dlwp" or "fcn3" (or any key in _MODEL_REGISTRY)
    output_path : str   output file (defaults to output.nc or output.zarr)
    grib_map    : dict  timestamp->GRIB mapping (defaults to GRIB_MAP_JAN2024)
    device      : str   "cuda" or "cpu"

    Returns
    -------
    model, ds, io  -- pass directly to earth2studio.run.deterministic
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(_MODEL_REGISTRY)}")

    model_cls, cache_subdir, io_type = _MODEL_REGISTRY[model_name]
    package = OfflinePackage(f"{CACHE_PATH}/{cache_subdir}")

    print(f"Loading {model_name.upper()} from {package.root_dir} ...")
    model = model_cls.load_model(package).to(device)

    grib_map = grib_map or GRIB_MAP_JAN2024
    print(f"Setting up LocalGFSSource ({len(grib_map)} timestep(s))...")
    ds = LocalGFSSource(grib_map)

    if output_path is None:
        output_path = f"output.{'zarr' if io_type == 'zarr' else 'nc'}"

    if io_type == "zarr":
        io = ZarrBackend(output_path)
    else:
        # mode='w' overwrites any existing file from a previous run
        io = NetCDF4Backend(output_path, backend_kwargs={"mode": "w"})

    print(f"Output -> {output_path}")
    return model, ds, io
