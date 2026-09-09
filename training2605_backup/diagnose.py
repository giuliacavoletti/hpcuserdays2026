"""
Run this first to see what keys your GRIB files expose vs what GFSLexicon expects.
Usage: python diagnose_grib.py
"""
import cfgrib
import numpy as np
from earth2studio.lexicon import GFSLexicon

GRIB_FILE = "/e/project1/training2605/earth2_cache/gfs/gfs.t00z.pgrb2.0p25.f000"

print("=" * 60)
print("KEYS ACTUALLY IN YOUR GRIB FILE (via cfgrib.open_datasets)")
print("=" * 60)
datasets = cfgrib.open_datasets(GRIB_FILE, backend_kwargs={"indexpath": ""})
grib_keys_found = set()
for ds in datasets:
    type_of_level = ds.attrs.get("GRIB_typeOfLevel", "")
    for var_name in ds.data_vars:
        da = ds[var_name]
        short_name = da.attrs.get("GRIB_shortName", var_name)
        lev_type   = da.attrs.get("GRIB_typeOfLevel", type_of_level)
        level_val  = da.attrs.get("GRIB_level", "?")
        cf_name    = da.attrs.get("GRIB_cfName", "")
        param_id   = da.attrs.get("GRIB_paramId", "")

        if lev_type == "heightAboveGround":
            level_str = f"{level_val} m above ground"
        elif lev_type == "isobaricInhPa":
            level_str = f"{level_val} mb"
        elif lev_type == "surface":
            level_str = "surface"
        else:
            level_str = lev_type

        key = f"{short_name}::{level_str}"
        grib_keys_found.add(key)
        print(f"  ds_varname={var_name:<15} shortName={short_name:<12} "
              f"cfName={cf_name:<20} level={level_str}")

print()
print("=" * 60)
print("KEYS EXPECTED BY GFSLexicon (for common DLWP variables)")
print("=" * 60)
# Sample of variables DLWP commonly requests
sample_vars = [
    "u10m", "v10m", "t2m", "sp", "msl",
    "z50", "z100", "z250", "z500", "z850", "z1000",
    "t50", "t100", "t250", "t500", "t850",
    "u100", "u250", "u500", "u850",
    "v100", "v250", "v500", "v850",
    "q50", "q100", "q250", "q500", "q700", "q850",
    "tcwv",
]
for v in sample_vars:
    try:
        gfs_key, _ = GFSLexicon[v]
        found = "found" if any(gfs_key in k or k in gfs_key for k in grib_keys_found) else " MISSING"
        print(f"  {v:<10} → GFSLexicon key: {gfs_key:<40} {found}")
    except KeyError:
        print(f"  {v:<10} → NOT IN GFSLexicon")

print()
print("=" * 60)
print("RAW GRIB KEYS (shortName::level) — copy these if you need to remap")
print("=" * 60)
for k in sorted(grib_keys_found):
    print(f"  {k}")