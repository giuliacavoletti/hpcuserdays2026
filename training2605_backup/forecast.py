import os
import numpy as np
from collections import OrderedDict
from earth2studio.models.px import Persistence
from earth2studio.data import Random
from earth2studio.io import ZarrBackend
from earth2studio.run import deterministic as run

# 1. Define your grid and variables (The "Manual" setup)
variables = ["t2m", "u10m"]
domain_coords = OrderedDict({
    "lat": np.linspace(-90, 90, 181),
    "lon": np.linspace(0, 360, 360, endpoint=False)
})

def main():
    print("Starting Corrected Zero-Dependency Smoke Test...")
    os.makedirs("outputs", exist_ok=True)

    # 2. Pass the required arguments to Persistence
    # It needs to know which variables to "persist" and on what grid.
    model = Persistence(variable=variables, domain_coords=domain_coords)

    # 3. Pass the same grid to the Random data source
    data_source = Random(domain_coords=domain_coords)

    # 4. Standard IO Backend
    io_handler = ZarrBackend("outputs/minimal_test.zarr")

    # 5. Run the forecast
    print("Running 5-step persistence forecast on a 1-degree global grid...")
    run(["2025-01-01T00:00:00"], 5, model, data_source, io_handler)
    
    print("Success! Results saved to outputs/minimal_test.zarr")

if __name__ == "__main__":
    main()