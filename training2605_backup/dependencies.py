import torch
from earth2studio.models.px import SFNO
from earth2studio.data import GFS

# 1. Check GPU
if torch.cuda.is_available():
    print(f"Using Device: {torch.cuda.get_device_name(0)}")
else:
    print("GPU NOT FOUND! Check your apptainer --nv flag.")

# 2. Load Model 
# Note: Syntax changed in 0.12.x from .load_model_package to direct instantiation
print("Loading SFNO model layers and weights...")
try:
    package = SFNO.load_default_package()
    model = SFNO(package).to("cuda") # This is where torch-harmonics is checked
    print("SUCCESS: SFNO is loaded and on GPU!")
except Exception as e:
    print(f"ERROR DURING LOADING: {e}")

# 3. Check model device
try:
    print(f"Model is on: {next(model.parameters()).device}")
except:
    pass