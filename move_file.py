import os
import shutil

# Create folders
os.makedirs("PINN_Data/train", exist_ok=True)
os.makedirs("PINN_Data/validate", exist_ok=True)

# Updated logic: train on 75_450 and 250_600; validate on 400_800
for file in os.listdir("."):
    if file.startswith("PINN_Formatted") and file.endswith(".csv"):
        if "75_450" in file or "250_600" in file:
            shutil.move(file, f"PINN_Data/train/{file}")
        elif "400_800" in file:
            shutil.move(file, f"PINN_Data/validate/{file}")
