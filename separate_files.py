import os
import shutil

# Paths for the datasets
train_folder = "PINN_Data/train"
val_folder = "PINN_Data/val"

# Create necessary subfolders
os.makedirs("PINN_Data/train/surface", exist_ok=True)
os.makedirs("PINN_Data/train/depth", exist_ok=True)
os.makedirs("PINN_Data/val/surface", exist_ok=True)
os.makedirs("PINN_Data/val/depth", exist_ok=True)

# Function to detect and move files
def separate_files(source_folder, target_surface, target_depth):
    for f in os.listdir(source_folder):
        if f.endswith(".csv"):
            if "surface" in f.lower():
                shutil.move(os.path.join(source_folder, f), os.path.join(target_surface, f))
            elif "depth" in f.lower():
                shutil.move(os.path.join(source_folder, f), os.path.join(target_depth, f))

# Separate the files
separate_files(train_folder, "PINN_Data/train/surface", "PINN_Data/train/depth")
separate_files(val_folder, "PINN_Data/val/surface", "PINN_Data/val/depth")

print("Datasets successfully separated into:")
print("- Train: Surface and Depth")
print("- Val: Surface and Depth")
