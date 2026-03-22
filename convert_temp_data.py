import numpy as np
import matplotlib.pyplot as plt

filename = 'Test.txt'

# === Step 1: Skip header and load only numeric lines ===
data_lines = []
with open(filename, 'r') as f:
    for line in f:
        if line.strip() == "" or line.startswith("%"):
            continue  # Skip header and empty lines
        tokens = line.strip().split()
        try:
            nums = list(map(float, tokens))
            data_lines.append(nums)
        except:
            continue  # Skip malformed lines

# === Step 2: Convert to NumPy array ===
data_array = np.array(data_lines)

# === Step 3: Check if shape is valid ===
print(f"Loaded data shape: {data_array.shape}")
if data_array.shape[0] < 2 or data_array.shape[1] < 2:
    raise ValueError("Data is too small or flat. Check file contents.")

# === Step 4: Define space and time ===
n_time_steps = data_array.shape[0]
n_space_points = data_array.shape[1]

time = np.linspace(0, 3e-3, n_time_steps)        # 0 to 3 ms
space = np.linspace(0, 5e-3, n_space_points)     # 0 to 5 mm

# === Step 5: Plot ===
plt.figure(figsize=(10, 6))
plt.contourf(space * 1e3, time * 1e3, data_array, levels=50, cmap='hot')
plt.xlabel('Arc Length [mm]')
plt.ylabel('Time [ms]')
plt.title('Temperature T(x, t)')
plt.colorbar(label='Temperature [K]')
plt.tight_layout()
plt.show()

import pandas as pd

# Optional: create headers for arc length positions
space = np.linspace(0, 5e-3, data_array.shape[1])  # 0 to 5 mm
headers = [f"{x*1e3:.2f} mm" for x in space]       # header in mm

# Save to CSV
df = pd.DataFrame(data_array, columns=headers)
df.to_csv("temperature_data.csv", index=False)
print("Saved as temperature_data.csv")

