import pandas as pd
import re

# Step 1: Read the text file
with open("time_steps.txt", "r") as file:
    lines = file.readlines()

# Step 2: Extract numeric time values using regex
time_values = []
for line in lines:
    match = re.match(r"^\s*([\d\.\+\-E]+)", line)  # handles scientific notation too
    if match:
        time_values.append(float(match.group(1)))

# Step 3: Create a DataFrame
df_time = pd.DataFrame(time_values, columns=["Time (s)"])

# Step 4: Save to CSV
df_time.to_csv("comsol_time_steps.csv", index=False)
