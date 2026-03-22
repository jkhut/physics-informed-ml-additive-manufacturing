import pandas as pd
import numpy as np



# Load the file (assuming it's CSV)
df = pd.read_csv("temperature_data.csv")  # Replace with actual filename
time_steps = pd.read_csv("comsol_time_steps.csv")
# Get column 0 as a Series
col0 = df.iloc[:, 0]
col1 = df.iloc[:, 1]
time_series = time_steps.iloc[:, 0]
all_times = time_series.tolist()
# Find all indexes where value is 0
zero_indices = col0[col0 == 0].index.tolist()

new_dict = {"location": df.iloc[zero_indices[0]:zero_indices[1], 0].values.tolist()}

i = 0
while i < len(all_times)-1:
    new_dict[all_times[i]] = df.iloc[zero_indices[i]:zero_indices[i+1], 1].values.tolist()
    i += 1

new_dict[all_times[-1]] = df.iloc[zero_indices[-1]:, 1].values.tolist()
# Assuming new_dict is already populated
df_out = pd.DataFrame.from_dict(new_dict)

# Save to CSV
df_out.to_csv("extracted_temperature_by_time.csv", index=False)




