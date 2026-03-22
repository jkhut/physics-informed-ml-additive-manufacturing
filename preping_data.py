import pandas as pd
import numpy as np
import os

def get_top_and_depth_files():
    return [f for f in os.listdir(".") if f.lower().startswith(("top", "depth")) and f.endswith(".csv")]

def convert_to_flat_format(df):
    z_values = df["location"].values
    t_values = df.columns[1:].astype(float).values
    T_matrix = df.iloc[:, 1:].values

    Z, T = np.meshgrid(z_values, t_values, indexing="ij")
    z_flat = Z.flatten() * 1e-6  # µm to m
    t_flat = T.flatten()
    T_flat = T_matrix.flatten()

    return pd.DataFrame({"z [m]": z_flat,"t [s]": t_flat, "T [K]": T_flat})


# Convert and save all matched files
for file in get_top_and_depth_files():
    raw_df = pd.read_csv(file)
    flat_df = convert_to_flat_format(raw_df)
    out_filename = f"PINN_Formatted_{file}"
    flat_df.to_csv(out_filename, index=False)
    print(f"Converted and saved: {out_filename}")
