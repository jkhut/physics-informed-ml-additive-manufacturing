import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

# -------------------------------------------------------
# CONFIG: Update these before running
COMSOL_DATA_PATH = 'PINN_Data/val/surface/PINN_Formatted_top_surface_400_800.csv'
MODEL_PATH = 'pinn_model_physics_constrained.pth'
inner_radius = 0.000200   # Inner radius in meters
outer_radius = 0.000400   # Outer radius in meters
total_time = 0.003        # Time snapshot to validate at (seconds)
alpha = 8.4e-5            # Thermal diffusivity of Aluminum
# -------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Testing on: {device}")

comsol_data = pd.read_csv(COMSOL_DATA_PATH)
comsol_data = comsol_data[comsol_data['t [s]'] == total_time]
print(f"Filtered to {len(comsol_data)} points at t = {total_time}s")

z_comsol = comsol_data['z [m]'].values
T_comsol = comsol_data['T [K]'].values

scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
print("Scalers loaded.")

z_t_pairs = np.column_stack((z_comsol, np.full_like(z_comsol, total_time)))
z_t_normalized = scaler_X.transform(z_t_pairs)
X_test = torch.tensor(z_t_normalized, dtype=torch.float32, requires_grad=True).to(device)


class PhysicsNeuron(nn.Module):
    def __init__(self, alpha):
        super(PhysicsNeuron, self).__init__()
        self.alpha = alpha

    def forward(self, z, t, T_pred, inner_radius, outer_radius, Q):
        if not z.requires_grad:
            z = z.clone().detach().requires_grad_(True)
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)

        dT_dt = torch.autograd.grad(T_pred, t, grad_outputs=torch.ones_like(T_pred),
                                    create_graph=True, allow_unused=True)[0]
        dT_dz = torch.autograd.grad(T_pred, z, grad_outputs=torch.ones_like(T_pred),
                                    create_graph=True, allow_unused=True)[0]

        if dT_dt is None:
            dT_dt = torch.zeros_like(t).to(device)
        if dT_dz is None:
            dT_dz = torch.zeros_like(z).to(device)

        d2T_dz2 = torch.autograd.grad(dT_dz, z, grad_outputs=torch.ones_like(dT_dz),
                                      create_graph=True, allow_unused=True)[0]
        if d2T_dz2 is None:
            d2T_dz2 = torch.zeros_like(z).to(device)

        epsilon = 1e-5
        physics_residual = dT_dt - self.alpha * (d2T_dz2 + (1 / (z + epsilon)) * dT_dz) - Q
        return physics_residual


class PhysicsConstrainedNN(nn.Module):
    def __init__(self, alpha):
        super(PhysicsConstrainedNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )
        self.physics_neuron = PhysicsNeuron(alpha)

    def forward(self, inputs, inner_radius, outer_radius, Q):
        z, t = inputs[:, 0:1], inputs[:, 1:2]
        if not z.requires_grad:
            z = z.clone().detach().requires_grad_(True)
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)
        T_pred = self.layers(inputs)
        physics_residual = self.physics_neuron(z, t, T_pred, inner_radius, outer_radius, Q)
        return T_pred, physics_residual


model = PhysicsConstrainedNN(alpha=alpha).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print("Model loaded successfully.")

with torch.no_grad():
    absorptivity = 0.1
    energy = 7.9
    duration = 0.003
    P = energy / duration
    A = 3.1415 * (outer_radius ** 2 - inner_radius ** 2)
    Q = (P * absorptivity) / A
    predictions, _ = model(X_test, inner_radius, outer_radius, Q)

predicted_temperatures = scaler_y.inverse_transform(predictions.cpu().numpy())

mae = mean_absolute_error(T_comsol, predicted_temperatures)
rmse = mean_squared_error(T_comsol, predicted_temperatures, squared=False)
print(f"\nValidation Metrics:")
print(f"MAE:  {mae:.4f} K")
print(f"RMSE: {rmse:.4f} K")

plt.figure(figsize=(8, 6))
plt.plot(z_comsol * 1e6, T_comsol, label='COMSOL Temperature [K]', linestyle='--')
plt.plot(z_comsol * 1e6, predicted_temperatures, label='PINN Prediction [K]', linestyle='-')
plt.legend()
plt.title(f"Temperature Distribution at t = {total_time}s")
plt.xlabel("Radial Distance from Center [µm]")
plt.ylabel("Temperature [K]")
plt.grid(True)
plt.show()
