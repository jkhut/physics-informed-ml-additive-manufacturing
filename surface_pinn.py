import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# -------------------------------------------------------
# CONFIG: Set these to your local dataset folder paths
TRAIN_PATH = 'PINN_Data/train/surface'
VAL_PATH = 'PINN_Data/val/surface'
# -------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")


def load_data(path):
    """Load and concatenate all CSV files from a directory."""
    data_list = []
    for file in os.listdir(path):
        if file.endswith('.csv'):
            data = pd.read_csv(os.path.join(path, file))
            data_list.append(data)
    return pd.concat(data_list, axis=0)


train_data = load_data(TRAIN_PATH)
val_data = load_data(VAL_PATH)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(train_data[['z [m]', 't [s]']].values)
y_train = scaler_y.fit_transform(train_data[['T [K]']].values)
X_val = scaler_X.transform(val_data[['z [m]', 't [s]']].values)
y_val = scaler_y.transform(val_data[['T [K]']].values)

X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32, requires_grad=True).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)


class PINNModel(nn.Module):
    def __init__(self):
        super(PINNModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.layers(x)


def physics_loss(model, inputs):
    """Compute physics-informed loss based on heat conduction in cylindrical coordinates."""
    z, t = inputs[:, 0:1], inputs[:, 1:2]
    inputs = torch.cat((z, t), dim=1)
    T_pred = model(inputs)

    dT_dt = torch.autograd.grad(T_pred, t, grad_outputs=torch.ones_like(t).to(device),
                                create_graph=True, allow_unused=True)[0]
    dT_dz = torch.autograd.grad(T_pred, z, grad_outputs=torch.ones_like(z).to(device),
                                create_graph=True, allow_unused=True)[0]
    d2T_dz2 = torch.autograd.grad(dT_dz, z, grad_outputs=torch.ones_like(z).to(device),
                                  create_graph=True, allow_unused=True)[0]

    # Laser & material parameters
    alpha = 8.4e-5       # Thermal diffusivity of Aluminum
    E = 7.9              # Energy in Joules
    duration = 0.003     # Pulse duration in seconds
    P = E / duration     # Power (W)
    R_inner = 0.0000375  # Inner radius: 75 µm
    R_outer = 0.000225   # Outer radius: 450 µm
    A = 3.1415 * (R_outer ** 2 - R_inner ** 2)
    I = (0.1 * P) / A

    Q = torch.zeros_like(z).to(device)
    Q[(z >= R_inner) & (z <= R_outer)] = I

    physics_residual = dT_dt - alpha * (d2T_dz2 + (1 / (z + 1e-5)) * dT_dz) - Q
    return torch.mean(physics_residual ** 2)


model = PINNModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 500
print_interval = 10

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    predictions = model(X_train)
    data_loss = criterion(predictions, y_train)
    phys_loss = physics_loss(model, X_train)
    loss = data_loss + phys_loss
    loss.backward()
    optimizer.step()

    if epoch % print_interval == 0:
        print(f'Epoch {epoch}: Data Loss = {data_loss.item():.6f} | '
              f'Physics Loss = {phys_loss.item():.6f} | Total Loss = {loss.item():.6f}')

model.eval()
with torch.no_grad():
    val_predictions = model(X_val)
    val_loss = criterion(val_predictions, y_val)
    print(f'\nFinal Validation Loss: {val_loss.item():.6f}')

torch.save(model.state_dict(), 'pinn_model_cuda.pth')
print("\nModel saved as pinn_model_cuda.pth")
