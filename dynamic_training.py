import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import random
import joblib

# -------------------------------------------------------
# CONFIG: Set these to your local dataset folder paths
TRAIN_PATH_1 = 'PINN_Data/train/surface'
TRAIN_PATH_2 = 'PINN_Data/train/surface_125_300'
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


train_data_1 = load_data(TRAIN_PATH_1)
train_data_2 = load_data(TRAIN_PATH_2)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

combined_X = pd.concat([train_data_1[['z [m]', 't [s]']], train_data_2[['z [m]', 't [s]']]])
combined_y = pd.concat([train_data_1[['T [K]']], train_data_2[['T [K]']]])

X_combined = scaler_X.fit_transform(combined_X.values)
y_combined = scaler_y.fit_transform(combined_y.values)

X_train_1 = X_combined[:len(train_data_1)]
y_train_1 = y_combined[:len(train_data_1)]
X_train_2 = X_combined[len(train_data_1):]
y_train_2 = y_combined[len(train_data_1):]

joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("Scalers saved.")

X_train_1 = torch.tensor(X_train_1, dtype=torch.float32, requires_grad=True).to(device)
y_train_1 = torch.tensor(y_train_1, dtype=torch.float32).to(device)
X_train_2 = torch.tensor(X_train_2, dtype=torch.float32, requires_grad=True).to(device)
y_train_2 = torch.tensor(y_train_2, dtype=torch.float32).to(device)


class PhysicsNeuron(nn.Module):
    def __init__(self, alpha):
        super(PhysicsNeuron, self).__init__()
        self.alpha = alpha

    def forward(self, z, t, T_pred, inner_radius, outer_radius, Q):
        z = z.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        dT_dt = torch.autograd.grad(T_pred, t, grad_outputs=torch.ones_like(t),
                                    create_graph=True, allow_unused=True)[0]
        dT_dz = torch.autograd.grad(T_pred, z, grad_outputs=torch.ones_like(z),
                                    create_graph=True, allow_unused=True)[0]

        if dT_dt is None:
            dT_dt = torch.zeros_like(t).to(device)
        if dT_dz is None:
            dT_dz = torch.zeros_like(z).to(device)

        if dT_dz.requires_grad:
            d2T_dz2 = torch.autograd.grad(dT_dz, z, grad_outputs=torch.ones_like(z),
                                          create_graph=True, allow_unused=True)[0]
        else:
            d2T_dz2 = torch.zeros_like(z).to(device)

        if d2T_dz2 is None:
            d2T_dz2 = torch.zeros_like(z).to(device)

        epsilon = 1e-3
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
        T_pred = self.layers(inputs)
        physics_residual = self.physics_neuron(z, t, T_pred, inner_radius, outer_radius, Q)
        return T_pred, physics_residual


model = PhysicsConstrainedNN(alpha=8.4e-5).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
print_interval = 100

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    if random.choice([0, 1]) == 0:
        X_train, y_train = X_train_1, y_train_1
        inner_radius, outer_radius = 0.0000375, 0.000225
    else:
        X_train, y_train = X_train_2, y_train_2
        inner_radius, outer_radius = 0.000125, 0.000300

    absorptivity = 0.1
    energy = 7.9
    duration = 0.003
    P = energy / duration
    A = 3.1415 * (outer_radius ** 2 - inner_radius ** 2)
    Q = (P * absorptivity) / A

    predictions, physics_residual = model(X_train, inner_radius, outer_radius, Q)
    data_loss = criterion(predictions, y_train)
    physics_loss = torch.mean(physics_residual ** 2)
    total_loss = data_loss + physics_loss
    total_loss.backward()
    optimizer.step()

    if epoch % print_interval == 0:
        print(f'Epoch {epoch}: Data Loss = {data_loss.item():.6f} | '
              f'Physics Loss = {physics_loss.item():.6f} | Total Loss = {total_loss.item():.6f}')

torch.save(model.state_dict(), 'pinn_model_physics_constrained.pth')
print("\nModel saved as pinn_model_physics_constrained.pth")
