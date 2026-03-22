import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# -------------------------------------------------------
# CONFIG: Set this to your local validation dataset path
TEST_PATH = 'PINN_Data/val/surface'
MODEL_PATH = 'pinn_model_cuda.pth'
# -------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Testing on: {device}")


def load_data(path):
    data_list = []
    for file in os.listdir(path):
        if file.endswith('.csv'):
            data = pd.read_csv(os.path.join(path, file))
            data_list.append(data)
    return pd.concat(data_list, axis=0)


test_data = load_data(TEST_PATH)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_X.fit(test_data[['z [m]', 't [s]']].values)
scaler_y.fit(test_data[['T [K]']].values)

X_test = scaler_X.transform(test_data[['z [m]', 't [s]']].values)
y_test = scaler_y.transform(test_data[['T [K]']].values)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)


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


model = PINNModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print("Model loaded successfully.")

with torch.no_grad():
    predictions = model(X_test)

predicted_temperatures = scaler_y.inverse_transform(predictions.cpu().numpy())
true_temperatures = scaler_y.inverse_transform(y_test.cpu().numpy())

test_data['Predicted Temperature [K]'] = predicted_temperatures
test_data['True Temperature [K]'] = true_temperatures
test_data.to_csv('PINN_Test_Results.csv', index=False)
print("Test results saved to PINN_Test_Results.csv")

plt.figure(figsize=(8, 6))
plt.plot(true_temperatures, label='True Temperature [K]', linestyle='--')
plt.plot(predicted_temperatures, label='Predicted Temperature [K]', linestyle='-')
plt.legend()
plt.title("PINN Prediction vs True Values")
plt.xlabel("Sample Index")
plt.ylabel("Temperature [K]")
plt.grid(True)
plt.show()
