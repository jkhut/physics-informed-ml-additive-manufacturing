import torch
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Clear any cached memory
torch.cuda.empty_cache()

# Load the model
model = torch.load('pinn_model_physics_constrained.pth')
model.eval()

# Load the scaler for inverse transformation
scaler_y = joblib.load('scaler_y.pkl')

# Load validation data
X_val_1, y_val_1 = torch.load('validation_data_1.pt')

# === Perform predictions in batches to avoid memory overflow ===
batch_size = 5000
predicted_temp = []
for i in range(0, len(X_val_1), batch_size):
    X_batch = X_val_1[i:i + batch_size]
    with torch.no_grad():
        preds, _ = model(X_batch, 0.0000375, 0.000225, 0)
        predicted_temp.append(preds.cpu().numpy())

# Concatenate all the batches
predicted_temp = np.concatenate(predicted_temp, axis=0)
actual_temp = y_val_1.cpu().numpy()

# Inverse transform to original temperature scale
predicted_temp = scaler_y.inverse_transform(predicted_temp)
actual_temp = scaler_y.inverse_transform(actual_temp)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(predicted_temp, label="Predicted Temperature", linestyle='--')
plt.plot(actual_temp, label="Actual Temperature", linestyle='-')
plt.legend()
plt.title("Temperature Prediction vs Actual (Validation Set 1)")
plt.xlabel("Sample Index")
plt.ylabel("Temperature (K)")
plt.grid(True)
plt.show()
