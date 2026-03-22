# Physics-Informed Neural Network (PINN) for Additive Manufacturing

> Predicting temperature distribution during ring-shaped laser welding using a Physics-Informed Neural Network trained on COMSOL Multiphysics simulation data.  
>  AI in Additive Manufacturing |  Advanced Manufacturing and Functional Devices (AMFD) Laboratory, Arizona State University

---

## 📌 Project Overview

This project develops a **Physics-Informed Neural Network (PINN)** to predict temperature distribution during **ring-shaped laser welding** — a process used in additive manufacturing.
Rather than relying purely on data, the model embeds the **heat conduction equation in cylindrical coordinates** directly into the loss function, ensuring predictions are physically 
consistent even with limited training data.

Training data is generated from **COMSOL Multiphysics simulations**, and the PINN is validated against ground-truth simulation outputs across multiple laser ring geometries.

---

## 🎯 Objectives

1. Generate temperature field data from COMSOL Multiphysics simulations of ring-shaped laser welding
2. Preprocess and format simulation data into PINN-compatible training datasets
3. Train a physics-constrained neural network that satisfies the heat equation
4. Validate predictions against unseen COMSOL simulation data
5. Generalize the model across different laser ring geometries (inner/outer radii)

---

## 🔬 Physics Background

The model is governed by the **heat conduction equation in cylindrical coordinates:**

```
∂T/∂t = α (∂²T/∂z² + (1/z) ∂T/∂z) + Q(z,t)
```

Where:
- `T` = Temperature (K)
- `t` = Time (s)
- `z` = Radial distance from center (m)
- `α` = Thermal diffusivity of Aluminum = 8.4×10⁻⁵ m²/s
- `Q` = Heat source term from the ring laser

**Laser Parameters:**
| Parameter | Value |
|-----------|-------|
| Energy | 7.9 J |
| Pulse duration | 3 ms |
| Power | 2633 W |
| Absorptivity | 10% |

**Ring geometries trained on:**
| Dataset | Inner Radius | Outer Radius |
|---------|-------------|-------------|
| Ring 1 | 37.5 µm | 225 µm |
| Ring 2 | 125 µm | 300 µm |
| Validation | 200 µm | 400 µm |

---

## 🏗️ Model Architecture

A fully connected neural network with physics-informed loss:

```
Input: [z (m), t (s)]  →  4 hidden layers (50 neurons, Tanh)  →  Output: T (K)
```

**Loss function:**
```
Total Loss = Data Loss (MSE) + Physics Loss (heat equation residual)
```

The physics residual penalizes solutions that violate the governing heat equation, enforcing physical consistency during training.

---

## 📁 Repository Structure

```
├── surface_pinn.py          # PINN training on surface temperature data
├── dynamic_training.py      # Training with dynamic switching between ring geometries
├── testing.py               # Model inference and result visualization
├── validation_script.py     # Validation against COMSOL ground truth with MAE/RMSE
├── val_original.py           # Quick validation on saved validation tensors
├── convert_temp_data.py     # Convert raw COMSOL .txt output to CSV
├── formatting_temp.py        # Format temperature data by time steps
├── preping_data.py          # Flatten 2D temperature matrices to PINN input format
├── separate_files.py        # Organize data into train/val/surface/depth folders
├── move_file.py             # Move formatted files into the correct dataset folders
├── time_steps_extraction.py # Extract COMSOL time steps from text file
├── comsol_time_steps.csv    # Extracted COMSOL simulation time steps
└── project_solution.mph     # COMSOL Multiphysics simulation file
```

---

## 🔄 Data Pipeline

```
COMSOL Simulation (.mph)
        ↓
Export raw temperature data (.txt)
        ↓
convert_temp_data.py  →  temperature_data.csv
        ↓
formating_temp.py     →  extracted_temperature_by_time.csv
        ↓
preping_data.py       →  PINN_Formatted_*.csv  (z, t, T columns)
        ↓
move_file.py + separate_files.py  →  PINN_Data/train/surface & val/surface
        ↓
surface_pinn.py / dynamic_training.py  →  trained model (.pth)
        ↓
validation_script.py  →  MAE, RMSE, plots
```

---

## 🚀 Usage

**1. Prepare data:**
```bash
python convert_temp_data.py
python formating_temp.py
python preping_data.py
python move_file.py
python separate_files.py
```

**2. Train the PINN:**
```bash
# Single geometry training
python surface_pinn.py

# Multi-geometry dynamic training
python dynamic_training.py
```

**3. Test and validate:**
```bash
python testing.py
python validation_script.py
```

> **Note:** Update the `CONFIG` paths at the top of each script to point to your local data directory before running.

---

## 🛠️ Requirements

```bash
pip install torch pandas numpy scikit-learn matplotlib joblib
```

---

## 👥 Team

**Arizona State University — Ira A. Fulton Schools of Engineering**  
Advanced Manufacturing and Functional Devices (AMFD) Laboratory

---

## 📂 Data Note

COMSOL simulation data files are not included in this repository due to file size. The `.mph` simulation file is provided to regenerate the data. Processed CSV training data can be regenerated using the data pipeline scripts above.
