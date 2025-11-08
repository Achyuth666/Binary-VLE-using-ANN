ANN Surrogate Modelling for Binary VLE (Ethanol–Water System)

A project submitted as part of the **FOSSEE Screening Task**, focusing on building and comparing **Artificial Neural Network (ANN)** and **Physics-Informed Neural Network (PINN)** models to predict vapor–liquid equilibrium (VLE) behavior for a **binary azeotropic system (Ethanol–Water)**.


📘 Project Overview

The goal of this project is to develop an **Artificial Neural Network (ANN)** model capable of predicting the **vapor composition (y)** in a binary azeotropic system, given inputs such as **liquid mole fraction (x)**, **temperature (T)**, and **pressure (P)**.

The project further enhances this model into a **Physically Informed Neural Network (PINN)** by embedding thermodynamic laws — specifically the **modified Raoult’s law** — into the model’s loss function, improving physical consistency and predictive accuracy.

🧩 Key Objectives

* Build and train an ANN to predict vapor composition in a binary VLE system.
* Incorporate physical laws into the learning process using a **custom PINN loss function**.
* Compare the performance of ANN and PINN based on MSE, RMSE, and predicted azeotropic composition.
* Validate the model’s predictions against experimental/simulated data.

📊 Dataset Description

**Source:** Simulated using **DWSIM** (open-source chemical process simulator)

**System:** Ethanol–Water
**Property package:** NRTL (Non-Random Two-Liquid)
**Type:** Isobaric (1 atm) VLE data

| Feature | Description                                            |
| ------- | ------------------------------------------------------ |
| **X**   | Mole fraction of ethanol in liquid phase (0–1)         |
| **T**   | Temperature (°C), decreases with ethanol concentration |
| **P**   | Pressure (atm), approximately constant (1 atm)         |
| **Y**   | Mole fraction of ethanol in vapor phase                |

⚙️ Methodology

### **1. Data Preparation**

* Generated VLE data in DWSIM using a flash separator and binary phase envelope.
* Cleaned and preprocessed the dataset using **pandas**.
* Split into **training**, **testing**, and **validation** sets.

### **2. Baseline ANN Model**

* **Architecture:** Simple feedforward neural network
* **Optimizer:** Adam
* **Loss:** Mean Squared Error (MSE)
* **Performance:**

  * MSE: 0.0021
  * RMSE: 0.04915
  * Predicted Azeotrope: x = 0.9200, y = 0.9138, T = 78.42°C

### **3. Physics-Informed Neural Network (PINN)**

* Custom loss function incorporating **modified Raoult’s law** and **Antoine’s equation**.
* Predicts activity coefficients (γ) instead of direct vapor composition.
* **Optimizer:** Adam (lr = 0.001)
* **Training:** 300 epochs
* **Performance:**

  * MSE: 1.745 × 10⁻⁵
  * RMSE: 0.00310
  * Predicted Azeotrope: x = 0.844, y = 0.844, T = 78.39°C

---

## 🧮 Custom Loss Function (PINN)

[
L_{total} = L_{data} + \lambda L_{physical}
]

where:
[
L_{data} = \frac{1}{N} \sum (y_{pred} - y_{data})^2
]
[
L_{physical} = \frac{1}{N} \sum \left(\frac{P_{pred} - P_{data}}{P_{data}}\right)^2
]

λ = 0.5 (weighting factor for physical loss)

---

## 📈 Results Summary

| Model            | MSE        | RMSE    | Azeotrope (x,y) |
| ---------------- | ---------- | ------- | --------------- |
| **Baseline ANN** | 0.0021     | 0.04915 | (0.920, 0.9138) |
| **PINN**         | 0.00001745 | 0.00310 | (0.844, 0.844)  |

✅ **PINN outperforms the baseline ANN** by reducing error by over 100× and achieving physically consistent results.

---

## 🧰 Tech Stack

* **Language:** Python
* **Libraries:** TensorFlow / Keras, NumPy, Pandas, Matplotlib
* **Simulation Tool:** DWSIM
* **Environment:** Jupyter Notebook


## 🧑‍💻 Author

**Achyuth Sreenath Haresamudram**
VIT Bhopal University
📧 [achyuth.23bai10584@vitbhopal.ac.in](mailto:achyuth.23bai10584@vitbhopal.ac.in)


## 🚀 Future Work

* Extend the model for **multi-component systems**.
* Integrate **uncertainty quantification** for data-driven thermodynamic predictions.
* Explore **transfer learning** across different binary systems.
