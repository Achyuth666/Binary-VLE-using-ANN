# --------------------------------------------------------------
# FOSSEE - IIT Bombay Autumn Internship - Screening Task
# Submitted by: Achyuth Sreenath Haaresamudram
# VIT Bhopal University
# Email: achyuth.23bai10584@vitbhopal.ac.in
# --------------------------------------------------------------


# --------------------------------------------------------------
# STEP 1: Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
tf.random.set_seed(10)
from tensorflow import keras
import os
# --------------------------------------------------------------


# --------------------------------------------------------------
# STEP 2: Data Collection and Pre-Processing (performing EDA)
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'VLE_Data.csv')

vle_data = pd.read_csv(data_path)

# Displaying some characteristics of the dataset
print(vle_data.shape)
print(vle_data.head())  
print(vle_data.describe())
print(vle_data.info())

# Checking for duplicates
print(f"Number of duplicate rows: {vle_data.duplicated().sum()}")
# Removing the duplicates (if any)
vle_data = vle_data.drop_duplicates()

# checking for null values
print(vle_data.isnull().sum())
# removing the null values found in the dataset
vle_data['X'].fillna(vle_data['X'].median(), inplace=True)
vle_data['Y'].fillna(vle_data['Y'].median(), inplace=True)
vle_data['T'].fillna(vle_data['T'].mean(), inplace=True)
vle_data['P'].fillna(vle_data['P'].median(), inplace=True)
print(vle_data.isnull().sum())

# Visualizing the distributions of features
sns.pairplot(vle_data)
plt.show()

# --------------------------------------------------------------


# --------------------------------------------------------------
# STEP 3: Separating Features and Target
X = vle_data.drop('Y', axis=1)
Y = vle_data[['Y']]        
# --------------------------------------------------------------


# --------------------------------------------------------------
# STEP 4: Splitting the Data into Training and Testing Sets
X_temp, X_val, Y_temp, Y_val = train_test_split(X, Y, test_size=0.1, random_state=2)
X_train, X_test, Y_train, Y_test = train_test_split(X_temp, Y_temp, test_size=0.2, random_state=2)
print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
# --------------------------------------------------------------


# --------------------------------------------------------------
# STEP 5: Standarding the data (features)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_val_std = scaler.transform(X_val)
X_test_std = scaler.transform(X_test)
# --------------------------------------------------------------


# --------------------------------------------------------------
# STEP 6: Building a Baseline Neural Network Model
baseline_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X_train_std.shape[1],)),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
baseline_model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_squared_error']
)
baseline_model.summary()
history1 = baseline_model.fit(
    X_train_std,
    Y_train,
    validation_data=(X_val_std, Y_val),
    epochs=50,
    batch_size=32
)
train_loss, train_mae = baseline_model.evaluate(X_train_std, Y_train, verbose=0)
print(f"Train Loss: {train_loss}")
print(f"Train Accuracy (Mean Absolute Error): {train_mae}")

test_loss, test_mae = baseline_model.evaluate(X_test_std, Y_test, verbose=0)
print(f"Test Loss: {test_loss}")

val_loss, val_mae = baseline_model.evaluate(X_val_std, Y_val, verbose=0)
print(f"Validation Loss: {val_loss}")

Y_pred = baseline_model.predict(X_test_std)

# Visualizing the results
plt.scatter(Y_test, Y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1])
plt.show()

# Plotting the loss curves for training and validation
plt.figure(figsize=(10,6))
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Baseline ANN Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Training data', 'Validation data'], loc='lower right')
plt.show()

# Detecting azeotropic point where X ≈ Y_pred ---> i.e finding the index where |Y_pred - X| is minimum
diff = np.abs(Y_pred.flatten() - X_test['X'].values)
azeotrope_index = np.argmin(diff)
azeotrope_x = X_test.iloc[azeotrope_index]['X']
azeotrope_t = X_test.iloc[azeotrope_index]['T']
azeotrope_y = Y_pred.flatten()[azeotrope_index]
print("Azeotropic point (x ≈ y_pred):")
print(f"x = {azeotrope_x:.5f}, y_pred = {azeotrope_y:.5f}, T = {azeotrope_t:.2f}")

# Calculating RMSE for baseline model
rmse_baseline = np.sqrt(mean_squared_error(Y_test, Y_pred))
print(f"RMSE for Baseline model: {rmse_baseline}")

# --------------------------------------------------------------


# --------------------------------------------------------------
# STEP 7: Building the PINN Model (Physics-Informed Neural Network)

# STEP 7.1: Prepare data for PINN
# Extract raw inputs (training data)
X_train_raw = X_train.astype(np.float32)
X_val_raw = X_val.astype(np.float32)
X_test_raw = X_test.astype(np.float32)

# Targets (remain raw/unscaled)
y_train_PINN = np.column_stack([Y_train['Y'].values.astype(np.float32),
                                X_train['P'].values.astype(np.float32),
                                X_train['X'].values.astype(np.float32),
                                X_train['T'].values.astype(np.float32)])

y_val_PINN   = np.column_stack([Y_val['Y'].values.astype(np.float32),
                                X_val['P'].values.astype(np.float32),
                                X_val['X'].values.astype(np.float32),
                                X_val['T'].values.astype(np.float32)])

y_test_PINN  = np.column_stack([Y_test['Y'].values.astype(np.float32),
                                X_test['P'].values.astype(np.float32),
                                X_test['X'].values.astype(np.float32),
                                X_test['T'].values.astype(np.float32)])

# STEP 7.2: Define constants for Antoine equation
A_ethanol, B_ethanol, C_ethanol = 8.20417, 1642.89, 230.3
A_water, B_water, C_water = 8.07131, 1730.63, 233.426
def p_sat_mmHg_tf(T_C, A, B, C):
    return tf.pow(10.0, A - B / (C + T_C))

# STEP 7.3: Define custom loss function for PINN ---> mathematical aspect is presented in the report file
def PINN_loss(lambda_physics=1.0):
    def loss(y1, gamma_pred):
        # y_true contains the true vapor composition (Y) values
        # gamma_pred contains the predicted activity coefficients from the PINN

        y1_data = tf.cast(y1[:, 0], tf.float32)
        P_data = tf.cast(y1[:, 1], tf.float32)
        x1 = tf.cast(y1[:, 2], tf.float32)
        T_C = tf.cast(y1[:, 3], tf.float32)

        gamma_ethanol = tf.cast(gamma_pred[:, 0], tf.float32)
        gamma_water = tf.cast(gamma_pred[:, 1], tf.float32)

        p1_mm = tf.pow(10.0, A_ethanol - B_ethanol / (C_ethanol + T_C))
        p2_mm = tf.pow(10.0, A_water - B_water / (C_water + T_C))

        p1_atm = p1_mm / 760.0
        p2_atm = p2_mm / 760.0

        # modified Raoult's law
        p_part1 = x1 * gamma_ethanol * p1_atm
        p_part2 = (1.0 - x1) * gamma_water * p2_atm

        # total pressure
        P_pred = p_part1 + p_part2

        # vapor phase mole fraction
        y1_pred = p_part1 / P_pred

        # loss fucntions
        data_mse = tf.reduce_mean(tf.square(y1_pred - y1_data))
        physics_mse = tf.reduce_mean(tf.square(P_pred - P_data))

        # total loss (final MSE)
        return data_mse + lambda_physics * physics_mse
    return loss

# STEP 7.4: Building the PINN model with the custom loss function
input_layer = keras.layers.Input(shape=(X_train.shape[1],), name='input_layer')
h = keras.layers.Dense(64, activation='relu')(input_layer)
h = keras.layers.Dense(64, activation='relu')(h)
h = keras.layers.Dense(32, activation='relu')(h)
log_gamma_linear = keras.layers.Dense(
    2,
    activation='linear',
    bias_initializer=keras.initializers.Zeros(),
    name='log_gamma'
)(h)
gamma_output = keras.layers.Activation('exponential', name='gamma_output')(log_gamma_linear)
PINN_model = keras.Model(inputs=input_layer, outputs=gamma_output)
PINN_model.summary()
PINN_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=PINN_loss(lambda_physics=0.5)
)
history2 = PINN_model.fit(
    X_train_raw,
    y_train_PINN,
    validation_data=(X_val_raw, y_val_PINN),
    epochs=300,
    batch_size=32,
)

# STEP 7.5: Evaluating the PINN model
train_loss = PINN_model.evaluate(X_train_raw, y_train_PINN, verbose=0)
print(f"Train Loss: {train_loss}")

val_loss = PINN_model.evaluate(X_val_raw, y_val_PINN, verbose=0)
print(f"Validation Loss: {val_loss}")

test_loss = PINN_model.evaluate(X_test_raw, y_test_PINN, verbose=0)
print(f"Test Loss: {test_loss}")

# STEP 7.6: Predicting on training set to find azeotropic point
y1_train_pred_gamma = PINN_model.predict(X_train_raw)
y1_train_pred = []
for i in range(len(y1_train_pred_gamma)):
    gamma_ethanol = y1_train_pred_gamma[i, 0]
    gamma_water = y1_train_pred_gamma[i, 1]
    x1 = X_train.iloc[i]['X']
    T_C = X_train.iloc[i]['T']
    P_data = X_train.iloc[i]['P']
    p1_atm = (10**(A_ethanol - B_ethanol / (C_ethanol + T_C)))/ 760.0
    p2_atm = (10**(A_water - B_water / (C_water + T_C))) / 760.0
    p_part1 = x1 * gamma_ethanol * p1_atm
    p_part2 = (1.0 - x1) * gamma_water * p2_atm
    P_pred = p_part1 + p_part2
    y1_pred = p_part1 / P_pred
    y1_train_pred.append(y1_pred)
y1_train_pred = np.array(y1_train_pred)

# STEP 7.7: Predicting on test set
y1_test_pred_gamma = PINN_model.predict(X_test_raw)
y1_test_pred = []
for i in range(len(y1_test_pred_gamma)):
    gamma_ethanol = y1_test_pred_gamma[i, 0]
    gamma_water = y1_test_pred_gamma[i, 1]
    x1 = X_test.iloc[i]['X']
    T_C = X_test.iloc[i]['T']
    P_data = X_test.iloc[i]['P']
    p1_atm = (10**(A_ethanol - B_ethanol / (C_ethanol + T_C))) / 760.0
    p2_atm = (10**(A_water - B_water / (C_water + T_C))) / 760.0
    p_part1 = x1 * gamma_ethanol * p1_atm
    p_part2 = (1.0 - x1) * gamma_water * p2_atm
    P_pred = p_part1 + p_part2
    y1_pred = p_part1 / P_pred
    y1_test_pred.append(y1_pred)
y1_test_pred = np.array(y1_test_pred)


# STEP 7.8: Visualizing the results
plt.scatter(Y_test, y1_test_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1])
plt.show()

# STEP 7.9: Plotting the loss curves for training and validation
plt.figure(figsize=(10,6))
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('PINN Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Training data', 'Validation data'], loc='lower right')
plt.show()

# STEP 7.10: Detecting azeotropic point where X ≈ Y_pred for PINN model
x1_train_original = X_train_raw['X'].values
diff_PINN = np.abs(y1_train_pred.flatten() - x1_train_original)
azeotrope_index_PINN = np.argmin(diff_PINN)
azeotrope_x_PINN = X_train_raw.iloc[azeotrope_index_PINN]['X']
azeotrope_t_PINN = X_train_raw.iloc[azeotrope_index_PINN]['T']
azeotrope_y_PINN = y1_train_pred.flatten()[azeotrope_index_PINN] 
print("Azeotropic point (x ≈ y_pred) for PINN model:")  
print(f"x = {azeotrope_x_PINN:.5f}, y_pred = {azeotrope_y_PINN:.5f}, T = {azeotrope_t_PINN:.2f}")

# STEP 7.11: Calculating RMSE for PINN model
rmse_PINN = np.sqrt(mean_squared_error(Y_test, y1_test_pred))
print(f"RMSE for PINN model: {rmse_PINN}")
# --------------------------------------------------------------


# --------------------------------------------------------------
# STEP 8: Saving the trained model
PINN_model.save('pinn_vle_model.h5')
# --------------------------------------------------------------


# --------------------------------------------------------------
# STEP 9: Comparing the PINN model with the baseline model - plotting a graph between the loss values of both the models
plt.figure(figsize=(10,6))
plt.plot(history2.history['loss'], label='PINN Training Loss')
plt.plot(history1.history['loss'], label='Baseline Training Loss')
plt.plot(history2.history['val_loss'], label='PINN Validation Loss')
plt.plot(history1.history['val_loss'], label='Baseline Validation Loss')
plt.title('Model Loss Comparison')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
# --------------------------------------------------------------


# --------------------------------------------------------------
# STEP 10: Loading the saved model and making a predictive system
loaded_model = keras.models.load_model('pinn_vle_model.h5', custom_objects={'loss': PINN_loss(lambda_physics=1.0)})
user_input_x = float(input("Enter liquid mole fraction (x1) between 0 and 1: "))
user_input_T = float(input("Enter temperature (T) in °C: "))
user_input_P = float(input("Enter pressure (P) in atm: "))
user_input_as_numpy_array = np.array([[user_input_x, user_input_T, user_input_P]], dtype=np.float32)
predicted_gamma = loaded_model.predict(user_input_as_numpy_array)
print(f"Predicted activity coefficients: gamma_ethanol = {predicted_gamma[0,0]:.4f}, gamma_water = {predicted_gamma[0,1]:.4f}")
# Now that gamma_ethanol and gamma_water are known, we compute y1 and P using modified Raoult's law
gamma_ethanol = predicted_gamma[0,0]
gamma_water = predicted_gamma[0,1]
x1 = user_input_x  
T_C = user_input_T
p1_mm = 10**(A_ethanol - B_ethanol / (C_ethanol + T_C))
p2_mm = 10**(A_water - B_water / (C_water + T_C))
p1_atm = p1_mm / 760.0
p2_atm = p2_mm / 760.0
p_part1 = x1 * gamma_ethanol * p1_atm
p_part2 = (1.0 - x1) * gamma_water * p2_atm
P_pred = p_part1 + p_part2
y1_pred = p_part1 / P_pred
print(f"Predicted vapor mole fraction: {y1_pred:.4f}")
print(f"Predicted total pressure: {P_pred:.4f} atm")
# --------------------------------------------------------------


# --------------------------------------------------------------
'''
THE END
Thank you for considering my submission.
I have attached the VLE dataset, this code file, and the Project Report (PDF) in the zip folder as instructed.
Additionally, I have attached a requirements.txt file listing all the necessary libraries and their versions used in this project.
Also, I have made a folder containing the graphs and plots generated, for your reference.
'''
# --------------------------------------------------------------
# --------------------------------------------------------------
# --------------------------------------------------------------