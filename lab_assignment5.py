import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Function to train the model with given data
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to predict results with the trained model
def predict_values(model, X_to_predict):
    predictions = model.predict(X_to_predict)
    return predictions

# Function to calculate metrics (error and accuracy values)
def calculate_metrics(real, predicted):
    mse = mean_squared_error(real, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(real, predicted)
    r2 = r2_score(real, predicted)
    # Returns all metrics as a tuple
    return mse, rmse, mape, r2

# =========================
# MAIN PROGRAM STARTS HERE
# =========================

# ---------- USING ONLY ONE FEATURE (A1, A2) ----------

# Example simple data, just for practice
# X_train contains one "input" for each example
X_train_single = np.array([[1], [2], [3], [4], [5]])
y_train_single = np.array([10, 20, 30, 40, 50])   # Target: doubles the input for easy understanding

# X_test contains new "inputs" the model hasn’t seen before
X_test_single = np.array([[6], [7]])
y_test_single = np.array([60, 70])

# Train linear regression model using just one feature
model_one = train_model(X_train_single, y_train_single)

# Predict for training data and test data
y_train_pred_single = predict_values(model_one, X_train_single)
y_test_pred_single = predict_values(model_one, X_test_single)

# Calculate performance numbers for train and test data
train_metrics_single = calculate_metrics(y_train_single, y_train_pred_single)
test_metrics_single = calculate_metrics(y_test_single, y_test_pred_single)

print("Results using ONE attribute: ")
print("Train - MSE, RMSE, MAPE, R2:", train_metrics_single)
print("Test  - MSE, RMSE, MAPE, R2:", test_metrics_single)
print()

# ----------- USING MULTIPLE FEATURES (A3) -----------
# Now use two "inputs" for each example
# Again, dummy example for practice
X_train_multi = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])
y_train_multi = np.array([11, 12, 13, 14, 15])  # Target, just for example

X_test_multi = np.array([[6, 0], [7, -1]])
y_test_multi = np.array([16, 17])

# Train linear regression model using multiple features
model_multi = train_model(X_train_multi, y_train_multi)

# Predict for new data
y_train_pred_multi = predict_values(model_multi, X_train_multi)
y_test_pred_multi = predict_values(model_multi, X_test_multi)

# Calculate metrics again
train_metrics_multi = calculate_metrics(y_train_multi, y_train_pred_multi)
test_metrics_multi = calculate_metrics(y_test_multi, y_test_pred_multi)

print("Results using MULTIPLE attributes: ")
print("Train - MSE, RMSE, MAPE, R2:", train_metrics_multi)
print("Test  - MSE, RMSE, MAPE, R2:", test_metrics_multi)
