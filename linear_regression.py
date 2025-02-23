import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate dataset (y = 2x + 1)
X = np.array([[i] for i in range(10)])  # Reshape to 2D array
y = np.array([2 * i + 1 for i in range(10)])

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Print model parameters
print(f"Intercept: {model.intercept_}, Slope: {model.coef_[0]}")

# Predict
y_pred = model.predict(X)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', linestyle='dashed', label='Predicted')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Linear Regression: y = 2x + 1")
plt.show()
