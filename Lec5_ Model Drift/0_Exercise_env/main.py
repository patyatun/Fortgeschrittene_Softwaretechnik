from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Function to create dataset with a specified drift
def create_dataset(n_samples=1000, drift_factor=0.0):
    X, y = make_classification(n_samples=n_samples, n_features=20, n_classes=2, random_state=42)
    # Introducing drift by shifting the feature values
    X_drifted = X + drift_factor * np.random.rand(*X.shape)
    return X_drifted, y

# Create initial training dataset
X_train, y_train = create_dataset()

# Case 1: Large drift
# The model's accuracy dropped to 70%, suggesting a significant impact due to the large drift, which might necessitate retraining.

# Create a test dataset with significant drift
X_test_large_drift, y_test_large_drift = create_dataset(drift_factor=5.0)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set with large drift
y_pred_large_drift = model.predict(X_test_large_drift)
accuracy_large_drift = accuracy_score(y_test_large_drift, y_pred_large_drift)
print("Large drift, new accuracy: ", accuracy_large_drift)

# Case 2 (Small Drift): The model maintained a relatively high accuracy of 86.7%, 
# indicating that the small drift did not critically impair its performance, and retraining may not be necessary at this stage.

# Case 2: Small drift
# Create a test dataset with minor drift
X_test_small_drift, y_test_small_drift = create_dataset(drift_factor=1.0)

# Evaluate the model on the test set with small drift
y_pred_small_drift = model.predict(X_test_small_drift)
accuracy_small_drift = accuracy_score(y_test_small_drift, y_pred_small_drift)
accuracy_small_drift

print("Small drift, new accuracy: ", accuracy_small_drift)
