import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset (replace with your data loading code)
X = np.load("p1/X.npy")
y = np.load("p1/y.npy")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a range of C values to test
C_values = [0.1, 1.0, 10.0]

# Initialize lists to store results
accuracies = []
selected_features = []

# Iterate over different C values
for C in C_values:
    # Create a logistic regression model with the current C value
    model = LogisticRegression(C=C, solver='liblinear', random_state=42)
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = model.predict(X_test)
    
    # Calculate accuracy and store it
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    
    # Store the selected features (coefficients)
    selected_features.append(model.coef_)

# Visualize the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(C_values, accuracies, marker='o')
plt.xlabel('C (Regularization Strength)')
plt.ylabel('Accuracy')
plt.title('Model Accuracy vs. Regularization Strength')

# Visualize the selected features for each C value
plt.subplot(1, 2, 2)
for i, C in enumerate(C_values):
    plt.plot(selected_features[i].ravel(), label=f'C={C}')

plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Selected Features vs. Regularization Strength')
plt.legend()
plt.tight_layout()

# Save the image with the same name as the class name (e.g., "class_name.png")
class_name = "regularization_strength_effect"  # Replace with your class name
plt.savefig(f'{class_name}.png')

# Show the plot
plt.show()
