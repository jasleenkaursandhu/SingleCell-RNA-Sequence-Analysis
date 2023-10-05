import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data from X.npy
raw_data = np.load('p2_unsupervised/X.npy')

# Apply log2 transform to each entry (adding 1 to avoid log(0))
processed_data = np.log2(raw_data + 1)

# Apply PCA with 50 principal components
pca = PCA(n_components=50)
X_pca = pca.fit_transform(processed_data)

# Running K-means on PCA data with 50 principal components and 4 clusters
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_pca)

# Split the data into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(processed_data, cluster_labels, test_size=0.2, random_state=42)

# Create and fit a logistic regression model
logistic_reg = LogisticRegression(multi_class='ovr', solver='liblinear', penalty='l1', C=1.0, random_state=42)
logistic_reg.fit(X_train, y_train)

# Predict cluster assignments on the validation set
y_pred = logistic_reg.predict(X_val)

# Calculate accuracy on the validation set
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Calculate and report additional evaluation metrics
precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_val, y_pred)

print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1 Score: {f1:.4f}")
print("Confusion Matrix:\n", conf_matrix)

# Now, we need to retrain the logistic regression model to obtain coefficients
# Use the entire dataset for training
logistic_reg_full = LogisticRegression(multi_class='ovr', solver='liblinear', penalty='l1', C=1.0, random_state=42)
logistic_reg_full.fit(processed_data, cluster_labels)

# Get the coefficients
coefficients = logistic_reg_full.coef_

# Save the coefficients to a file for future use
np.save('coefficients.npy', coefficients)