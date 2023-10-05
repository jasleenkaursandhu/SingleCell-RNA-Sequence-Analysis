import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the logistic regression coefficients from question 2 
coefficients = np.load('coefficients.npy')

# Top 100 features by taking features with the largest magnitude coefficients
coef_abs = np.abs(coefficients)
max_coefs = np.max(coef_abs, axis=0)
features = np.argpartition(max_coefs, -100)[-100:]

# Loading all evaluation sets 
X_train = np.load('p2_evaluation/X_train.npy')
y_train = np.load('p2_evaluation/y_train.npy')
X_test = np.load('p2_evaluation/X_test.npy')
y_test = np.load('p2_evaluation/y_test.npy')

# Making logarithmic transformation log2(x+1)
X_train_log = np.log2(X_train + 1)
X_test_log = np.log2(X_test + 1)

# Standardizing data by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_log)
X_test_scaled = scaler.transform(X_test_log)

# Fitting top 100 features on standardized evaluation data with cross-validation
lr2 = LogisticRegressionCV(penalty="l2", solver="liblinear", Cs=[.001, .1, 1, 5, 10, 50, 100], cv=5)
lr2.fit(X_train_scaled[:, features], y_train)

# Score for standardized feature selection method
accuracy_lr2 = lr2.score(X_test_scaled[:, features], y_test)
print(f"Accuracy with top 100 coefficient features: {accuracy_lr2:.4f}")

# Selecting genes with high variance as features
variances = np.var(X_train_log, axis=0)
variance_features = np.argpartition(variances, -100)[-100:]

# Fitting high variance features on standardized evaluation data with cross-validation
lr3 = LogisticRegressionCV(penalty="l2", solver="liblinear", Cs=[1, 3, 5, 10], cv=5)
lr3.fit(X_train_scaled[:, variance_features], y_train)

# Score for high variance feature selection method
accuracy_lr3 = lr3.score(X_test_scaled[:, variance_features], y_test)
print(f"Accuracy with high variance features: {accuracy_lr3:.4f}")

# Selecting genes randomly
random_features = np.random.choice(np.arange(X_train_log.shape[1]), size=100, replace=False)

# Fitting 100 random features on standardized evaluation data with cross-validation
lr4 = LogisticRegressionCV(penalty="l2", solver="liblinear", Cs=[1, 3, 5, 10], cv=5)
lr4.fit(X_train_scaled[:, random_features], y_train)

# Score for random feature selection method
accuracy_lr4 = lr4.score(X_test_scaled[:, random_features], y_test)
print(f"Accuracy with random features: {accuracy_lr4:.4f}")

# Plotting Histograms of the variances of features selected by Lasso Regression and the top 100 variances
plt.figure(figsize=(8, 5))
plt.title('Histograms of Variances of Lasso and top100 Variances')
plt.hist(variances[features], color="blue", alpha=0.3, bins=20, label="Top 100 Coefficients")
plt.hist(variances[variance_features], color="red", alpha=0.3, bins=20, label="High Variance")
plt.xlabel('Variance')
plt.ylabel('Frequency')
plt.legend()
# Save the image with the same name as the class name (e.g., "class_name.png")
class_name = "feature_selection_and_evaluation"  # Replace with your class name
plt.savefig(f'{class_name}.png')

# Show the plot
plt.show()
plt.show()
