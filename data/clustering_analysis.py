import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the data from X.npy
raw_data = np.load('p2_unsupervised/X.npy')

# Apply log2 transform to each entry (adding 1 to avoid log(0))
processed_data = np.log2(raw_data + 1)

# Apply PCA with 50 principal components
pca = PCA(n_components=50)
X_pca = pca.fit_transform(processed_data)

# Running K-means on PCA data with 50 principal components
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_pca)
                for k in range(1, 10)]

# Calculate the inertia for each K-Means model
inertias = [model.inertia_ for model in kmeans_per_k]

# Plot the inertia and find the optimal number of clusters (elbow method)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.title("Elbow Method: Finding Optimal Clusters in K-Means")
plt.grid(True)

# Silhouette score of K-Means on PCA data with 50 principal components
silhouette_scores = [silhouette_score(X_pca, model.labels_)
                     for model in kmeans_per_k[1:]]

# Plot the silhouette scores to find the optimal number of clusters
plt.subplot(1, 2, 2)
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.title("Silhouette Score Analysis: Evaluating Cluster Quality in K-Means")
plt.grid(True)

plt.tight_layout()
plt.show()

# Running DBSCAN on PCA data with 50 principal components and 4 clusters
dbscan = DBSCAN(eps=0.5, min_samples=5) 
db_labels = dbscan.fit_predict(X_pca)

# Visualize the results of K-Means with a scatter plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_per_k[3].labels_, cmap=plt.cm.Spectral)
plt.xlabel("$z_1$", fontsize=12)
plt.ylabel("$z_2$", fontsize=12, rotation=0)
plt.title('K-Means Clustering of Brain Cells', fontsize=14)

# Visualize the results of DBSCAN with a scatter plot
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=db_labels, cmap=plt.cm.Spectral)
plt.xlabel("$z_1$", fontsize=12)
plt.ylabel("$z_2$", fontsize=12, rotation=0)
plt.title('DBSCAN Clustering of Brain Cells', fontsize=14)

plt.tight_layout()
plt.show()
