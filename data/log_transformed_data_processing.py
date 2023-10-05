import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE  # Import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data from X.npy
raw_data = np.load('X.npy')

# Apply log2 transform to each entry (adding 1 to avoid log(0))
processed_data = np.log2(raw_data + 1)

# Number of clusters (you should replace this with the optimal K from the Elbow Method)
num_clusters = 5

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
cluster_assignments = kmeans.fit_predict(processed_data)

# Compute the mean value of data points in each cluster
cluster_means = []
for i in range(num_clusters):
    cluster_data = processed_data[cluster_assignments == i]
    mean = np.mean(cluster_data, axis=0)
    cluster_means.append(mean)

cluster_means = np.array(cluster_means)

# Perform PCA on cluster means for visualization
pca = PCA(n_components=2)
cluster_means_pca = pca.fit_transform(cluster_means)

# Perform MDS on cluster means for visualization
mds = MDS(n_components=2)
cluster_means_mds = mds.fit_transform(cluster_means)

# Perform T-SNE on cluster means for visualization
tsne = TSNE(n_components=2, perplexity=40)  # Use the same perplexity as mentioned
cluster_means_tsne = tsne.fit_transform(cluster_means)

# Plot the PCA visualization of cluster means
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.scatter(cluster_means_pca[:, 0], cluster_means_pca[:, 1], s=100)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("PCA Visualization of Cluster Means")
plt.grid(True)

# Plot the MDS visualization of cluster means
plt.subplot(1, 3, 2)
plt.scatter(cluster_means_mds[:, 0], cluster_means_mds[:, 1], s=100)
plt.xlabel("First MDS Component")
plt.ylabel("Second MDS Component")
plt.title("MDS Visualization of Cluster Means")
plt.grid(True)

# Plot the T-SNE visualization of cluster means
plt.subplot(1, 3, 3)
plt.scatter(cluster_means_tsne[:, 0], cluster_means_tsne[:, 1], s=100)
plt.xlabel("First T-SNE Component")
plt.ylabel("Second T-SNE Component")
plt.title("T-SNE Visualization of Cluster Means")
plt.grid(True)

plt.tight_layout()
plt.show()
