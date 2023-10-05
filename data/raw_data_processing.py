import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data from X.npy (assuming this is the original, unprocessed data)
raw_data = np.load('X.npy')

# Number of clusters for K-Means clustering (replace with your desired number)
num_clusters_kmeans = 5

# Perform PCA on the raw data to obtain the top two principal components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(raw_data)

# Perform MDS on the raw data for visualization
mds = MDS(n_components=2)
mds_result = mds.fit_transform(raw_data)

# Perform T-SNE on the raw data for visualization
tsne = TSNE(n_components=2, perplexity=40, random_state=0)
tsne_result = tsne.fit_transform(raw_data)

# Perform K-Means clustering on T-SNE embeddings
kmeans = KMeans(n_clusters=num_clusters_kmeans, random_state=0)
cluster_assignments = kmeans.fit_predict(tsne_result)

# Plot PCA visualization
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(pca_result[:, 0], pca_result[:, 1], s=10)
plt.title("PCA Visualization")

# Plot MDS visualization
plt.subplot(1, 3, 2)
plt.scatter(mds_result[:, 0], mds_result[:, 1], s=10)
plt.title("MDS Visualization")

# Plot T-SNE visualization with cluster colors
plt.subplot(1, 3, 3)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_assignments, cmap='viridis', s=10)
plt.title("T-SNE Visualization with Clusters")

plt.tight_layout()
plt.show()
