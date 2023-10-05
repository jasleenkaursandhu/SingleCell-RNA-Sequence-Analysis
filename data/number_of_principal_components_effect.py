import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib as mpl
from sklearn.cluster import KMeans  # Import KMeans for clustering

# Loading dataset 1
X = np.load("p1/X.npy")
y = np.load("p1/y.npy")

# Making logarithmic transformation
X_log = np.log2(X + 1)

# Creating a list of PC counts to evaluate
pc_counts = [10, 50, 100, 250]

# Creating a single figure with subplots
fig, axes = plt.subplots(1, len(pc_counts), figsize=(20, 5))

# Color map for consistency
cmap = mpl.cm.Spectral

# Iterating over different PC counts
for i, n_pcs in enumerate(pc_counts):
    # Transforming data with PCA
    z = PCA(n_components=n_pcs).fit_transform(X_log)
    
    # Clustering using KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(z)
    
    ax = axes[i]
    scatter = ax.scatter(z[:, 0], z[:, 1], c=cluster_labels, cmap=cmap)
    ax.set_title('PCA with {} PCs'.format(n_pcs))
    ax.set_xlabel("$z_1$", fontsize=18)
    ax.set_ylabel("$z_2$", fontsize=18, rotation=0)
    
# Save plots as image files
for i, n_pcs in enumerate(pc_counts):
    fig = axes[i].get_figure()
    fig.savefig(f'pca_clusters_{n_pcs}.png')

plt.tight_layout()
plt.show()
