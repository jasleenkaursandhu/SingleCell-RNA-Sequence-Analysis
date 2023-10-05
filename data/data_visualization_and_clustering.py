import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data from X.npy
raw_data = np.load('p2_unsupervised/X.npy')

# Check the shape of the data
print("Data shape:", raw_data.shape)

# Apply log2 transform to each entry (adding 1 to avoid log(0))
processed_data = np.log2(raw_data + 1)

# Get the number of cells (rows) and genes (columns)
num_cells, num_genes = processed_data.shape

# Find the largest entry in the first column of the processed data
largest_entry = np.max(processed_data[:, 0])

# Print the results
print(f"Number of Cells (number of rows): {num_cells}")
print(f"Number of Genes (number of columns): {num_genes}")
print(f"Largest entry in the first column of processed data: {largest_entry:.5f}")

# Apply PCA with 85% explained variance directly
pca_85 = PCA(n_components=0.85).fit_transform(processed_data)

# Visualize data with t-SNE on the first 50 PCs
tsne_first_50 = TSNE(n_components=2, perplexity=40, random_state=42).fit_transform(pca_85)

# Visualize data with MDS
mds = MDS(n_components=2, random_state=42).fit_transform(processed_data)

# K-Means Clustering with 4 clusters on PCA-reduced data with 85% explained variance
kmeans_4 = KMeans(n_clusters=4).fit(pca_85)
kmeans_pred_4 = kmeans_4.fit_predict(pca_85)
print("Inertia with 4 clusters:", kmeans_4.inertia_)

titles = ["PCA (85% Explained Variance)", "MDS", "t-SNE (First 50 PCs)"]

plt.figure(figsize=(15, 5))

# Define unique image names
image_names = ["pca_85_explained_variance.png", "mds_visualization.png", "tsne_first_50_pcs.png"]

for subplot, title, z, clusters, image_name in zip((131, 132, 133), titles, (pca_85, mds, tsne_first_50), (kmeans_pred_4, kmeans_pred_4, kmeans_pred_4), image_names):
    plt.subplot(subplot)
    plt.title(title, fontsize=14)
    plt.scatter(z[:, 0], z[:, 1], c=clusters, cmap=plt.cm.Spectral)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

    # Save the image with the specified name
    plt.savefig(image_name)

plt.tight_layout()
plt.show()
