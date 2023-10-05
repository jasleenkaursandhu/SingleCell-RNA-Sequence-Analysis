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

titles = ["PCA (85% Explained Variance)", "MDS", "t-SNE (First 50 PCs)"]

plt.figure(figsize=(15, 5))

for i, (subplot, title, z) in enumerate(zip((131, 132, 133), titles, (pca_85, mds, tsne_first_50))):
    plt.subplot(subplot)
    plt.title(title, fontsize=14)
    
    # Remove cluster coloring by not specifying c=clusters
    plt.scatter(z[:, 0], z[:, 1], cmap=plt.cm.Spectral)
    
    plt.xlabel("$z_1$", fontsize=18)
    if i == 0:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

# Save the images
plt.savefig("pca_85_explained_variance.png")
plt.savefig("mds_visualization.png")
plt.savefig("tsne_first_50_pcs.png")

plt.tight_layout()
plt.show()
