import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib as mpl

# Loading dataset 1
X = np.load("p1/X.npy")
y = np.load("p1/y.npy")

# Making logarithmic transformation
X_log = np.log2(X + 1)

# Transforming data with PCA
z = PCA(n_components=500).fit_transform(X_log)

# Creating a single figure with subplots
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

# Color map for consistency
cmap = mpl.cm.Spectral

# Plotting t-SNE with different numbers of principal components
for i, n_pcs in enumerate([10, 50, 100, 250, 500]):
    z_tsne = TSNE(n_components=2, perplexity=40).fit_transform(z[:, 0:n_pcs])
    ax = axes[i]
    scatter = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y, cmap=cmap)
    ax.set_title('t-SNE with {} PCs'.format(n_pcs))
    ax.set_xlabel("$z_1$", fontsize=18)
    ax.set_ylabel("$z_2$", fontsize=18, rotation=0)
    
    # Adding labels to data points
    for label in np.unique(y):
        ax.scatter([], [], c=cmap(label), label=f'Class {label}')
    
    ax.legend()

# Save plots as image files
for i, n_pcs in enumerate([10, 50, 100, 250, 500]):
    fig = axes[i].get_figure()
    fig.savefig(f'tsne_pcs_{n_pcs}.png')

plt.tight_layout()
plt.show()
