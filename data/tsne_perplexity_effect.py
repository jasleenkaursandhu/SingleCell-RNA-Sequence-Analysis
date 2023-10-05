import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib as mpl

# Loading dataset 1
X = np.load("p1/X.npy")
y = np.load("p1/y.npy")

# Making logarithmic transformation
X_log = np.log2(X + 1)

# Creating a single figure with subplots
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

# Color map for consistency
cmap = mpl.cm.Spectral

# Plotting t-SNE with different perplexity values
perplexity_values = [10, 30, 50, 100, 200]  # Adjust the perplexity values as needed

for i, perplexity in enumerate(perplexity_values):
    tsne = TSNE(n_components=2, perplexity=perplexity)
    z_tsne = tsne.fit_transform(X_log)
    
    ax = axes[i]
    scatter = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y, cmap=cmap)
    ax.set_title('t-SNE with Perplexity {}'.format(perplexity))
    ax.set_xlabel("$z_1$", fontsize=18)
    ax.set_ylabel("$z_2$", fontsize=18, rotation=0)
    
    # Adding labels to data points
    for label in np.unique(y):
        ax.scatter([], [], c=cmap(label), label=f'Class {label}')
    
    ax.legend()

# Save plots as image files
for i, perplexity in enumerate(perplexity_values):
    fig = axes[i].get_figure()
    fig.savefig(f'tsne_perplexity_{perplexity}.png')

plt.tight_layout()
plt.show()
