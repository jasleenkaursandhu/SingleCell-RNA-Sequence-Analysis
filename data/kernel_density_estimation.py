import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from X.npy
raw_data = np.load('X.npy')

# Apply log2 transform to each entry (adding 1 to avoid log(0))
processed_data = np.log2(raw_data + 1)

# Select a gene for KDE plot (replace 0 with the desired gene index)
gene_index = 0
selected_gene = processed_data[:, gene_index]

# Create a KDE plot
sns.kdeplot(selected_gene, shade=True)
plt.xlabel(f'Expression of Gene {gene_index + 1}')
plt.ylabel('Density')
plt.title(f'KDE Plot for Gene {gene_index + 1}')
plt.show()
