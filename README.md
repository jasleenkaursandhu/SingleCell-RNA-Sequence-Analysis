# Single-Cell RNA-Seq Analysis

## Project Purpose

This repository contains an analysis of a single-cell RNA-seq dataset obtained from the mouse neocortex, a brain region responsible for higher-level functions. The primary objectives of this project are to unveil hierarchical structures within the dataset, discover critical genes governing brain functions, and explore the relationships between individual cells.

## Repository Structure

The project is divided into multiple parts, each focusing on different aspects of data exploration, clustering, and feature selection.

### Problem 1: Small Subset Analysis

In this section, we explore a small, labeled subset of the dataset, referred to as "p1." The main objectives include:

- Visualizing the data with various techniques.
- Identifying cluster structures.
- Evaluating the ground truth clustering labels provided.

### Problem 2: Unsupervised Analysis

This section utilizes the "p2_unsupervised" dataset, which contains only the count matrix. The goals include:

- Applying unsupervised clustering methods to uncover hidden structures.
- Preparing the data for feature selection.

### Problem 2: Feature Selection and Classification

In Problem 2 (Evaluation), we work with a labeled training and test set from the "p2_evaluation" dataset. The tasks include:

- Feature selection to identify informative genes.
- Building a classification model to distinguish cell types.
- Evaluating the model's performance.

### Problem 3: Sensitivity Analysis

In the final part of the project, we revisit decisions made during the analysis, such as T-SNE hyperparameters and the number of clusters chosen. The goal is to assess the robustness of our results to these choices.

## Visualization Overview

This repository provides various visualizations to help interpret the single-cell RNA-seq data:

- **PCA (Principal Component Analysis):** Visualizes data using the top two principal components, revealing broad data structure.

- **MDS (Multidimensional Scaling):** Provides two-dimensional representations using MDS, helping identify spatial relationships among cells.

- **T-SNE (t-distributed Stochastic Neighbor Embedding):** Projects data into two dimensions using T-SNE, emphasizing local structures and clusters.

## Getting Started

To run the code and visualize the data, follow these steps:

1. Clone the repository to your local machine.
2. Install the required Python libraries listed in the `requirements.txt` file.
3. Explore each problem folder for detailed code and visualizations.

## Acknowledgments

I acknowledge the MIT Institute for providing the dataset used in this analysis.



