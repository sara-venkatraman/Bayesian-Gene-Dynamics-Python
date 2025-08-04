from llr2 import LLR2
from plotting import plotGenes

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import networkx as nx


# Read gene expression data and prior adjacency matrix
geneData = pd.read_csv("Data/geneData.csv", index_col = 0).values.tolist
priorMatrix = pd.read_csv("Data/priorMatrix.csv", index_col = 0)

# Save gene names and define hours corresponding to each time point
geneNames = geneData.index
hours = [0, 1, 2, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 30, 36, 42, 48]

# Reshape gene expression data into a list of lists
geneDataList = geneData.values.tolist()

# === Compute lead-lag R^2 matrices ===

# These lines also print out the runtime, coming out to about 3.9 and 2.4 mins., respectively
LLR2Bayes = LLR2(geneDataList, hours, bayes = True, priorMatrix = priorMatrix, writeToCSV = True)
LLR2OLS = LLR2(geneDataList, hours, bayes = False, writeToCSV = True)

# OR: Read the matrices in from CSV if they have already been computed
LLR2Bayes = np.loadtxt("LLR2_Matrices/BayesLLR2.csv", delimiter = ",")
LLR2OLS = np.loadtxt("LLR2_Matrices/OLSLLR2.csv", delimiter = ",")

# === Hierarchical clustering ===

# Perform clustering on condensed distance matrix
hierClust = linkage(squareform(1 - LLR2Bayes), method = "ward")

# Plot dendrogram
plt.figure(figsize = (10, 7))
hierClustDend = dendrogram(hierClust)
plt.show()

# Divide genes into clusters
subGroups = fcluster(hierClust, t = 16, criterion = "maxclust")
(clusterNumber, clusterCounts) = np.unique(subGroups, return_counts = True)

# Plot the gene trajectories in each cluster
plotColors = ["darkorange", "dodgerblue", "forestgreen", "darkmagenta", 
"teal", "saddlebrown", "navy", "darkgoldenrod", "blueviolet", "darkred", 
"olivedrab", "darkslategray", "cornflowerblue", "rosybrown", "cadetblue", "crimson"] * 2

fig, axes = plt.subplots(4, 4, figsize = (12, 9))

for i, ax in enumerate(axes.flatten()):
  genesInCluster = np.where(subGroups == (i + 1))[0]
  clusterData = [geneDataList[i] for i in genesInCluster]
  plotTitle = "Cluster " + str(i + 1) + " (" + str(clusterCounts[i]) + " genes)"
  fig = plotGenes(clusterData, hours, plotColors = [plotColors[i]] * len(genesInCluster), lineOpacity = 0.15, axis = ax,
  xAxisLabel = "Time", yAxisLabel = "Expression", plotTitle = plotTitle)

plt.tight_layout()
fig.savefig("ClustersLLR2.png", format = "png", dpi = 300)

# === Network statistics - Bayesian network ===

# Create adjacency matrix by thresholding R^2 values at 0.9
adjacencyBayes = (LLR2Bayes > 0.9) + 0

# Zero out diagonal to avoid self-loops and construct graph
np.fill_diagonal(adjacencyBayes, 0)
graphBayes = nx.from_numpy_array(adjacencyBayes)

# Draw network figure
plt.figure(figsize = (5, 4))
plt.title("Gene network computed with Bayesian LLR2\n(i.e. with prior information)")
nx.draw(graphBayes, nx.spring_layout(graphBayes, k = 0.05, seed = 123), with_labels = False, 
node_size = 12, edgecolors = "navy", edge_color = "gray", linewidths = 0.7, node_color = "orange")
plt.show()
plt.savefig("NetworkBayesLLR2.png", format = "png", dpi = 500)


# === Network statistics - Non-Bayesian network ===

# Create adjacency matrix by thresholding R^2 values at 0.9
adjacencyOLS = (LLR2OLS > 0.9) + 0

# Zero out diagonal to avoid self-loops and construct graph
np.fill_diagonal(adjacencyOLS, 0)
graphOLS = nx.from_numpy_array(adjacencyOLS)

# Draw network figure
plt.figure(figsize = (5, 4))
plt.title("Gene network computed with OLS LLR2\n(i.e. without prior information)")
nx.draw(graphOLS, nx.spring_layout(graphOLS, k = 0.25, seed = 123), with_labels = False, 
node_size = 12, edgecolors = "navy", edge_color = "gray", linewidths = 0.7, node_color = "orange", alpha = 0.7)
plt.show()
plt.savefig("NetworkOLSLLR2.png", format = "png", dpi = 500)





