import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv('./data/flexfield_fitness_cleaned.csv')

# Get rid of 'Customer ID', don't want to include in clustering analysis
features = df.drop(columns=['Customer ID'])

# Set number of clusters to 4
k = 4

kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(features)

# Add cluster labels to data
df['Cluster'] = kmeans.labels_

df.to_csv('./data/flexfield_fitness_with_clusters.csv', index=False)

# Perform PCA for 2D visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features)

pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = df['Cluster']

plt.figure(figsize=(10, 7))
for cluster in range(k):
    subset = pca_df[pca_df['Cluster'] == cluster]
    plt.scatter(subset['PC1'], subset['PC2'], label=f'Cluster {cluster}', s=50)

plt.title('K-means Clustering with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.savefig('./figs/clustering.png')





