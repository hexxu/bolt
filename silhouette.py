import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer

df = pd.read_csv('./data/flexfield_fitness_cleaned.csv')

features = df.drop(columns=['Customer ID'])

k_values = range(2, 11)

fig, ax = plt.subplots(5, 2, figsize=(15,20))

ax = ax.flatten()

for idx, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k, random_state=0)
    
    q, mod = divmod(k, 2)

    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[idx])
    visualizer.fit(features) 

# Adjust layout for better spacing
plt.tight_layout()

# Save the entire figure with all subplots to a file
plt.savefig('./figs/silhouette_visualizations.png', dpi=300)
