## Steps
1) First cleaned the data by applying normalizations for continuous features and 1-hot encoding for categorical 
    - Script for this is [clean_data.py](./clean_data.py)
    - Cleaned data in [flexfield_fitness_cleaned.csv](./flexfield_fitness_cleaned.csv)
2) Used a mix of elbow and silhouette method to determine optimal k
    - [elbow.png](elbow.png)
    - [silhouette_visualizations.png](./silhouette_visualizations.png)
    - Explained [here](https://builtin.com/data-science/elbow-method#:~:text=The%20elbow%20method%20is%20a%20graphical%20method%20for%20finding%20the,the%20graph%20forms%20an%20elbow.)
3) Applied clustering
    - Script for this is [clustering.py](./clustering.py)
    - Also plotted the clusterings for k=4 and k=2 in [clustering.png](./clean_data.py) and [clustering_k_2.png](./clustering_k_2.png), respectively

## Additional Notes
To data for the clusterings is not 2D, therefore it's not possible to plot the clusters with the original structure. Instead, have to use 2D approximation, and so the clustering plots do not encompass the full structure of the data.