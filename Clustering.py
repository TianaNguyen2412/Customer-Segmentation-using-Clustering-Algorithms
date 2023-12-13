# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# -

# # Data Exploration

df = pd.read_csv('data.csv')
print(f'Shape of data: {df.shape}')
df.head()

df.info()

df.columns = ['CustomerID', 'Gender', 'Age', 'Annual Income', 'Spending Score']

df.iloc[:, 2:].describe().round(2)

df.isna().sum().sum()

df.duplicated().sum()

# ## Data Visualization

# +
plt.figure(figsize=(12, 4))
plt.tight_layout(pad=5)

plt.subplot(1, 3, 1)
sns.histplot(x='Age', data=df).set_title('Age Distribution')

plt.subplot(1, 3, 2)
sns.histplot(x='Annual Income', data=df).set_title('Annual Income Distribution')

plt.subplot(1, 3, 3)
sns.histplot(x='Spending Score', data=df).set_title('Spending Score Distribution')
# -

# Summary:
# <li>Most of the customers are under 50 years old with an annual income of under 78,000 USD.
# <li>Spending score is on a scale of 0 to 100, mainly falling into the range [35, 55].

sns.pairplot(data=df, x_vars=['Age'], y_vars=['Spending Score'], hue='Age', palette='Blues')

# There is a difference in spending scores among ages. While customers under 35 often spend a lot with Spending Score of above 40, older customers have lower spending scores, under 60.

sns.relplot(x='Annual Income', y='Spending Score', data=df, kind='scatter', hue='Gender')

# It appears to have 5 groups of customers with different behaviors based on annual income and spending score. We also do not observe a clear distinction regarding gender in this plot.

sns.boxplot(x='Gender', y='Spending Score', data=df).set_title('Spending Score by Gender')

# This boxplot provides us with a slight difference in spending scores between males and females. They all have the same average spending score, but spending score for females seems higher than that for males.

# # Clustering

features = df[['Gender', 'Age', 'Annual Income', 'Spending Score']].copy()

features = pd.get_dummies(features, columns=['Gender'])
features.head()

import warnings
warnings.filterwarnings('ignore')

# Determine the best clustering algorithm
bestSil = -1
for k in range(2, 6):
    print (f'k={k}')
    clus = [KMeans(n_clusters=k, n_init='auto'), 
            Birch(n_clusters=k), 
            AgglomerativeClustering(n_clusters=k), 
            SpectralClustering(n_clusters=k)]
    for cl in clus:
        res = cl.fit(features)
        sil = silhouette_score(features, res.labels_)
        print (f"{(str(cl).split('(')[0]+'   ')[:8]} with k={str(k)}: {str(round(sil,4))}")
        if (sil > bestSil):
            bestSil = sil
            bestCl = cl
            bestK = k
print('***********************************************')
print ('Best algorithm is....... ' + str(bestCl).split('(')[0][:6] + ' with k=' + str(bestK))
print('************************')
print ('With Silhouette Score of ' + str(round(bestSil, 4)))

kmeans = KMeans(n_clusters=5, n_init='auto', random_state=0)
kmeans.fit(features)

features['Cluster'] = kmeans.labels_+1
features.head(5)

# Get summary information on the clusters' characteristics
features.groupby('Cluster').mean()

# Eyeballing the results, we realize the difference among customer groups mainly originates from values of annual income and spending score.
# <br>Therefore, customers are divided into 5 groups as below:
# <li>Cluster 1: Customers have a high annual income but a low spending score.
# <li>Cluster 2: Customers have a medium-high annual income and spending score.
# <li>Cluster 3: Customers have a low-medium annual income and spending score.
# <li>Cluster 4: Customers have a low annual income but high spending score.
# <li>Cluster 5: Customers have a high annual income and spending score.

# ## Dimensionality Reduction

# Now we will use PCA (Principal Component Analysis) and t-SNE (t-distributed stochastic neighbor embedding) to reduce the dimensionality of the dataset for visualization. To do that, we need to convert the features from a 5-dimensional space to a 2-dimensional space.

# +
# PCA with 2 components
pca = PCA(n_components=2).fit_transform(features)
features['PCA1'] = pca[:, 0]
features['PCA2'] = pca[:, 1]

# TSNE with 2 components
tsne = TSNE(n_components=2).fit_transform(features)
features['TSNE1'] = tsne[:, 0]
features['TSNE2'] = tsne[:, 1]

# +
plt.figure(figsize=(10, 5))
plt.tight_layout(pad=5)

plt.subplot(1, 2, 1)
sns.scatterplot(x='PCA1', y='PCA2', data=features, hue='Cluster', palette='RdYlBu').set_title('Visualization using PCA')

plt.subplot(1, 2, 2)
sns.scatterplot(x='TSNE1', y='TSNE2', data=features, hue='Cluster', palette='RdYlBu').set_title('Visualization using TSNE')
# -

# Although there is some overlap between clusters 2 and 3 from visualization using PCA, the separation among 5 clusters is clear by the TSNE method. Therefore, categorizing customers into 5 groups makes sense.
