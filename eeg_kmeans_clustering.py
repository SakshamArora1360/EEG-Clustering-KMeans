import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(r"D:\Saksham\Desktop\Gen Ai\Data FIles(Practice)\EEG_data.csv")

# Drop irrelevant columns
drop_cols = [
    'SubjectID', 'VideoID', 'Raw', 'Theta', 'Alpha1', 'Alpha2',
    'Beta1', 'Beta2', 'Gamma1', 'Gamma2',
    'predefinedlabel', 'user-definedlabeln'
]
df.drop(columns=drop_cols, inplace=True)

# Features
X = df.copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow & Silhouette method
Ks = range(2, 11)
inertias, sils = [], []

for k in Ks:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, labels))

# Plot Elbow & Silhouette Score
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(Ks, inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(Ks, sils, marker='o')
plt.title('Silhouette Score')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.grid()

plt.tight_layout()
plt.show()

# Best K based on Silhouette
best_k = Ks[np.argmax(sils)]
print(f'The best K value according to Silhouette Score is: {best_k}')

# Final KMeans model
kmeans = KMeans(n_clusters=best_k, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot Clusters (Attention vs Mediation)
plt.figure(figsize=(7, 5))
sns.scatterplot(
    x='Attention', y='Mediation', hue='Cluster',
    data=df, palette='Set1', s=100
)

# Plot centroids (scaled back to original feature space)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centers[:, 0], centers[:, 1],
    s=300, c='yellow', edgecolors='black', label='Centroids'
)

plt.title('EEG Brainwave Clustering: Attention vs Mediation')
plt.legend()
plt.show()
