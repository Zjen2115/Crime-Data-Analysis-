import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import numpy as np

crime = pd.read_csv("Crime.csv", low_memory=False)

crime = crime[crime['State'] == "MD"].copy()

crime['Start_Date_Time'] = pd.to_datetime(crime['Start_Date_Time'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
crime['End_Date_Time'] = pd.to_datetime(crime['End_Date_Time'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
crime['Dispatch Date / Time'] = pd.to_datetime(crime['Dispatch Date / Time'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')

crime['Crime Duration'] = (crime['End_Date_Time'] - crime['Start_Date_Time']).dt.total_seconds()
crime["Hour"] = crime["Start_Date_Time"].dt.hour

features = ['Crime Duration', 'Hour']

X = crime[features].dropna().values

print(f"Number of samples: {X.shape[0]}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if X_scaled.shape[0] > 10000:
    np.random.seed(0)
    indices = np.random.choice(range(X_scaled.shape[0]), size=10000, replace=False)
    X_sample = X_scaled[indices]
else:
    X_sample = X_scaled

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method - Crime Data')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"Silhouette Score for k={k}: {score:.3f}")

n_clusters = 4

print("\nKMeans Clustering\n")
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
clf = kmeans.fit(X)
centroids = clf.cluster_centers_
score = silhouette_score(X, clf.labels_)
print("Centroids:\n", centroids)
print("Silhouette Score:", score)

y_kmeans = clf.labels_

plt.figure(figsize=(8, 5))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='purple', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='orange', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='blue', label='Cluster 4')

plt.scatter(centroids[:, 0], centroids[:, 1], s=400, c='red', marker='x', label='Centroids')
plt.title("KMeans Clustering - Crime Data")
plt.xlabel("Crime Duration (seconds)")
plt.ylabel("Hour")
plt.legend()
plt.show()

print("\nAgglomerative Clustering\n")
agglo_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
agglo_labels = agglo_model.fit_predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(X[agglo_labels == 0, 0], X[agglo_labels == 0, 1], s=100, c='purple', label='Cluster 1')
plt.scatter(X[agglo_labels == 1, 0], X[agglo_labels == 1, 1], s=100, c='orange', label='Cluster 2')
plt.scatter(X[agglo_labels == 2, 0], X[agglo_labels == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[agglo_labels == 3, 0], X[agglo_labels == 3, 1], s=100, c='blue', label='Cluster 4')

plt.title("Agglomerative Clustering - Crime Data")
plt.xlabel("Crime Duration (seconds)")
plt.ylabel("Hour")
plt.legend()
plt.show()

# Dendrogram
linkage_matrix = linkage(X, method='ward')
plt.figure(figsize=(14, 7))
dendrogram(linkage_matrix, truncate_mode='lastp', p=20, color_threshold=0)
plt.title("Hierarchical Clustering Dendrogram (ward) - Crime Data")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()