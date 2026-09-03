from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = []
for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    results.append([k, model.inertia_, silhouette_score(X_scaled, labels)])

metrics = pd.DataFrame(results, columns=["k","inertia","silhouette_score"])
best_k = int(metrics.loc[metrics["silhouette_score"].idxmax(),"k"])

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df = X.copy()
df["cluster"] = clusters

print("Dataset shape:", X.shape)
print("\nK selection:")
print(metrics)
print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index())
print("\nCluster profiles:")
print(df.groupby("cluster")[iris.feature_names].mean().round(2))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,5))
for c in sorted(df["cluster"].unique()):
    mask=df["cluster"]==c
    plt.scatter(X_pca[mask,0],X_pca[mask,1],label=f"Cluster {c}")
centers=pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0],centers[:,1],marker="X",s=180,label="Centroids")
plt.title(f"K-Means Clusters with PCA (k={best_k})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(); plt.tight_layout(); plt.show()

print("\nOptional external comparison (labels not used for training):")
print("Adjusted Rand Index:", round(adjusted_rand_score(y,clusters),3))

df.to_csv("task3_iris_clustered.csv",index=False)
metrics.to_csv("task3_k_selection_metrics.csv",index=False)
df.groupby("cluster")[iris.feature_names].mean().round(2).to_csv("task3_cluster_profiles.csv")
