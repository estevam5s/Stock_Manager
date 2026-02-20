import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def find_optimal_clusters(data, max_clusters=10):
    inertias = []
    silhouettes = []
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(data, kmeans.labels_))
    
    optimal_k = K_range[silhouettes.index(max(silhouettes))]
    
    return optimal_k, inertias, silhouettes


def perform_clustering(df, features, n_clusters=4):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaled_data)
    
    return df, kmeans, scaler


def analyze_clusters(df, cluster_col="cluster"):
    print("\n" + "="*50)
    print("         ANÁLISE DE CLUSTERS")
    print("="*50)
    
    cluster_stats = df.groupby(cluster_col).agg({
        "current_stock": ["mean", "std"],
        "monthly_sales": ["mean", "std"],
        "inventory_value": ["mean", "std"]
    }).round(2)
    
    print(cluster_stats)
    print("="*50)
    
    return cluster_stats


def plot_clusters_2d(df, feature1, feature2, cluster_col="cluster", save_path=None):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df[feature1], df[feature2], 
                         c=df[cluster_col], cmap="viridis", alpha=0.6)
    plt.colorbar(scatter, label=cluster_col)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f"Clusters: {feature1} vs {feature2}")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo em: {save_path}")
    
    plt.close()
