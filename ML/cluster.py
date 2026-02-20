import sys
import os
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config


def find_optimal_clusters(data, max_clusters=10):
    inertias = []
    silhouettes = []
    K_range = range(2, max_clusters + 1)
    
    print("\n🔍 Buscando número ideal de clusters...")
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(data, kmeans.labels_)
        silhouettes.append(sil)
        print(f"   k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil:.3f}")
    
    optimal_k = K_range[silhouettes.index(max(silhouettes))]
    print(f"\n✓ Número ideal de clusters: {optimal_k}")
    
    return optimal_k, inertias, silhouettes


def perform_clustering(data, n_clusters=4, find_optimal=False):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    if find_optimal:
        n_clusters, inertias, silhouettes = find_optimal_clusters(scaled_data)
    
    print(f"\n📊 Executando K-Means com {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    silhouette_avg = silhouette_score(scaled_data, clusters)
    print(f"✓ Silhouette Score: {silhouette_avg:.3f}")
    
    return clusters, kmeans, scaler


def analyze_clusters(df, cluster_col="cluster"):
    print("\n" + "="*60)
    print("        ANÁLISE DE CLUSTERS")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != cluster_col]
    
    if len(numeric_cols) == 0:
        print("⚠️ Nenhuma coluna numérica para análise")
        return
    
    cluster_stats = df.groupby(cluster_col)[numeric_cols[:6]].agg(["mean", "std", "min", "max"])
    
    print("\nEstatísticas por Cluster:")
    print("-" * 60)
    
    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id]
        print(f"\n🔹 Cluster {cluster_id} ({len(cluster_data)} produtos)")
        
        if "current_stock" in cluster_data.columns:
            avg_stock = cluster_data["current_stock"].mean()
            print(f"   Estoque médio: {avg_stock:.1f}")
        
        if "monthly_sales" in cluster_data.columns:
            avg_sales = cluster_data["monthly_sales"].mean()
            print(f"   Vendas mensais médias: {avg_sales:.1f}")
        
        if "inventory_value" in cluster_data.columns:
            avg_value = cluster_data["inventory_value"].mean()
            print(f"   Valor inventário médio: R$ {avg_value:.2f}")
    
    print("\n" + "="*60)
    
    return cluster_stats


def plot_clusters(df, features, output_dir="ML/models"):
    print(f"\n📈 Gerando visualizações...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if len(features) >= 2:
        plt.figure(figsize=(12, 8))
        
        x_feat = features[0]
        y_feat = features[1] if len(features) > 1 else features[0]
        
        scatter = plt.scatter(df[x_feat], df[y_feat], 
                             c=df["cluster"], cmap="viridis", 
                             alpha=0.7, s=100)
        plt.colorbar(scatter, label="Cluster")
        plt.xlabel(x_feat)
        plt.ylabel(y_feat)
        plt.title(f"Clusters: {x_feat} vs {y_feat}")
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, "cluster_visualization.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✓ Visualização salva em: {output_path}")
    
    plt.figure(figsize=(10, 6))
    cluster_counts = df["cluster"].value_counts().sort_index()
    plt.bar(cluster_counts.index, cluster_counts.values, color=plt.cm.viridis(np.linspace(0, 1, len(cluster_counts))))
    plt.xlabel("Cluster")
    plt.ylabel("Número de Produtos")
    plt.title("Distribuição de Produtos por Cluster")
    plt.xticks(cluster_counts.index)
    
    output_path = os.path.join(output_dir, "cluster_distribution.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Distribuição salva em: {output_path}")


def save_model(kmeans, scaler, output_path):
    model_data = {
        "kmeans": kmeans,
        "scaler": scaler
    }
    joblib.dump(model_data, output_path)
    print(f"\n✓ Modelo de clustering salvo em: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Clusterização de produtos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python ML/cluster.py ML/data/dados_estoque.csv
  python ML/cluster.py ML/data/dados_estoque.csv --clusters 4
  python ML/cluster.py ML/data/dados_estoque.csv --find-optimal
  python ML/cluster.py ML/data/dados_estoque.csv --output ML/models/clustering.pkl
        """
    )
    parser.add_argument("data_path", help="Caminho do arquivo CSV")
    parser.add_argument("--clusters", "-k", type=int, default=4, help="Número de clusters (padrão: 4)")
    parser.add_argument("--find-optimal", "-f", action="store_true", help="Encontrar número ideal de clusters")
    parser.add_argument("--output", "-o", help="Caminho para salvar modelo")
    parser.add_argument("--no-plot", action="store_true", help="Não gerar visualizações")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("        CLUSTERIZAÇÃO - KMEANS")
    print(f"{'='*60}\n")
    
    if not os.path.exists(args.data_path):
        print(f"❌ ERRO: Arquivo não encontrado: {args.data_path}")
        sys.exit(1)
    
    df = pd.read_csv(args.data_path)
    print(f"✓ {len(df)} registros carregados")
    
    feature_cols = ["current_stock", "minimum_stock", "maximum_stock", 
                   "monthly_sales", "sales_last_7_days", "lead_time_days", 
                   "unit_cost", "inventory_value"]
    
    available_features = [f for f in feature_cols if f in df.columns]
    
    if len(available_features) < 2:
        print("⚠️ Features insuficientes, usando todas as numéricas...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        available_features = list(numeric_cols)
    
    print(f"Features utilizadas: {available_features}")
    
    data = df[available_features].fillna(0)
    
    clusters, kmeans, scaler = perform_clustering(
        data, 
        n_clusters=args.clusters, 
        find_optimal=args.find_optimal
    )
    
    df["cluster"] = clusters
    
    analyze_clusters(df, "cluster")
    
    if not args.no_plot:
        plot_clusters(df, available_features)
    
    if args.output:
        save_model(kmeans, scaler, args.output)
    else:
        output_path = args.data_path.replace(".csv", "_clustered.csv")
        df.to_csv(output_path, index=False)
        print(f"\n✓ Dados com clusters salvos em: {output_path}")
    
    print(f"\n{'='*60}")
    print("        CLUSTERIZAÇÃO CONCLUÍDA!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
