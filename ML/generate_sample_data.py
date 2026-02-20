import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import create_features


def generate_sample_data(n_samples=200):
    np.random.seed(42)
    
    categories = ["Eletrônicos", "Vestuário", "Alimentos", "Bebidas", "Cosméticos", 
                  "Ferramentas", "Móveis", "Papelaria", "Esportes", "Brinquedos"]
    suppliers = ["Fornecedor A", "Fornecedor B", "Fornecedor C", "Fornecedor D", "Fornecedor E"]
    regions = ["SP", "RJ", "MG", "RS", "BA", "PR", "SC", "PE"]
    responsables = ["João", "Maria", "Pedro", "Ana", "Carlos"]
    
    data = {
        "product_id": [f"P{str(i).zfill(4)}" for i in range(1, n_samples + 1)],
        "category": np.random.choice(categories, n_samples),
        "supplier": np.random.choice(suppliers, n_samples),
        "region": np.random.choice(regions, n_samples),
        "responsible": np.random.choice(responsables, n_samples),
    }
    
    minimum_stock = np.random.randint(10, 200, n_samples)
    maximum_stock = minimum_stock * np.random.randint(3, 6, n_samples)
    current_stock = np.random.randint(0, maximum_stock * 1.5, n_samples)
    
    data["minimum_stock"] = minimum_stock
    data["maximum_stock"] = maximum_stock
    data["current_stock"] = current_stock
    
    monthly_sales = np.random.randint(5, 500, n_samples)
    data["monthly_sales"] = monthly_sales
    
    data["sales_last_7_days"] = ((monthly_sales / 30) * 7 * np.random.uniform(0.5, 1.5, n_samples)).astype(int)
    data["sales_last_30_days"] = monthly_sales
    
    data["lead_time_days"] = np.random.randint(3, 30, n_samples)
    data["unit_cost"] = np.round(np.random.uniform(5, 500, n_samples), 2)
    data["seasonality_index"] = np.round(np.random.uniform(0.5, 1.5, n_samples), 2)
    data["demand_trend"] = np.round(np.random.uniform(-0.3, 0.3, n_samples), 2)
    
    df = pd.DataFrame(data)
    
    risk_levels = []
    for _, row in df.iterrows():
        if row["current_stock"] < (row["minimum_stock"] * 0.5):
            risk_levels.append("critical")
        elif row["current_stock"] < row["minimum_stock"]:
            risk_levels.append("high")
        elif row["current_stock"] < (row["maximum_stock"] * 0.3):
            risk_levels.append("medium")
        elif row["current_stock"] > row["maximum_stock"]:
            risk_levels.append("medium")
        else:
            risk_levels.append("low")
    
    df["target_risk_level"] = risk_levels
    
    df = create_features(df)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dados_estoque.csv")
    
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*50}")
    print("     DADOS DE EXEMPLO GERADOS")
    print(f"{'='*50}")
    print(f"Arquivo: {output_path}")
    print(f"Registros: {len(df)}")
    print(f"Colunas: {len(df.columns)}")
    print(f"\nDistribuição do target:")
    print(df["target_risk_level"].value_counts())
    print(f"{'='*50}\n")


if __name__ == "__main__":
    generate_sample_data(200)
