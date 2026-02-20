import pandas as pd
import numpy as np
import os


def generate_rich_data(n_samples=1000):
    np.random.seed(42)
    
    categories = ["Eletrônicos", "Vestuário", "Alimentos", "Bebidas", "Cosméticos", 
                  "Ferramentas", "Móveis", "Papelaria", "Esportes", "Brinquedos",
                  "Automotivo", "Livros", "Medicamentos", "Jardinagem", "Eletrodomésticos"]
    suppliers = ["Fornecedor A", "Fornecedor B", "Fornecedor C", "Fornecedor D", "Fornecedor E",
                 "Fornecedor F", "Fornecedor G", "Fornecedor H", "Fornecedor I", "Fornecedor J"]
    regions = ["SP", "RJ", "MG", "RS", "BA", "PR", "SC", "PE", "CE", "GO"]
    responsables = ["João", "Maria", "Pedro", "Ana", "Carlos", "Juliana", "Roberto", "Fernanda"]
    
    data = {
        "product_id": [f"P{str(i).zfill(5)}" for i in range(1, n_samples + 1)],
        "category": np.random.choice(categories, n_samples),
        "supplier": np.random.choice(suppliers, n_samples),
        "region": np.random.choice(regions, n_samples),
        "responsible": np.random.choice(responsables, n_samples),
    }
    
    minimum_stock = np.random.randint(10, 500, n_samples)
    maximum_stock = minimum_stock * np.random.randint(3, 8, n_samples)
    
    current_stock = []
    for min_s, max_s in zip(minimum_stock, maximum_stock):
        rand = np.random.random()
        if rand < 0.15:
            current_stock.append(int(np.random.randint(0, max(1, int(min_s * 0.5)))))
        elif rand < 0.30:
            current_stock.append(int(np.random.randint(max(1, int(min_s * 0.5)), min_s + 1)))
        elif rand < 0.50:
            current_stock.append(int(np.random.randint(min_s + 1, max_s + 1)))
        else:
            current_stock.append(int(np.random.randint(max_s + 1, max_s * 2)))
    
    data["minimum_stock"] = minimum_stock
    data["maximum_stock"] = maximum_stock
    data["current_stock"] = current_stock
    
    monthly_sales = np.random.randint(5, 1000, n_samples)
    monthly_sales = monthly_sales * np.random.uniform(0.5, 1.5, n_samples)
    data["monthly_sales"] = monthly_sales.astype(int)
    
    seasonal_factors = np.random.uniform(0.5, 1.5, n_samples)
    data["sales_last_7_days"] = ((monthly_sales / 30) * 7 * seasonal_factors).astype(int)
    data["sales_last_30_days"] = (monthly_sales * seasonal_factors).astype(int)
    
    data["lead_time_days"] = np.random.randint(1, 45, n_samples)
    data["unit_cost"] = np.round(np.random.uniform(1, 1000, n_samples), 2)
    data["seasonality_index"] = np.round(np.random.uniform(0.3, 2.0, n_samples), 2)
    data["demand_trend"] = np.round(np.random.uniform(-0.5, 0.5, n_samples), 2)
    
    df = pd.DataFrame(data)
    
    risk_levels = []
    for _, row in df.iterrows():
        stock_ratio = row["current_stock"] / row["minimum_stock"] if row["minimum_stock"] > 0 else 0
        stock_vs_max = row["current_stock"] / row["maximum_stock"] if row["maximum_stock"] > 0 else float('inf')
        
        if row["current_stock"] < (row["minimum_stock"] * 0.3):
            risk_levels.append("critical")
        elif row["current_stock"] < (row["minimum_stock"] * 0.6):
            risk_levels.append("high")
        elif stock_ratio < 1.0:
            risk_levels.append("high")
        elif stock_vs_max > 1.2:
            risk_levels.append("medium")
        elif stock_ratio < 1.5:
            risk_levels.append("medium")
        else:
            risk_levels.append("low")
    
    df["target_risk_level"] = risk_levels
    
    df["stock_turnover_rate"] = np.where(
        df["current_stock"] > 0,
        df["monthly_sales"] / df["current_stock"],
        0
    )
    
    df["safety_stock_ratio"] = np.where(
        df["current_stock"] > 0,
        df["minimum_stock"] / df["current_stock"],
        0
    )
    
    df["stock_coverage_days"] = np.where(
        df["monthly_sales"] > 0,
        df["current_stock"] / (df["monthly_sales"] / 30),
        0
    )
    
    df["stock_pressure_index"] = np.where(
        df["current_stock"] > 0,
        df["sales_last_7_days"] / df["current_stock"],
        0
    )
    
    df["inventory_value"] = df["current_stock"] * df["unit_cost"]
    
    df["excess_stock"] = np.where(
        df["current_stock"] > df["maximum_stock"],
        df["current_stock"] - df["maximum_stock"],
        0
    )
    
    df["stock_deficit"] = np.where(
        df["current_stock"] < df["minimum_stock"],
        df["minimum_stock"] - df["current_stock"],
        0
    )
    
    df["sales_velocity"] = np.where(
        df["sales_last_30_days"] > 0,
        df["sales_last_7_days"] / (df["sales_last_30_days"] / 4),
        0
    )
    
    df["reorder_urgency"] = np.where(
        df["lead_time_days"] > 0,
        (df["minimum_stock"] - df["current_stock"]) / df["lead_time_days"],
        0
    )
    
    df["daily_sales"] = df["monthly_sales"] / 30
    
    df["days_until_stockout"] = np.where(
        df["daily_sales"] > 0,
        df["current_stock"] / df["daily_sales"],
        999
    )
    
    df["stock_value_ratio"] = np.where(
        df["inventory_value"] > 0,
        df["monthly_sales"] / df["inventory_value"] * 30,
        0
    )
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dados_ricos.csv")
    
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print("     DADOS RICOS GERADOS")
    print(f"{'='*60}")
    print(f"Arquivo: {output_path}")
    print(f"Registros: {len(df)}")
    print(f"Colunas: {len(df.columns)}")
    print(f"\nDistribuição do target:")
    print(df["target_risk_level"].value_counts())
    print(f"\nEstatísticas:")
    print(f"  - Estoque médio: {df['current_stock'].mean():.1f}")
    print(f"  - Vendas mensais médias: {df['monthly_sales'].mean():.1f}")
    print(f"  - Valor inventário total: R$ {df['inventory_value'].sum():,.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    generate_rich_data(1000)
