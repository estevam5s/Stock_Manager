import pandas as pd
import numpy as np
import os


def estimate_size(n_samples):
    sample = generate_sample(n_samples=100)
    sample.to_csv('temp_size_check.csv', index=False)
    size = os.path.getsize('temp_size_check.csv')
    os.remove('temp_size_check.csv')
    return size


def generate_sample(n_samples, seed=42):
    np.random.seed(seed)
    
    categories = ["Eletrônicos", "Vestuário", "Alimentos", "Bebidas", "Cosméticos", 
                  "Ferramentas", "Móveis", "Papelaria", "Esportes", "Brinquedos",
                  "Automotivo", "Livros", "Medicamentos", "Jardinagem", "Eletrodomésticos",
                  "Informática", "Periféricos", "Acessórios", "Roupas", "Calçados"]
    suppliers = [f"Fornecedor {chr(65+i)}" for i in range(26)]
    regions = ["SP", "RJ", "MG", "RS", "BA", "PR", "SC", "PE", "CE", "GO", "AM", "PA", "MA"]
    responsables = ["João", "Maria", "Pedro", "Ana", "Carlos", "Juliana", "Roberto", "Fernanda",
                    "Lucas", "Beatriz", "Gabriel", "Larissa", "Rafael", "Camila", "Bruno"]
    
    data = {
        "product_id": [f"P{str(i).zfill(8)}" for i in range(1, n_samples + 1)],
        "category": np.random.choice(categories, n_samples),
        "supplier": np.random.choice(suppliers, n_samples),
        "region": np.random.choice(regions, n_samples),
        "responsible": np.random.choice(responsables, n_samples),
    }
    
    minimum_stock = np.random.randint(5, 2000, n_samples)
    maximum_stock = (minimum_stock * np.random.uniform(2, 10, n_samples)).astype(int)
    
    current_stock = []
    for min_s, max_s in zip(minimum_stock, maximum_stock):
        rand = np.random.random()
        if rand < 0.10:
            current_stock.append(int(np.random.randint(0, max(1, int(min_s * 0.3)))))
        elif rand < 0.20:
            current_stock.append(int(np.random.randint(max(1, int(min_s * 0.3)), min_s + 1)))
        elif rand < 0.35:
            current_stock.append(int(np.random.randint(min_s + 1, max(min_s + 2, int(max_s * 0.4) + 1))))
        elif rand < 0.55:
            current_stock.append(int(np.random.randint(max(min_s + 2, int(max_s * 0.4) + 1), max_s + 1)))
        else:
            current_stock.append(int(np.random.randint(max_s + 1, max_s * 3)))
    
    data["minimum_stock"] = minimum_stock
    data["maximum_stock"] = maximum_stock
    data["current_stock"] = current_stock
    
    monthly_sales = (np.random.randint(1, 5000, n_samples) * np.random.uniform(0.3, 3.0, n_samples)).astype(int)
    data["monthly_sales"] = monthly_sales
    
    seasonal_factors = np.random.uniform(0.2, 2.5, n_samples)
    data["sales_last_7_days"] = ((monthly_sales / 30) * 7 * seasonal_factors).astype(int)
    data["sales_last_30_days"] = (monthly_sales * seasonal_factors).astype(int)
    
    data["lead_time_days"] = np.random.randint(1, 60, n_samples)
    data["unit_cost"] = np.round(np.random.uniform(0.5, 5000, n_samples), 2)
    data["seasonality_index"] = np.round(np.random.uniform(0.1, 3.0, n_samples), 3)
    data["demand_trend"] = np.round(np.random.uniform(-1.0, 1.0, n_samples), 3)
    
    df = pd.DataFrame(data)
    
    risk_levels = []
    for _, row in df.iterrows():
        stock_ratio = row["current_stock"] / row["minimum_stock"] if row["minimum_stock"] > 0 else 0
        stock_vs_max = row["current_stock"] / row["maximum_stock"] if row["maximum_stock"] > 0 else float('inf')
        
        if row["current_stock"] < (row["minimum_stock"] * 0.2):
            risk_levels.append("critical")
        elif row["current_stock"] < (row["minimum_stock"] * 0.5):
            risk_levels.append("critical")
        elif stock_ratio < 0.5:
            risk_levels.append("critical")
        elif stock_ratio < 0.8:
            risk_levels.append("high")
        elif stock_ratio < 1.0:
            risk_levels.append("high")
        elif stock_vs_max > 1.5:
            risk_levels.append("medium")
        elif stock_ratio < 1.5:
            risk_levels.append("medium")
        else:
            risk_levels.append("low")
    
    df["target_risk_level"] = risk_levels
    
    df["stock_turnover_rate"] = np.where(df["current_stock"] > 0, df["monthly_sales"] / df["current_stock"], 0)
    df["safety_stock_ratio"] = np.where(df["current_stock"] > 0, df["minimum_stock"] / df["current_stock"], 0)
    df["stock_coverage_days"] = np.where(df["monthly_sales"] > 0, df["current_stock"] / (df["monthly_sales"] / 30), 0)
    df["stock_pressure_index"] = np.where(df["current_stock"] > 0, df["sales_last_7_days"] / df["current_stock"], 0)
    df["inventory_value"] = df["current_stock"] * df["unit_cost"]
    df["excess_stock"] = np.where(df["current_stock"] > df["maximum_stock"], df["current_stock"] - df["maximum_stock"], 0)
    df["stock_deficit"] = np.where(df["current_stock"] < df["minimum_stock"], df["minimum_stock"] - df["current_stock"], 0)
    df["sales_velocity"] = np.where(df["sales_last_30_days"] > 0, df["sales_last_7_days"] / (df["sales_last_30_days"] / 4), 0)
    df["reorder_urgency"] = np.where(df["lead_time_days"] > 0, (df["minimum_stock"] - df["current_stock"]) / df["lead_time_days"], 0)
    df["daily_sales"] = df["monthly_sales"] / 30
    df["days_until_stockout"] = np.where(df["daily_sales"] > 0, df["current_stock"] / df["daily_sales"], 999)
    df["stock_value_ratio"] = np.where(df["inventory_value"] > 0, df["monthly_sales"] / df["inventory_value"] * 30, 0)
    
    df["avg_daily_demand"] = df["monthly_sales"] / 30
    df["reorder_point"] = df["avg_daily_demand"] * df["lead_time_days"]
    df["safety_stock_needed"] = df["avg_daily_demand"] * df["lead_time_days"] * 0.5
    df["optimal_order_quantity"] = np.sqrt(2 * df["monthly_sales"] * 10 / df["unit_cost"])
    df["holding_cost"] = df["inventory_value"] * 0.2
    df["ordering_cost"] = df["monthly_sales"] / df["avg_daily_demand"] * 10
    df["total_cost"] = df["holding_cost"] + df["ordering_cost"]
    df["stockout_risk"] = np.where(df["current_stock"] < df["reorder_point"], (df["reorder_point"] - df["current_stock"]) / df["reorder_point"], 0)
    df["service_level"] = 1 - (df["stock_deficit"] / (df["minimum_stock"] + 1))
    df["fill_rate"] = np.where(df["monthly_sales"] > 0, (df["current_stock"] - df["stock_deficit"]) / df["monthly_sales"], 1)
    
    return df


def generate_massive_data(target_mb=50):
    print(f"\n{'='*60}")
    print("     GERANDO DATASET MASSIVO")
    print(f"{'='*60}")
    
    print("\n📊 Estimando tamanho...")
    sample_size = estimate_size(100)
    print(f"   Amostra de 100 registros: {sample_size / 1024:.2f} KB")
    
    target_bytes = target_mb * 1024 * 1024
    estimated_records = int(target_bytes / sample_size * 100)
    estimated_records = min(estimated_records, 500000)
    estimated_records = max(estimated_records, 50000)
    
    print(f"   Alvo: ~{target_mb}MB")
    print(f"   Registros estimados: {estimated_records:,}")
    
    print(f"\n💾 Gerando {estimated_records:,} registros...")
    df = generate_sample(estimated_records)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dados_massivos.csv")
    
    df.to_csv(output_path, index=False)
    file_size = os.path.getsize(output_path)
    
    print(f"\n{'='*60}")
    print("     DATASET MASSIVO GERADO")
    print(f"{'='*60}")
    print(f"Arquivo: {output_path}")
    print(f"Tamanho: {file_size / (1024*1024):.2f} MB")
    print(f"Registros: {len(df):,}")
    print(f"Colunas: {len(df.columns)}")
    print(f"\nDistribuição do target:")
    print(df["target_risk_level"].value_counts())
    print(f"\nEstatísticas:")
    print(f"  - Estoque médio: {df['current_stock'].mean():.1f}")
    print(f"  - Vendas mensais médias: {df['monthly_sales'].mean():.1f}")
    print(f"  - Valor inventário total: R$ {df['inventory_value'].sum():,.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    generate_massive_data(50)
