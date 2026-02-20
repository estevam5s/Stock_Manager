import sqlite3
import pandas as pd
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import create_features


def export_from_sqlite(db_path, output_path):
    print(f"1. Conectando ao banco: {db_path}")
    conn = sqlite3.connect(db_path)
    
    print(f"2. Carregando dados da tabela 'estoque'")
    df = pd.read_sql_query("SELECT * FROM estoque", conn)
    conn.close()
    
    print(f"   - {len(df)} registros carregados")
    
    df = df.rename(columns={
        "id": "product_id",
        "produto": "product_name",
        "grupo": "category",
        "medida": "unit_measure",
        "lote": "batch",
        "estoque": "current_stock",
        "estoque_mín": "minimum_stock",
        "fornecedor": "supplier",
        "responsável": "responsible",
        "entradas": "entries",
        "custo_unit": "unit_cost",
        "saídas": "sales_last_30_days",
        "valor_venda": "unit_price"
    })
    
    df["maximum_stock"] = df["minimum_stock"] * 3
    
    df["monthly_sales"] = df["sales_last_30_days"]
    df["sales_last_7_days"] = (df["monthly_sales"] / 30) * 7
    df["lead_time_days"] = 7
    df["region"] = "SP"
    df["seasonality_index"] = 1.0
    df["demand_trend"] = 0.0
    
    df["target_risk_level"] = df.apply(
        lambda x: "critical" if x["current_stock"] < (x["minimum_stock"] * 0.5)
        else "high" if x["current_stock"] < x["minimum_stock"]
        else "medium" if x["current_stock"] < (x["maximum_stock"] * 0.3)
        else "low",
        axis=1
    )
    
    required_cols = [
        "product_id", "category", "supplier", "region",
        "current_stock", "minimum_stock", "maximum_stock",
        "monthly_sales", "lead_time_days", "unit_cost",
        "sales_last_7_days", "sales_last_30_days",
        "seasonality_index", "demand_trend", "target_risk_level"
    ]
    
    for col in required_cols:
        if col not in df.columns:
            if col in ["current_stock", "minimum_stock", "maximum_stock"]:
                df[col] = 0
            elif col in ["monthly_sales", "sales_last_7_days", "sales_last_30_days"]:
                df[col] = 0
            elif col in ["lead_time_days", "unit_cost"]:
                df[col] = 0
            elif col == "seasonality_index":
                df[col] = 1.0
            elif col == "demand_trend":
                df[col] = 0.0
            else:
                df[col] = "unknown"
    
    df = df[required_cols]
    
    print(f"3. Aplicando Feature Engineering")
    df = create_features(df)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"4. Dados exportados para: {output_path}")
    print(f"   - Total de registros: {len(df)}")
    print(f"   - Colunas: {len(df.columns)}")
    
    print(f"\nDistribuição do target:")
    print(df["target_risk_level"].value_counts())


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "stockDatabase.db")
    output_path = os.path.join(base_dir, "ML", "data", "dados_estoque.csv")
    
    if os.path.exists(db_path):
        export_from_sqlite(db_path, output_path)
    else:
        print(f"Banco de dados não encontrado: {db_path}")
        print("Execute generate_sample_data.py para criar dados de exemplo")
