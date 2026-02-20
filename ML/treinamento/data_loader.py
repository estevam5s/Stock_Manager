import os
import sqlite3
import pandas as pd
import numpy as np


class DataLoader:
    @staticmethod
    def load_csv(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        return pd.read_csv(filepath)
    
    @staticmethod
    def load_from_sqlite(db_path, table_name="estoque"):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Banco não encontrado: {db_path}")
        
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df
    
    @staticmethod
    def prepare_stock_data(df):
        df = df.copy()
        
        df["current_stock"] = df["estoque"].fillna(0)
        df["minimum_stock"] = df["estoque_mín"].fillna(0)
        df["maximum_stock"] = df["estoque_mín"] * 3
        df["monthly_sales"] = df.get("saídas", df.get("saidas", 0))
        df["sales_last_7_days"] = (df["monthly_sales"] / 30) * 7
        df["sales_last_30_days"] = df["monthly_sales"]
        df["lead_time_days"] = 7
        df["unit_cost"] = df["custo_unit"].fillna(0)
        df["category"] = df.get("grupo", "unknown")
        df["supplier"] = df.get("fornecedor", "unknown")
        df["region"] = "SP"
        df["product_id"] = df.get("id", range(1, len(df) + 1))
        
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
        
        return df
    
    @staticmethod
    def generate_sample_data(n_samples=1000):
        np.random.seed(42)
        
        categories = ["Eletrônicos", "Vestuário", "Alimentos", "Bebidas", "Cosméticos",
                      "Ferramentas", "Móveis", "Papelaria", "Esportes", "Brinquedos"]
        suppliers = [f"Fornecedor {chr(65+i)}" for i in range(10)]
        regions = ["SP", "RJ", "MG", "RS", "BA", "PR", "SC", "PE"]
        
        data = {
            "product_id": [f"P{str(i).zfill(6)}" for i in range(1, n_samples + 1)],
            "category": np.random.choice(categories, n_samples),
            "supplier": np.random.choice(suppliers, n_samples),
            "region": np.random.choice(regions, n_samples),
        }
        
        minimum_stock = np.random.randint(10, 500, n_samples)
        maximum_stock = minimum_stock * np.random.randint(3, 6, n_samples)
        current_stock = np.random.randint(0, maximum_stock * 2, n_samples)
        
        data["minimum_stock"] = minimum_stock
        data["maximum_stock"] = maximum_stock
        data["current_stock"] = current_stock
        
        monthly_sales = np.random.randint(5, 500, n_samples)
        data["monthly_sales"] = monthly_sales
        data["sales_last_7_days"] = ((monthly_sales / 30) * 7).astype(int)
        data["sales_last_30_days"] = monthly_sales
        
        data["lead_time_days"] = np.random.randint(3, 30, n_samples)
        data["unit_cost"] = np.round(np.random.uniform(5, 500, n_samples), 2)
        
        df = pd.DataFrame(data)
        
        risk_levels = []
        for _, row in df.iterrows():
            if row["current_stock"] < (row["minimum_stock"] * 0.5):
                risk_levels.append("critical")
            elif row["current_stock"] < row["minimum_stock"]:
                risk_levels.append("high")
            elif row["current_stock"] < (row["maximum_stock"] * 0.3):
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
        
        return df
    
    @staticmethod
    def get_data_summary(df):
        summary = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=["object"]).columns),
            "missing_values": df.isnull().sum().sum(),
        }
        
        if "target_risk_level" in df.columns:
            summary["target_distribution"] = df["target_risk_level"].value_counts().to_dict()
        
        return summary
