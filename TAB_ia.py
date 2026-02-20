import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
import customtkinter as ctk
from tkinter import ttk
from tkinter import messagebox

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from con_database import Database
from functions_base import FunctionsExtras


class TabIA(FunctionsExtras):
    def __init__(self, container):
        self.container = container
        self.model = None
        self.df_produtos = None
        
        self.colors = {
            "critical": "#FF4444",
            "high": "#FF8800", 
            "medium": "#FFBB00",
            "low": "#00CC66"
        }
        
        self.setup_ui()
        self.load_model()
        self.load_and_predict()
    
    def setup_ui(self):
        self.frame_title = ctk.CTkFrame(self.container, fg_color="transparent")
        self.frame_title.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            self.frame_title,
            text="🤖 Análise de Estoque com IA",
            font=("Cascadia Code", 20, "bold")
        ).pack(side="left")
        
        self.btn_refresh = ctk.CTkButton(
            self.frame_title,
            text="🔄 Atualizar Predições",
            font=("Cascadia Code", 12, "bold"),
            command=self.load_and_predict,
            width=150,
            height=30
        )
        self.btn_refresh.pack(side="right", padx=10)
        
        self.info_frame = ctk.CTkFrame(self.container)
        self.info_frame.pack(fill="x", padx=10, pady=5)
        
        self.lbl_status = ctk.CTkLabel(
            self.info_frame,
            text="📊 Carregando modelo...",
            font=("Cascadia Code", 12)
        )
        self.lbl_status.pack(side="left", padx=10, pady=5)
        
        self.stats_frame = ctk.CTkFrame(self.container)
        self.stats_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        columns = ("ID", "Produto", "Estoque", "Mínimo", "Risco IA", "Confiança")
        self.tree = ttk.Treeview(self.stats_frame, columns=columns, show="headings", height=18)
        
        self.tree.heading("ID", text="ID")
        self.tree.heading("Produto", text="Produto")
        self.tree.heading("Estoque", text="Estoque")
        self.tree.heading("Mínimo", text="Mín.")
        self.tree.heading("Risco IA", text="Nível de Risco")
        self.tree.heading("Confiança", text="Confiança")
        
        self.tree.column("ID", width=40, anchor="center")
        self.tree.column("Produto", width=250, anchor="w")
        self.tree.column("Estoque", width=70, anchor="center")
        self.tree.column("Mínimo", width=60, anchor="center")
        self.tree.column("Risco IA", width=100, anchor="center")
        self.tree.column("Confiança", width=80, anchor="center")
        
        scrollbar = ttk.Scrollbar(self.stats_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.tree.tag_configure("critical", background="#FFCCCC", foreground="#CC0000")
        self.tree.tag_configure("high", background="#FFE5CC", foreground="#CC6600")
        self.tree.tag_configure("medium", background="#FFFFCC", foreground="#CC9900")
        self.tree.tag_configure("low", background="#CCFFCC", foreground="#006600")
        
        self.alerts_frame = ctk.CTkFrame(self.container, fg_color="#2B2B2B")
        self.alerts_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            self.alerts_frame,
            text="⚠️ Alertas de Risco",
            font=("Cascadia Code", 14, "bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        self.alerts_text = ctk.CTkTextbox(self.alerts_frame, height=60, font=("Cascadia Code", 11))
        self.alerts_text.pack(fill="x", padx=10, pady=5)
        self.alerts_text.configure(state="disabled")
    
    def load_model(self):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ML", "models", "estoque_model.pkl")
        
        if not os.path.exists(model_path):
            self.lbl_status.configure(text="⚠️ Modelo não encontrado. Execute o treinamento primeiro!", text_color="orange")
            return False
        
        try:
            self.model = joblib.load(model_path)
            self.lbl_status.configure(text="✅ Modelo IA carregado com sucesso!", text_color="green")
            return True
        except Exception as e:
            self.lbl_status.configure(text=f"❌ Erro ao carregar modelo: {str(e)}", text_color="red")
            return False
    
    def load_and_predict(self):
        if self.model is None:
            if not self.load_model():
                return
        
        try:
            conn = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stockDatabase.db"))
            df = pd.read_sql_query("SELECT * FROM estoque", conn)
            conn.close()
            
            if df.empty:
                self.lbl_status.configure(text="⚠️ Nenhum produto cadastrado!", text_color="orange")
                return
            
            df = self.prepare_features(df)
            predictions = self.model.predict(df)
            probabilities = self.model.predict_proba(df)
            confidences = [max(p) * 100 for p in probabilities]
            
            self.display_results(df, predictions, confidences)
            self.generate_alerts(df, predictions, confidences)
            
            self.lbl_status.configure(text=f"✅ Predições atualizadas! {len(df)} produtos analisados.", text_color="green")
            
        except Exception as e:
            self.lbl_status.configure(text=f"❌ Erro: {str(e)}", text_color="red")
    
    def prepare_features(self, df):
        df = df.copy()
        
        df["current_stock"] = df["estoque"].fillna(0)
        df["minimum_stock"] = df["estoque_mín"].fillna(0)
        df["maximum_stock"] = df["estoque_mín"] * 3
        df["monthly_sales"] = df["saídas"] if "saídas" in df.columns else df["saidas"] if "saidas" in df.columns else 0
        df["sales_last_7_days"] = (df["monthly_sales"] / 30) * 7
        df["sales_last_30_days"] = df["monthly_sales"]
        df["lead_time_days"] = 7
        df["unit_cost"] = df["custo_unit"].fillna(0)
        
        df["category"] = df.get("grupo", "unknown")
        df["supplier"] = df.get("fornecedor", "unknown")
        df["region"] = "SP"
        df["responsible"] = df.get("responsável", "unknown")
        
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
        
        required_cols = ["product_id", "category", "supplier", "region", "responsible",
                        "current_stock", "minimum_stock", "maximum_stock",
                        "monthly_sales", "lead_time_days", "unit_cost",
                        "sales_last_7_days", "sales_last_30_days",
                        "stock_turnover_rate", "safety_stock_ratio", 
                        "stock_coverage_days", "stock_pressure_index", "inventory_value"]
        
        for col in required_cols:
            if col not in df.columns:
                if "stock" in col or "sales" in col:
                    df[col] = 0.0
                elif "ratio" in col or "index" in col or "days" in col or "rate" in col or "value" in col:
                    df[col] = 0.0
                elif "lead_time" in col:
                    df[col] = 7
                elif "cost" in col:
                    df[col] = 0.0
                else:
                    df[col] = "unknown"
        
        return df
    
    def display_results(self, df, predictions, confidences):
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for idx, row in df.iterrows():
            risk = predictions[idx]
            conf = confidences[idx]
            
            self.tree.insert("", "end", values=(
                row.get("id", idx + 1),
                row.get("produto", f"Produto {idx + 1}")[:30],
                int(row.get("current_stock", 0)),
                int(row.get("minimum_stock", 0)),
                risk.upper(),
                f"{conf:.1f}%"
            ), tags=(risk,))
    
    def generate_alerts(self, df, predictions, confidences):
        self.alerts_text.configure(state="normal")
        self.alerts_text.delete("1.0", "end")
        
        alerts = []
        
        for idx, (pred, conf) in enumerate(zip(predictions, confidences)):
            produto = df.iloc[idx].get("produto", f"Produto {idx + 1}")
            
            if pred == "critical":
                alerts.append(f"🔴 CRÍTICO: {produto} (Estoque: {int(df.iloc[idx].get('current_stock', 0))}) - Confiança: {conf:.1f}%")
            elif pred == "high":
                alerts.append(f"🟠 ALTO RISCO: {produto} - Confiança: {conf:.1f}%")
        
        if alerts:
            self.alerts_text.insert("1.0", "\n".join(alerts[:10]))
        else:
            self.alerts_text.insert("1.0", "✅ Nenhum alerta de risco crítico!")
        
        self.alerts_text.configure(state="disabled")
