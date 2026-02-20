import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
import threading
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import customtkinter as ctk
from tkinter import ttk, filedialog, messagebox
from tkinter import *
from PIL import Image

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class TrainGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("🤖 Treinamento de Modelo ML - Análise de Estoque")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        self.df = None
        self.model = None
        self.results = None
        self.training_thread = None
        self.stop_training = False
        
        self.colors = {
            "primary": "#2B2B2B",
            "secondary": "#3B3B3B",
            "accent": "#0078D4",
            "success": "#00CC66",
            "warning": "#FFBB00",
            "error": "#FF4444",
            "critical": "#FF0000"
        }
        
        self.setup_layout()
        
    def setup_layout(self):
        self.main_frame = ctk.CTkScrollableFrame(self.root, fg_color=self.colors["primary"])
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.header()
        self.data_source_frame()
        self.model_config_frame()
        self.target_feature_frame()
        self.preview_frame()
        self.training_frame()
        self.metrics_frame()
        self.action_buttons()
        
    def header(self):
        header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 20))
        
        title = ctk.CTkLabel(
            header_frame,
            text="🤖 TREINAMENTO DE MODELO ML - ANÁLISE DE ESTOQUE",
            font=("Cascadia Code", 24, "bold")
        )
        title.pack(side="left")
        
        version = ctk.CTkLabel(
            header_frame,
            text="v2.0 | Professional Edition",
            font=("Cascadia Code", 12),
            text_color="gray"
        )
        version.pack(side="right")
        
    def data_source_frame(self):
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(fill="x", pady=5)
        
        title = ctk.CTkLabel(
            frame,
            text="📂 FONTE DE DADOS",
            font=("Cascadia Code", 16, "bold")
        )
        title.pack(anchor="w", padx=15, pady=(10, 5))
        
        options_frame = ctk.CTkFrame(frame, fg_color="transparent")
        options_frame.pack(fill="x", padx=15, pady=5)
        
        self.var_upload = ctk.BooleanVar(value=True)
        self.var_generate = ctk.BooleanVar(value=False)
        self.var_sqlite = ctk.BooleanVar(value=False)
        self.var_manual = ctk.BooleanVar(value=False)
        
        ctk.CTkCheckBox(
            options_frame,
            text="📤 Upload CSV",
            variable=self.var_upload,
            command=self.on_data_source_change
        ).pack(side="left", padx=10)
        
        ctk.CTkCheckBox(
            options_frame,
            text="🎲 Gerar Dados",
            variable=self.var_generate,
            command=self.on_data_source_change
        ).pack(side="left", padx=10)
        
        ctk.CTkCheckBox(
            options_frame,
            text="🗄️ SQLite",
            variable=self.var_sqlite,
            command=self.on_data_source_change
        ).pack(side="left", padx=10)
        
        ctk.CTkCheckBox(
            options_frame,
            text="✏️ Manual",
            variable=self.var_manual,
            command=self.on_data_source_change
        ).pack(side="left", padx=10)
        
        self.data_options_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.data_options_frame.pack(fill="x", padx=15, pady=10)
        
        self.load_data_options()
        
        self.data_info_frame = ctk.CTkFrame(frame, fg_color=self.colors["secondary"])
        self.data_info_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        self.lbl_data_info = ctk.CTkLabel(
            self.data_info_frame,
            text="📭 Nenhum dado carregado",
            font=("Cascadia Code", 12)
        )
        self.lbl_data_info.pack(padx=10, pady=10)
        
    def load_data_options(self):
        for widget in self.data_options_frame.winfo_children():
            widget.destroy()
            
        if self.var_upload.get():
            btn_frame = ctk.CTkFrame(self.data_options_frame, fg_color="transparent")
            btn_frame.pack(fill="x", pady=5)
            
            ctk.CTkButton(
                btn_frame,
                text="📂 Selecionar Arquivo CSV",
                command=self.upload_csv,
                width=200
            ).pack(side="left", padx=5)
            
        if self.var_generate.get():
            gen_frame = ctk.CTkFrame(self.data_options_frame, fg_color="transparent")
            gen_frame.pack(fill="x", pady=5)
            
            ctk.CTkLabel(gen_frame, text="Quantidade:").pack(side="left", padx=5)
            
            self.var_quantity = ctk.StringVar(value="1000")
            combo = ctk.CTkOptionMenu(
                gen_frame,
                values=["100", "1,000", "10,000", "100,000", "1,000,000"],
                variable=self.var_quantity,
                width=150
            )
            combo.pack(side="left", padx=5)
            
            ctk.CTkButton(
                gen_frame,
                text="🎲 Gerar Dados",
                command=self.generate_data,
                width=150
            ).pack(side="left", padx=5)
            
        if self.var_sqlite.get():
            sqlite_frame = ctk.CTkFrame(self.data_options_frame, fg_color="transparent")
            sqlite_frame.pack(fill="x", pady=5)
            
            ctk.CTkButton(
                sqlite_frame,
                text="🗄️ Exportar do SQLite",
                command=self.export_sqlite,
                width=200
            ).pack(side="left", padx=5)
            
        if self.var_manual.get():
            manual_frame = ctk.CTkFrame(self.data_options_frame, fg_color="transparent")
            manual_frame.pack(fill="x", pady=5)
            
            ctk.CTkButton(
                manual_frame,
                text="✏️ Adicionar Dados Manualmente",
                command=self.manual_data_entry,
                width=200
            ).pack(side="left", padx=5)
            
    def on_data_source_change(self):
        self.load_data_options()
        
    def upload_csv(self):
        filename = filedialog.askopenfilename(
            title="Selecionar Arquivo CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.df = pd.read_csv(filename)
                self.update_data_info(f"📄 Arquivo: {os.path.basename(filename)} | Registros: {len(self.df):,} | Colunas: {len(self.df.columns)}")
                self.update_preview()
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao carregar arquivo: {str(e)}")
                
    def generate_data(self):
        quantity_map = {
            "100": 100,
            "1,000": 1000,
            "10,000": 10000,
            "100,000": 100000,
            "1,000,000": 1000000
        }
        
        n_samples = quantity_map.get(self.var_quantity.get(), 1000)
        
        try:
            self.df = self.generate_sample_data(n_samples)
            self.update_data_info(f"🎲 Dados gerados | Registros: {len(self.df):,} | Colunas: {len(self.df.columns)}")
            self.update_preview()
            messagebox.showinfo("Sucesso", f"Dados gerados com {len(self.df):,} registros!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar dados: {str(e)}")
            
    def generate_sample_data(self, n_samples):
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
        
        df["stock_turnover_rate"] = np.where(df["current_stock"] > 0, df["monthly_sales"] / df["current_stock"], 0)
        df["safety_stock_ratio"] = np.where(df["current_stock"] > 0, df["minimum_stock"] / df["current_stock"], 0)
        df["stock_coverage_days"] = np.where(df["monthly_sales"] > 0, df["current_stock"] / (df["monthly_sales"] / 30), 0)
        df["stock_pressure_index"] = np.where(df["current_stock"] > 0, df["sales_last_7_days"] / df["current_stock"], 0)
        df["inventory_value"] = df["current_stock"] * df["unit_cost"]
        
        return df
        
    def export_sqlite(self):
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stockDatabase.db")
        
        if not os.path.exists(db_path):
            messagebox.showerror("Erro", "Banco de dados não encontrado!")
            return
            
        try:
            conn = sqlite3.connect(db_path)
            self.df = pd.read_sql_query("SELECT * FROM estoque", conn)
            conn.close()
            
            self.df["minimum_stock"] = self.df["estoque_mín"]
            self.df["maximum_stock"] = self.df["estoque_mín"] * 3
            self.df["current_stock"] = self.df["estoque"]
            self.df["monthly_sales"] = self.df["saídas"]
            self.df["sales_last_7_days"] = (self.df["monthly_sales"] / 30) * 7
            self.df["sales_last_30_days"] = self.df["monthly_sales"]
            self.df["lead_time_days"] = 7
            self.df["unit_cost"] = self.df["custo_unit"]
            self.df["category"] = self.df["grupo"]
            self.df["supplier"] = self.df["fornecedor"]
            self.df["region"] = "SP"
            self.df["product_id"] = self.df["id"]
            
            self.df["stock_turnover_rate"] = np.where(self.df["current_stock"] > 0, self.df["monthly_sales"] / self.df["current_stock"], 0)
            self.df["safety_stock_ratio"] = np.where(self.df["current_stock"] > 0, self.df["minimum_stock"] / self.df["current_stock"], 0)
            self.df["stock_coverage_days"] = np.where(self.df["monthly_sales"] > 0, self.df["current_stock"] / (self.df["monthly_sales"] / 30), 0)
            self.df["stock_pressure_index"] = np.where(self.df["current_stock"] > 0, self.df["sales_last_7_days"] / self.df["current_stock"], 0)
            self.df["inventory_value"] = self.df["current_stock"] * self.df["unit_cost"]
            
            self.df["target_risk_level"] = self.df.apply(
                lambda x: "critical" if x["current_stock"] < (x["minimum_stock"] * 0.5)
                else "high" if x["current_stock"] < x["minimum_stock"]
                else "medium" if x["current_stock"] < (x["maximum_stock"] * 0.3)
                else "low", axis=1
            )
            
            self.update_data_info(f"🗄️ SQLite | Registros: {len(self.df):,} | Colunas: {len(self.df.columns)}")
            self.update_preview()
            messagebox.showinfo("Sucesso", f"Dados exportados: {len(self.df)} registros")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao exportar: {str(e)}")
            
    def manual_data_entry(self):
        messagebox.showinfo("Entrada Manual", "Funcionalidade de entrada manual de dados")
        
    def update_data_info(self, text):
        self.lbl_data_info.configure(text=text)
        
    def model_config_frame(self):
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(fill="x", pady=5)
        
        title = ctk.CTkLabel(
            frame,
            text="⚙️ CONFIGURAÇÃO DO MODELO",
            font=("Cascadia Code", 16, "bold")
        )
        title.pack(anchor="w", padx=15, pady=(10, 5))
        
        config_grid = ctk.CTkFrame(frame, fg_color="transparent")
        config_grid.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(config_grid, text="Modelo:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.var_model = ctk.StringVar(value="random_forest")
        combo = ctk.CTkOptionMenu(
            config_grid,
            values=["random_forest", "gradient_boosting"],
            variable=self.var_model,
            width=200
        )
        combo.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        ctk.CTkLabel(config_grid, text="n_estimators:").grid(row=0, column=2, padx=10, pady=5, sticky="e")
        self.var_n_estimators = ctk.StringVar(value="100")
        entry = ctk.CTkEntry(config_grid, textvariable=self.var_n_estimators, width=100)
        entry.grid(row=0, column=3, padx=10, pady=5, sticky="w")
        
        ctk.CTkLabel(config_grid, text="max_depth:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.var_max_depth = ctk.StringVar(value="10")
        entry = ctk.CTkEntry(config_grid, textvariable=self.var_max_depth, width=100)
        entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        ctk.CTkLabel(config_grid, text="learning_rate:").grid(row=1, column=2, padx=10, pady=5, sticky="e")
        self.var_learning_rate = ctk.StringVar(value="0.1")
        entry = ctk.CTkEntry(config_grid, textvariable=self.var_learning_rate, width=100)
        entry.grid(row=1, column=3, padx=10, pady=5, sticky="w")
        
        ctk.CTkLabel(config_grid, text="test_size (%):").grid(row=2, column=0, padx=10, pady=5, sticky="e")
        self.var_test_size = ctk.StringVar(value="20")
        entry = ctk.CTkEntry(config_grid, textvariable=self.var_test_size, width=100)
        entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        
        self.var_random_state = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            config_grid,
            text="Random State (reprodutibilidade)",
            variable=self.var_random_state
        ).grid(row=2, column=2, padx=10, pady=5, sticky="w")
        
    def target_feature_frame(self):
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(fill="x", pady=5)
        
        title = ctk.CTkLabel(
            frame,
            text="🎯 SELEÇÃO DE TARGET E FEATURES",
            font=("Cascadia Code", 16, "bold")
        )
        title.pack(anchor="w", padx=15, pady=(10, 5))
        
        target_frame = ctk.CTkFrame(frame, fg_color="transparent")
        target_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(target_frame, text="Target (coluna alvo):").pack(side="left", padx=5)
        self.combo_target = ctk.CTkOptionMenu(target_frame, values=[], width=250)
        self.combo_target.pack(side="left", padx=5)
        
        self.feature_vars = {}
        self.features_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.features_frame.pack(fill="x", padx=15, pady=(0, 10))
        
    def update_preview(self):
        if self.df is None:
            return
            
        columns = list(self.df.columns)
        self.combo_target.configure(values=columns)
        
        if "target_risk_level" in columns:
            self.combo_target.set("target_risk_level")
            
        for widget in self.features_frame.winfo_children():
            widget.destroy()
            
        self.feature_vars.clear()
        
        ctk.CTkLabel(self.features_frame, text="Features:").pack(anchor="w", pady=5)
        
        row = 0
        col = 0
        for col_name in columns[:15]:
            if col_name != self.combo_target.get():
                var = ctk.BooleanVar(value=True)
                cb = ctk.CTkCheckBox(
                    self.features_frame,
                    text=col_name[:20],
                    variable=var,
                    width=150
                )
                cb.grid(row=row, column=col, padx=5, pady=2, sticky="w")
                self.feature_vars[col_name] = var
                
                col += 1
                if col >= 4:
                    col = 0
                    row += 1
                    
    def preview_frame(self):
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(fill="both", expand=True, pady=5)
        
        title = ctk.CTkLabel(
            frame,
            text="📊 PREVIEW DOS DADOS",
            font=("Cascadia Code", 16, "bold")
        )
        title.pack(anchor="w", padx=15, pady=(10, 5))
        
        toolbar = ctk.CTkFrame(frame, fg_color="transparent")
        toolbar.pack(fill="x", padx=15, pady=5)
        
        self.lbl_preview_count = ctk.CTkLabel(toolbar, text="Nenhum dado carregado", text_color="gray")
        self.lbl_preview_count.pack(side="left")
        
        ctk.CTkButton(
            toolbar,
            text="📥 Exportar CSV",
            command=self.export_preview,
            width=120
        ).pack(side="right", padx=5)
        
        ctk.CTkButton(
            toolbar,
            text="📈 Estatísticas",
            command=self.show_statistics,
            width=120
        ).pack(side="right", padx=5)
        
        self.preview_tree_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.preview_tree_frame.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        
        columns = ["#", "Coluna", "Tipo", "Nulos", "Únicos", "Exemplo"]
        self.preview_tree = ttk.Treeview(self.preview_tree_frame, columns=columns, show="headings", height=8)
        
        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=100)
            
        scrollbar = ttk.Scrollbar(self.preview_tree_frame, orient="vertical", command=self.preview_tree.yview)
        self.preview_tree.configure(yscrollcommand=scrollbar.set)
        
        self.preview_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def export_preview(self):
        if self.df is None:
            messagebox.showwarning("Aviso", "Nenhum dado para exportar")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Salvar CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            self.df.to_csv(filename, index=False)
            messagebox.showinfo("Sucesso", f"Dados exportados para {filename}")
            
    def show_statistics(self):
        if self.df is None:
            messagebox.showwarning("Aviso", "Nenhum dado carregado")
            return
            
        stats_window = ctk.CTkToplevel(self.root)
        stats_window.title("Estatísticas dos Dados")
        stats_window.geometry("600x500")
        
        scroll = ctk.CTkScrollableFrame(stats_window)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            scroll,
            text="📊 ESTATÍSTICAS DO DATASET",
            font=("Cascadia Code", 18, "bold")
        ).pack(pady=10)
        
        ctk.CTkLabel(
            scroll,
            text=f"Total de Registros: {len(self.df):,}",
            font=("Cascadia Code", 14)
        ).pack(pady=5)
        
        ctk.CTkLabel(
            scroll,
            text=f"Total de Colunas: {len(self.df.columns)}",
            font=("Cascadia Code", 14)
        ).pack(pady=5)
        
        ctk.CTkLabel(
            scroll,
            text="Distribuição do Target:",
            font=("Cascadia Code", 14, "bold")
        ).pack(pady=(20, 5))
        
        target_col = self.combo_target.get()
        if target_col and target_col in self.df.columns:
            for label, count in self.df[target_col].value_counts().items():
                ctk.CTkLabel(
                    scroll,
                    text=f"  {label}: {count:,} ({count/len(self.df)*100:.1f}%)",
                    font=("Cascadia Code", 12)
                ).pack(pady=2)
                
    def training_frame(self):
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(fill="x", pady=5)
        
        title = ctk.CTkLabel(
            frame,
            text="🏋️ TREINAMENTO",
            font=("Cascadia Code", 16, "bold")
        )
        title.pack(anchor="w", padx=15, pady=(10, 5))
        
        self.progress_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.progress_frame.pack(fill="x", padx=15, pady=10)
        
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, height=20)
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.set(0)
        
        self.lbl_progress = ctk.CTkLabel(
            self.progress_frame,
            text="Aguardando início do treinamento...",
            font=("Cascadia Code", 12)
        )
        self.lbl_progress.pack(pady=5)
        
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        self.btn_train = ctk.CTkButton(
            btn_frame,
            text="🚀 INICIAR TREINAMENTO",
            command=self.start_training,
            font=("Cascadia Code", 14, "bold"),
            height=40,
            fg_color=self.colors["success"],
            hover_color="#00AA55"
        )
        self.btn_train.pack(side="left", padx=5)
        
        self.btn_stop = ctk.CTkButton(
            btn_frame,
            text="⏹️ PARAR",
            command=self.stop_training_func,
            state="disabled",
            height=40,
            fg_color=self.colors["error"],
            hover_color="#CC3333"
        )
        self.btn_stop.pack(side="left", padx=5)
        
    def start_training(self):
        if self.df is None:
            messagebox.showwarning("Aviso", "Carregue os dados primeiro!")
            return
            
        target_col = self.combo_target.get()
        if not target_col:
            messagebox.showwarning("Aviso", "Selecione a coluna target!")
            return
            
        selected_features = [col for col, var in self.feature_vars.items() if var.get()]
        if not selected_features:
            messagebox.showwarning("Aviso", "Selecione pelo menos uma feature!")
            return
            
        self.btn_train.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.stop_training = False
        
        self.training_thread = threading.Thread(target=self.train_model, args=(target_col, selected_features))
        self.training_thread.start()
        
    def train_model(self, target_col, features):
        try:
            self.update_progress(0.1, "Preparando dados...")
            
            X = self.df[features].copy()
            y = self.df[target_col].copy()
            
            X = X.fillna(0)
            
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
            
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.impute import SimpleImputer
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import train_test_split
            
            numerical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_transformer, numerical_cols),
                    ("cat", categorical_transformer, categorical_cols)
                ]
            )
            
            self.update_progress(0.2, "Dividindo dados...")
            
            test_size = float(self.var_test_size.get()) / 100
            random_state = 42 if self.var_random_state.get() else None
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            self.update_progress(0.3, f"Treinando modelo... (Treino: {len(X_train):,}, Teste: {len(X_test):,})")
            
            model_type = self.var_model.get()
            
            if model_type == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                n_estimators = int(self.var_n_estimators.get())
                max_depth = int(self.var_max_depth.get()) if self.var_max_depth.get() else None
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                from sklearn.ensemble import GradientBoostingClassifier
                learning_rate = float(self.var_learning_rate.get())
                max_depth = int(self.var_max_depth.get()) if self.var_max_depth.get() else None
                model = GradientBoostingClassifier(
                    n_estimators=int(self.var_n_estimators.get()),
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=42
                )
            
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])
            
            pipeline.fit(X_train, y_train)
            
            self.update_progress(0.7, "Avaliando modelo...")
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            
            y_pred = pipeline.predict(X_test)
            
            results = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "classes": list(pipeline.classes_)
            }
            
            self.update_progress(0.85, "Salvando modelo...")
            
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ML", "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "estoque_model.pkl")
            joblib.dump(pipeline, model_path)
            
            self.model = pipeline
            self.results = results
            
            self.update_progress(1.0, f"Treinamento concluído! Modelo salvo em: {model_path}")
            
            self.root.after(0, self.show_results)
            
        except Exception as e:
            self.update_progress(0, f"Erro: {str(e)}")
            messagebox.showerror("Erro", f"Erro durante treinamento: {str(e)}")
            
        finally:
            self.root.after(0, lambda: self.btn_train.configure(state="normal"))
            self.root.after(0, lambda: self.btn_stop.configure(state="disabled"))
            
    def update_progress(self, value, text):
        self.root.after(0, lambda: self.progress_bar.set(value))
        self.root.after(0, lambda: self.lbl_progress.configure(text=text))
        
    def show_results(self):
        if self.results is None:
            return
            
        self.display_metrics(self.results)
        messagebox.showinfo("Sucesso", "Treinamento concluído com sucesso!")
        
    def stop_training_func(self):
        self.stop_training = True
        self.update_progress(0, "Treinamento interrompido!")
        
    def metrics_frame(self):
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(fill="both", expand=True, pady=5)
        
        title = ctk.CTkLabel(
            frame,
            text="📈 MÉTRICAS E RESULTADOS",
            font=("Cascadia Code", 16, "bold")
        )
        title.pack(anchor="w", padx=15, pady=(10, 5))
        
        metrics_grid = ctk.CTkFrame(frame, fg_color="transparent")
        metrics_grid.pack(fill="both", expand=True, padx=15, pady=10)
        
        self.metrics_labels = {}
        
        metric_names = ["accuracy", "precision", "recall", "f1"]
        metric_titles = ["Accuracy", "Precision", "Recall", "F1-Score"]
        
        for i, (name, title) in enumerate(zip(metric_names, metric_titles)):
            card = ctk.CTkFrame(metrics_grid)
            card.grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
            
            ctk.CTkLabel(
                card,
                text=title,
                font=("Cascadia Code", 12),
                text_color="gray"
            ).pack(pady=(10, 5))
            
            lbl_value = ctk.CTkLabel(
                card,
                text="--",
                font=("Cascadia Code", 24, "bold"),
                text_color=self.colors["accent"]
            )
            lbl_value.pack(pady=(0, 10))
            
            self.metrics_labels[name] = lbl_value
            
        self.confusion_frame = ctk.CTkFrame(metrics_grid)
        self.confusion_frame.grid(row=0, column=4, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(
            self.confusion_frame,
            text="Matriz de Confusão",
            font=("Cascadia Code", 12),
            text_color="gray"
        ).pack(pady=(10, 5))
        
        self.lbl_confusion = ctk.CTkLabel(
            self.confusion_frame,
            text="Aguardando\ntreinamento...",
            font=("Cascadia Code", 10)
        )
        self.lbl_confusion.pack(pady=20)
        
    def display_metrics(self, results):
        for name, label in self.metrics_labels.items():
            value = results.get(name, 0)
            label.configure(text=f"{value*100:.1f}%")
            
        if "confusion_matrix" in results:
            cm = results["confusion_matrix"]
            classes = results.get("classes", [])
            cm_text = "Matriz de Confusão:\n\n"
            if len(classes) > 0:
                cm_text += "   " + "  ".join([f"{c[:5]:>6}" for c in classes]) + "\n"
                for i, row in enumerate(cm):
                    cm_text += f"{classes[i][:5]:>5} " + "  ".join([f"{v:>6}" for v in row]) + "\n"
            else:
                for row in cm:
                    cm_text += " ".join([f"{v:>4}" for v in row]) + "\n"
                    
            self.lbl_confusion.configure(text=cm_text)
            
    def action_buttons(self):
        frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        frame.pack(fill="x", pady=20)
        
        ctk.CTkButton(
            frame,
            text="💾 SALVAR MODELO",
            command=self.save_model,
            width=180,
            height=40,
            font=("Cascadia Code", 12, "bold")
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            frame,
            text="📂 CARREGAR MODELO",
            command=self.load_model,
            width=180,
            height=40,
            font=("Cascadia Code", 12, "bold")
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            frame,
            text="🔄 RE-TREINAR",
            command=self.retrain,
            width=180,
            height=40,
            font=("Cascadia Code", 12, "bold"),
            fg_color=self.colors["warning"],
            hover_color="#CC9900"
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            frame,
            text="📊 VISUALIZAR FEATURES",
            command=self.show_feature_importance,
            width=180,
            height=40,
            font=("Cascadia Code", 12, "bold")
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            frame,
            text="❌ SAIR",
            command=self.root.destroy,
            width=120,
            height=40,
            font=("Cascadia Code", 12, "bold"),
            fg_color=self.colors["error"],
            hover_color="#CC3333"
        ).pack(side="right", padx=10)
        
    def save_model(self):
        if self.model is None:
            messagebox.showwarning("Aviso", "Nenhum modelo treinado para salvar")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Salvar Modelo",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filename:
            joblib.dump(self.model, filename)
            messagebox.showinfo("Sucesso", f"Modelo salvo em: {filename}")
            
    def load_model(self):
        filename = filedialog.askopenfilename(
            title="Carregar Modelo",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.model = joblib.load(filename)
                messagebox.showinfo("Sucesso", f"Modelo carregado de: {filename}")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao carregar modelo: {str(e)}")
                
    def retrain(self):
        if self.df is None:
            messagebox.showwarning("Aviso", "Carregue os dados primeiro!")
            return
        self.start_training()
        
    def show_feature_importance(self):
        if self.model is None:
            messagebox.showwarning("Aviso", "Treine um modelo primeiro!")
            return
            
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            importances = self.model.feature_importances_
            
            feature_window = ctk.CTkToplevel(self.root)
            feature_window.title("Importância das Features")
            feature_window.geometry("500x400")
            
            scroll = ctk.CTkScrollableFrame(feature_window)
            scroll.pack(fill="both", expand=True, padx=10, pady=10)
            
            ctk.CTkLabel(
                scroll,
                text="📊 IMPORTÂNCIA DAS FEATURES",
                font=("Cascadia Code", 16, "bold")
            ).pack(pady=10)
            
            sorted_idx = np.argsort(importances)[::-1]
            
            for i in sorted_idx[:15]:
                bar_length = importances[i] * 100
                frame = ctk.CTkFrame(scroll, fg_color="transparent")
                frame.pack(fill="x", pady=2)
                
                ctk.CTkLabel(
                    frame,
                    text=f"Feature {i}",
                    width=150,
                    anchor="w"
                ).pack(side="left")
                
                bar = ctk.CTkProgressBar(frame, height=10)
                bar.pack(side="left", fill="x", expand=True, padx=5)
                bar.set(importances[i])
                
                ctk.CTkLabel(
                    frame,
                    text=f"{importances[i]*100:.1f}%",
                    width=60
                ).pack(side="right")
        else:
            messagebox.showinfo("Info", "Este modelo não suporta importância de features")
            
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = TrainGUI()
    app.run()
