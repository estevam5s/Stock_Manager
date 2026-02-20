import sys
import os
import pandas as pd
import numpy as np
import joblib
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import customtkinter as ctk
from tkinter import ttk, filedialog, messagebox
from tkinter import *

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")


class RestaurantTrainGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("🏪 Análise Financeira - Restaurante | Treinamento ML")
        self.root.geometry("1300x850")
        self.root.minsize(1100, 750)
        
        self.df = None
        self.model = None
        self.results = None
        
        self.colors = {
            "primary": "#1a1a2e",
            "secondary": "#16213e",
            "accent": "#00d4aa",
            "success": "#00cc66",
            "warning": "#ffbb00",
            "error": "#ff4444",
            "info": "#0099ff"
        }
        
        self.setup_layout()
        
    def setup_layout(self):
        self.main_frame = ctk.CTkScrollableFrame(self.root, fg_color=self.colors["primary"])
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.header()
        self.data_source_frame()
        self.analysis_type_frame()
        self.model_config_frame()
        self.preview_frame()
        self.training_frame()
        self.metrics_frame()
        self.action_buttons()
        
    def header(self):
        header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 20))
        
        title = ctk.CTkLabel(
            header_frame,
            text="🏪 ANÁLISE FINANCEIRA - RESTAURANTE | TREINAMENTO ML",
            font=("Cascadia Code", 22, "bold")
        )
        title.pack(side="left")
        
        version = ctk.CTkLabel(
            header_frame,
            text="v1.0 | Professional Edition",
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
        
        ctk.CTkCheckBox(
            options_frame,
            text="📤 Upload CSV",
            variable=self.var_upload,
            command=self.load_data_options
        ).pack(side="left", padx=15)
        
        ctk.CTkCheckBox(
            options_frame,
            text="🎲 Gerar Dados (100K)",
            variable=self.var_generate,
            command=self.load_data_options
        ).pack(side="left", padx=15)
        
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
                width=220,
                fg_color=self.colors["info"]
            ).pack(side="left", padx=5)
            
        if self.var_generate.get():
            gen_frame = ctk.CTkFrame(self.data_options_frame, fg_color="transparent")
            gen_frame.pack(fill="x", pady=5)
            
            ctk.CTkButton(
                gen_frame,
                text="🎲 Gerar 100K Registros",
                command=self.generate_data,
                width=220,
                fg_color=self.colors["warning"]
            ).pack(side="left", padx=5)
            
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
                messagebox.showinfo("Sucesso", f"Arquivo carregado com {len(self.df):,} registros!")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao carregar: {str(e)}")
                
    def generate_data(self):
        try:
            from data_generator import generate_financial_data
            self.df = generate_financial_data(100000)
            self.update_data_info(f"🎲 Dados gerados | Registros: {len(self.df):,} | Colunas: {len(self.df.columns)}")
            self.update_preview()
            messagebox.showinfo("Sucesso", f"Dados gerados com {len(self.df):,} registros!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar dados: {str(e)}")
            
    def update_data_info(self, text):
        self.lbl_data_info.configure(text=text)
        
    def analysis_type_frame(self):
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(fill="x", pady=5)
        
        title = ctk.CTkLabel(
            frame,
            text="🔍 TIPO DE ANÁLISE",
            font=("Cascadia Code", 16, "bold")
        )
        title.pack(anchor="w", padx=15, pady=(10, 5))
        
        analysis_frame = ctk.CTkFrame(frame, fg_color="transparent")
        analysis_frame.pack(fill="x", padx=15, pady=10)
        
        self.analysis_vars = {}
        
        analyses = [
            ("Previsão de Vendas", "previsao_vendas", "📈"),
            ("Análise de Lucratividade", "lucratividade", "💰"),
            ("Previsão de Custos", "previsao_custos", "💵"),
            ("Análise de Cliente", "analise_cliente", "👥"),
            ("Detecção de Anomalias", "deteccao_anomalias", "⚠️")
        ]
        
        for text, key, icon in analyses:
            var = ctk.BooleanVar(value=(key == "previsao_vendas"))
            cb = ctk.CTkCheckBox(
                analysis_frame,
                text=f"{icon} {text}",
                variable=var,
                command=self.on_analysis_change
            )
            cb.pack(side="left", padx=10)
            self.analysis_vars[key] = var
            
    def on_analysis_change(self):
        selected = [k for k, v in self.analysis_vars.items() if v.get()]
        if selected:
            target_map = {
                "previsao_vendas": "venda_diaria",
                "lucratividade": "lucratividade",
                "previsao_custos": "custo_ingredientes",
                "analise_cliente": "classificacao_venda",
                "deteccao_anomalias": "classificacao_venda"
            }
            target = target_map.get(selected[0], "venda_diaria")
            if target in self.df.columns and self.combo_target:
                self.combo_target.set(target)
                
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
        
        ctk.CTkLabel(config_grid, text="Modelo:").grid(row=0, column=0, padx=10, pady=8, sticky="e")
        self.var_model = ctk.StringVar(value="random_forest")
        combo = ctk.CTkOptionMenu(
            config_grid,
            values=["random_forest", "gradient_boosting"],
            variable=self.var_model,
            width=200
        )
        combo.grid(row=0, column=1, padx=10, pady=8, sticky="w")
        
        ctk.CTkLabel(config_grid, text="Tarefa:").grid(row=0, column=2, padx=10, pady=8, sticky="e")
        self.var_task = ctk.StringVar(value="classification")
        combo = ctk.CTkOptionMenu(
            config_grid,
            values=["classification", "regression"],
            variable=self.var_task,
            width=180
        )
        combo.grid(row=0, column=3, padx=10, pady=8, sticky="w")
        
        ctk.CTkLabel(config_grid, text="n_estimators:").grid(row=1, column=0, padx=10, pady=8, sticky="e")
        self.var_n_estimators = ctk.StringVar(value="100")
        entry = ctk.CTkEntry(config_grid, textvariable=self.var_n_estimators, width=100)
        entry.grid(row=1, column=1, padx=10, pady=8, sticky="w")
        
        ctk.CTkLabel(config_grid, text="max_depth:").grid(row=1, column=2, padx=10, pady=8, sticky="e")
        self.var_max_depth = ctk.StringVar(value="15")
        entry = ctk.CTkEntry(config_grid, textvariable=self.var_max_depth, width=100)
        entry.grid(row=1, column=3, padx=10, pady=8, sticky="w")
        
        ctk.CTkLabel(config_grid, text="test_size (%):").grid(row=2, column=0, padx=10, pady=8, sticky="e")
        self.var_test_size = ctk.StringVar(value="20")
        entry = ctk.CTkEntry(config_grid, textvariable=self.var_test_size, width=100)
        entry.grid(row=2, column=1, padx=10, pady=8, sticky="w")
        
    def preview_frame(self):
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(fill="both", expand=True, pady=5)
        
        title = ctk.CTkLabel(
            frame,
            text="📊 PREVIEW DOS DADOS",
            font=("Cascadia Code", 16, "bold")
        )
        title.pack(anchor="w", padx=15, pady=(10, 5))
        
        target_frame = ctk.CTkFrame(frame, fg_color="transparent")
        target_frame.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(target_frame, text="Target (alvo):").pack(side="left", padx=5)
        self.combo_target = ctk.CTkOptionMenu(target_frame, values=[], width=250)
        self.combo_target.pack(side="left", padx=5)
        
        toolbar = ctk.CTkFrame(frame, fg_color="transparent")
        toolbar.pack(fill="x", padx=15, pady=5)
        
        self.lbl_preview_count = ctk.CTkLabel(toolbar, text="Nenhum dado carregado", text_color="gray")
        self.lbl_preview_count.pack(side="left")
        
        ctk.CTkButton(
            toolbar,
            text="📥 Exportar",
            command=self.export_preview,
            width=100
        ).pack(side="right", padx=5)
        
        ctk.CTkButton(
            toolbar,
            text="📈 Stats",
            command=self.show_statistics,
            width=100
        ).pack(side="right", padx=5)
        
        self.preview_tree_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.preview_tree_frame.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        
        columns = ["#", "Coluna", "Tipo", "Nulos", "Exemplo"]
        self.preview_tree = ttk.Treeview(self.preview_tree_frame, columns=columns, show="headings", height=8)
        
        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=120)
            
        scrollbar = ttk.Scrollbar(self.preview_tree_frame, orient="vertical", command=self.preview_tree.yview)
        self.preview_tree.configure(yscrollcommand=scrollbar.set)
        
        self.preview_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def update_preview(self):
        if self.df is None:
            return
            
        columns = list(self.df.columns)
        self.combo_target.configure(values=columns)
        
        default_targets = ["venda_diaria", "lucratividade", "custo_ingredientes", "classificacao_venda", "margem_lucro"]
        for target in default_targets:
            if target in columns:
                self.combo_target.set(target)
                break
                
        self.lbl_preview_count.configure(text=f"Showing 100 of {len(self.df):,} records")
        
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
            
        for i, col in enumerate(columns[:20]):
            nulls = self.df[col].isnull().sum()
            example = str(self.df[col].iloc[0])[:30] if len(self.df) > 0 else ""
            dtype = str(self.df[col].dtype)
            self.preview_tree.insert("", "end", values=(i+1, col, dtype, nulls, example))
            
    def export_preview(self):
        if self.df is None:
            return
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if filename:
            self.df.to_csv(filename, index=False)
            messagebox.showinfo("Sucesso", "Dados exportados!")
            
    def show_statistics(self):
        if self.df is None:
            return
            
        stats_window = ctk.CTkToplevel(self.root)
        stats_window.title("Estatísticas")
        stats_window.geometry("600x500")
        
        scroll = ctk.CTkScrollableFrame(stats_window)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(scroll, text="📊 ESTATÍSTICAS FINANCEIRAS", font=("Cascadia Code", 18, "bold")).pack(pady=10)
        
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if "venda_diaria" in self.df.columns:
            ctk.CTkLabel(scroll, text=f"Venda Média: R$ {self.df['venda_diaria'].mean():.2f}").pack(pady=2)
            ctk.CTkLabel(scroll, text=f"Venda Total: R$ {self.df['venda_diaria'].sum():,.2f}").pack(pady=2)
            
        if "ticket_medio" in self.df.columns:
            ctk.CTkLabel(scroll, text=f"Ticket Médio: R$ {self.df['ticket_medio'].mean():.2f}").pack(pady=2)
            
        if "margem_lucro" in self.df.columns:
            ctk.CTkLabel(scroll, text=f"Margem Média: {self.df['margem_lucro'].mean():.1f}%").pack(pady=2)
            
        if "lucratividade" in self.df.columns:
            ctk.CTkLabel(scroll, text="\nDistribuição de Lucratividade:", font=("Cascadia Code", 14, "bold")).pack(pady=5)
            for label, count in self.df['lucratividade'].value_counts().items():
                ctk.CTkLabel(scroll, text=f"  {label}: {count:,} ({count/len(self.df)*100:.1f}%)").pack(pady=1)
                
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
            text="Aguardando início...",
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
            height=45,
            fg_color=self.colors["success"],
            hover_color="#00aa55"
        )
        self.btn_train.pack(side="left", padx=5, fill="x", expand=True)
        
    def start_training(self):
        if self.df is None:
            messagebox.showwarning("Aviso", "Carregue os dados primeiro!")
            return
            
        target_col = self.combo_target.get()
        if not target_col:
            messagebox.showwarning("Aviso", "Selecione o target!")
            return
            
        self.btn_train.configure(state="disabled")
        
        self.training_thread = threading.Thread(target=self.train_model, args=(target_col,))
        self.training_thread.start()
        
    def train_model(self, target_col):
        try:
            self.update_progress(0.1, "Preparando dados...")
            
            feature_cols = [col for col in self.df.columns if col != target_col]
            
            X = self.df[feature_cols].copy()
            y = self.df[target_col].copy()
            
            X = X.fillna(0)
            
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
            
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.impute import SimpleImputer
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import train_test_split
            
            numerical_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            
            categorical_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            
            preprocessor = ColumnTransformer([
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols)
            ])
            
            self.update_progress(0.2, "Dividindo dados...")
            
            test_size = float(self.var_test_size.get()) / 100
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            self.update_progress(0.3, f"Treinando... (Treino: {len(X_train):,})")
            
            model_type = self.var_model.get()
            task = self.var_task.get()
            
            if task == "classification":
                if model_type == "random_forest":
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(
                        n_estimators=int(self.var_n_estimators.get()),
                        max_depth=int(self.var_max_depth.get()) if self.var_max_depth.get() else None,
                        random_state=42, n_jobs=-1
                    )
                else:
                    from sklearn.ensemble import GradientBoostingClassifier
                    model = GradientBoostingClassifier(
                        n_estimators=int(self.var_n_estimators.get()),
                        max_depth=int(self.var_max_depth.get()) if self.var_max_depth.get() else None,
                        random_state=42
                    )
            else:
                if model_type == "random_forest":
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(
                        n_estimators=int(self.var_n_estimators.get()),
                        max_depth=int(self.var_max_depth.get()) if self.var_max_depth.get() else None,
                        random_state=42, n_jobs=-1
                    )
                else:
                    from sklearn.ensemble import GradientBoostingRegressor
                    model = GradientBoostingRegressor(
                        n_estimators=int(self.var_n_estimators.get()),
                        max_depth=int(self.var_max_depth.get()) if self.var_max_depth.get() else None,
                        random_state=42
                    )
            
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            
            self.update_progress(0.7, "Avaliando modelo...")
            
            y_pred = pipeline.predict(X_test)
            
            if task == "classification":
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                self.results = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
                }
            else:
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                self.results = {
                    "mae": mean_absolute_error(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "r2": r2_score(y_test, y_pred)
                }
            
            self.update_progress(0.85, "Salvando modelo...")
            
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "modelo_restaurante.pkl")
            joblib.dump(pipeline, model_path)
            
            self.model = pipeline
            
            self.update_progress(1.0, f"✅ Concluído! Modelo salvo!")
            
            self.root.after(0, self.display_results)
            self.root.after(0, lambda: messagebox.showinfo("Sucesso", "Treinamento concluído!"))
            
        except Exception as e:
            self.update_progress(0, f"Erro: {str(e)}")
            messagebox.showerror("Erro", str(e))
            
        finally:
            self.root.after(0, lambda: self.btn_train.configure(state="normal"))
            
    def update_progress(self, value, text):
        self.root.after(0, lambda: self.progress_bar.set(value))
        self.root.after(0, lambda: self.lbl_progress.configure(text=text))
        
    def display_results(self):
        if not self.results:
            return
            
        task = self.var_task.get()
        
        if task == "classification":
            for name, label in self.metrics_labels.items():
                if name in self.results:
                    label.configure(text=f"{self.results[name]*100:.1f}%")
        else:
            self.metrics_labels["accuracy"].configure(text=f"{self.results.get('r2', 0)*100:.1f}%")
            self.metrics_labels["precision"].configure(text=f"{self.results.get('mae', 0):.2f}")
            self.metrics_labels["recall"].configure(text=f"{self.results.get('rmse', 0):.2f}")
            
    def metrics_frame(self):
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(fill="both", expand=True, pady=5)
        
        title = ctk.CTkLabel(
            frame,
            text="📈 MÉTRICAS",
            font=("Cascadia Code", 16, "bold")
        )
        title.pack(anchor="w", padx=15, pady=(10, 5))
        
        metrics_grid = ctk.CTkFrame(frame, fg_color="transparent")
        metrics_grid.pack(fill="both", expand=True, padx=15, pady=10)
        
        self.metrics_labels = {}
        
        metric_config = [
            ("accuracy", "Accuracy / R²"),
            ("precision", "Precision / MAE"),
            ("recall", "Recall / RMSE"),
            ("f1", "F1-Score")
        ]
        
        for i, (name, title) in enumerate(metric_config):
            card = ctk.CTkFrame(metrics_grid)
            card.grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
            
            ctk.CTkLabel(card, text=title, font=("Cascadia Code", 11), text_color="gray").pack(pady=(8, 4))
            
            lbl = ctk.CTkLabel(card, text="--", font=("Cascadia Code", 22, "bold"), text_color=self.colors["accent"])
            lbl.pack(pady=(0, 8))
            
            self.metrics_labels[name] = lbl
            
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
            command=self.start_training,
            width=180,
            height=40,
            font=("Cascadia Code", 12, "bold"),
            fg_color=self.colors["warning"]
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            frame,
            text="❌ SAIR",
            command=self.root.destroy,
            width=120,
            height=40,
            font=("Cascadia Code", 12, "bold"),
            fg_color=self.colors["error"]
        ).pack(side="right", padx=10)
        
    def save_model(self):
        if self.model is None:
            messagebox.showwarning("Aviso", "Nenhum modelo treinado")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle", "*.pkl")])
        if filename:
            joblib.dump(self.model, filename)
            messagebox.showinfo("Sucesso", "Modelo salvo!")
            
    def load_model(self):
        filename = filedialog.askopenfilename(filetypes=[("Pickle", "*.pkl")])
        if filename:
            try:
                self.model = joblib.load(filename)
                messagebox.showinfo("Sucesso", "Modelo carregado!")
            except Exception as e:
                messagebox.showerror("Erro", str(e))
                
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = RestaurantTrainGUI()
    app.run()
