import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


class Visualizer:
    def __init__(self, output_dir="ML/models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('dark_background')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, cm, class_labels, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "confusion_matrix.png")
        
        plt.figure(figsize=(10, 8))
        
        if len(class_labels) > 0:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_labels, yticklabels=class_labels)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.title('Matriz de Confusão', fontsize=16, fontweight='bold')
        plt.ylabel('Real')
        plt.xlabel('Predito')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_feature_importance(self, feature_names, importances, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "feature_importance.png")
        
        indices = np.argsort(importances)[::-1][:15]
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(indices)))
        
        plt.barh(range(len(indices)), importances[indices], color=colors)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        
        plt.xlabel('Importância', fontsize plt.ylabel=12)
       ('Feature', fontsize=12)
        plt.title('Top 15 Features Mais Importantes', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_target_distribution(self, df, target_col, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "target_distribution.png")
        
        plt.figure(figsize=(10, 6))
        
        if target_col in df.columns:
            counts = df[target_col].value_counts()
            colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
            
            plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            plt.title(f'Distribuição de {target_col}', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_numeric_distributions(self, df, columns, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "numeric_distributions.png")
        
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(columns):
            if col in df.columns:
                axes[i].hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='white', alpha=0.7)
                axes[i].set_title(col, fontsize=10)
                axes[i].set_xlabel('')
        
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle('Distribuição das Variáveis Numéricas', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_correlation_matrix(self, df, columns, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "correlation_matrix.png")
        
        numeric_df = df[columns].select_dtypes(include=[np.number])
        
        plt.figure(figsize=(12, 10))
        
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=0.5)
        
        plt.title('Matriz de Correlação', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_training_history(self, history, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "training_history.png")
        
        plt.figure(figsize=(12, 5))
        
        if 'loss' in history:
            plt.subplot(1, 2, 1)
            plt.plot(history['loss'], label='Loss', color='red')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Val Loss', color='orange')
            plt.title('Loss durante o Treinamento')
            plt.xlabel('Época')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        if 'accuracy' in history:
            plt.subplot(1, 2, 2)
            plt.plot(history['accuracy'], label='Accuracy', color='blue')
            if 'val_accuracy' in history:
                plt.plot(history['val_accuracy'], label='Val Accuracy', color='green')
            plt.title('Accuracy durante o Treinamento')
            plt.xlabel('Época')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_dashboard(self, df, target_col=None, model=None, X_test=None, y_test=None):
        dashboard_path = os.path.join(self.output_dir, "dashboard.png")
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Dashboard de Análise ML', fontsize=20, fontweight='bold')
        
        if target_col and target_col in df.columns:
            plt.subplot(2, 3, 1)
            counts = df[target_col].value_counts()
            plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
            plt.title(f'Distribuição: {target_col}')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        if len(numeric_cols) > 0:
            plt.subplot(2, 3, 2)
            df[numeric_cols].mean().plot(kind='bar', color='steelblue')
            plt.title('Média das Variáveis Numéricas')
            plt.xticks(rotation=45, ha='right')
        
        if len(numeric_cols) > 1:
            plt.subplot(2, 3, 3)
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=plt.gca())
            plt.title('Correlações')
        
        if len(numeric_cols) >= 2:
            plt.subplot(2, 3, 4)
            plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.5, s=10)
            plt.xlabel(numeric_cols[0])
            plt.ylabel(numeric_cols[1])
            plt.title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
        
        plt.subplot(2, 3, 5)
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            missing.plot(kind='bar', color='coral')
            plt.title('Valores Faltantes')
            plt.xticks(rotation=45, ha='right')
        
        plt.subplot(2, 3, 6)
        plt.text(0.5, 0.8, f'Total Registros: {len(df):,}', fontsize=14, ha='center', transform=plt.gca().transAxes)
        plt.text(0.5, 0.6, f'Total Colunas: {len(df.columns)}', fontsize=14, ha='center', transform=plt.gca().transAxes)
        plt.text(0.5, 0.4, f'Colunas Numéricas: {len(numeric_cols)}', fontsize=14, ha='center', transform=plt.gca().transAxes)
        plt.axis('off')
        plt.title('Resumo')
        
        plt.tight_layout()
        plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return dashboard_path
