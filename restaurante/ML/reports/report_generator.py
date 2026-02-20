import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


class ReportGenerator:
    def __init__(self, output_dir="restaurante/ML/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('dark_background')
        sns.set_palette("husl")
        
    def generate_financial_report(self, df, model_metrics=None):
        report_path = os.path.join(self.output_dir, "relatorio_financeiro.html")
        
        html = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Relatório Financeiro - Restaurante</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #1a1a2e; color: #fff; }}
        h1 {{ color: #00d4aa; border-bottom: 2px solid #00d4aa; padding-bottom: 10px; }}
        h2 {{ color: #00d4aa; margin-top: 30px; }}
        .metric {{ display: inline-block; background: #16213e; padding: 20px; margin: 10px; border-radius: 10px; text-align: center; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #00d4aa; }}
        .metric-label {{ font-size: 14px; color: #888; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background: #16213e; color: #00d4aa; }}
        .critical {{ color: #ff4444; }}
        .high {{ color: #ff8800; }}
        .medium {{ color: #ffbb00; }}
        .low {{ color: #00cc66; }}
        .chart {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>📊 Relatório Financeiro - Restaurante</h1>
    <p>Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
    
    <h2>💰 Métricas Gerais</h2>
"""
        
        if 'venda_diaria' in df.columns:
            html += f"""
    <div class="metric">
        <div class="metric-value">R$ {df['venda_diaria'].sum():,.2f}</div>
        <div class="metric-label">Venda Total</div>
    </div>
    <div class="metric">
        <div class="metric-value">R$ {df['venda_diaria'].mean():,.2f}</div>
        <div class="metric-label">Média por Dia</div>
    </div>
"""
        
        if 'lucro_bruto' in df.columns:
            html += f"""
    <div class="metric">
        <div class="metric-value">R$ {df['lucro_bruto'].sum():,.2f}</div>
        <div class="metric-label">Lucro Total</div>
    </div>
    <div class="metric">
        <div class="metric-value">{df['margem_lucro'].mean():.1f}%</div>
        <div class="metric-label">Margem Média</div>
    </div>
"""
        
        if 'classificacao_risco' in df.columns:
            html += f"""
    <h2>⚠️ Análise de Risco</h2>
    <table>
        <tr><th>Nível</th><th>Quantidade</th><th>Percentual</th></tr>
"""
            for level, count in df['classificacao_risco'].value_counts().items():
                pct = count / len(df) * 100
                html += f"""
        <tr>
            <td class="{level}">{level.upper()}</td>
            <td>{count:,}</td>
            <td>{pct:.1f}%</td>
        </tr>
"""
            html += "</table>"
        
        if 'categoria' in df.columns:
            html += f"""
    <h2>🍽️ Vendas por Categoria</h2>
    <table>
        <tr><th>Categoria</th><th>Vendas Totais</th><th>Média</th></tr>
"""
            cat_stats = df.groupby('categoria')['venda_diaria'].agg(['sum', 'mean']).sort_values('sum', ascending=False)
            for cat, row in cat_stats.iterrows():
                html += f"""
        <tr><td>{cat}</td><td>R$ {row['sum']:,.2f}</td><td>R$ {row['mean']:,.2f}</td></tr>
"""
            html += "</table>"
        
        if 'mes' in df.columns:
            html += f"""
    <h2>📅 Vendas por Mês</h2>
    <table>
        <tr><th>Mês</th><th>Vendas</th><th>Lucro</th></tr>
"""
            month_stats = df.groupby('mes')['venda_diaria'].sum()
            for mes, venda in month_stats.items():
                html += f"""
        <tr><td>Mês {mes}</td><td>R$ {venda:,.2f}</td><td>-</td></tr>
"""
            html += "</table>"
        
        if model_metrics:
            html += f"""
    <h2>🤖 Métricas do Modelo ML</h2>
    <div class="metric">
        <div class="metric-value">{model_metrics.get('accuracy', 0)*100:.2f}%</div>
        <div class="metric-label">Accuracy</div>
    </div>
    <div class="metric">
        <div class="metric-value">{model_metrics.get('precision', 0)*100:.2f}%</div>
        <div class="metric-label">Precision</div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ Relatório HTML salvo: {report_path}")
        return report_path
    
    def generate_charts(self, df):
        charts = []
        
        if 'venda_diaria' in df.columns and 'mes' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            monthly = df.groupby('mes')['venda_diaria'].mean()
            monthly.plot(kind='bar', ax=ax, color='#00d4aa')
            ax.set_title('Venda Média por Mês', fontsize=16)
            ax.set_xlabel('Mês')
            ax.set_ylabel('Venda Média (R$)')
            path = os.path.join(self.output_dir, "chart_vendas_mes.png")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            charts.append(path)
            
        if 'classificacao_risco' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 8))
            df['classificacao_risco'].value_counts().plot(
                kind='pie', ax=ax, autopct='%1.1f%%',
                colors=['#00cc66', '#ffbb00', '#ff8800', '#ff4444']
            )
            ax.set_title('Distribuição de Risco', fontsize=16)
            path = os.path.join(self.output_dir, "chart_risco.png")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            charts.append(path)
            
        if 'categoria' in df.columns and 'venda_diaria' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            cat_sales = df.groupby('categoria')['venda_diaria'].sum().sort_values(ascending=True)
            cat_sales.plot(kind='barh', ax=ax, color='#00d4aa')
            ax.set_title('Vendas por Categoria', fontsize=16)
            ax.set_xlabel('Venda Total (R$)')
            path = os.path.join(self.output_dir, "chart_categorias.png")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            charts.append(path)
            
        if 'dia_semana' in df.columns and 'venda_diaria' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            days = df.groupby('dia_semana')['venda_diaria'].mean()
            days.plot(kind='bar', ax=ax, color='#00d4aa')
            ax.set_title('Venda Média por Dia da Semana', fontsize=16)
            ax.set_xlabel('Dia (0=Segunda, 6=Domingo)')
            ax.set_ylabel('Venda Média (R$)')
            path = os.path.join(self.output_dir, "chart_dias_semana.png")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            charts.append(path)
        
        print(f"✓ {len(charts)} gráficos gerados")
        return charts
    
    def generate_full_report(self, df, model_metrics=None):
        print("\n" + "="*60)
        print("     GERANDO RELATÓRIO COMPLETO")
        print("="*60 + "\n")
        
        self.generate_charts(df)
        report_path = self.generate_financial_report(df, model_metrics)
        
        print(f"\n{'='*60}")
        print("     RELATÓRIO CONCLUÍDO")
        print(f"{'='*60}")
        print(f"Arquivos salvos em: {self.output_dir}")
        
        return report_path


if __name__ == "__main__":
    import joblib
    
    data_path = "restaurante/ML/data/dados_financeiros.csv"
    model_path = "restaurante/ML/models/modelo_restaurante.pkl"
    
    print("Carregando dados...")
    df = pd.read_csv(data_path)
    
    model_data = joblib.load(model_path)
    metrics = {
        'accuracy': model_data.get('accuracy', 0),
        'precision': model_data.get('precision', 0)
    }
    
    generator = ReportGenerator()
    generator.generate_full_report(df, metrics)
