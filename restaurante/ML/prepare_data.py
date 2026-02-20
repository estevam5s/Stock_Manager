import json
import pandas as pd
import numpy as np
import os


def load_real_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    vendas = data.get('vendas', [])
    
    records = []
    for v in vendas:
        date_str = v.get('date', '')
        value = v.get('value', 0)
        
        parts = date_str.split('/')
        if len(parts) == 3:
            day = int(parts[0])
            month = int(parts[1])
            year = int(parts[2])
            
            records.append({
                'dia': day,
                'mes': month,
                'ano': year,
                'venda_diaria': value
            })
    
    df = pd.DataFrame(records)
    return df


def expand_data_for_ml(df):
    expanded = []
    
    for idx, row in df.iterrows():
        dia = int(row['dia'])
        mes = int(row['mes'])
        ano = int(row['ano'])
        
        try:
            dia_semana = pd.Timestamp(f"{ano}-{mes}-{dia}").weekday()
        except:
            dia_semana = 0
        
        base_features = {
            'dia': dia,
            'mes': mes,
            'ano': ano,
            'dia_semana': dia_semana,
            'e_fim_semana': 1 if dia_semana in [5, 6] else 0,
            'e_feriado': 1 if mes in [12, 1] and dia >= 20 else 0,
            'e_inicio_mes': 1 if dia <= 7 else 0,
            'e_fim_mes': 1 if dia >= 23 else 0,
            'trimestre': (mes - 1) // 3 + 1,
            'venda_diaria': row['venda_diaria']
        }
        
        expanded.append(base_features)
    
    return pd.DataFrame(expanded)


def generate_enhanced_data(csv_output_path, n_enhance=50000):
    json_path = 'restaurante/relatorio-vendas-pedacinho-do-ceu/dados_vendas.json'
    
    if os.path.exists(json_path):
        print("Carregando dados reais do JSON...")
        df_real = load_real_data(json_path)
        print(f"Dados reais carregados: {len(df_real)} registros")
        
        df_expanded = expand_data_for_ml(df_real)
        print(f"Dados expandidos: {len(df_expanded)} registros")
        
        base_sales = df_expanded['venda_diaria'].mean()
    else:
        print("Dados reais não encontrados, gerando dados sintéticos...")
        base_sales = 5000
        df_expanded = pd.DataFrame()
    
    np.random.seed(42)
    
    categories = ["Lanches", "Pizzas", "Massas", "Saladas", "Sobremesas",
                 "Bebidas", "Carnes", "Refrigerantes", "Sucos", "Café"]
    
    expanded_data = []
    
    for i in range(n_enhance):
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)
        year = np.random.randint(2020, 2025)
        
        weekday = pd.Timestamp(f"{year}-{month}-{day}").weekday()
        is_weekend = 1 if weekday in [5, 6] else 0
        is_holiday = 1 if month in [12, 1] and day >= 20 else 0
        is_friday = 1 if weekday == 4 else 0
        
        base = base_sales
        
        if is_weekend:
            base *= np.random.uniform(1.2, 1.8)
        if is_holiday:
            base *= np.random.uniform(1.3, 2.0)
        if is_friday:
            base *= np.random.uniform(1.1, 1.5)
            
        if month in [12, 1]:
            base *= np.random.uniform(1.2, 1.6)
        elif month in [6, 7]:
            base *= np.random.uniform(0.8, 1.0)
            
        if day <= 7:
            base *= np.random.uniform(0.7, 0.9)
        elif day >= 23:
            base *= np.random.uniform(0.9, 1.1)
            
        sale = base * np.random.uniform(0.5, 1.5)
        
        custo = sale * np.random.uniform(0.25, 0.40)
        lucro = sale - custo
        margem = (lucro / sale * 100) if sale > 0 else 0
        
        categoria = np.random.choice(categories)
        
        expanded_data.append({
            'dia': day,
            'mes': month,
            'ano': year,
            'dia_semana': weekday,
            'e_fim_semana': is_weekend,
            'e_feriado': is_holiday,
            'e_sexta': is_friday,
            'trimestre': (month - 1) // 3 + 1,
            'e_inicio_mes': 1 if day <= 7 else 0,
            'e_fim_mes': 1 if day >= 23 else 0,
            'venda_diaria': round(sale, 2),
            'custo_ingredientes': round(custo, 2),
            'lucro_bruto': round(lucro, 2),
            'margem_lucro': round(margem, 2),
            'categoria': categoria,
            'num_pedidos': int(np.random.randint(20, 150)),
            'ticket_medio': round(sale / np.random.randint(20, 150), 2),
            'temperatura': round(np.random.uniform(18, 38), 1),
            'promocao': np.random.choice([0, 1], p=[0.85, 0.15]),
            'desconto': round(sale * np.random.uniform(0, 0.1), 2) if np.random.random() > 0.7 else 0
        })
    
    df = pd.DataFrame(expanded_data)
    
    df['classificacao_venda'] = df['venda_diaria'].apply(
        lambda x: 'alta' if x > 8000 else 'media' if x > 4000 else 'baixa'
    )
    
    df['classificacao_lucro'] = df['margem_lucro'].apply(
        lambda x: 'excelente' if x > 50 else 'boa' if x > 35 else 'regular' if x > 20 else 'ruim'
    )
    
    df['classificacao_risco'] = df.apply(
        lambda r: 'critico' if r['venda_diaria'] < 2000 or r['margem_lucro'] < 15
        else 'alto' if r['venda_diaria'] < 4000 or r['margem_lucro'] < 25
        else 'medio' if r['venda_diaria'] < 6000 or r['margem_lucro'] < 40
        else 'baixo', axis=1
    )
    
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df.to_csv(csv_output_path, index=False)
    
    print(f"\n{'='*60}")
    print("     DADOS FINANCEIROS DO RESTAURANTE GERADOS")
    print(f"{'='*60}")
    print(f"Arquivo: {csv_output_path}")
    print(f"Registros: {len(df):,}")
    print(f"Colunas: {len(df.columns)}")
    print(f"\nEstatísticas Financeiras:")
    print(f"  - Venda média: R$ {df['venda_diaria'].mean():.2f}")
    print(f"  - Venda total: R$ {df['venda_diaria'].sum():,.2f}")
    print(f"  - Ticket médio: R$ {df['ticket_medio'].mean():.2f}")
    print(f"  - Margem média: {df['margem_lucro'].mean():.1f}%")
    print(f"  - Lucro total: R$ {df['lucro_bruto'].sum():,.2f}")
    print(f"\nClassificação de Vendas:")
    print(df['classificacao_venda'].value_counts())
    print(f"\nClassificação de Lucro:")
    print(df['classificacao_lucro'].value_counts())
    print(f"\nClassificação de Risco:")
    print(df['classificacao_risco'].value_counts())
    print(f"{'='*60}\n")
    
    return df


if __name__ == "__main__":
    output_path = "restaurante/ML/data/dados_financeiros.csv"
    generate_enhanced_data(output_path, 50000)
