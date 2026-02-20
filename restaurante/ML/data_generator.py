import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def generate_financial_data(n_samples=100000):
    print(f"\n{'='*60}")
    print("     GERANDO DADOS FINANCEIROS DO RESTAURANTE")
    print(f"{'='*60}")
    
    np.random.seed(42)
    
    categories = [
        "Lanches", "Pizzas", "Massas", "Saladas", "Sobremesas",
        "Bebidas Alcoólicas", "Refrigerantes", "Sucos", "Café", "Lanches Naturais",
        "Frutos do Mar", "Carnes", "Vegetariano", "Kids", "Executivo"
    ]
    
    pratos = {
        "Lanches": ["Hambúrguer Artesanal", "Cheeseburger", "Duplo Burger", "Chicken Burger", "Veggie Burger"],
        "Pizzas": ["Margherita", "Calabresa", "Portuguesa", "Frango Catupiry", "Quatro Queijos"],
        "Massas": ["Spaghetti Carbonara", "Lasanha", "Ravioli", "Fettuccine", "Penne"],
        "Saladas": ["Caesar", "Grega", "Quinoa", "Tropical", "Verde"],
        "Sobremesas": ["Pudim", "Tiramisu", "Mousse", "Sorvete", "Brownie"],
        "Bebidas Alcoólicas": ["Cerveja", "Vinho", "Caipirinha", "Coquetel", "Whisky"],
        "Refrigerantes": ["Coca-Cola", "Guaraná", "Sprite", "Fanta", "Água"],
        "Sucos": ["Laranja", "Limão", "Abacaxi", "Verde", "Misto"],
        "Café": ["Expresso", "Cappuccino", "Latte", "Mocha", "Irish Coffee"],
        "Lanches Naturais": ["Sanduíche Natural", "Wrap", "Torrada", "Panini", "Baguete"],
        "Frutos do Mar": ["Peixe", "Camarão", "Lagosta", "Salmão", "Mariscos"],
        "Carnes": ["Filé Mignon", "Picanha", "Costela", "Fraldinha", "Alcatra"],
        "Vegetariano": ["Hambúrguer de Grão", "Tofu", "Tempeh", "Lasanha Vegetariana", "Curry"],
        "Kids": ["Mini Burger", "Nuggets", "Macarrão", "Pizza Kids", "Suco"],
        "Executivo": ["Prato do Dia", "Menu Light", "Combo Empresarial", "Almoço Executivo", "Janta"]
    }
    
    turnos = ["Almoço", "Jantar", "Café da Manhã", "Lanche da Tarde", "Happy Hour"]
    dias_semana = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
    formas_pagamento = ["Dinheiro", "PIX", "Débito", "Crédito", "VR", "VA"]
    bairros = ["Centro", "Jardim América", "Vila Nova", "Parque Industrial", "Bairro Alto", "Condado"]
    
    start_date = datetime(2023, 1, 1)
    
    data = {
        "id": list(range(1, n_samples + 1)),
        "data": [start_date + timedelta(days=i % 365, hours=i % 24) for i in range(n_samples)],
    }
    
    data["categoria_prato"] = np.random.choice(categories, n_samples)
    data["nome_prato"] = [np.random.choice(pratos.get(cat, ["Prato"])) for cat in data["categoria_prato"]]
    data["turno"] = np.random.choice(turnos, n_samples)
    data["dia_semana"] = np.random.choice(dias_semana, n_samples)
    data["forma_pagamento"] = np.random.choice(formas_pagamento, n_samples, p=[0.15, 0.25, 0.2, 0.25, 0.1, 0.05])
    data["bairro"] = np.random.choice(bairros, n_samples)
    
    base_prices = {
        "Lanches": (25, 45), "Pizzas": (45, 80), "Massas": (35, 60),
        "Saladas": (25, 40), "Sobremesas": (15, 30), "Bebidas Alcoólicas": (15, 40),
        "Refrigerantes": (5, 12), "Sucos": (10, 20), "Café": (8, 18),
        "Lanches Naturais": (20, 35), "Frutos do Mar": (50, 100),
        "Carnes": (45, 90), "Vegetariano": (30, 50), "Kids": (25, 40),
        "Executivo": (25, 45)
    }
    
    prices = []
    costs = []
    for cat in data["categoria_prato"]:
        price_range = base_prices.get(cat, (20, 50))
        price = np.random.uniform(price_range[0], price_range[1])
        cost = price * np.random.uniform(0.25, 0.45)
        prices.append(round(price, 2))
        costs.append(round(cost, 2))
    
    data["preco_venda"] = prices
    data["custo_ingredientes"] = costs
    
    data["num_itens"] = np.random.randint(1, 8, n_samples)
    data["num_pedidos"] = np.random.randint(1, 5, n_samples)
    
    sale_values = []
    for i in range(n_samples):
        num_items = data["num_itens"][i]
        num_orders = data["num_pedidos"][i]
        avg_price = data["preco_venda"][i]
        
        variations = np.random.uniform(0.7, 1.3)
        sale = num_items * num_orders * avg_price * variations
        sale_values.append(round(sale, 2))
    
    data["venda_diaria"] = sale_values
    
    data["custo_total"] = [c * data["num_itens"][i] * data["num_pedidos"][i] 
                          for i, c in enumerate(data["custo_ingredientes"])]
    
    data["lucro_bruto"] = [data["venda_diaria"][i] - data["custo_total"][i] 
                          for i in range(n_samples)]
    
    data["margem_lucro"] = [(data["lucro_bruto"][i] / data["venda_diaria"][i] * 100) 
                           if data["venda_diaria"][i] > 0 else 0 for i in range(n_samples)]
    
    data["desconto"] = [round(np.random.uniform(0, v * 0.15), 2) for v in data["venda_diaria"]]
    
    hora_pico = []
    for i in range(n_samples):
        hora = data["data"][i].hour
        if 12 <= hora <= 14 or 18 <= hora <= 21:
            hora_pico.append(hora + np.random.uniform(-0.5, 0.5))
        else:
            hora_pico.append(hora + np.random.uniform(-2, 2))
    data["hora_pico"] = [round(h, 1) for h in hora_pico]
    
    data["temperatura"] = np.round(np.random.uniform(15, 40, n_samples), 1)
    
    data["feriado"] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    data["promocao"] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    
    data["cliente_novo"] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    data["avaliacao"] = np.round(np.random.uniform(3.5, 5.0, n_samples), 1)
    data["tempo_espera_min"] = np.random.randint(5, 45, n_samples)
    
    df = pd.DataFrame(data)
    
    df["lucratividade"] = df.apply(
        lambda x: "excelente" if x["margem_lucro"] > 40
        else "boa" if x["margem_lucro"] > 25
        else "regular" if x["margem_lucro"] > 15
        else "ruim",
        axis=1
    )
    
    df["classificacao_venda"] = df.apply(
        lambda x: "alta" if x["venda_diaria"] > 500
        else "media" if x["venda_diaria"] > 200
        else "baixa",
        axis=1
    )
    
    df["dia_util"] = df["dia_semana"].apply(
        lambda x: 1 if x in ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"] else 0
    )
    
    df["final_semana"] = df["dia_semana"].apply(
        lambda x: 1 if x in ["Sábado", "Domingo"] else 0
    )
    
    df["preco_medio_item"] = df["venda_diaria"] / (df["num_itens"] * df["num_pedidos"])
    
    df["ticket_medio"] = df["venda_diaria"] / df["num_pedidos"]
    
    df = df.drop(columns=["data"])
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dados_restaurante.csv")
    
    df.to_csv(output_path, index=False)
    file_size = os.path.getsize(output_path)
    
    print(f"\n{'='*60}")
    print("     DADOS FINANCEIROS GERADOS")
    print(f"{'='*60}")
    print(f"Arquivo: {output_path}")
    print(f"Tamanho: {file_size / (1024*1024):.2f} MB")
    print(f"Registros: {len(df):,}")
    print(f"Colunas: {len(df.columns)}")
    print(f"\nCategorias: {df['categoria_prato'].nunique()}")
    print(f"Pratos: {df['nome_prato'].nunique()}")
    print(f"\nDistribuição de Lucratividade:")
    print(df['lucratividade'].value_counts())
    print(f"\nEstatísticas Financeiras:")
    print(f"  - Venda média: R$ {df['venda_diaria'].mean():.2f}")
    print(f"  - Ticket médio: R$ {df['ticket_medio'].mean():.2f}")
    print(f"  - Margem média: {df['margem_lucro'].mean():.1f}%")
    print(f"  - Lucro total: R$ {df['lucro_bruto'].sum():,.2f}")
    print(f"{'='*60}\n")
    
    return df


if __name__ == "__main__":
    generate_financial_data(100000)
