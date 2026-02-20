import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

TEST_SIZE = 0.2
RANDOM_STATE = 42

CATEGORICAL_COLS = ["categoria_prato", "turno", "dia_semana", "forma_pagamento", "bairro"]
NUMERICAL_COLS = [
    "venda_diaria", "num_pedidos", "ticket_medio", "custo_ingredientes",
    "lucro_bruto", "margem_lucro", "hora_pico", "num_itens", "desconto",
    "temperatura", "feriado", "promocao"
]

TARGET_COL = "target_venda"

ANALYSIS_TYPES = [
    "previsao_vendas",
    "analise_lucratividade",
    "previsao_custos",
    "analise_cliente",
    "deteccao_anomalias"
]
