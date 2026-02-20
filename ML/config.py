import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

TEST_SIZE = 0.2
RANDOM_STATE = 42

CATEGORICAL_COLS = ["grupo", "medida", "fornecedor", "responsável", "ativo"]
NUMERICAL_COLS = [
    "estoque", "estoque_mín", "entradas", "saídas",
    "custo_unit", "valor_venda", "repor"
]

TARGET_COL = "target_risk_level"

FEATURE_ENGINEERING_COLS = [
    "stock_turnover_rate",
    "safety_stock_ratio", 
    "stock_coverage_days",
    "stock_pressure_index",
    "inventory_value"
]
