import sys
import os
import argparse
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from feature_engineering import create_features
from preprocess import handle_missing_values, create_preprocessor
from model import get_model, create_pipeline
from utils import evaluate_model, plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Treinar modelo de análise de estoque")
    parser.add_argument("data_path", help="Caminho do arquivo CSV")
    parser.add_argument("target_col", help="Nome da coluna target")
    parser.add_argument("model_type", choices=["random_forest", "gradient_boosting"],
                       help="Tipo de modelo")
    
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print("     TREINAMENTO - ANÁLISE DE ESTOQUE ML")
    print(f"{'='*50}\n")
    
    print(f"1. Carregando dados: {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"   - {len(df)} registros carregados")
    print(f"   - Colunas: {list(df.columns)}")
    
    print(f"\n2. Aplicando Feature Engineering")
    df = create_features(df)
    print(f"   - Features criadas: {len(df.columns)} total")
    
    target_col = args.target_col
    if target_col not in df.columns:
        print(f"\nERRO: Coluna target '{target_col}' não encontrada!")
        sys.exit(1)
    
    feature_cols = [col for col in df.columns if col != target_col]
    
    categorical_cols = [col for col in config.CATEGORICAL_COLS if col in feature_cols]
    numerical_cols = [col for col in config.NUMERICAL_COLS + config.FEATURE_ENGINEERING_COLS 
                     if col in feature_cols]
    
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    print(f"\n3. Preparando dados")
    X = df[feature_cols]
    y = df[target_col]
    
    X = handle_missing_values(X)
    
    print(f"   - Features numéricas: {len(numerical_cols)}")
    print(f"   - Features categóricas: {len(categorical_cols)}")
    print(f"   - Target: {target_col}")
    print(f"   - Classes: {list(y.unique())}")
    
    print(f"\n4. Criando pipeline de preprocessing")
    preprocessor = create_preprocessor(numerical_cols, categorical_cols)
    
    print(f"\n5. Treinando modelo: {args.model_type}")
    model = get_model(args.model_type, config.RANDOM_STATE)
    
    pipeline = create_pipeline(preprocessor, model)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    print(f"   - Treino: {len(X_train)} | Teste: {len(X_test)}")
    pipeline.fit(X_train, y_train)
    
    print(f"\n6. Avaliando modelo")
    y_pred = pipeline.predict(X_test)
    
    results, cm = evaluate_model(y_test, y_pred, class_labels=list(y.unique()))
    
    if cm is not None:
        cm_path = os.path.join(config.MODELS_DIR, "confusion_matrix.png")
        plot_confusion_matrix(cm, list(y.unique()), cm_path)
    
    print(f"\n7. Salvando modelo")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_path = os.path.join(config.MODELS_DIR, "estoque_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"   - Modelo salvo em: {model_path}")
    
    print(f"\n{'='*50}")
    print("     TREINAMENTO CONCLUÍDO!")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
