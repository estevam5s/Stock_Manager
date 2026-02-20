import sys
import os
import argparse
import pandas as pd
import joblib
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from feature_engineering import create_features
from preprocess import handle_missing_values, create_preprocessor
from model import get_classification_model, get_regression_model, create_pipeline


def main():
    parser = argparse.ArgumentParser(description="Treinar modelo de análise financeira do restaurante")
    parser.add_argument("data_path", help="Caminho do arquivo CSV")
    parser.add_argument("target_col", help="Nome da coluna target")
    parser.add_argument("model_type", choices=["random_forest", "gradient_boosting"], help="Tipo de modelo")
    parser.add_argument("--task", choicesregression"], default=["classification", "="classification", help="Tipo de tarefa")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("     TREINAMENTO - ANÁLISE FINANCEIRA RESTAURANTE")
    print(f"{'='*60}\n")
    
    print(f"1. Carregando dados: {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"   - {len(df)} registros carregados")
    
    print(f"\n2. Aplicando Feature Engineering")
    df = create_features(df)
    print(f"   - Features criadas: {len(df.columns)} total")
    
    target_col = args.target_col
    if target_col not in df.columns:
        print(f"\nERRO: Coluna target '{target_col}' não encontrada!")
        sys.exit(1)
    
    feature_cols = [col for col in df.columns if col != target_col]
    
    categorical_cols = [col for col in config.CATEGORICAL_COLS if col in feature_cols]
    numerical_cols = [col for col in config.NUMERICAL_COLS + list(df.select_dtypes(include=[np.number]).columns) if col in feature_cols]
    numerical_cols = list(set(numerical_cols) - {target_col})
    
    print(f"\n3. Preparando dados")
    X = df[feature_cols]
    y = df[target_col]
    
    X = handle_missing_values(X)
    
    print(f"   - Features numéricas: {len(numerical_cols)}")
    print(f"   - Features categóricas: {len(categorical_cols)}")
    print(f"   - Target: {target_col}")
    
    print(f"\n4. Criando pipeline de preprocessing")
    preprocessor = create_preprocessor(numerical_cols, categorical_cols)
    
    print(f"\n5. Treinando modelo: {args.model_type} ({args.task})")
    
    if args.task == "classification":
        model = get_classification_model(args.model_type, config.RANDOM_STATE)
    else:
        model = get_regression_model(args.model_type, config.RANDOM_STATE)
    
    pipeline = create_pipeline(preprocessor, model)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    print(f"   - Treino: {len(X_train)} | Teste: {len(X_test)}")
    pipeline.fit(X_train, y_train)
    
    print(f"\n6. Avaliando modelo")
    
    if args.task == "classification":
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        y_pred = pipeline.predict(X_test)
        
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }
        
        print(f"\n   Accuracy:  {results['accuracy']:.4f}")
        print(f"   Precision: {results['precision']:.4f}")
        print(f"   Recall:    {results['recall']:.4f}")
        print(f"   F1-Score:  {results['f1']:.4f}")
        
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_pred = pipeline.predict(X_test)
        
        results = {
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred)
        }
        
        print(f"\n   MAE:  {results['mae']:.4f}")
        print(f"   MSE:  {results['mse']:.4f}")
        print(f"   RMSE: {results['rmse']:.4f}")
        print(f"   R²:   {results['r2']:.4f}")
    
    print(f"\n7. Salvando modelo")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_path = os.path.join(config.MODELS_DIR, "modelo_restaurante.pkl")
    joblib.dump(pipeline, model_path)
    print(f"   - Modelo salvo em: {model_path}")
    
    print(f"\n{'='*60}")
    print("     TREINAMENTO CONCLUÍDO!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
