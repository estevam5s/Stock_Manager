import sys
import os
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config


def predict(model_path, data_path=None, single_data=None):
    print(f"\n{'='*60}")
    print("     PREDIÇÃO - ANÁLISE FINANCEIRA RESTAURANTE")
    print(f"{'='*60}\n")
    
    if not os.path.exists(model_path):
        print(f"ERRO: Modelo não encontrado em: {model_path}")
        sys.exit(1)
    
    print(f"1. Carregando modelo: {model_path}")
    pipeline = joblib.load(model_path)
    print(f"   - Modelo carregado com sucesso!")
    
    if data_path:
        print(f"\n2. Carregando dados: {data_path}")
        df = pd.read_csv(data_path)
        print(f"   - {len(df)} registros")
        
        predictions = pipeline.predict(df)
        
        print(f"\n3. Predições:")
        print("-" * 40)
        
        for i, pred in enumerate(predictions[:10]):
            print(f"   Registro {i+1}: {pred}")
        
        df["predicted"] = predictions
        
        output_path = data_path.replace(".csv", "_predictions.csv")
        df.to_csv(output_path, index=False)
        print(f"\n   Resultado salvo em: {output_path}")
    
    elif single_data:
        print(f"\n2. Predição de dados únicos")
        df = pd.DataFrame([single_data])
        
        prediction = pipeline.predict(df)[0]
        
        print(f"\n3. Resultado:")
        print("-" * 40)
        print(f"   Predição: {prediction}")
    
    print(f"\n{'='*60}")
    print("     PREDIÇÃO CONCLUÍDA!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predizer análise financeira")
    parser.add_argument("model_path", help="Caminho do modelo .pkl")
    parser.add_argument("--data", "-d", help="Caminho do arquivo CSV")
    args = parser.parse_args()
    
    predict(args.model_path, args.data)
