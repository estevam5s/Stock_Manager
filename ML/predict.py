import sys
import os
import argparse
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config


def predict(model_path, data_path=None, single_data=None):
    print(f"\n{'='*50}")
    print("     PREDIÇÃO - ANÁLISE DE ESTOQUE ML")
    print(f"{'='*50}\n")
    
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
        probabilities = pipeline.predict_proba(df)
        
        print(f"\n3. Predições:")
        print("-" * 40)
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            confidence = max(prob) * 100
            print(f"   Registro {i+1}: {pred} (confiança: {confidence:.1f}%)")
        
        df["predicted_risk_level"] = predictions
        df["confidence_score"] = [max(p) for p in probabilities]
        
        output_path = data_path.replace(".csv", "_predictions.csv")
        df.to_csv(output_path, index=False)
        print(f"\n   Resultado salvo em: {output_path}")
    
    elif single_data:
        print(f"\n2. Predição de dados únicos")
        df = pd.DataFrame([single_data])
        
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0]
        confidence = max(probability) * 100
        
        print(f"\n3. Resultado:")
        print("-" * 40)
        print(f"   Predição: {prediction}")
        print(f"   Confiança: {confidence:.1f}%")
        print(f"   Probabilidades por classe:")
        classes = pipeline.classes_
        for cls, prob in zip(classes, probability):
            print(f"      - {cls}: {prob*100:.1f}%")
    
    else:
        print("ERRO: Forneça data_path ou single_data")
        sys.exit(1)
    
    print(f"\n{'='*50}")
    print("     PREDIÇÃO CONCLUÍDA!")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Predizer risco de estoque")
    parser.add_argument("model_path", help="Caminho do modelo .pkl")
    parser.add_argument("--data", "-d", help="Caminho do arquivo CSV com dados")
    parser.add_argument("--json", "-j", help="Dados em formato JSON (ex: '{\"col1\": valor1, ...}')")
    
    args = parser.parse_args()
    
    single_data = None
    if args.json:
        import json
        try:
            single_data = json.loads(args.json)
        except json.JSONDecodeError as e:
            print(f"ERRO: JSON inválido - {e}")
            sys.exit(1)
    
    predict(args.model_path, args.data, single_data)


if __name__ == "__main__":
    main()
