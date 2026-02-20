import sys
import os
import argparse
import pandas as pd
import joblib
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config


def predict(model_path, data_path=None, single_data=None, verbose=False, output_path=None):
    print(f"\n{'='*60}")
    print("        PREDIÇÃO - ANÁLISE DE ESTOQUE ML")
    print(f"{'='*60}\n")
    
    if not os.path.exists(model_path):
        print(f"❌ ERRO: Modelo não encontrado em: {model_path}")
        sys.exit(1)
    
    print(f"📂 Carregando modelo: {model_path}")
    pipeline = joblib.load(model_path)
    print(f"   ✓ Modelo carregado com sucesso!")
    
    if hasattr(pipeline, 'classes_'):
        classes = pipeline.classes_
    else:
        classes = []
    
    if data_path:
        print(f"\n📊 Carregando dados: {data_path}")
        df = pd.read_csv(data_path)
        print(f"   ✓ {len(df)} registros carregados")
        
        predictions = pipeline.predict(df)
        probabilities = pipeline.predict_proba(df)
        confidences = [max(p) * 100 for p in probabilities]
        
        print(f"\n🔮 Predições:")
        print("-" * 60)
        
        if verbose:
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                confidence = max(prob) * 100
                print(f"   [{i+1:3d}] {pred:10s} | Confiança: {confidence:5.1f}%")
                for cls, p in zip(classes, prob):
                    print(f"          {cls}: {p*100:5.1f}%")
                print()
        else:
            summary = {}
            for pred, conf in zip(predictions, confidences):
                if pred not in summary:
                    summary[pred] = {"count": 0, "avg_confidence": 0}
                summary[pred]["count"] += 1
                summary[pred]["avg_confidence"] += conf
            
            for risk in ["critical", "high", "medium", "low"]:
                if risk in summary:
                    avg = summary[risk]["avg_confidence"] / summary[risk]["count"]
                    print(f"   {risk.upper():10s}: {summary[risk]['count']:3d} produtos | Média: {avg:.1f}%")
        
        df["predicted_risk_level"] = predictions
        df["confidence_score"] = confidences
        
        if output_path:
            final_output = output_path
        else:
            final_output = data_path.replace(".csv", "_predictions.csv")
        
        df.to_csv(final_output, index=False)
        print(f"\n✓ Resultado salvo em: {final_output}")
        
        critical_products = df[df["predicted_risk_level"] == "critical"]
        if not critical_products.empty:
            print(f"\n⚠️  ATENÇÃO: {len(critical_products)} produtos em nível CRÍTICO!")
            for _, row in critical_products.head(5).iterrows():
                prod_name = row.get("product_id", "Produto")
                print(f"      - {prod_name}")
    
    elif single_data:
        print(f"\n🎯 Predição de dados únicos")
        df = pd.DataFrame([single_data])
        
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0]
        confidence = max(probability) * 100
        
        print(f"\n📋 Resultado:")
        print("-" * 40)
        print(f"   Predição: {prediction}")
        print(f"   Confiança: {confidence:.1f}%")
        
        if len(classes) > 0:
            print(f"\n   Probabilidades por classe:")
            for cls, prob in zip(classes, probability):
                bar = "█" * int(prob * 10)
                print(f"      {cls:10s}: {prob*100:5.1f}% {bar}")
    
    else:
        print("❌ ERRO: Forneça --data ou --json")
        print("\nUso:")
        print("  python predict.py modelo.pkl --data dados.csv")
        print("  python predict.py modelo.pkl --json '{\"col\": valor}'")
        print("  python predict.py modelo.pkl --data dados.csv --verbose")
        print("  python predict.py modelo.pkl --data dados.csv -o resultado.csv")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("        PREDIÇÃO CONCLUÍDA!")
    print(f"{'='*60}\n")


def interactive_mode(model_path):
    print(f"\n{'='*60}")
    print("        MODO INTERATIVO")
    print(f"{'='*60}\n")
    
    pipeline = joblib.load(model_path)
    print("Forneça os dados do produto (Enter para padrão):\n")
    
    data = {}
    data["current_stock"] = float(input("Estoque atual: ") or 0)
    data["minimum_stock"] = float(input("Estoque mínimo: ") or 10)
    data["maximum_stock"] = float(input("Estoque máximo: ") or 50)
    data["monthly_sales"] = float(input("Vendas mensais: ") or 0)
    data["sales_last_7_days"] = float(input("Vendas últimos 7 dias: ") or 0)
    data["sales_last_30_days"] = float(input("Vendas últimos 30 dias: ") or 0)
    data["lead_time_days"] = float(input("Lead time (dias): ") or 7)
    data["unit_cost"] = float(input("Custo unitário: ") or 0)
    data["category"] = input("Categoria: ") or "unknown"
    data["supplier"] = input("Fornecedor: ") or "unknown"
    data["region"] = input("Região: ") or "SP"
    
    predict(model_path, single_data=data, verbose=True)


def main():
    parser = argparse.ArgumentParser(
        description="Predizer risco de estoque",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python predict.py ML/models/estoque_model.pkl --data ML/data/dados_estoque.csv
  python predict.py ML/models/estoque_model.pkl --data dados.csv --verbose
  python predict.py ML/models/estoque_model.pkl --data dados.csv -o resultado.csv
  python predict.py ML/models/estoque_model.pkl --json '{"current_stock": 5, "minimum_stock": 50}'
  python predict.py ML/models/estoque_model.pkl --interactive
        """
    )
    parser.add_argument("model_path", help="Caminho do modelo .pkl")
    parser.add_argument("--data", "-d", help="Caminho do arquivo CSV com dados")
    parser.add_argument("--json", "-j", help="Dados em formato JSON")
    parser.add_argument("--output", "-o", help="Caminho de saída para resultados")
    parser.add_argument("--verbose", "-v", action="store_true", help="Modo verboso")
    parser.add_argument("--interactive", "-i", action="store_true", help="Modo interativo")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.model_path)
        return
    
    single_data = None
    if args.json:
        try:
            single_data = json.loads(args.json)
        except json.JSONDecodeError as e:
            print(f"❌ ERRO: JSON inválido - {e}")
            sys.exit(1)
    
    predict(args.model_path, args.data, single_data, args.verbose, args.output)


if __name__ == "__main__":
    main()
