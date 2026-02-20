import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime


class RestaurantAnalyzer:
    def __init__(self, model_path="restaurante/ML/models/modelo_restaurante.pkl"):
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model_data = joblib.load(self.model_path)
            self.model = self.model_data.get('model')
            print(f"✓ Modelo carregado: {self.model_path}")
            print(f"  Classes: {self.model_data.get('classes', [])}")
            print(f"  Accuracy: {self.model_data.get('accuracy', 0)*100:.2f}%")
        else:
            print(f"✗ Modelo não encontrado: {self.model_path}")
            
    def predict(self, features):
        if self.model is None:
            return None
            
        df = pd.DataFrame([features])
        
        feature_cols = self.model_data.get('feature_cols', [])
        
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        df = df[feature_cols]
        df = df.fillna(0)
        
        for col in df.select_dtypes(include=['object']).columns:
            if col in self.model_data.get('categorical_encoder', {}):
                le = self.model_data['categorical_encoder'][col]
                df[col] = df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
        
        prediction = self.model.predict(df)[0]
        probabilities = self.model.predict_proba(df)[0]
        
        return {
            'prediction': prediction,
            'probabilities': dict(zip(self.model.classes_, probabilities)),
            'confidence': max(probabilities) * 100
        }
    
    def predict_batch(self, df):
        if self.model is None:
            return None
            
        feature_cols = self.model_data.get('feature_cols', [])
        
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        df = df[feature_cols]
        df = df.fillna(0)
        
        predictions = self.model.predict(df)
        
        return predictions
    
    def analyze_day(self, dia, mes, ano, categoria, temperatura=25, promocao=0, desconto=0):
        features = {
            'dia': dia,
            'mes': mes,
            'ano': ano,
            'dia_semana': datetime(ano, mes, dia).weekday(),
            'e_fim_semana': 1 if datetime(ano, mes, dia).weekday() in [5, 6] else 0,
            'e_feriado': 0,
            'e_sexta': 1 if datetime(ano, mes, dia).weekday() == 4 else 0,
            'trimestre': (mes - 1) // 3 + 1,
            'e_inicio_mes': 1 if dia <= 7 else 0,
            'e_fim_mes': 1 if dia >= 23 else 0,
            'venda_diaria': 0,
            'custo_ingredientes': 0,
            'lucro_bruto': 0,
            'margem_lucro': 0,
            'categoria': categoria,
            'num_pedidos': 0,
            'ticket_medio': 0,
            'temperatura': temperatura,
            'promocao': promocao,
            'desconto': desconto
        }
        
        return self.predict(features)
    
    def get_predictions_from_data(self, csv_path):
        if not os.path.exists(csv_path):
            print(f"✗ Arquivo não encontrado: {csv_path}")
            return None
            
        df = pd.read_csv(csv_path)
        predictions = self.predict_batch(df)
        
        return predictions


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Análise de Restaurante")
    parser.add_argument("--predict", "-p", help="Predizer para um dia específico (JSON)")
    parser.add_argument("--data", "-d", help="CSV com dados para predição")
    parser.add_argument("--report", "-r", action="store_true", help="Gerar relatório")
    
    args = parser.parse_args()
    
    analyzer = RestaurantAnalyzer()
    
    if args.predict:
        data = json.loads(args.predict)
        result = analyzer.analyze_day(
            dia=data.get('dia', 1),
            mes=data.get('mes', 1),
            ano=data.get('ano', 2024),
            categoria=data.get('categoria', 'Lanches'),
            temperatura=data.get('temperatura', 25),
            promocao=data.get('promocao', 0),
            desconto=data.get('desconto', 0)
        )
        if result:
            print(f"\n📊 Resultado da Predição:")
            print(f"   Risco: {result['prediction']}")
            print(f"   Confiança: {result['confidence']:.1f}%")
            print(f"   Probabilidades:")
            for k, v in result['probabilities'].items():
                print(f"      {k}: {v*100:.1f}%")
    
    elif args.data:
        predictions = analyzer.get_predictions_from_data(args.data)
        if predictions is not None:
            print(f"\n✓ Predições geradas para {len(predictions)} registros")
            print(f"   Primeiros 10: {predictions[:10]}")
    
    elif args.report:
        from reports.report_generator import ReportGenerator
        
        df = pd.read_csv("restaurante/ML/data/dados_financeiros.csv")
        metrics = {
            'accuracy': analyzer.model_data.get('accuracy', 0),
            'precision': analyzer.model_data.get('precision', 0)
        }
        
        generator = ReportGenerator()
        generator.generate_full_report(df, metrics)
    
    else:
        print("""
Uso:
  python analyze.py --predict '{"dia": 25, "mes": 12, "ano": 2024, "categoria": "Lanches"}'
  python analyze.py --data dados.csv
  python analyze.py --report
        """)


if __name__ == "__main__":
    main()
