import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys


def train_model():
    data_path = "restaurante/ML/data/dados_financeiros.csv"
    model_path = "restaurante/ML/models/modelo_restaurante.pkl"
    
    print(f"\n{'='*60}")
    print("     TREINAMENTO - ANÁLISE FINANCEIRA RESTAURANTE")
    print(f"{'='*60}\n")
    
    print(f"1. Carregando dados: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   - {len(df)} registros carregados")
    print(f"   - {len(df.columns)} colunas")
    
    target_col = "classificacao_risco"
    
    if target_col not in df.columns:
        print(f"   Target não encontrado, usando classificação de venda")
        target_col = "classificacao_venda"
    
    print(f"   - Target: {target_col}")
    
    categorical_cols = ['categoria']
    numerical_cols = ['dia', 'mes', 'ano', 'dia_semana', 'e_fim_semana', 'e_feriado',
                      'e_sexta', 'trimestre', 'e_inicio_mes', 'e_fim_mes',
                      'venda_diaria', 'custo_ingredientes', 'lucro_bruto', 'margem_lucro',
                      'num_pedidos', 'ticket_medio', 'temperatura', 'promocao', 'desconto']
    
    feature_cols = [c for c in numerical_cols if c in df.columns]
    feature_cols += [c for c in categorical_cols if c in df.columns]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    X = X.fillna(0)
    
    print(f"\n2. Features: {len(feature_cols)}")
    print(f"   - Numéricas: {len([c for c in numerical_cols if c in X.columns])}")
    print(f"   - Categóricas: {len([c for c in categorical_cols if c in X.columns])}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n3. Treinando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f"\n4. Avaliando modelo...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n   📊 MÉTRICAS:")
    print(f"   - Accuracy:  {accuracy*100:.2f}%")
    print(f"   - Precision: {precision*100:.2f}%")
    print(f"   - Recall:    {recall*100:.2f}%")
    print(f"   - F1-Score:  {f1*100:.2f}%")
    
    classes = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    print(f"\n   📋 Classes: {classes}")
    print(f"   📈 Matriz de Confusão:")
    print(cm)
    
    print(f"\n5. Salvando modelo...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'classes': classes,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'categorical_encoder': {}
    }
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            model_data['categorical_encoder'][col] = le
    
    joblib.dump(model_data, model_path)
    print(f"   - Modelo salvo em: {model_path}")
    
    print(f"\n6. Feature Importance (Top 10):")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"   {i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")
    
    print(f"\n{'='*60}")
    print("     TREINAMENTO CONCLUÍDO!")
    print(f"{'='*60}\n")
    
    return model_data


if __name__ == "__main__":
    train_model()
