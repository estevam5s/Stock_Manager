# MegaDataAI

<p align="center">
  <img src="https://img.shields.io/pypi/v/mega-data-ai?style=flat-square" alt="PyPI Version">
  <img src="https://img.shields.io/pypi/l/mega-data-ai?style=flat-square" alt="License">
  <img src="https://img.shields.io/pypi/pyversions/mega-data-ai?style=flat-square" alt="Python Version">
</p>

> **MegaDataAI** é um motor de inteligência artificial de alto desempenho para análise de grandes volumes de dados, oferecendo classificação, clustering e análise estatística.

## Recursos

- **Classificação** - Random Forest e Gradient Boosting
- **Clustering** - K-Means para análise não supervisionada
- **Análise Estatística** - Resumo, correlação, detecção de outliers
- **Processamento** - Suporte a CSV, JSON, Parquet, Excel
- **Alta Performance** - Otimizado para grandes datasets

## Instalação

### Via pip (em breve)
```bash
pip install mega-data-ai
```

### Instalação local
```bash
# Clone o repositório
cd mega-data-ai

# Instale as dependências
pip install -r requirements.txt

# Ou instale o pacote
pip install -e .
```

## Uso Rápido

### Treinamento

```python
from mega_data_ai import MegaDataAI

# Inicializar a IA
ai = MegaDataAI()

# Carregar dados
df = ai.load_data('seus_dados.csv')

# Treinar modelo
result = ai.train(df, target_column='sua_coluna_alvo')

print(f"Acurácia: {result['metrics']['accuracy'] * 100:.2f}%")

# Salvar modelo
ai.save_model('meu_modelo.pkl')
```

### Predição

```python
from mega_data_ai import MegaDataAI

# Carregar modelo treinado
ai = MegaDataAI()
ai.load_model('meu_modelo.pkl')

# Predizer uma amostra
result = ai.predict({
    'feature_0': 0.5,
    'feature_1': -0.3,
    'value': 500,
    'quantity': 50
})

print(f"Predição: {result['prediction']}")
print(f"Confiança: {result['confidence']}%")
```

### Predição em Lote

```python
import pandas as pd

# Carregar novos dados
df_novos = pd.read_csv('novos_dados.csv')

# Predizer em lote
resultados = ai.predict_batch(df_novos)

for r in resultados:
    print(f"{r['prediction']} - {r['confidence']}%")
```

## API

```python
from mega_data_ai import load_and_train, load_and_predict

# Treinar em uma linha
result = load_and_train('dados.csv', 'target', 'modelo.pkl')

# Predizer em uma linha
predictions = load_and_predict('novos_dados.csv', 'modelo.pkl')
```

## Documentação das Classes

### MegaDataAI

Classe principal que combina todas as funcionalidades.

```python
ai = MegaDataAI()

# Métodos disponíveis:
ai.load_data(filepath)           # Carrega dados de arquivo
ai.train(data, target)           # Treina o modelo
ai.predict(data)                 # Prediz uma amostra
ai.predict_batch(data)          # Prediz múltiplas amostras
ai.save_model(filepath)          # Salva o modelo
ai.load_model(filepath)         # Carrega o modelo
```

### DataAnalyzer

Para análise exploratória de dados.

```python
analyzer = DataAnalyzer()
df = analyzer.load_data('dados.csv')

# Métodos:
summary = analyzer.get_summary()              # Resumo estatístico
correlation = analyzer.get_correlation_matrix() # Matriz de correlação
```

### ClassificationEngine

Para treinamento de modelos de classificação.

```python
classifier = ClassificationEngine()

# Preparar dados
X_train, X_test, y_train, y_test = classifier.prepare_data(df, 'target')

# Treinar
classifier.train(X_train, y_train, model_type='random_forest')

# Avaliar
metrics = classifier.evaluate(X_test, y_test)

# Salvar/Carregar
classifier.save_model('modelo.pkl')
classifier.load_model('modelo.pkl')
```

### ClusteringEngine

Para clustering não supervisionado.

```python
clustering = ClusteringEngine()
clustering.fit(dados, n_clusters=5)

labels = clustering.predict(dados)
centers = clustering.get_cluster_centers()
```

## Treinamento via Linha de Comando

```bash
# Treinar com dados existentes
python train.py

# Treinar com seus dados
python train.py dados.csv modelo.pkl 10000

# Parâmetros:
#   dados.csv    - Arquivo de dados de entrada
#   modelo.pkl   - Nome do arquivo de modelo de saída
#   10000        - Número de amostras (opcional)
```

## API REST

Iniciar o servidor:

```bash
python app.py
```

Endpoints disponíveis:

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Interface web |
| POST | `/api/analyze` | Analisar dados |
| POST | `/api/train` | Treinar modelo |
| POST | `/api/predict` | Fazer predições |
| GET | `/api/stats` | Estatísticas |

### Exemplo de uso da API

```bash
# Treinar modelo
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{"data_file": "dados.csv", "target": "risk_level"}'

# Fazer predição
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [{"feature_0": 0.5, "value": 500}]}'
```

## Formatos de Arquivo Suportados

- **CSV** - `.csv`
- **JSON** - `.json`
- **Parquet** - `.parquet`
- **Excel** - `.xlsx`, `.xls`

## Estrutura dos Dados

O modelo foi treinado com os seguintes dados:

```
training_data.csv
├── feature_0 a feature_14  (15 features numéricas)
├── category                (categórico: A, B, C, D, E)
├── status                 (categórico: active, inactive, pending, completed)
├── region                 (categórico: North, South, East, West)
├── value                  (numérico: 0-1000)
├── quantity              (numérico: 1-100)
└── risk_level            (TARGET: low, medium, high, critical)
```

## Métricas do Modelo

- **Acurácia**: ~87%
- **F1 Score**: ~87%
- **Precision**: ~88%
- **Recall**: ~87%

## Requisitos

```
pandas >= 1.0.0
numpy >= 1.20.0
scikit-learn >= 1.0.0
joblib >= 1.0.0
```

## Contribuição

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## Licença

MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Autores

- **CyberDataAI Team** - *Trabalho inicial*

---

<p align="center">Made with ❤️ by CyberDataAI</p>
