# 🧠 PROMPT COMPLETO --- CRIAÇÃO DE IA AVANÇADA PARA ANÁLISE DE ESTOQUE

Você é um engenheiro de Machine Learning sênior.

Sua tarefa é desenvolver um sistema completo de Machine Learning para
análise avançada de estoque, inspirado na arquitetura do MegaDataAI,
utilizando Python, pandas e scikit-learn.

O sistema deve gerar um modelo treinável e salvar o modelo final em
formato `.pkl`.

------------------------------------------------------------------------

# 🎯 OBJETIVO

Criar um motor inteligente capaz de:

-   Prever risco de ruptura de estoque
-   Detectar excesso de estoque
-   Classificar nível de criticidade (low, medium, high, critical)
-   Analisar padrões de consumo
-   Identificar produtos com comportamento anômalo
-   Permitir treinamento, avaliação e predição
-   Ser modular e pronto para produção

------------------------------------------------------------------------

# 📁 ESTRUTURA OBRIGATÓRIA DO PROJETO

Criar automaticamente a seguinte estrutura:

/ML │ ├── train.py ├── predict.py ├── preprocess.py ├──
feature_engineering.py ├── model.py ├── clustering.py ├── utils.py ├──
config.py ├── requirements.txt │ └── models/ └── estoque_model.pkl

------------------------------------------------------------------------

# 📊 ESTRUTURA DOS DADOS DE ENTRADA

O dataset deve conter colunas como:

-   product_id (categórico)
-   category (categórico)
-   supplier (categórico)
-   region (categórico)
-   current_stock (numérico)
-   minimum_stock (numérico)
-   maximum_stock (numérico)
-   monthly_sales (numérico)
-   lead_time_days (numérico)
-   unit_cost (numérico)
-   sales_last_7\_days (numérico)
-   sales_last_30_days (numérico)
-   seasonality_index (numérico)
-   demand_trend (numérico)
-   target_risk_level (TARGET: low, medium, high, critical)

------------------------------------------------------------------------

# 🧠 INTELIGÊNCIA AVANÇADA (OBRIGATÓRIO)

## Feature Engineering

Criar novas variáveis:

-   stock_turnover_rate = monthly_sales / current_stock
-   safety_stock_ratio = minimum_stock / current_stock
-   stock_coverage_days = current_stock / (monthly_sales / 30)
-   stock_pressure_index = sales_last_7\_days / current_stock
-   inventory_value = current_stock \* unit_cost

------------------------------------------------------------------------

## Pré-processamento

-   Tratamento de valores nulos
-   Encoding automático de variáveis categóricas
-   Padronização de dados numéricos
-   Separação treino/teste (80/20)
-   Pipeline com ColumnTransformer
-   Uso de Pipeline do sklearn

------------------------------------------------------------------------

## Modelos Suportados

Permitir escolha entre:

-   RandomForestClassifier
-   GradientBoostingClassifier

Implementar parâmetro para escolha do modelo no train.py.

------------------------------------------------------------------------

## Métricas de Avaliação

Calcular:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion Matrix

Exibir no terminal após o treinamento.

------------------------------------------------------------------------

## Salvamento do Modelo

Salvar o modelo treinado em:

/ML/models/estoque_model.pkl

Utilizar:

joblib.dump(model, filepath)

------------------------------------------------------------------------

# 🏋️ TREINAMENTO VIA LINHA DE COMANDO

O arquivo `train.py` deve permitir execução:

python ML/train.py dados_estoque.csv target_risk_level random_forest

Parâmetros:

1.  Caminho do dataset
2.  Nome da coluna target
3.  Tipo do modelo (random_forest ou gradient_boosting)

Fluxo:

-   Carregar dados
-   Executar feature engineering
-   Pré-processar
-   Treinar modelo
-   Avaliar
-   Salvar modelo

------------------------------------------------------------------------

# 🔮 PREDIÇÃO

O arquivo `predict.py` deve permitir:

python ML/predict.py ML/models/estoque_model.pkl novos_dados.csv

Deve retornar:

-   Predição
-   Probabilidade
-   Score de confiança

Permitir também predição individual via dicionário.

------------------------------------------------------------------------

# 📊 CLUSTERIZAÇÃO OPCIONAL

Implementar módulo `clustering.py` usando:

-   KMeans

Objetivo:

-   Segmentar produtos por comportamento
-   Identificar grupos de risco
-   Permitir execução opcional via linha de comando

------------------------------------------------------------------------

# ⚙️ REQUISITOS

Gerar automaticamente o arquivo:

requirements.txt

Com:

pandas numpy scikit-learn joblib matplotlib seaborn

------------------------------------------------------------------------

# 🚀 RESULTADO FINAL ESPERADO

Um sistema de Machine Learning completo, estruturado, inteligente e
treinável, salvo em `.pkl`, capaz de realizar análise avançada de
estoque empresarial.

O código deve ser robusto, profissional e organizado.
