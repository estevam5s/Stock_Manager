#!/bin/bash

# Script para executar análise do restaurante

echo "=========================================="
echo "  ANÁLISE FINANCEIRA - RESTAURANTE"
echo "=========================================="
echo ""

if [ "$1" == "report" ]; then
    echo "📊 Gerando relatório..."
    source venv/bin/activate
    python restaurante/ML/analyze.py --report

elif [ "$1" == "predict" ]; then
    echo "🔮 Fazendo predição..."
    source venv/bin/activate
    python restaurante/ML/analyze.py --predict "$2"

elif [ "$1" == "train" ]; then
    echo "🏋️ Treinando modelo..."
    source venv/bin/activate
    python restaurante/ML/prepare_data.py
    python restaurante/ML/train_model.py

elif [ "$1" == "gui" ]; then
    echo "🖥️ Abrindo interface..."
    source venv/bin/activate
    python restaurante/ML/train_gui.py

elif [ "$1" == "help" ]; then
    echo "Comandos disponíveis:"
    echo ""
    echo "  ./run.sh train          - Treinar modelo com dados"
    echo "  ./run.sh report        - Gerar relatório"
    echo "  ./run.sh gui           - Abrir interface"
    echo '  ./run.sh predict \'{"dia":25,"mes":12}\' - Predizer'
    echo "  ./run.sh help          - Mostrar ajuda"
    echo ""

else
    echo "=========================================="
    echo "  RESTAURANTE ML - Menu Principal"
    echo "=========================================="
    echo ""
    echo "1) Treinar Modelo"
    echo "2) Gerar Relatório"
    echo "3) Interface Gráfica"
    echo "4) Sair"
    echo ""
    read -p "Escolha uma opção: " opt
    
    case $opt in
        1) ./run.sh train ;;
        2) ./run.sh report ;;
        3) ./run.sh gui ;;
        *) exit ;;
    esac
fi
