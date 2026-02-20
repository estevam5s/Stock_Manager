#!/usr/bin/env python3
import sys
import os
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def main():
    print(f"\n{'='*60}")
    print("     AUTOMACAO COMPLETA - ML ESTOQUE")
    print(f"{'='*60}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_dir, "data", "dados_estoque.csv")
    
    if not os.path.exists(data_file):
        print("\n📦 Gerando dados de exemplo...")
        run_command(f"python {base_dir}/generate_sample_data.py", "Gerando dados")
    else:
        print(f"\n✓ Dados já existentes: {data_file}")
    
    print("\n🏋️ Treinando modelo...")
    ret = run_command(
        f"python {base_dir}/train.py {data_file} target_risk_level random_forest",
        "Treinando modelo Random Forest"
    )
    
    if ret != 0:
        print("❌ Treinamento falhou!")
        sys.exit(1)
    
    print("\n🔮 Fazendo predições...")
    run_command(
        f"python {base_dir}/predict.py {base_dir}/models/estoque_model.pkl --data {data_file}",
        "Predizendo risco"
    )
    
    print("\n📊 Executando clustering...")
    run_command(
        f"python {base_dir}/cluster.py {data_file} --clusters 4",
        "Clusterização K-Means"
    )
    
    print(f"\n{'='*60}")
    print("  AUTOMACAO COMPLETA CONCLUIDA!")
    print(f"{'='*60}")
    print("\nArquivos gerados:")
    print(f"  - Modelo: {base_dir}/models/estoque_model.pkl")
    print(f"  - Predicoes: {data_file.replace('.csv', '_predictions.csv')}")
    print(f"  - Clusters: {data_file.replace('.csv', '_clustered.csv')}")
    print(f"  - Visualizacoes: {base_dir}/models/cluster_*.png")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
