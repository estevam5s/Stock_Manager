import pandas as pd
import numpy as np


def create_features(df):
    df = df.copy()
    
    if "preco_venda" in df.columns and "custo_ingredientes" in df.columns:
        df["margem_real"] = np.where(
            df["preco_venda"] > 0,
            (df["preco_venda"] - df["custo_ingredientes"]) / df["preco_venda"] * 100,
            0
        )
    
    if "venda_diaria" in df.columns and "num_pedidos" in df.columns:
        df["ticket_medio"] = np.where(
            df["num_pedidos"] > 0,
            df["venda_diaria"] / df["num_pedidos"],
            0
        )
    
    if "venda_diaria" in df.columns and "num_itens" in df.columns:
        df["valor_por_item"] = np.where(
            df["num_itens"] > 0,
            df["venda_diaria"] / df["num_itens"],
            0
        )
    
    if "lucro_bruto" in df.columns and "custo_ingredientes" in df.columns:
        df["roi"] = np.where(
            df["custo_ingredientes"] > 0,
            df["lucro_bruto"] / df["custo_ingredientes"] * 100,
            0
        )
    
    if "temperatura" in df.columns and "venda_diaria" in df.columns:
        df["venda_por_temperatura"] = df["venda_diaria"] * df["temperatura"] / 25
    
    if "hora_pico" in df.columns:
        df["horario_pico_score"] = np.where(
            (df["hora_pico"] >= 12) & (df["hora_pico"] <= 14), 3,
            np.where(
                (df["hora_pico"] >= 18) & (df["hora_pico"] <= 21), 3,
                np.where(
                    (df["hora_pico"] >= 11) & (df["hora_pico"] <= 22), 2, 1
                )
            )
        )
    
    if "feriado" in df.columns and "venda_diaria" in df.columns:
        df["venda_feriado"] = df["feriado"] * df["venda_diaria"]
    
    if "promocao" in df.columns and "desconto" in df.columns:
        df["desconto_efetivo"] = df["promocao"] * df["desconto"]
    
    if "avaliacao" in df.columns:
        df["nota_boa"] = (df["avaliacao"] >= 4.5).astype(int)
        df["nota_ruim"] = (df["avaliacao"] < 4.0).astype(int)
    
    if "tempo_espera_min" in df.columns:
        df["espera_longo"] = (df["tempo_espera_min"] > 30).astype(int)
        df["espera_rapido"] = (df["tempo_espera_min"] <= 15).astype(int)
    
    if "dia_semana" in df.columns:
        df["e_fds"] = df["dia_semana"].isin(["Sábado", "Domingo"]).astype(int)
        df["e_semana"] = df["dia_semana"].isin(["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]).astype(int)
    
    return df
