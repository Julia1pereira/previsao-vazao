import pandas as pd
import logging
from datetime import date
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Tratamento de dados')

def transformar_dataframe(path_df_vazao, path_df_clima, colunas, diretorio_saida):
    """
    Une os dataframes de vazão e dados climáticos com base em 'data'.
    Aplica transformação dos dados de vazão.

    Retorna um DataFrame com as colunas selecionadas além do index da data.
    """
    df_vazao = pd.read_parquet(path_df_vazao)
    df_clima = pd.read_parquet(path_df_clima)

    df_vazao = tratar_dados_vazao(df_vazao)

    # padronização de datas
    df_clima['data'] = pd.to_datetime(df_clima['data'])

    df = pd.merge(df_vazao, df_clima, left_index=True, right_on=['data'], how='inner')
    df.set_index('data', inplace=True)
    df.sort_index()

    # Aplicação da engenharia de atributos
    df = aplicar_atributos(df[colunas])

    # Salva dataframe
    filename = f'{diretorio_saida}dados_processados_{df["co_estacao"].iloc[0]}_{date.today()}.parquet'
    df.to_parquet(filename)

def remover_duplicatas_vazao(df):
    logger.info("Removendo duplicatas")

    # 1. Remover duplicatas com vazão nula
    # Mantém as duplicatas onde pelo menos um tem vazão válida
    mask_dupe_null = df.index.duplicated(keep=False) & df['vazao'].isnull()
    df = df[~mask_dupe_null]

    # 2. Remover duplicatas restantes, mantendo a primeira
    df.drop_duplicates(subset='data_vazao', keep='first', inplace=True)

    # Garantir unicidade da coluna data
    if not df['data_vazao'].is_unique:
        raise ValueError("Ainda existem datas duplicadas após o tratamento.")

    return df

def adicionar_dados_faltantes_vazao(df):
    logger.info("Adicionando dados faltantes")

    # 3. Remover linhas onde vazão é nula
    df.dropna(subset=['vazao'], inplace=True)

    # 4. Indexar pelo campo data
    df.set_index('data_vazao', inplace=True)
    df.sort_index(inplace=True)

    # 5. Criar o range completo de datas
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_range)

    # 6. Interpolar vazão nos dias faltantes
    df['vazao'] = df['vazao'].interpolate(method='time', limit_direction='both')

    # 7. Preencher colunas categóricas
    cat_cols = df.columns.drop('vazao')

    for col in cat_cols:
        unique_vals = df[col].dropna().unique()

        if len(unique_vals) == 1:
            # Todas são iguais → preencher
            df[col] = df[col].fillna(unique_vals[0])
        else:
            raise ValueError(
                f"A coluna '{col}' possui valores distintos, "
                "o que não deveria ocorrer para uma única estação."
            )

    # 8. Validar que não restaram nulos
    if df.isnull().any().any():
        raise ValueError("Ainda existem valores nulos após a interpolação e preenchimento.")

    return df

def tratar_dados_vazao(df):
    df = remover_duplicatas_vazao(df)
    df = adicionar_dados_faltantes_vazao(df)

    return df

def gerar_estacao(data):
    ano = data.year
    if pd.Timestamp(ano, 3, 20) <= data < pd.Timestamp(ano, 6, 21):
        return 'outono'
    elif pd.Timestamp(ano, 6, 21) <= data < pd.Timestamp(ano, 9, 22):
        return 'inverno'
    elif pd.Timestamp(ano, 9, 22) <= data < pd.Timestamp(ano, 12, 21):
        return 'primavera'
    else:
        return 'verao'

def aplicar_atributos(df):
    # 1. Lags (memória)
    df["vazao_t1"] = df["vazao"].shift(1)
    df["umidade_t1"] = df["precipitacao"].shift(1)

    # 2. Rolling
    for col in ["vazao", "precipitacao", "umidade_relativa"]:
        for w in [3, 6, 9]:
            df[f"{col}_roll{w}_mean"]  = df[col].rolling(w).mean()
            df[f"{col}_roll{w}_sum"]   = df[col].rolling(w).sum()

    df = df.ffill().bfill()

    # 3. Sazonalidade
    df["mes"] = df.index.month
    df["mes_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    df['estacao'] = df.index.map(gerar_estacao)

    df = pd.get_dummies(df, columns=['estacao'], prefix='estacao')

    return df
