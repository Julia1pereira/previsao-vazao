import logging
from datetime import date
from typing import Optional, Sequence

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INTERPOLATION_NULL_THRESHOLD = 0.2


def transformar_dataframe(
    path_df_vazao: str,
    path_df_clima: str,
    path_df_previsao: str,
    colunas: Sequence[str],
    diretorio_saida: str,
) -> Optional[pd.DataFrame]:
    """
    Lê parquet de vazão, clima e previsão, trata cada fonte, une pelos carimbos de data,
    cria atributos derivados, salva o parquet final e retorna (estacao, caminho_arquivo).
    Se o tratamento de vazão falhar ou esvaziar a série, interrompe e retorna None.
    """
    df_vazao = pd.read_parquet(path_df_vazao).copy()
    df_clima = pd.read_parquet(path_df_clima).copy()
    df_previsao = pd.read_parquet(path_df_previsao).copy()

    df_clima = tratar_dados_clima(df_clima)

    df_vazao = tratar_dados_vazao(df_vazao)
    if df_vazao is None or df_vazao.empty:
        logger.warning("Fluxo interrompido: dataframe de vazao vazio após tratamento.")
        return None

    df_merged = pd.merge(df_clima, df_vazao, how="left", on="data")
    df_merged = pd.concat([df_merged, df_previsao], ignore_index=True)

    df_merged = df_merged.set_index("data").sort_index()
    df_processado = aplicar_atributos(df_merged[colunas])

    estacao = df_processado['co_estacao'].iloc[0]
    filename = f"{diretorio_saida}dados_processados_{estacao}_{date.today()}.parquet"
    df_processado.to_parquet(filename)
    logger.info("Dados processados salvos em %s", filename)

    return estacao, filename


def remover_datas(df: pd.DataFrame) -> None:
    """
    Converte data_vazao para datetime (se necessário) e filtra linhas anteriores a
    1981-01-01. Retorna None; a filtragem é feita sobre a referência local.

    Para facilitar o treino do modelo, datas anteriores são excluídas já que não
    se tem dados climáticos para o período, e também evita que a interpolaçao seja
    feita nesse período.
    """
    cutoff = pd.Timestamp(1981, 1, 1)

    if not pd.api.types.is_datetime64_any_dtype(df["data_vazao"]):
        df = df.assign(data_vazao=pd.to_datetime(df["data_vazao"]))

    df = df[df["data_vazao"] >= cutoff]


def remover_duplicatas_vazao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicatas de data privilegiando registros com vazão preenchida, descarta
    duplicatas com vazão nula e mantém a primeira ocorrência restante. Lança erro se
    ainda houver datas duplicadas ao final.
    """
    df_limpo = df.copy()

    mask_dupe_null = df_limpo.index.duplicated(keep=False) & df_limpo["vazao"].isnull()
    df_limpo = df_limpo[~mask_dupe_null]

    df_limpo = df_limpo.drop_duplicates(subset="data_vazao", keep="first")

    if not df_limpo["data_vazao"].is_unique:
        raise ValueError("Ainda existem datas duplicadas após o tratamento.")

    return df_limpo


def adicionar_dados_faltantes_vazao(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Reindexa vazão em base diária contínua, interpola valores faltantes no tempo,
    aborta (retorna None) se a fração interpolada exceder INTERPOLATION_NULL_THRESHOLD
    e lança erro caso restem nulos após a interpolação.

    O limite de interpolação é importante para controlar a quantidade de dados falsos
    gerados dentro do dataset.
    """
    df_tratado = df.copy()

    df_tratado = df_tratado.dropna(subset=["vazao"])

    df_tratado = df_tratado.set_index("data_vazao").sort_index()

    full_range = pd.date_range(start=df_tratado.index.min(), end=df_tratado.index.max(), freq="D")
    df_tratado = df_tratado.reindex(full_range)

    dias_preenchidos = df_tratado["vazao"].isna().sum()
    df_tratado["vazao"] = df_tratado["vazao"].interpolate(method="time", limit_direction="both")

    if dias_preenchidos > len(df_tratado) * INTERPOLATION_NULL_THRESHOLD:
        logger.warning(
            "Interpolação preencheu %.1f%% (>%.0f%%) dos valores. Dados de vazão picotados.",
            (dias_preenchidos / len(df_tratado)) * 100,
            INTERPOLATION_NULL_THRESHOLD * 100,
        )
        return None

    if df_tratado["vazao"].isnull().any():
        raise ValueError("Ainda existem valores nulos após a interpolação.")

    return df_tratado


def tratar_dados_vazao(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Pipeline de vazão: normaliza datas, remove duplicatas privilegiando valores válidos,
    reindexa para base diária contínua, interpola lacunas e retorna dataframe final com
    coluna data; se a interpolação não for confiável, retorna None.
    """
    remover_datas(df)
    df_tratado = remover_duplicatas_vazao(df)
    df_tratado = adicionar_dados_faltantes_vazao(df_tratado)

    if df_tratado is None:
        return None

    df_tratado = df_tratado.reset_index().rename(columns={"index": "data"})

    return df_tratado


def tratar_dados_clima(df: pd.DataFrame) -> pd.DataFrame:
    """
    Copia o dataframe climático, garante ordenação por data e substitui valores numéricos
    negativos pela média móvel de 3 dias defasada em 3 posições.

    Geralmente os dados de clima têm valores negativos quanto mais se aproxima do dia atual,
    não ultrapassando 3 dias de dados negativos.
    """
    df_clima = df.copy()

    if "data" in df_clima.columns:
        df_clima["data"] = pd.to_datetime(df_clima["data"])
        df_clima = df_clima.sort_values("data")

    colunas_numericas = df_clima.select_dtypes(include=["number"]).columns

    for coluna in colunas_numericas:
        # shift de 3 dias considerando que pode ter mais de 1 dia com valores negativos no dataset climático
        medias_previas = df_clima[coluna].shift(3).rolling(window=3, min_periods=1).mean()
        negativos = df_clima[coluna] < 0
        df_clima.loc[negativos, coluna] = medias_previas[negativos]

    return df_clima


def adicionar_lags(df: pd.DataFrame) -> None:
    """Cria defasagens de 1 dia para vazão, precipitação e umidade relativa (colunas *_t1)."""
    df["vazao_t1"] = df["vazao"].shift(1)
    df["umidade_t1"] = df["umidade_relativa"].shift(1)
    df["precipitacao_t1"] = df["precipitacao"].shift(1)


def adicionar_janelas(df: pd.DataFrame) -> None:
    """
    Calcula janelas móveis (média e soma) para 3, 6 e 9 dias das colunas
    vazao, precipitacao e umidade_relativa.
    """
    for col in ["vazao", "precipitacao", "umidade_relativa"]:
        for window in [3, 6, 9]:
            df[f"{col}_roll{window}_mean"] = df[col].rolling(window).mean()
            df[f"{col}_roll{window}_sum"] = df[col].rolling(window).sum()


def gerar_estacao(data: pd.Timestamp) -> str:
    """Retorna a estação do ano para a data informada com base nos solstícios/equinócios."""
    ano = data.year

    if pd.Timestamp(ano, 3, 20) <= data < pd.Timestamp(ano, 6, 21):
        return "outono"
    if pd.Timestamp(ano, 6, 21) <= data < pd.Timestamp(ano, 9, 22):
        return "inverno"
    if pd.Timestamp(ano, 9, 22) <= data < pd.Timestamp(ano, 12, 21):
        return "primavera"
    return "verao"


def adicionar_sazonalidade(df: pd.DataFrame) -> None:
    """
    Usa o índice datetime para criar mês, encoding trigonométrico anual (mes_sin/mes_cos)
    e dummies das estações do ano.
    """
    df["mes"] = df.index.month
    df["mes_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    df["estacao"] = df.index.map(gerar_estacao)
    estacoes_dummies = pd.get_dummies(df["estacao"], prefix="estacao")
    df.drop(columns=["estacao"], inplace=True)
    df[estacoes_dummies.columns] = estacoes_dummies


def preencher_dados_categoricos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Faz forward-fill das colunas que não começam com 'vazao' para propagar variáveis
    categóricas ou numéricas auxiliares ao longo do tempo.
    """
    colunas_preencher = [col for col in df.columns if not col.startswith("vazao")]
    df[colunas_preencher] = df[colunas_preencher].ffill()

    return df


def aplicar_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de feature engineering: cria lags, janelas móveis, preenche inícios de série,
    forward-fill para demais colunas e adiciona sazonalidade trigonométrica e dummies.
    """
    df = df.copy()

    adicionar_lags(df)
    adicionar_janelas(df)
    df = df.bfill()  # preenche lags/rolling iniciais
    df = preencher_dados_categoricos(df)
    adicionar_sazonalidade(df)

    return df
