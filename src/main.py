"""Pipeline completo: coleta, tratamento e treino do LSTM final."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

from models.lstm_vazao import train_final_lstm
from process.coletar_dados import collect_data
from process.coletor_dados_climaticos import NasaPowerCollector, OpenMeteoCollector
from process.coletor_dados_estacao import EstacaoDataCollector
from process.tratamento_dados import transformar_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("pipeline")

RAW_DIR = Path("data_lake/raw")
PROCESSED_DIR = Path("data_lake/processed")
MODELS_DIR = Path("models")

COLUNAS_ALVO = [
    "vazao",
    "precipitacao",
    "temp_media",
    "umidade_relativa",
    "codigo_bacia",
    "codigo_sub_bacia",
    "co_estacao",
    "latitude",
    "longitude",
    "cidade",
    "estado",
    "rio",
]


def _reset_dir(path: Path) -> None:
    """Remove e recria um diretório (recursivo) para garantir estado limpo antes do fluxo."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _build_collectors() -> Tuple[EstacaoDataCollector, NasaPowerCollector, OpenMeteoCollector]:
    """
    Instancia coletores de vazão (PostgreSQL), clima histórico (NASA POWER) e previsão
    (Open-Meteo) com os parâmetros padrão esperados pelo pipeline.
    """
    banco_vazao = EstacaoDataCollector(
        user="postgres",
        password="admin",
        host="localhost",
        port=5432,
        database="banco_vazao",
    )

    nasa_collector = NasaPowerCollector(
        interval="daily",
        parameters="T2M,PRECTOT,RH2M",
        community="AG",
    )

    open_meteo = OpenMeteoCollector()

    return banco_vazao, nasa_collector, open_meteo


def _collect_raw_files(
    coletor_nasa: NasaPowerCollector,
    coletor_open: OpenMeteoCollector,
    coletor_estacao: EstacaoDataCollector,
) -> List[Tuple[str, str]]:
    """
    Limpa o diretório raw, coleta dados de clima/vazão/previsão para cada estação e
    retorna a lista de caminhos parquet gerados.
    """
    _reset_dir(RAW_DIR)

    files = collect_data(coletor_nasa, coletor_open, coletor_estacao, str(RAW_DIR) + "/")
    log.info("Coleta concluída | estações processadas: %s", len(files))

    return files


def _process_pairs(files: Iterable[Tuple[str, str, str]]) -> List[Tuple[str, Path]]:
    """
    Limpa o diretório processed, trata cada trio (vazão, clima, previsão), enriquece
    com atributos e devolve lista de (estacao, caminho_parquet_processado).
    """
    _reset_dir(PROCESSED_DIR)
    processed: List[Tuple[str, Path]] = []

    for file_vazao, file_clima, file_previsao in files:
        log.info("Processando estação | vazao")
        file = transformar_dataframe(file_vazao, file_clima, file_previsao, COLUNAS_ALVO, str(PROCESSED_DIR) + "/")

        if file == None:
            log.warning('Dataset vazio, provavelmente dados picotados.')
            continue
        
        processed.append(file)

    return processed


def _train_models(processed_info: Iterable[Tuple[str, Path]]) -> None:
    """
    Percorre os datasets processados, treina o LSTM final de cada estação e registra
    métricas (MAE/RMSE/R2); ignora estações com falha de leitura ou treino.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for estacao, processed_path in processed_info:

        log.info("Treinando LSTM final para estação %s", estacao)
        df = None

        try:
            df = __import__("pandas").read_parquet(processed_path)
        except Exception as e:  # pragma: no cover - depende de ambiente
            log.error("Falha ao ler parquet %s: %s", processed_path, e)
            continue

        model_dir = MODELS_DIR / f"estacao_{estacao}"

        try:
            metrics = train_final_lstm(df, model_dir=model_dir)
            log.info(
                "Treino concluído | estacao %s | MAE=%.4f | RMSE=%.4f | R2=%.4f",
                estacao,
                metrics["mae"],
                metrics["rmse"],
                metrics["r2"],
            )
        except Exception as e:  # pragma: no cover - depende de ambiente
            log.error("Falha no treino da estação %s: %s", estacao, e)


def main() -> None:
    """Pipeline end-to-end: coleta dados brutos, processa atributos e treina modelos finais."""
    coletor_estacao, coletor_nasa, coletor_open = _build_collectors()
    raw_files = _collect_raw_files(coletor_nasa, coletor_open, coletor_estacao)
    processed_info = _process_pairs(raw_files)
    _train_models(processed_info)


if __name__ == "__main__":
    main()
