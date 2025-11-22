from datetime import date
import pandas as pd
import logging

from process.coletor_dados_estacao import EstacaoDataCollector
from process.coletor_dados_climaticos import NasaPowerCollector


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Coleta de dados')

def collect_data(coletor_nasa: NasaPowerCollector, coletor_estacao: EstacaoDataCollector, diretorio_saida: str) -> list:
    """
    Coleta dados das estações hidrométricas e dados climáticos da NASA POWER.
    Salva os dados coletados em arquivos Parquet.
    Estações sem dados de vazão ou dados climáticos são ignoradas, ou seja, não são salvas no dataset.
    """
    logger.info('Iniciando a coleta de dados das estações e dados climáticos da NASA POWER.')


    df_stations = coletor_estacao.fetch_stations()
    files = []

    for station in df_stations.itertuples():
        try:
            station_id = station.co_seq_estacao

            logger.info(f'Coletando dados para a estação {station_id}')

            if station.primeira_vazao is None or station.ultima_vazao is None:
                logger.warning(f'Ignorando estação {station_id} por falta de dados de vazão.')
                continue
            
            latitude = station.latitude
            longitude = station.longitude
            start_date = station.primeira_vazao.strftime('%Y%m%d')
            end_date = station.ultima_vazao.strftime('%Y%m%d')
        except Exception as e:
            logger.error(f'Erro ao processar a estação {station}: {e}. Provavelmente as colunas estão com nomes inesperados.')

        df_clima = coletor_nasa.get_data(
            lon=longitude,
            lat=latitude,
            start=start_date,
            end=end_date
        )

        if df_clima is None:
            logger.warning(f'Ignorando estação {station[0]} por falta de dados climáticos na NASA POWER.')
            continue

        # Processa e salva os dados da NASA coletados
        df_clima['cod_estacao'] = station_id
        filename_clima = f'{diretorio_saida}dados_climatico_{station_id}_{date.today()}.parquet'
        df_clima.to_parquet(filename_clima)

        # Coleta e salva os dados de vazão da estação
        df_vazao = coletor_estacao.collect_data_of_station(station_id)
        filename_estacoes = f'{diretorio_saida}dados_vazao_{station_id}_{date.today()}.parquet'
        df_vazao.to_parquet(filename_estacoes)

        files.append((filename_estacoes, filename_clima))

    return files
