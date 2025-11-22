from process.coletar_dados import collect_data
from process.coletor_dados_estacao import EstacaoDataCollector
from process.coletor_dados_climaticos import NasaPowerCollector
from process.tratamento_dados import transformar_dataframe
import os
import shutil


banco_vazao = EstacaoDataCollector(
    user='postgres',
    password='admin',
    host='localhost',
    port=5432,
    database='banco_vazao'
)

nasa_collector = NasaPowerCollector(
    interval='daily',
    parameters='T2M,PRECTOT,RH2M',
    community='AG'
)

diretorio_saida_raw = './data_lake/raw/'
diretorio_saida_processado = './data_lake/processed/'

# # limpa o diretório de saída antes de salvar novos arquivos
# if os.path.exists(diretorio_saida_raw):
#     shutil.rmtree(diretorio_saida_raw)
# os.makedirs(diretorio_saida_raw)

# files = collect_data(nasa_collector, banco_vazao, diretorio_saida_raw)

files = [('./data_lake/raw/dados_vazao_3936_2025-11-17.parquet', './data_lake/raw/dados_climatico_3936_2025-11-17.parquet'),]

for file_vazao, file_clima in files:
    print(f'Arquivo de dados climáticos salvo em: {file_clima}')
    print(f'Arquivo de dados de vazão salvo em: {file_vazao}')

    colunas_alvo = ["vazao", "precipitacao", "temp_media", "umidade_relativa", "codigo_bacia", "codigo_sub_bacia", "co_estacao", "latitude", "longitude", "cidade", "estado", "rio"]
    df = transformar_dataframe(file_vazao, file_clima, colunas_alvo, diretorio_saida_processado)

