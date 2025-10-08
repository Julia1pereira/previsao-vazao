# Script: coletar_dados.py
# Descrição: baixa séries históricas da NAZA com as precipitações diárias.

from io import StringIO
import requests
import pandas as pd

def coletar_dados_nasa(lat, lon, start, end):
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECOT,T2M,RH2M&community=RE&longitude={lon}&latitude={lat}&start={start}&end={end}&format=CSV"
    response = requests.get(url)
    
    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text), skiprows=11)

        # muda o formato da data para YYYY-MM-DD e coloca como index
        df['data'] = pd.to_datetime(df['YEAR'].astype(str) + df['DOY'].astype(str), format='%Y%j')
        df.set_index('data', inplace=True)

        # renomea colunas e selecionar apenas as necessárias
        df.rename(columns={'PRECOT': 'precipitacao', 'T2M': 'temp_media', 'RH2M': 'umidade_relativa'}, inplace=True)
        df = df[['precipitacao', 'temp_media', 'umidade_relativa']]
        
        return df
    else:
        print("Erro ao baixar os dados:", response.status_code)
        return None

# fazer um for para cada estação do sua latitude e longitude, puxando a data mínima e máxima de coleta de dados