from coletar_dados import NasaPowerCollector
import pandas as pd
from io import StringIO

class DataProcessorNASA(NasaPowerCollector):
    def __init__(self):
        self.interval = 'daily'
        self.parameters = 'T2M,PRECTOT,RH2M'
        self.community = 'AG'

    def process(self, response):
        # Exemplo de processamento: preencher valores ausentes com a média
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text), skiprows=11)

            # muda o formato da data para YYYY-MM-DD e coloca como index
            df['data'] = pd.to_datetime(df['YEAR'].astype(str) + df['DOY'].astype(str), format='%Y%j')
            df.set_index('data', inplace=True)

            # renomea colunas e selecionar apenas as necessárias
            df.rename(columns={'PRECTOT': 'precipitacao', 'T2M': 'temp_media', 'RH2M': 'umidade_relativa'}, inplace=True)
            df = df[['precipitacao', 'temp_media', 'umidade_relativa']]
            
            return df
        else:
            raise RuntimeError(f"Erro ao baixar os dados: {response.status_code}")

    def save_to_parquet(self, df, path, filename):
        import os
        try:
            df.to_parquet(os.path.join(path, filename), index=True)
        except Exception as e:
            print(f"Erro ao salvar o arquivo parquet: {e}")

class DataProcessorEstacao:
    def __init__(self):
        self.user = 'postgres'
        self.password = 'admin'
        self.host = 'localhost'
        self.port = '5432'
        self.database = 'banco_vazao'

    def process(self, data):
        # eliminar os buracos de cada estação
        pass