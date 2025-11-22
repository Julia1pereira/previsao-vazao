import requests
from io import StringIO
import pandas as pd

class NasaPowerCollector:
    """
    Classe para coletar dados da API NASA POWER.

    args:
        interval (str): Intervalo de tempo dos dados (e.g., 'daily', 'monthly').
        parameters (str): Parâmetros a serem coletados, separados por vírgula (e.g., 'T2M,PRECTOT').
        community (str): Comunidade de usuários (e.g., 'AG' para agricultura).
    """
    def __init__(self, interval, parameters, community):
        self.interval = interval
        self.parameters = parameters
        self.community = community
        self.format = "CSV"
        self.units = "metric"
        self.user = "user"
        self.header = True
        self.time_standard = "utc"

    def get_base_url(self):
        return f'https://power.larc.nasa.gov/api/temporal/{self.interval}/point'

    def get_data(self, lon, lat, start, end, **kwargs):
        """
        Coleta os dados da API NASA POWER.

        args:
            lon (float): Longitude do ponto de interesse.
            lat (float): Latitude do ponto de interesse.
            start (str): Data de início no formato 'YYYYMMDD'.
            end (str): Data de fim no formato 'YYYYMMDD'.
            **kwargs: Parâmetros adicionais para a requisição.

        note:
            A API NASA POWER não possui dados antes de 1981, então se a data de início for anterior, ela será ajustada para '19810101'.
            Se a data de fim também for menor, retorna None
        """
        url = self.get_base_url()

        if start < '19810101':
            start = '19810101'
            if end < start:
                return None

        params = {
            'parameters': self.parameters,
            'community': self.community,
            'longitude': lon,
            'latitude': lat,
            'start': start,
            'end': end,
            'format': self.format,
            'units': self.units,
            'user': self.user,
            'header': self.header,
            'time-standard': self.time_standard,
        }

        params.update(kwargs)

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Erro ao baixar os dados: {e}")

        return self.create_dataframe(response)

    def create_dataframe(self, response):
        df = pd.read_csv(StringIO(response.text), skiprows=11)

        df['data'] = pd.to_datetime(df['YEAR'].astype(str) + df['DOY'].astype(str), format='%Y%j')

        coluna_map = {
            'PRECTOTCORR': 'precipitacao',
            'T2M': 'temp_media',
            'RH2M': 'umidade_relativa',
        }

        rename_dict = {coluna: coluna_map[coluna] for coluna in df.columns if coluna in coluna_map}

        df.rename(columns=rename_dict, inplace=True)

        colunas_finais = list(rename_dict.values()) + ['data']
        df = df[colunas_finais]

        return df
