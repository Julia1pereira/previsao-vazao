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

    def _get_base_url(self):
        return f'https://power.larc.nasa.gov/api/temporal/{self.interval}/point'
    
    def _check_date(self, date_str):
        if not isinstance(date_str, str) and len(date_str) != 8:
            raise ValueError("Data deve estar no formato 'YYYYMMDD'.")

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
            A API NASA POWER não possui dados antes de 1981.
        """
        url = self._get_base_url()

        self._check_date(start)
        self._check_date(end)

        if start < '19810101':
            raise ValueError("A data de início não pode ser anterior a 1981-01-01.")

        if end <= start:
            raise ValueError("A data de fim não pode ser anterior à data de início.")

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


class OpenMeteoCollector:
    """
    Coletor da API Open-Meteo para dados diários.

    args:
        daily_parameters (list[str]): Lista de variáveis diárias a coletar.
    """

    def __init__(self, daily_parameters=None):
        self.daily_parameters = daily_parameters or [
            "temperature_2m_mean",
            "precipitation_sum",
            "relative_humidity_2m_mean",
        ]
        self.timezone = "America/Sao_Paulo"

    def _get_base_url(self):
        return "https://api.open-meteo.com/v1/forecast"

    def get_data(self, lon, lat, **kwargs):
        """
        Coleta dados climáticos diários do Open-Meteo.

        args:
            lon (float): Longitude do ponto.
            lat (float): Latitude do ponto.
            start (str): Data inicial no formato 'YYYYMMDD'.
            end (str): Data final no formato 'YYYYMMDD'.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(self.daily_parameters),
            "timezone": self.timezone,
        }

        params.update(kwargs)

        try:
            response = requests.get(self._get_base_url(), params=params)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Erro ao baixar os dados: {e}")

        return self.create_dataframe(response.json())

    def create_dataframe(self, data):
        daily = data.get("daily")
        if not daily or "time" not in daily:
            raise ValueError("Resposta da API Open-Meteo inválida ou sem dados diários.")

        df = pd.DataFrame(daily)
        df["data"] = pd.to_datetime(df["time"])

        coluna_map = {
            "precipitation_sum": "precipitacao",
            "temperature_2m_mean": "temp_media",
            "relative_humidity_2m_mean": "umidade_relativa",
        }

        rename_dict = {coluna: coluna_map[coluna] for coluna in df.columns if coluna in coluna_map}
        df.rename(columns=rename_dict, inplace=True)

        colunas_finais = [col for col in ("precipitacao", "temp_media", "umidade_relativa") if col in df.columns]
        colunas_finais.append("data")

        return df[colunas_finais]
