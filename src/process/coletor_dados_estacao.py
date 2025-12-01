import sqlalchemy
import pandas as pd

FILTER_BACIA = True
COD_BACIA = 8

class EstacaoDataCollector:
    def __init__(self, user: str, password: str, host: str, port: int, database: str):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.driver = "psycopg2"
        self.engine = sqlalchemy.create_engine(self._make_connection_string())

    def _make_connection_string(self):
        """
        Gera a URL de conexão para um banco PostgreSQL compatível com SQLAlchemy.

        Exemplo:
            postgresql+psycopg2://usuario:senha@localhost:5432/banco

        Args:
            user (str): Usuário do banco de dados.
            password (str): Senha do banco de dados.
            host (str): Endereço do servidor (localhost, IP ou domínio).
            port (int): Porta do banco (default: 5432).
            database (str): Nome do banco de dados.
            driver (str): Driver usado pelo SQLAlchemy (default: 'psycopg2').

        Returns:
            str: URL completa de conexão.
        """

        return f"postgresql+{self.driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def fetch_stations(self):
        with self.engine.connect() as connection:
            where_clauses = []
            params = {}

            if FILTER_BACIA:
                where_clauses.append("te.codigo_bacia = :codigo_bacia")
                params["codigo_bacia"] = COD_BACIA

            where_clauses.append("te.operando = 1")
            where_sql = " and ".join(where_clauses)

            query = sqlalchemy.text(
                f"""
                    select te.co_seq_estacao, te.nome, te.latitude, te.longitude,
                           min(tvd.data_vazao) as primeira_vazao, max(tvd.data_vazao) as ultima_vazao
                    from tb_estacao te
                    full join tb_resumo_mensal trm on te.co_seq_estacao = trm.co_estacao
                    full join tb_vazao_diaria tvd on trm.co_seq_resumo_mensal = tvd.co_resumo_mensal
                    where {where_sql}
                    group by co_seq_estacao , nome, latitude, longitude;
                """
            )
            result = connection.execute(query, params)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df

    def collect_data_of_station(self, station_id):
        with self.engine.connect() as connection:
            query = sqlalchemy.text(
                """
                    select tvd.data_vazao, tvd.vazao, te.codigo_bacia, te.codigo_sub_bacia,
                           trm.co_estacao, te.latitude, te.longitude, tc.nome as cidade,
                           te2.nome as estado, tr.nome as rio
                    from tb_vazao_diaria tvd
                    full join tb_resumo_mensal trm on tvd.co_resumo_mensal = trm.co_seq_resumo_mensal
                    full join tb_estacao te on trm.co_estacao = te.co_seq_estacao
                    full join tb_cidade tc on te.co_cidade = tc.co_seq_cidade 
                    full join tb_estado te2 on tc.co_estado = te2.co_seq_estado
                    full join tb_rio tr on te.co_rio = tr.co_seq_rio
                    where te.co_seq_estacao = :station_id
                    order by tvd.data_vazao;
                """
            )
            result = connection.execute(query, {"station_id": station_id})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
