## Previsao de Vazao com LSTM + Coleta Climatica

Pipeline completo para coletar dados hidrometricos, enriquecer com clima/historico/previsao, tratar features e treinar um modelo LSTM final para previsao diaria de vazao.

### Arquitetura
- `src/main.py`: orquestrador end-to-end (coleta -> tratamento -> treino).
- `src/process/coletar_dados.py`: coordena coletores e salva parquets brutos em `data_lake/raw`.
- `src/process/coletor_dados_estacao.py`: busca vazao e metadados em PostgreSQL.
- `src/process/coletor_dados_climaticos.py`: coleta historico NASA POWER e previsao Open-Meteo.
- `src/process/tratamento_dados.py`: pipeline de limpeza, interpolacao, lags, janelas, sazonalidade.
- `src/models/lstm_vazao.py`: treino/carga/predict do modelo final (hiperparametros do tuner).
- Notebooks em `notebooks/`: exploracao e tuning (ex.: `02_treinamento_modelo.ipynb`).

### Dependencias
- Python 3.11+ (recomendado).
- PostgreSQL com esquema contendo tabelas `tb_estacao`, `tb_resumo_mensal`, `tb_vazao_diaria`, `tb_cidade`, `tb_estado`, `tb_rio` populadas.
- Acesso HTTP para APIs NASA POWER e Open-Meteo.

Instalacao rapida:
```bash
python -m venv .venv
source .venv/bin/activate  # no Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Configuracao
Credenciais/hosts do banco estao definidas em `src/main.py` via `_build_collectors`. Ajuste para seu ambiente:
```python
banco_vazao = EstacaoDataCollector(
    user="postgres",
    password="admin",
    host="localhost",
    port=5432,
    database="banco_vazao",
)
```
Se preferir, altere para ler de variaveis de ambiente antes de instanciar o collector.

### Como rodar o pipeline completo
Gera brutos em `data_lake/raw`, processados em `data_lake/processed` e modelos em `models/estacao_*`:
```bash
PYTHONPATH=src python -m src.main
```

### Etapas do pipeline
1) **Coleta** (`_collect_raw_files`):
   - Para cada estacao valida, baixa clima historico da NASA POWER e previsao do Open-Meteo.
   - Exporta parquets `dados_vazao_*`, `dados_climatico_*`, `dados_previsao_*` em `data_lake/raw/`.
2) **Tratamento** (`_process_pairs`):
   - Limpeza de vazao (datas min 1981, deduplicacao, interpolacao controlada).
   - Correcao de clima (negativos corrigidos por media movel defasada).
   - Engenheira de atributos: lags, janelas 3/6/9 dias, sazonalidade seno/cosseno e dummies de estacao.
   - Salva parquet processado por estacao em `data_lake/processed/`.
3) **Treino** (`_train_models`):
   - Separa holdout (20%), treina LSTM com early stopping usando hiperparametros de `BEST_PARAMS` em `src/models/lstm_vazao.py`.
   - Salva pesos `.keras` e artefatos (scalers/colunas/janela) em `models/estacao_*`.
   - Loga metricas MAE, RMSE e R2 do holdout.

### Reutilizando modelo treinado
Exemplo de carga e previsao incremental:
```python
from pathlib import Path
import pandas as pd
from src.models.lstm_vazao import load_final_lstm, predict_with_final_lstm

model, artifacts = load_final_lstm(model_dir=Path("models/estacao_9724"))
df_novo = pd.read_parquet("data_lake/processed/dados_processados_9724_YYYY-MM-DD.parquet")
preds = predict_with_final_lstm(df_novo, model, artifacts)
print(preds.tail())
```

### Tuning e notebooks
- `notebooks/02_treinamento_modelo.ipynb` traz o tuning com Bayesian Optimization, suporte opcional a camada CNN e reproducao do treino final. Rode o notebook se quiser explorar outro espaco de hiperparametros.

### Estrutura de diretorios (gerados)
- `data_lake/raw/`: parquets brutos de vazao/clima/previsao por estacao.
- `data_lake/processed/`: parquets tratados e enriquecidos.
- `models/estacao_*`: pesos `.keras` e `lstm_vazao_scalers.pkl` por estacao.

### Observacoes
- O pipeline assume conectividade com o banco e as APIs; falhas de rede ou esquemas divergentes geram skips no log.
- A interpolacao de vazao aborta se mais de 20% da serie ficar preenchida artificialmente (veja `INTERPOLATION_NULL_THRESHOLD`).
- Ajuste `COLUNAS_ALVO` em `src/main.py` se adicionar/remover features no tratamento.
- Para reduzir custo computacional, é possível reduzir o pipeline somente a estação de uma bacia específica. Em `src/process/coletor_dados_estacao.py` ajuste `FILTER_BACIA` para true e adiciona o códico da bacia em `COD_BACIA`.
