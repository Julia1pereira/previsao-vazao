# Plano de Implementação do Projeto

Este documento descreve o planejamento técnico do desenvolvimento do sistema de previsão de vazão de rios brasileiros utilizando Inteligência Artificial.

## 1. Objetivo Geral

Construir um modelo de aprendizado de máquina capaz de prever a vazão diária de rios brasileiros a partir de dados hidrológicos (ANA) e climáticos (NASA POWER / Open-Meteo), integrando-o a uma interface web funcional.

## 2. Arquitetura Geral

A solução será composta por:

- **Pipeline de coleta de dados**
  - Hidrológicos (ANA – vazão diária)
  - Climáticos (NASA POWER e Open-Meteo)
- **Pipeline de processamento**
  - Limpeza, padronização e criação de features derivadas
- **Modelos de IA**
  - Baseline: Regressão Linear / Random Forest
  - Rede Neural: LSTM
- **API/Backend**
  - Servirá previsões a partir do modelo treinado
- **Frontend**
  - Visualização de histórico e previsões

## 3. Etapas do Desenvolvimento

### 3.1 Coleta de Dados
- Implementar rotina de extração da vazão diária via ANA.
- Coletar variáveis climáticas: precipitação, temperatura e umidade.
- Criar scripts para unificar os conjuntos de dados.

### 3.2 Preparação dos Dados
- Tratamento de valores faltantes.
- Padronização temporal.
- Feature engineering com:
  - Lags (t−1, t−3…)
  - Médias móveis (3, 6, 9 dias)
  - Somas acumuladas
- Normalização / padronização.

### 3.3 Modelagem
- Treinar modelos baseline para comparação.
- Treinar LSTM com janelas temporais (windows).
- Realizar validação cruzada temporal (TimeSeriesSplit).
- Avaliar com métricas: MAE, RMSE, MAPE.

### 3.4 Deploy do Modelo
- Salvar modelo final em formato `.pkl` ou `.h5`.
- Criar endpoint REST (`/prever`) recebendo dados climáticos e retornando vazão estimada.

### 3.5 Interface Web
- Página para:
  - Visualização do histórico da vazão (já implementada).
  - Consultar previsões para os próximos dias.
- Comunicação via API.

## 4. Tecnologias Utilizadas

- **Python** (Pandas, NumPy, Scikit-Learn, TensorFlow/Keras)
- **Open-Meteo API** (previsão climática)
- **NASA POWER API** (dados históricos)
- **ANA Hidroweb** (vazão)
- **FastAPI** para backend

## 5. Cronograma (alinhado ao Plano de Ensino)

| Semana | Tarefa |
|-------|--------|
| 1–3   | Definição do problema e levantamento de dados |
| 4–7   | Coleta + limpeza + criação das features |
| 8–11  | Modelagem e testes experimentais |
| 12–14 | Implementação da API e integração com o frontend |
| 15–16 | Avaliação, resultados e relatório final |
| 17    | Preparação para apresentação |
| 18    | Finalização e entrega |

## 6. Resultados Esperados

- Previsões de vazão com bom desempenho (erro aceitável para uso prático).
- Sistema funcional acessível via navegador.
- Relatório científico completo conforme exigido na disciplina.
