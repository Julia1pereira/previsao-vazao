"""Modelo LSTM final para previsão de vazão diária.

Baseado no notebook de treinamento (02_treinamento_modelo.ipynb) usando os melhores
hiperparâmetros encontrados no tuner.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Hiperparâmetros finais obtidos no tuner
BEST_PARAMS = {
    "lstm_units": 112,
    "num_layers": 1,  # mantido para documentação; o modelo final usa uma camada
    "dropout": 0.1,
    "dense_units": 80,
    "learning_rate": 0.001,
}

WINDOW = 30
TARGET_COLUMN = "vazao"
MODEL_COLUMNS = [
    "vazao",
    "precipitacao",
    "temp_media",
    "umidade_relativa",
    "vazao_t1",
    "umidade_t1",
    "vazao_roll3_mean",
    "vazao_roll3_sum",
    "vazao_roll6_mean",
    "vazao_roll6_sum",
    "vazao_roll9_mean",
    "vazao_roll9_sum",
    "precipitacao_roll3_mean",
    "precipitacao_roll3_sum",
    "precipitacao_roll6_mean",
    "precipitacao_roll6_sum",
    "precipitacao_roll9_mean",
    "precipitacao_roll9_sum",
    "umidade_relativa_roll3_mean",
    "umidade_relativa_roll3_sum",
    "umidade_relativa_roll6_mean",
    "umidade_relativa_roll6_sum",
    "umidade_relativa_roll9_mean",
    "umidade_relativa_roll9_sum",
    "mes",
    "mes_sin",
    "mes_cos",
    "estacao_inverno",
    "estacao_outono",
    "estacao_primavera",
    "estacao_verao",
]

MODEL_FILENAME = "lstm_vazao.keras"
ARTIFACTS_FILENAME = "lstm_vazao_scalers.pkl"


def _sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante ordenação temporal sem mutar o dataframe de entrada, usando a coluna data
    (ou o índice datetime) para ordenar e definir o índice temporal.
    """
    if "data" in df.columns:
        ordered = df.copy()
        ordered["data"] = pd.to_datetime(ordered["data"])
        return ordered.sort_values("data").set_index("data")

    if pd.api.types.is_datetime64_any_dtype(df.index):
        return df.sort_index()

    return df.copy()


def _select_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Valida presença das colunas exigidas pelo modelo e retorna apenas essas colunas."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes para o modelo LSTM: {missing}")
    return df[list(columns)]


def _prepare_dataframe(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Ordena, filtra colunas do modelo e descarta linhas com alvo nulo."""
    ordered = _sort_by_date(df)
    filtered = _select_columns(ordered, columns)
    return filtered.dropna(subset=[TARGET_COLUMN])


def _scale_dataframe(
    df: pd.DataFrame, scaler: MinMaxScaler | None = None
) -> Tuple[pd.DataFrame, MinMaxScaler, MinMaxScaler]:
    """
    Escala features com MinMax; se scaler for None, faz fit+transform e gera target_scaler
    separado para a coluna alvo. Retorna dataframe escalado, scaler de features e scaler do alvo.
    """
    if scaler is None:
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(df)
    else:
        scaled_values = scaler.transform(df)

    scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)
    target_scaler = MinMaxScaler().fit(df[[TARGET_COLUMN]])

    return scaled_df, scaler, target_scaler


def _create_lstm_dataset(data: pd.DataFrame, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constrói janelas deslizantes (X) e alvo (y) para LSTM com janela fixa; lança erro
    se não houver dados suficientes para ao menos uma janela completa.
    """
    target_idx = data.columns.get_loc(TARGET_COLUMN)
    X, y = [], []

    values = data.values
    for i in range(window, len(values)):
        X.append(values[i - window : i])
        y.append(values[i, target_idx])

    if not X:
        raise ValueError("Dados insuficientes para criar janelas LSTM.")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_final_model(input_shape: Tuple[int, int]) -> Sequential:
    """Monta e compila o LSTM final com hiperparâmetros definidos em BEST_PARAMS."""
    model = Sequential()
    model.add(
        LSTM(
            units=BEST_PARAMS["lstm_units"],
            activation="tanh",
            return_sequences=False,
            input_shape=input_shape,
        )
    )
    model.add(Dropout(BEST_PARAMS["dropout"]))
    model.add(Dense(BEST_PARAMS["dense_units"], activation="relu"))
    model.add(Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BEST_PARAMS["learning_rate"]),
        loss="mse",
    )
    return model


def train_final_lstm(
    df: pd.DataFrame, model_dir: str | Path = "models"
) -> Dict[str, float]:
    """
    Treina o modelo final: prepara dados, escala, cria janelas, separa holdout,
    treina com early stopping, calcula métricas em teste, salva pesos e artefatos.
    """
    prepared = _prepare_dataframe(df, MODEL_COLUMNS)
    scaled_df, scaler, target_scaler = _scale_dataframe(prepared)
    X, y = _create_lstm_dataset(scaled_df, window=WINDOW)

    split = int(len(X) * 0.8)
    if split == 0 or split >= len(X):
        raise ValueError("Volume de dados insuficiente para split de treino e teste.")

    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = build_final_model((WINDOW, scaled_df.shape[1]))
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_test_real = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_real = target_scaler.inverse_transform(y_pred_scaled).flatten()

    mse_value = mean_squared_error(y_test_real, y_pred_real)
    metrics = {
        "mae": float(mean_absolute_error(y_test_real, y_pred_real)),
        "mse": float(mse_value),
        "rmse": float(np.sqrt(mse_value)),
        "r2": float(r2_score(y_test_real, y_pred_real)),
    }

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / MODEL_FILENAME)

    artifacts = {
        "scaler": scaler,
        "target_scaler": target_scaler,
        "columns": prepared.columns.tolist(),
        "window": WINDOW,
    }

    with open(model_dir / ARTIFACTS_FILENAME, "wb") as f:
        pickle.dump(artifacts, f)

    return metrics


def load_final_lstm(model_dir: str | Path = "models") -> Tuple[Sequential, Dict]:
    """Carrega pesos (.keras) e artefatos (scalers/cols/janela) do diretório informado."""
    model_dir = Path(model_dir)
    model = tf.keras.models.load_model(model_dir / MODEL_FILENAME)

    with open(model_dir / ARTIFACTS_FILENAME, "rb") as f:
        artifacts = pickle.load(f)

    return model, artifacts


def predict_with_final_lstm(
    df: pd.DataFrame, model: Sequential, artifacts: Dict
) -> pd.Series:
    """
    Usa artefatos (scalers, colunas, janela) e o modelo carregado para gerar previsões
    em série temporal, retornando a sequência desnormalizada alinhada pelo índice.
    """
    columns = artifacts["columns"]
    window = artifacts["window"]
    scaler: MinMaxScaler = artifacts["scaler"]
    target_scaler: MinMaxScaler = artifacts["target_scaler"]

    prepared = _prepare_dataframe(df, columns)
    scaled_df, _, _ = _scale_dataframe(prepared, scaler=scaler)
    X, _ = _create_lstm_dataset(scaled_df, window=window)

    preds_scaled = model.predict(X, verbose=0)
    preds = target_scaler.inverse_transform(preds_scaled).flatten()
    prediction_index = prepared.index[window:]

    return pd.Series(preds, index=prediction_index, name="vazao_predita")
