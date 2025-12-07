"""API para servir previsões de vazão usando o modelo LSTM final.

Gera previsões para os próximos 7 dias a partir de hoje. Se faltarem valores de
vazão recentes, usa a média histórica (mesmo dia/mês em anos anteriores) ou
aceita valores enviados pelo usuário.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel

from src.models.lstm_vazao import MODEL_FILENAME, load_final_lstm


class PrevisaoRequest(BaseModel):
    estacao: int
    vazoes_passadas: Optional[List[float]] = None  # ordem antiga->recente (termina em ontem)


app = FastAPI()

PROCESSED_DIR = Path("data_lake/processed")
MODELS_DIR = Path("models")
VAZAO_COL = "vazao"

def _latest_parquet(estacao: int) -> Path:
    arquivos = sorted(PROCESSED_DIR.glob(f"dados_processados_{float(estacao)}_2025-11-27.parquet"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum parquet processado encontrado para estacao {estacao}")
    return arquivos[-1]


def _load_dataset(estacao: int) -> pd.DataFrame:
    path = _latest_parquet(estacao)
    df = pd.read_parquet(path)
    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"])
        df = df.set_index("data").sort_index()
    elif not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("Dataset sem coluna/índice de data.")
    return df.sort_index()


def _seasonal_mean(df: pd.DataFrame, ref_date: pd.Timestamp) -> float:
    mask = (df.index.month == ref_date.month) & (df.index.day == ref_date.day) & (df.index < ref_date)
    serie = df.loc[mask, VAZAO_COL].dropna()
    if not serie.empty:
        return float(serie.mean())
    serie_total = df[VAZAO_COL].dropna()
    if not serie_total.empty:
        return float(serie_total.mean())
    return 0.0


def _inject_user_vazoes(df: pd.DataFrame, vazoes: List[float], today: pd.Timestamp) -> None:
    if not vazoes:
        return
    start_date = today - pd.Timedelta(days=len(vazoes))
    for idx, valor in enumerate(vazoes):
        dia = start_date + pd.Timedelta(days=idx)
        df.loc[dia, VAZAO_COL] = valor


def _fill_recent_missing(df: pd.DataFrame, today: pd.Timestamp) -> None:
    recent_mask = (df.index < today) & df[VAZAO_COL].isna()
    missing_dates = df.index[recent_mask]
    for dia in missing_dates:
        df.loc[dia, VAZAO_COL] = _seasonal_mean(df, dia)


def _date_features(d: pd.Timestamp) -> Dict[str, float]:
    mes = d.month
    return {
        "mes": mes,
        "mes_sin": math.sin(2 * math.pi * mes / 12),
        "mes_cos": math.cos(2 * math.pi * mes / 12),
        "estacao_inverno": 1.0 if date(d.year, 6, 21) <= d.date() < date(d.year, 9, 22) else 0.0,
        "estacao_outono": 1.0 if date(d.year, 3, 20) <= d.date() < date(d.year, 6, 21) else 0.0,
        "estacao_primavera": 1.0 if date(d.year, 9, 22) <= d.date() < date(d.year, 12, 21) else 0.0,
        "estacao_verao": 1.0 if d.date() >= date(d.year, 12, 21) or d.date() < date(d.year, 3, 20) else 0.0,
    }


def _ensure_row(df: pd.DataFrame, dia: pd.Timestamp) -> None:
    if dia in df.index:
        return
    base = df.iloc[-1].copy()
    base[VAZAO_COL] = np.nan
    for k, v in _date_features(dia).items():
        if k in base.index:
            base[k] = v
    df.loc[dia] = base


def _recompute_vazao_features(df: pd.DataFrame) -> None:
    df.sort_index(inplace=True)
    df["vazao_t1"] = df[VAZAO_COL].shift(1)
    for w in [3, 6, 9]:
        df[f"vazao_roll{w}_mean"] = df[VAZAO_COL].rolling(w).mean()
        df[f"vazao_roll{w}_sum"] = df[VAZAO_COL].rolling(w).sum()


def _load_model(estacao: int):
    model_dir = MODELS_DIR / f"estacao_{estacao}.0"
    if not (model_dir / MODEL_FILENAME).exists():
        raise FileNotFoundError(f"Modelo não encontrado para estacao {estacao}")
    return load_final_lstm(model_dir)


def _prepare_for_forecast(df: pd.DataFrame, vazoes_user: Optional[List[float]]) -> pd.DataFrame:
    hoje = pd.Timestamp.today().normalize()
    df = df.copy()
    _inject_user_vazoes(df, vazoes_user or [], hoje)
    _fill_recent_missing(df, hoje)
    _recompute_vazao_features(df)
    return df


def _predict_next_days(df: pd.DataFrame, model, artifacts, horizon: int = 7) -> List[Dict[str, float]]:
    cols = artifacts["columns"]
    scaler = artifacts["scaler"]
    target_scaler = artifacts["target_scaler"]
    window = artifacts["window"]

    df = df.copy().sort_index()
    hoje = pd.Timestamp.today().normalize()
    resultados = []

    for offset in range(0, horizon):
        dia = hoje + pd.Timedelta(days=offset)
        _ensure_row(df, dia)

        window_df = df.loc[df.index < dia].tail(window)
        if len(window_df) < window:
            raise HTTPException(status_code=400, detail="Dados insuficientes para janela do modelo.")

        window_slice = window_df[cols]
        if window_slice.isnull().any().any():
            raise HTTPException(
                status_code=400,
                detail=f"Existem valores faltantes na janela antes de {dia.date()}",
            )

        window_scaled = scaler.transform(window_slice)
        pred_scaled = model.predict(window_scaled[np.newaxis, ...], verbose=0)
        pred = float(target_scaler.inverse_transform(pred_scaled)[0, 0])

        df.at[dia, VAZAO_COL] = pred
        _recompute_vazao_features(df)

        resultados.append({"data": dia.date().isoformat(), "vazao_prevista": pred})

    return resultados


@app.post("/previsao")
def get_estacao_prediction(payload: PrevisaoRequest):
    try:
        df = _load_dataset(payload.estacao)
        model, artifacts = _load_model(payload.estacao)
        df_prep = _prepare_for_forecast(df, payload.vazoes_passadas)
        previsoes = _predict_next_days(df_prep, model, artifacts, horizon=7)
        return {"estacao": payload.estacao, "previsoes": previsoes}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - ambiente externo
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8081"))
    uvicorn.run(app, host="0.0.0.0", port=port)
