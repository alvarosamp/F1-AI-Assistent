"""IA local de engenheiro de corrida: identifica o carro da frente, recomenda
estratégia de pit/pneu e gera um briefing em linguagem natural com um modelo
HuggingFace rodando localmente (transformers, sem chamada de rede/API paga),
com fallback determinístico se o modelo não puder ser carregado/rodar a tempo.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Modelo pequeno o suficiente para rodar em CPU com latência razoável.
# Troque via env var HF_MODEL_ID por outro modelo instruct do HuggingFace Hub.
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
HF_MAX_NEW_TOKENS = int(os.environ.get("HF_MAX_NEW_TOKENS", "320"))

_pit_model = None
_tyre_model = None
_hf_pipeline = None


def _load_pit_model():
    global _pit_model
    if _pit_model is None:
        _pit_model = joblib.load(MODELS_DIR / "pit_model.pkl")
    return _pit_model


def _load_tyre_model():
    global _tyre_model
    if _tyre_model is None:
        _tyre_model = joblib.load(MODELS_DIR / "tyre_deg_model.pkl")
    return _tyre_model


def find_driver_ahead(session, driver: str, lap_number: int) -> dict | None:
    """Quem está na posição imediatamente à frente do piloto naquela volta
    (por Position real da corrida/sessão, não por gap instantâneo de telemetria)."""
    lap_rows = session.laps[session.laps["LapNumber"] == lap_number].dropna(subset=["Position"])
    if lap_rows.empty:
        return None
    ordered = lap_rows.sort_values("Position")
    drivers_in_order = ordered["Driver"].tolist()
    driver = driver.upper()
    if driver not in drivers_in_order:
        return None

    idx = drivers_in_order.index(driver)
    if idx == 0:
        return {"driver_ahead": None, "position": 1, "is_leader": True}

    ahead_driver = drivers_in_order[idx - 1]
    ahead_row = ordered.iloc[idx - 1]
    our_row = ordered.iloc[idx]
    gap_s = None
    if pd.notna(ahead_row.get("Time")) and pd.notna(our_row.get("Time")):
        gap_s = round((our_row["Time"] - ahead_row["Time"]).total_seconds(), 3)

    return {
        "driver_ahead": ahead_driver,
        "position": idx + 1,
        "is_leader": False,
        "gap_s": gap_s,
    }


def recommend_strategy(session, driver: str, lap_number: int) -> dict:
    laps = session.laps.pick_drivers(driver.upper())
    upto = laps[laps["LapNumber"] <= lap_number].sort_values("LapNumber")
    if upto.empty:
        raise ValueError(f"Nenhuma volta encontrada para {driver} até a volta {lap_number}.")

    current = upto.iloc[-1]
    tyre_life = float(current.get("TyreLife") or 0.0)
    compound = current.get("Compound")
    lap_no = int(current.get("LapNumber") or lap_number)

    last3 = upto.tail(3)
    lap_times_s = last3["LapTime"].dropna().dt.total_seconds()
    lap_time_mean_3 = float(lap_times_s.mean()) if not lap_times_s.empty else None

    lap_time_delta = None
    if len(upto) >= 2:
        prev_lt = upto.iloc[-2]["LapTime"]
        curr_lt = current["LapTime"]
        if pd.notna(prev_lt) and pd.notna(curr_lt):
            lap_time_delta = round((curr_lt - prev_lt).total_seconds(), 3)

    tyre_ratio = tyre_life / max(lap_no, 1)

    pit_probability = None
    try:
        model = _load_pit_model()
        features = [[tyre_life, lap_time_mean_3 or 0.0, lap_time_delta or 0.0, tyre_ratio]]
        pit_probability = round(float(model.predict_proba(features)[0][1]), 4)
    except Exception:
        pit_probability = None

    pace_trend_s_per_lap = None
    try:
        tyre_model = _load_tyre_model()
        coefs = tyre_model["coefficients"].get(str(compound).upper())
        if coefs:
            pace_trend_s_per_lap = round(float(coefs["coef_per_lap"]), 4)
    except Exception:
        pace_trend_s_per_lap = None

    if pit_probability is not None and pit_probability >= 0.6:
        action = "pit_now"
    elif pit_probability is not None and pit_probability >= 0.35:
        action = "pit_soon"
    elif tyre_life >= 22:
        action = "pit_soon"
    else:
        action = "stay_out"

    return {
        "driver": driver.upper(),
        "lap_number": lap_no,
        "compound": compound,
        "tyre_life": tyre_life,
        "lap_time_mean_3_s": round(lap_time_mean_3, 3) if lap_time_mean_3 is not None else None,
        "lap_time_delta_s": lap_time_delta,
        "pit_probability": pit_probability,
        "pace_trend_s_per_lap": pace_trend_s_per_lap,
        "recommended_action": action,
    }


ACTION_LABELS = {
    "pit_now": "Boxes agora",
    "pit_soon": "Preparar boxes nas próximas voltas",
    "stay_out": "Seguir na pista",
}


def _deterministic_briefing(context: dict[str, Any]) -> str:
    driver = context["driver"]
    ideal = context.get("ideal_lap") or {}
    ahead = context.get("ahead")
    strategy = context.get("strategy") or {}

    lines = [f"{driver}, aqui é o engenheiro."]

    gap = ideal.get("gap_to_ideal_s")
    if gap is not None:
        lines.append(f"Sua volta ideal seria {gap:.3f}s mais rápida que sua melhor volta real — ritmo está na mesa.")

    if ahead and ahead.get("driver_ahead"):
        gap_ahead = ahead.get("gap_s")
        if gap_ahead is not None:
            lines.append(f"{ahead['driver_ahead']} está {gap_ahead:.1f}s à sua frente na pista.")
        else:
            lines.append(f"{ahead['driver_ahead']} está à sua frente.")
    elif ahead and ahead.get("is_leader"):
        lines.append("Você está na liderança, sem ninguém à frente.")

    action_label = ACTION_LABELS.get(strategy.get("recommended_action"), "Seguir monitorando")
    tyre_life = strategy.get("tyre_life")
    compound = strategy.get("compound")
    lines.append(
        f"Pneu {compound}, {tyre_life:.0f} voltas de uso. Recomendação: {action_label}."
    )
    pit_prob = strategy.get("pit_probability")
    if pit_prob is not None:
        lines.append(f"Probabilidade de pit modelo: {pit_prob*100:.0f}%.")

    return " ".join(lines)


def _load_hf_pipeline():
    global _hf_pipeline
    if _hf_pipeline is None:
        from transformers import pipeline

        _hf_pipeline = pipeline(
            "text-generation",
            model=HF_MODEL_ID,
            device=-1,  # CPU; troque para 0 se houver GPU disponível no container
            torch_dtype="auto",
        )
    return _hf_pipeline


def generate_briefing(context: dict[str, Any]) -> dict:
    driver = context["driver"]
    ideal = context.get("ideal_lap") or {}
    ahead = context.get("ahead") or {}
    strategy = context.get("strategy") or {}

    prompt = f"""Você é um engenheiro de corrida de F1 falando no rádio com o piloto {driver} durante a corrida.

Dados atuais:
- Nossa posição na pista nesta volta: P{ahead.get('position')}
- Volta ideal (teórica, soma dos melhores setores): {ideal.get('ideal_lap_s')}s
- Melhor volta real cravada: {ideal.get('actual_best_lap_s')}s
- Gap para a volta ideal: {ideal.get('gap_to_ideal_s')}s
- Piloto à frente na pista: {ahead.get('driver_ahead') or 'ninguém, está na liderança'}
- Gap para o piloto à frente: {ahead.get('gap_s')}s
- Composto atual: {strategy.get('compound')}
- Voltas de uso do pneu: {strategy.get('tyre_life')}
- Tendência de ritmo do composto (s/volta): {strategy.get('pace_trend_s_per_lap')}
- Probabilidade de pit stop do modelo: {strategy.get('pit_probability')}
- Ação recomendada pelo modelo: {strategy.get('recommended_action')}

Dê um briefing curto (4-6 frases), no estilo de rádio de F1, direto e profissional, em português.
Comente o ritmo em relação à volta ideal, a situação com o carro da frente, e dê uma recomendação
clara de estratégia (boxes agora / preparar boxes / seguir na pista), focando em ajudar o piloto
a ganhar posições ou defender a posição atual."""

    try:
        pipe = _load_hf_pipeline()
        messages = [
            {"role": "system", "content": "Você é um engenheiro de corrida de F1 experiente, direto e profissional, sempre em português do Brasil."},
            {"role": "user", "content": prompt},
        ]
        output = pipe(
            messages,
            max_new_tokens=HF_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        generated = output[0]["generated_text"]
        text = generated[-1]["content"].strip() if isinstance(generated, list) else str(generated).strip()
        if text:
            return {"mode": "huggingface", "model": HF_MODEL_ID, "text": text}
    except Exception as exc:
        return {
            "mode": "deterministic_fallback",
            "model": None,
            "text": _deterministic_briefing(context),
            "error": str(exc),
        }

    return {"mode": "deterministic_fallback", "model": None, "text": _deterministic_briefing(context)}
