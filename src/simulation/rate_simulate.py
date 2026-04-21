"""
Simulador Monte Carlo de Corrida de F1 — Sprint 3 Entrega 4.

O QUE FAZ:
    Dado um grid de largada (lista de pilotos) e um GP, simula a corrida
    N vezes (default 10.000) e produz distribuição de posição final por
    piloto. Traduz em probabilidades de mercado: race winner, podium,
    top 6, top 10, points finish.

COMO FUNCIONA:
    Para cada simulação:
    1. Amostra se vai ter Safety Car nessa corrida (modelo SC por pista)
    2. Para cada volta:
       a. Amostra DNF de cada piloto (modelo DNF)
       b. Estima lap time de cada piloto usando o modelo v3 de lap time
       c. Adiciona ruído gaussiano proporcional à fase da corrida
       d. Se SC ativo, comprime gaps entre pilotos
    3. No final, ordena por tempo total acumulado → posição final
    4. Agrega posições finais de todas as simulações → probabilidades

MODELOS USADOS:
    - global_model_v3.pkl + global_target_encoder_v3.pkl (lap time)
    - dnf_model.pkl (probabilidade de abandono)
    - sc_model.pkl (probabilidade de safety car por pista)
    - tyre_deg_model.pkl (ajuste de degradação por compound)

LIMITAÇÕES HONESTAS:
    - Não simula ultrapassagens explicitamente (usa gap de tempo)
    - Não simula estratégia de pit stop (usa stint fixo de 2 paradas)
    - Não simula chuva (assume condições secas)
    - Ruído gaussiano é simplificação; distribuição real tem caudas pesadas
    - Pilotos novos (não no treino) usam fallback do target encoder

USO:
    from race_simulator import RaceSimulator
    sim = RaceSimulator()
    results = sim.simulate("Australian Grand Prix", grid=["VER","NOR","LEC",...], n_simulations=10000)
    print(results["probabilities"])
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

# Permite carregar artefatos pickled que referenciam o módulo `target_encoding`
SRC_DIR = PROJECT_ROOT / "src"
FEATURES_DIR = SRC_DIR / "features"
for _path in (SRC_DIR, FEATURES_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Ruído por fase da corrida (medido no diagnose_model.py Sprint 2):
# RMSE cresce de 0.32 no início pra 0.42 no fim. Fator walk-forward ≈ 1.8x.
NOISE_BASE = 0.58   # ≈ RMSE walk-forward do modelo v3
NOISE_SLOPE = 0.18  # cresce ~0.18s do início ao fim da corrida

# Compound encoding
COMPOUND_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}

# Estratégia de pit stop fixa (simplificação)
# Na realidade cada equipe tem estratégia diferente, mas pra v1 usamos
# uma estratégia "típica" de 2 paradas em corrida de ~55 voltas
DEFAULT_STRATEGY = [
    {"compound": "MEDIUM", "laps": 18},
    {"compound": "HARD", "laps": 20},
    {"compound": "MEDIUM", "laps": 17},
]

# Típico: corrida tem ~55-60 voltas
DEFAULT_RACE_LAPS = 55


class RaceSimulator:
    """Simulador Monte Carlo de corrida de F1."""

    def __init__(self, models_dir: Path | str | None = None):
        models = Path(models_dir) if models_dir else MODELS_DIR

        # Carrega modelos
        self.lap_model = joblib.load(models / "global_model_v3.pkl")

        feat_file = models / "global_feature_columns_v3.json"
        with open(feat_file, "r", encoding="utf-8") as f:
            self.feat_meta = json.load(f)

        self.encoder = joblib.load(models / "global_target_encoder_v3.pkl")

        dnf_data = joblib.load(models / "dnf_model.pkl")
        self.dnf_model = dnf_data["model"]
        self.dnf_features = dnf_data["features"]
        self.dnf_median = dnf_data["median_fill"]

        self.sc_model = joblib.load(models / "sc_model.pkl")
        self.tyre_model = joblib.load(models / "tyre_deg_model.pkl")

        self.all_features = self.feat_meta["all_features_in_order"]

    def _get_sc_probability(self, gp: str) -> float:
        return self.sc_model["gp_probs"].get(gp, self.sc_model["global_mean"])

    def _get_dnf_probability(self, driver: str, team: str, gp: str,
                              grid_pos: int, quali_pos: int, era: int) -> float:
        row = pd.DataFrame([{
            "dnf_rate_driver": self.dnf_median.get("dnf_rate_driver", 0.12),
            "dnf_rate_team": self.dnf_median.get("dnf_rate_team", 0.12),
            "dnf_rate_gp": self.dnf_median.get("dnf_rate_gp", 0.12),
            "regulation_era": era,
            "grid_position": grid_pos,
            "quali_position": quali_pos,
        }])
        return float(self.dnf_model.predict_proba(row)[0, 1])

    def _get_tyre_degradation(self, compound: str, tyre_life: int) -> float:
        comp = compound.upper()
        if comp not in self.tyre_model["coefficients"]:
            comp = "MEDIUM"
        coef = self.tyre_model["coefficients"][comp]
        return coef["coef_per_lap"] * tyre_life + coef["intercept"]

    def _build_lap_features(
        self,
        driver: str,
        team: str,
        gp: str,
        lap_number: int,
        total_laps: int,
        position: int,
        tyre_life: int,
        stint: int,
        compound: str,
        grid_pos: int,
        quali_pos: int,
        gap_to_pole_ms: float,
        is_sc: int,
        prev_lap_time: float | None,
        prev_speed_mean: float,
        era: int,
        avg_residual_recent: float,
    ) -> pd.DataFrame:
        """Constrói o vetor de features pra uma predição de lap time."""
        lap_pct = lap_number / total_laps
        compound_enc = COMPOUND_MAP.get(compound.upper(), 1)

        # Montar dict com TODOS os features na ordem certa
        row = {
            "LapNumber": lap_number,
            "Stint": stint,
            "LapNumber_pct": lap_pct,
            "tyre_x_progress": tyre_life * lap_pct,
            "compound_x_tyre": compound_enc * tyre_life,
            "TyreLife": tyre_life,
            "CompoundEncoded": compound_enc,
            "Position": position,
            "tyre_ratio": min(tyre_life / 30.0, 1.0),
            "stint_progress": min(tyre_life / 20.0, 1.0),
            "regulation_era": era,
            "is_sc": is_sc,
            "is_vsc": 0,
            "is_yellow": 0,
            "is_neutralized": is_sc,
            "laps_since_neutralization": 99 if is_sc == 0 else 0,
            "AirTemp": 25.0,
            "TrackTemp": 40.0,
            "Humidity": 50.0,
            "Pressure": 1013.0,
            "Rainfall": 0.0,
            "WindSpeed": 5.0,
            "temp_delta": 15.0,
            "quali_position": quali_pos,
            "grid_position": grid_pos,
            "gap_to_pole_ms": gap_to_pole_ms,
            "avg_residual_last_3_races": avg_residual_recent,
            "lap_time_prev": prev_lap_time if prev_lap_time else np.nan,
            "lap_time_mean_3_prev": prev_lap_time if prev_lap_time else np.nan,
            "lap_time_delta_prev": 0.0,
            "lap_time_std_5_prev": 0.3,
            "speed_mean_prev": prev_speed_mean,
            "speed_max_prev": 320.0,
            "speed_std_prev": 45.0,
            "throttle_mean_prev": 60.0,
            "throttle_std_prev": 30.0,
            "brake_ratio_prev": 0.18,
            "rpm_mean_prev": 10500.0,
            "gear_mean_prev": 5.5,
            "drs_ratio_prev": 0.25,
            "speed_mean_delta_prev": 0.0,
            "speed_max_delta_prev": 0.0,
            "throttle_mean_delta_prev": 0.0,
            "brake_ratio_delta_prev": 0.0,
            "degradation_score_prev": 0.0,
            "aggression_score_prev": 60.0 * 0.82,
            "efficiency_score_prev": prev_speed_mean / 10501.0,
            "drs_usage_intensity_prev": 0.25 * 320.0,
            "consistency_score_prev": 0.3,
        }

        # Target encoding
        enc_df = pd.DataFrame([{"Driver": driver, "Team": team, "gp": gp}])
        te_vals = self.encoder.transform(enc_df).iloc[0].to_dict()
        row.update(te_vals)

        # Montar DataFrame na ordem EXATA do modelo
        return pd.DataFrame([{f: row.get(f, 0.0) for f in self.all_features}])

    def simulate(
        self,
        gp: str,
        grid: list[dict],
        n_simulations: int = 10_000,
        total_laps: int = DEFAULT_RACE_LAPS,
        strategy: list[dict] | None = None,
        seed: int = 42,
        verbose: bool = True,
        report_every_sims: int = 250,
        report_every_laps: int | None = None,
        lap_log_simulation_index: int = 0,
    ) -> dict:
        """
        Simula a corrida n_simulations vezes.

        Args:
            gp: Nome do GP (ex: "Australian Grand Prix")
            grid: Lista de dicts, cada um com:
                {"driver": "VER", "team": "Red Bull Racing", "grid_pos": 1,
                 "quali_pos": 1, "gap_to_pole_ms": 0.0}
                Ordenado por posição de largada.
            n_simulations: número de simulações Monte Carlo
            total_laps: número de voltas da corrida
            strategy: lista de stints [{compound, laps}, ...] (default: 2 paradas)
            seed: seed aleatória pra reprodutibilidade
            verbose: imprime logs de progresso
            report_every_sims: imprime progresso a cada N simulações (se verbose)
            report_every_laps: se definido, imprime progresso por volta (somente na simulação
                indicada em lap_log_simulation_index)
            lap_log_simulation_index: índice (0-based) da simulação que terá log por volta

        Returns:
            dict com "probabilities" (por piloto, por posição) e "raw_results"
        """
        if strategy is None:
            strategy = DEFAULT_STRATEGY

        rng = np.random.default_rng(seed)
        n_drivers = len(grid)
        era = 2  # 2024

        # Pré-computa probabilidades de DNF e SC
        sc_prob = self._get_sc_probability(gp)
        dnf_probs = {}
        for entry in grid:
            drv = entry["driver"]
            dnf_probs[drv] = self._get_dnf_probability(
                drv, entry["team"], gp,
                entry["grid_pos"], entry["quali_pos"], era,
            )

        # Resultados: posição final de cada piloto em cada simulação
        position_counts = {entry["driver"]: np.zeros(n_drivers + 1) for entry in grid}
        dnf_counts = {entry["driver"]: 0 for entry in grid}

        sc_counts = 0

        t0 = time.perf_counter()
        last_report = t0

        if verbose:
            print(
                f"[INFO] Iniciando simulação: gp={gp!r}, sims={n_simulations:,}, laps={total_laps}, "
                f"report_every_sims={report_every_sims}, report_every_laps={report_every_laps}",
                flush=True,
            )

        for sim in range(n_simulations):
            if (
                verbose
                and report_every_sims
                and report_every_sims > 0
                and sim > 0
                and (sim % report_every_sims == 0)
            ):
                now = time.perf_counter()
                elapsed = now - t0
                sims_per_sec = sim / elapsed if elapsed > 0 else 0.0
                remaining = n_simulations - sim
                eta_sec = remaining / sims_per_sec if sims_per_sec > 0 else float("inf")

                # Probabilidades parciais de vitória (com base nas simulações já concluídas)
                win_probs = {d: float(position_counts[d][1] / sim) for d in position_counts}
                top_win = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                top_win_str = " | ".join(f"{d}:{p*100:4.1f}%" for d, p in top_win)

                sc_rate = (sc_counts / sim) if sim > 0 else 0.0

                # Agregado simples de DNF por sim (média de DNFs por simulação)
                total_dnfs_so_far = sum(dnf_counts.values())
                avg_dnfs_per_sim = (total_dnfs_so_far / sim) if sim > 0 else 0.0

                # Evita spammar se stdout estiver lento
                if now - last_report >= 0.5:
                    eta_min = eta_sec / 60.0
                    print(
                        f"Progresso: {sim:,}/{n_simulations:,} "
                        f"({sim/n_simulations*100:.1f}%) — "
                        f"{sims_per_sec:.1f} sims/s — ETA ~{eta_min:.1f} min",
                        f"  Top win (parcial): {top_win_str}",
                        f"  SC rate (parcial): {sc_rate*100:.1f}% | DNF/sim (média): {avg_dnfs_per_sim:.2f}",
                        flush=True,
                    )
                    last_report = now

            # Estado de cada piloto
            states = {}
            for entry in grid:
                drv = entry["driver"]
                states[drv] = {
                    "total_time": 0.0,
                    "dnf": False,
                    "position": entry["grid_pos"],
                    "tyre_life": 0,
                    "stint": 0,
                    "compound": strategy[0]["compound"],
                    "stint_laps_remaining": strategy[0]["laps"],
                    "strategy_idx": 0,
                    "prev_lap_time": None,
                    "speed_mean": 200.0,
                    "team": entry["team"],
                    "grid_pos": entry["grid_pos"],
                    "quali_pos": entry["quali_pos"],
                    "gap_to_pole_ms": entry.get("gap_to_pole_ms", 0.0),
                    "avg_residual_recent": entry.get("avg_residual_recent", 0.0),
                }

            # Amostra SC: em qual volta acontece? (simplificação: 1 SC max)
            has_sc = rng.random() < sc_prob
            sc_start_lap = int(rng.integers(5, max(total_laps - 5, 6))) if has_sc else -1
            sc_duration = int(rng.integers(3, 7)) if has_sc else 0
            if has_sc:
                sc_counts += 1

            if verbose and sim == lap_log_simulation_index:
                if has_sc:
                    print(
                        f"[SIM {sim}] Safety Car: start_lap={sc_start_lap}, duration={sc_duration}",
                        flush=True,
                    )
                else:
                    print(f"[SIM {sim}] Safety Car: não", flush=True)

            for lap in range(1, total_laps + 1):
                is_sc_lap = 1 if (sc_start_lap <= lap < sc_start_lap + sc_duration) else 0
                lap_pct = lap / total_laps
                noise_std = NOISE_BASE + NOISE_SLOPE * lap_pct

                if (
                    verbose
                    and report_every_laps
                    and report_every_laps > 0
                    and sim == lap_log_simulation_index
                    and (lap == 1 or lap % report_every_laps == 0 or lap == total_laps)
                ):
                    active_before = [(d, states[d]["total_time"]) for d in states if not states[d]["dnf"]]
                    active_before.sort(key=lambda x: x[1])
                    leader = active_before[0][0] if active_before else "-"
                    n_active = len(active_before)
                    print(
                        f"[SIM {sim}] Lap {lap:>2d}/{total_laps} | SC={is_sc_lap} | líder={leader} | ativos={n_active}",
                        flush=True,
                    )

                for entry in grid:
                    drv = entry["driver"]
                    st = states[drv]

                    if st["dnf"]:
                        continue

                    # Amostra DNF (prob distribuída uniformemente pelas voltas)
                    dnf_per_lap = dnf_probs[drv] / total_laps
                    if rng.random() < dnf_per_lap:
                        st["dnf"] = True
                        dnf_counts[drv] += 1
                        continue

                    # Tyre management
                    st["tyre_life"] += 1
                    st["stint_laps_remaining"] -= 1

                    # Pit stop?
                    if st["stint_laps_remaining"] <= 0 and st["strategy_idx"] < len(strategy) - 1:
                        st["strategy_idx"] += 1
                        next_stint = strategy[st["strategy_idx"]]
                        st["compound"] = next_stint["compound"]
                        st["stint_laps_remaining"] = next_stint["laps"]
                        st["tyre_life"] = 0
                        st["stint"] += 1
                        st["total_time"] += 22.0  # tempo de pit stop (~22s)

                    # Predição de lap time
                    features = self._build_lap_features(
                        driver=drv, team=st["team"], gp=gp,
                        lap_number=lap, total_laps=total_laps,
                        position=st["position"], tyre_life=st["tyre_life"],
                        stint=st["stint"], compound=st["compound"],
                        grid_pos=st["grid_pos"], quali_pos=st["quali_pos"],
                        gap_to_pole_ms=st["gap_to_pole_ms"],
                        is_sc=is_sc_lap,
                        prev_lap_time=st["prev_lap_time"],
                        prev_speed_mean=st["speed_mean"],
                        era=era, avg_residual_recent=st["avg_residual_recent"],
                    )

                    # Modelo prevê RESIDUAL, precisa adicionar baseline
                    residual = float(self.lap_model.predict(features)[0])

                    # Adiciona degradação do pneu
                    deg = self._get_tyre_degradation(st["compound"], st["tyre_life"])

                    # Lap time = baseline + residual + degradação + ruído
                    # baseline ≈ 90s (varia por pista, mas o residual já captura isso)
                    base_lap = 90.0
                    noise = rng.normal(0, noise_std)

                    if is_sc_lap:
                        lap_time = base_lap + 30.0  # SC = ~30s mais lento
                    else:
                        lap_time = base_lap + residual + deg + noise

                    st["total_time"] += lap_time
                    st["prev_lap_time"] = lap_time
                    st["speed_mean"] = 200.0 + rng.normal(0, 5)

                # Atualiza posições baseado no tempo acumulado
                active = [(d, states[d]["total_time"]) for d in states if not states[d]["dnf"]]
                active.sort(key=lambda x: x[1])
                for pos, (d, _) in enumerate(active, 1):
                    states[d]["position"] = pos

            # Registra posição final
            active = [(d, states[d]["total_time"]) for d in states if not states[d]["dnf"]]
            active.sort(key=lambda x: x[1])
            for pos, (d, _) in enumerate(active, 1):
                position_counts[d][pos] += 1
            for d in states:
                if states[d]["dnf"]:
                    position_counts[d][0] += 1  # posição 0 = DNF

        total_elapsed = time.perf_counter() - t0
        if total_elapsed > 0:
            print(
                f"[OK] Simulações concluídas em {total_elapsed/60.0:.2f} min "
                f"({n_simulations/total_elapsed:.1f} sims/s)",
                flush=True,
            )

        # Converte contagens em probabilidades
        probabilities = {}
        for drv in position_counts:
            total = n_simulations
            probs = {}
            probs["DNF"] = float(position_counts[drv][0] / total)
            for pos in range(1, n_drivers + 1):
                probs[f"P{pos}"] = float(position_counts[drv][pos] / total)

            # Mercados derivados
            probs["win"] = probs["P1"]
            probs["podium"] = sum(probs.get(f"P{p}", 0) for p in range(1, 4))
            probs["top6"] = sum(probs.get(f"P{p}", 0) for p in range(1, 7))
            probs["top10"] = sum(probs.get(f"P{p}", 0) for p in range(1, 11))
            probs["points"] = probs["top10"]

            probabilities[drv] = probs

        return {
            "gp": gp,
            "n_simulations": n_simulations,
            "total_laps": total_laps,
            "sc_probability": sc_prob,
            "probabilities": probabilities,
            "dnf_counts": {d: int(dnf_counts[d]) for d in dnf_counts},
        }


def format_results(results: dict) -> str:
    """Formata os resultados pra exibição no terminal."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"SIMULAÇÃO MONTE CARLO — {results['gp']}")
    lines.append(f"{'='*70}")
    lines.append(f"Simulações: {results['n_simulations']:,}")
    lines.append(f"Voltas: {results['total_laps']}")
    lines.append(f"P(Safety Car): {results['sc_probability']*100:.1f}%")

    lines.append(f"\n{'Driver':>5s}  {'Win':>6s}  {'Podium':>7s}  {'Top6':>6s}  {'Top10':>6s}  {'DNF':>5s}")
    lines.append("-" * 45)

    # Ordena por probabilidade de vitória
    probs = results["probabilities"]
    sorted_drivers = sorted(probs.keys(), key=lambda d: probs[d]["win"], reverse=True)

    for drv in sorted_drivers:
        p = probs[drv]
        lines.append(
            f"{drv:>5s}  {p['win']*100:5.1f}%  {p['podium']*100:6.1f}%  "
            f"{p['top6']*100:5.1f}%  {p['top10']*100:5.1f}%  {p['DNF']*100:4.1f}%"
        )

    return "\n".join(lines)


def main():
    """Demo: simula uma corrida com grid exemplo."""
    # Grid de exemplo (Australian GP 2024-style)
    grid = [
        {"driver": "VER", "team": "Red Bull Racing", "grid_pos": 1, "quali_pos": 1, "gap_to_pole_ms": 0},
        {"driver": "SAI", "team": "Ferrari", "grid_pos": 2, "quali_pos": 2, "gap_to_pole_ms": 150},
        {"driver": "LEC", "team": "Ferrari", "grid_pos": 3, "quali_pos": 3, "gap_to_pole_ms": 200},
        {"driver": "NOR", "team": "McLaren", "grid_pos": 4, "quali_pos": 4, "gap_to_pole_ms": 300},
        {"driver": "PIA", "team": "McLaren", "grid_pos": 5, "quali_pos": 5, "gap_to_pole_ms": 350},
        {"driver": "RUS", "team": "Mercedes", "grid_pos": 6, "quali_pos": 6, "gap_to_pole_ms": 400},
        {"driver": "HAM", "team": "Mercedes", "grid_pos": 7, "quali_pos": 7, "gap_to_pole_ms": 450},
        {"driver": "ALO", "team": "Aston Martin", "grid_pos": 8, "quali_pos": 8, "gap_to_pole_ms": 600},
        {"driver": "STR", "team": "Aston Martin", "grid_pos": 9, "quali_pos": 9, "gap_to_pole_ms": 700},
        {"driver": "PER", "team": "Red Bull Racing", "grid_pos": 10, "quali_pos": 10, "gap_to_pole_ms": 800},
    ]

    print("Carregando modelos...")
    sim = RaceSimulator()

    print("Simulando corrida (10.000 simulações)...")
    results = sim.simulate(
        gp="Australian Grand Prix",
        grid=grid,
        n_simulations=10_000,
        total_laps=58,
        seed=42,
    )

    print(format_results(results))

    # Salva resultados
    output_path = MODELS_DIR / "simulation_demo_results.json"
    # Converter pra JSON-safe
    json_results = {
        "gp": results["gp"],
        "n_simulations": results["n_simulations"],
        "sc_probability": results["sc_probability"],
        "probabilities": results["probabilities"],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n[OK] Resultados salvos em {output_path}")


if __name__ == "__main__":
    main()