"""
Simulador Monte Carlo v3 — CORRIGIDO.

PROBLEMA DO v2:
    Driver_te e Team_te do target encoding dão ~1.1s/volta de vantagem
    fixa ao Verstappen, independente do grid. Em 58 voltas = 64s de
    vantagem fictícia. Ele sempre ganha.

CORREÇÃO (3 mudanças):

    1. SUBSTITUIR Driver_te/Team_te pelo gap_to_pole_ms deste weekend.
       O gap pro pole REAL do qualifying já captura "quão rápido esse
       piloto está NESTE fim de semana". Se Norris classificou 0.1s
       atrás do VER, o gap é 0.1s — não 1.1s do encoding histórico.
       Implementação: Driver_te e Team_te são OVERRIDADOS no batch
       de features, substituídos por valores derivados do gap_to_pole.

    2. PENALIDADE DE TRÁFEGO nas primeiras 5 voltas.
       Largar atrás custa tempo real por "ar sujo" (~0.3s por carro
       à frente nas primeiras 3-5 voltas). Sem isso, o modelo ignora
       que largar em P10 custa ~15-20s nos primeiros 5 laps.

    3. DIFICULDADE DE ULTRAPASSAGEM.
       Ultrapassar é difícil em F1. Mesmo sendo mais rápido, um piloto
       atrás perde ~0.3s/volta em ar sujo. Isso limita a recuperação
       de posições e dá mais peso ao grid de largada.

RESULTADO ESPERADO:
    - Grid position passa a ter peso real nas probabilidades
    - VER em P1 ainda favorito, mas NÃO domina com 44%
    - VER em P10 NÃO ganha mais — as Ferraris/McLarens na frente seguram
    - Probabilidades mais parecidas com odds reais de casas de aposta
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

SRC_DIR = PROJECT_ROOT / "src"
FEATURES_DIR = SRC_DIR / "features"
for _path in (SRC_DIR, FEATURES_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

COMPOUND_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
NOISE_BASE = 0.58
NOISE_SLOPE = 0.18

DEFAULT_STRATEGY = [
    {"compound": "MEDIUM", "laps": 18},
    {"compound": "HARD", "laps": 20},
    {"compound": "MEDIUM", "laps": 17},
]
DEFAULT_RACE_LAPS = 55
PIT_STOP_TIME = 22.0

# ===== NOVOS PARÂMETROS DE CORREÇÃO =====
# Penalidade de ar sujo: cada carro à frente custa ~0.3s/volta nas primeiras voltas
DIRTY_AIR_PENALTY_PER_CAR = 0.3  # segundos por carro à frente
DIRTY_AIR_LAPS = 5               # nas primeiras N voltas o efeito é forte
DIRTY_AIR_DECAY = 0.6            # fator de decaimento por volta (0.3 → 0.18 → 0.11...)

# Dificuldade de ultrapassagem: piloto atrás de outro perde tempo
OVERTAKE_DIFFICULTY = 0.15  # segundos perdidos por volta quando "preso" atrás de outro carro

# Conversão gap_to_pole_ms -> Driver_te override
# Ideia: o pole position vira o "mais rápido" (Driver_te mais negativo)
# e cada ms de gap vira proporcional à escala do target encoding
GAP_TO_POLE_SCALE = 0.001  # 1ms de gap ≈ 0.001s de residual por volta


class RaceSimulator:
    def __init__(self, models_dir: Path | str | None = None):
        models = Path(models_dir) if models_dir else MODELS_DIR

        self.lap_model = joblib.load(models / "global_model_v3.pkl")

        with open(models / "global_feature_columns_v3.json", "r") as f:
            self.feat_meta = json.load(f)
        self.all_features = self.feat_meta["all_features_in_order"]
        self.n_features = len(self.all_features)
        self.feat_idx = {f: i for i, f in enumerate(self.all_features)}

        self.encoder = joblib.load(models / "global_target_encoder_v3.pkl")

        dnf_data = joblib.load(models / "dnf_model.pkl")
        self.dnf_model = dnf_data["model"]
        self.dnf_median = dnf_data["median_fill"]

        self.sc_model = joblib.load(models / "sc_model.pkl")
        self.tyre_model = joblib.load(models / "tyre_deg_model.pkl")

        self._deg_coefs = {}
        self._deg_intercepts = {}
        for comp_name, comp_enc in COMPOUND_MAP.items():
            if comp_name in self.tyre_model["coefficients"]:
                c = self.tyre_model["coefficients"][comp_name]
                self._deg_coefs[comp_enc] = c["coef_per_lap"]
                self._deg_intercepts[comp_enc] = c["intercept"]
            else:
                c = self.tyre_model["coefficients"].get("MEDIUM", {"coef_per_lap": -0.03, "intercept": 0.0})
                self._deg_coefs[comp_enc] = c["coef_per_lap"]
                self._deg_intercepts[comp_enc] = c["intercept"]

    def _precompute_fixed_features(self, grid: list[dict], gp: str) -> dict:
        n_drivers = len(grid)

        # Target encoding original (do treino)
        orig_driver_te = np.zeros(n_drivers)
        orig_team_te = np.zeros(n_drivers)
        gp_te_val = 0.0

        for i, entry in enumerate(grid):
            enc_row = pd.DataFrame([{"Driver": entry["driver"], "Team": entry["team"], "gp": gp}])
            te = self.encoder.transform(enc_row).iloc[0]
            orig_driver_te[i] = te["Driver_te"]
            orig_team_te[i] = te["Team_te"]
            gp_te_val = te["gp_te"]

        # CORREÇÃO 1: Override Driver_te baseado no gap_to_pole deste weekend
        # O pole setter (gap=0) recebe o Driver_te mais negativo do grid
        # Os outros recebem proporcionalmente ao gap
        gaps_ms = np.array([e.get("gap_to_pole_ms", 0.0) for e in grid], dtype=np.float64)

        # Usar o melhor Driver_te do grid como âncora (quem fez pole "merece" o melhor encoding)
        best_te = orig_driver_te.min()  # mais negativo = mais rápido

        # Cada piloto: Driver_te = best_te + gap_to_pole_ms * scale
        # Isso faz com que o pole seja o "mais rápido" e os gaps reflitam o weekend atual
        adjusted_driver_te = best_te + gaps_ms * GAP_TO_POLE_SCALE

        # Team_te: média entre o original e o ajustado (suaviza a transição)
        # Quando gap é pequeno, team_te importa menos
        adjusted_team_te = orig_team_te * 0.3  # reduz o peso do team encoding

        # DNF probabilities
        dnf_probs = np.zeros(n_drivers)
        for i, entry in enumerate(grid):
            dnf_row = pd.DataFrame([{
                "dnf_rate_driver": self.dnf_median.get("dnf_rate_driver", 0.12),
                "dnf_rate_team": self.dnf_median.get("dnf_rate_team", 0.12),
                "dnf_rate_gp": self.dnf_median.get("dnf_rate_gp", 0.12),
                "regulation_era": 2,
                "grid_position": entry["grid_pos"],
                "quali_position": entry["quali_pos"],
            }])
            dnf_probs[i] = float(self.dnf_model.predict_proba(dnf_row)[0, 1])

        grid_pos = np.array([e["grid_pos"] for e in grid], dtype=np.float64)
        quali_pos = np.array([e["quali_pos"] for e in grid], dtype=np.float64)
        gap_to_pole = np.array([e.get("gap_to_pole_ms", 0.0) for e in grid], dtype=np.float64)
        avg_res = np.array([e.get("avg_residual_recent", 0.0) for e in grid], dtype=np.float64)

        return {
            "driver_te": adjusted_driver_te,      # AJUSTADO pelo gap_to_pole
            "team_te": adjusted_team_te,            # REDUZIDO
            "gp_te": gp_te_val,
            "dnf_probs": dnf_probs,
            "grid_pos": grid_pos,
            "quali_pos": quali_pos,
            "gap_to_pole": gap_to_pole,
            "avg_res": avg_res,
        }

    def _build_batch_features(
        self, fixed, n_sims, n_drivers, lap_number, total_laps,
        positions, tyre_life, stint, compound_enc, is_sc, prev_lap_time,
    ):
        N = n_sims * n_drivers
        lap_pct = lap_number / total_laps

        pos_flat = positions.ravel()
        tl_flat = tyre_life.ravel()
        stint_flat = stint.ravel()
        comp_flat = compound_enc.ravel()
        prev_lt_flat = prev_lap_time.ravel()
        is_sc_broad = np.broadcast_to(is_sc[:, None], (n_sims, n_drivers)).ravel()

        def bc(arr_1d):
            return np.broadcast_to(arr_1d[None, :], (n_sims, n_drivers)).ravel()

        data = np.zeros((N, self.n_features), dtype=np.float64)
        fi = self.feat_idx

        data[:, fi["LapNumber"]] = lap_number
        data[:, fi["Stint"]] = stint_flat
        data[:, fi["LapNumber_pct"]] = lap_pct
        data[:, fi["tyre_x_progress"]] = tl_flat * lap_pct
        data[:, fi["compound_x_tyre"]] = comp_flat * tl_flat
        data[:, fi["TyreLife"]] = tl_flat
        data[:, fi["CompoundEncoded"]] = comp_flat
        data[:, fi["Position"]] = pos_flat
        data[:, fi["tyre_ratio"]] = np.minimum(tl_flat / 30.0, 1.0)
        data[:, fi["stint_progress"]] = np.minimum(tl_flat / 20.0, 1.0)
        data[:, fi["regulation_era"]] = 2
        data[:, fi["is_sc"]] = is_sc_broad
        data[:, fi["is_vsc"]] = 0
        data[:, fi["is_yellow"]] = 0
        data[:, fi["is_neutralized"]] = is_sc_broad
        data[:, fi["laps_since_neutralization"]] = np.where(is_sc_broad == 1, 0, 99)
        data[:, fi["AirTemp"]] = 25.0
        data[:, fi["TrackTemp"]] = 40.0
        data[:, fi["Humidity"]] = 50.0
        data[:, fi["Pressure"]] = 1013.0
        data[:, fi["Rainfall"]] = 0.0
        data[:, fi["WindSpeed"]] = 5.0
        data[:, fi["temp_delta"]] = 15.0
        data[:, fi["quali_position"]] = bc(fixed["quali_pos"])
        data[:, fi["grid_position"]] = bc(fixed["grid_pos"])
        data[:, fi["gap_to_pole_ms"]] = bc(fixed["gap_to_pole"])
        data[:, fi["avg_residual_last_3_races"]] = bc(fixed["avg_res"])
        data[:, fi["lap_time_prev"]] = prev_lt_flat
        data[:, fi["lap_time_mean_3_prev"]] = prev_lt_flat
        data[:, fi["lap_time_delta_prev"]] = 0.0
        data[:, fi["lap_time_std_5_prev"]] = 0.3
        data[:, fi["speed_mean_prev"]] = 200.0
        data[:, fi["speed_max_prev"]] = 320.0
        data[:, fi["speed_std_prev"]] = 45.0
        data[:, fi["throttle_mean_prev"]] = 60.0
        data[:, fi["throttle_std_prev"]] = 30.0
        data[:, fi["brake_ratio_prev"]] = 0.18
        data[:, fi["rpm_mean_prev"]] = 10500.0
        data[:, fi["gear_mean_prev"]] = 5.5
        data[:, fi["drs_ratio_prev"]] = 0.25
        data[:, fi["speed_mean_delta_prev"]] = 0.0
        data[:, fi["speed_max_delta_prev"]] = 0.0
        data[:, fi["throttle_mean_delta_prev"]] = 0.0
        data[:, fi["brake_ratio_delta_prev"]] = 0.0
        data[:, fi["degradation_score_prev"]] = 0.0
        data[:, fi["aggression_score_prev"]] = 49.2
        data[:, fi["efficiency_score_prev"]] = 200.0 / 10501.0
        data[:, fi["drs_usage_intensity_prev"]] = 80.0
        data[:, fi["consistency_score_prev"]] = 0.3
        data[:, fi["Driver_te"]] = bc(fixed["driver_te"])
        data[:, fi["Team_te"]] = bc(fixed["team_te"])
        data[:, fi["gp_te"]] = fixed["gp_te"]

        return pd.DataFrame(data, columns=self.all_features)

    def simulate(
        self, gp, grid, n_simulations=10_000, total_laps=DEFAULT_RACE_LAPS,
        strategy=None, seed=42, verbose=True,
    ):
        if strategy is None:
            strategy = DEFAULT_STRATEGY

        rng = np.random.default_rng(seed)
        n_drivers = len(grid)
        n_sims = n_simulations

        t0 = time.perf_counter()
        if verbose:
            print(f"[INFO] Pré-computando features (com correção gap_to_pole)...")

        fixed = self._precompute_fixed_features(grid, gp)
        sc_prob = self.sc_model["gp_probs"].get(gp, self.sc_model["global_mean"])

        strat_compound_enc = [COMPOUND_MAP.get(s["compound"].upper(), 1) for s in strategy]
        strat_laps = [s["laps"] for s in strategy]

        has_sc = rng.random(n_sims) < sc_prob
        sc_start = rng.integers(5, max(total_laps - 5, 6), size=n_sims)
        sc_duration = rng.integers(3, 7, size=n_sims)

        dnf_per_lap = fixed["dnf_probs"][None, :] / total_laps
        dnf_rolls = rng.random((total_laps, n_sims, n_drivers))

        noise_all = np.zeros((total_laps, n_sims, n_drivers))
        for lap in range(total_laps):
            lap_pct = (lap + 1) / total_laps
            noise_std = NOISE_BASE + NOISE_SLOPE * lap_pct
            noise_all[lap] = rng.normal(0, noise_std, size=(n_sims, n_drivers))

        total_times = np.zeros((n_sims, n_drivers))
        is_dnf = np.zeros((n_sims, n_drivers), dtype=bool)
        positions = np.tile(np.arange(1, n_drivers + 1, dtype=np.float64), (n_sims, 1))
        tyre_life = np.zeros((n_sims, n_drivers), dtype=np.float64)
        stint_idx = np.zeros((n_sims, n_drivers), dtype=int)
        compound_enc = np.full((n_sims, n_drivers), strat_compound_enc[0], dtype=np.float64)
        stint_laps_remaining = np.full((n_sims, n_drivers), strat_laps[0], dtype=int)
        prev_lap_time = np.full((n_sims, n_drivers), 90.0)

        if verbose:
            t_pre = time.perf_counter() - t0
            print(f"[INFO] Pré-computação: {t_pre:.1f}s. Iniciando {total_laps} voltas...")

        for lap_idx in range(total_laps):
            lap_number = lap_idx + 1
            is_sc_lap = (has_sc & (sc_start <= lap_number) & (lap_number < sc_start + sc_duration)).astype(np.float64)

            new_dnf = (dnf_rolls[lap_idx] < dnf_per_lap) & (~is_dnf)
            is_dnf |= new_dnf

            tyre_life += 1
            stint_laps_remaining -= 1

            needs_pit = (stint_laps_remaining <= 0) & (stint_idx < len(strategy) - 1)
            if needs_pit.any():
                stint_idx[needs_pit] += 1
                for si in range(len(strategy)):
                    mask = needs_pit & (stint_idx == si)
                    if mask.any():
                        compound_enc[mask] = strat_compound_enc[si]
                        stint_laps_remaining[mask] = strat_laps[si]
                tyre_life[needs_pit] = 0
                total_times[needs_pit] += PIT_STOP_TIME

            batch_df = self._build_batch_features(
                fixed, n_sims, n_drivers, lap_number, total_laps,
                positions, tyre_life, stint_idx.astype(np.float64),
                compound_enc, is_sc_lap, prev_lap_time,
            )

            residuals = self.lap_model.predict(batch_df).reshape(n_sims, n_drivers)

            # Degradação
            deg = np.zeros_like(tyre_life)
            for comp_enc_val, coef in self._deg_coefs.items():
                mask = (compound_enc == comp_enc_val)
                if mask.any():
                    deg[mask] = coef * tyre_life[mask] + self._deg_intercepts[comp_enc_val]

            # CORREÇÃO 2: Penalidade de tráfego nas primeiras voltas
            traffic_penalty = np.zeros((n_sims, n_drivers))
            if lap_number <= DIRTY_AIR_LAPS:
                decay_factor = DIRTY_AIR_DECAY ** (lap_number - 1)
                # Penalidade proporcional à posição (P1=0, P2=1 carro, P10=9 carros)
                cars_ahead = positions - 1  # quantos carros à frente
                traffic_penalty = cars_ahead * DIRTY_AIR_PENALTY_PER_CAR * decay_factor

            # CORREÇÃO 3: Dificuldade de ultrapassagem (ar sujo constante)
            # Pilotos fora do top 3 perdem um pouco por estarem em tráfego
            overtake_penalty = np.where(positions > 3, OVERTAKE_DIFFICULTY, 0.0)

            # Lap time final
            base_lap = 90.0
            lap_times = np.where(
                is_sc_lap[:, None] == 1,
                base_lap + 30.0,
                base_lap + residuals + deg + noise_all[lap_idx] + traffic_penalty + overtake_penalty,
            )

            lap_times[is_dnf] = 0.0
            total_times += lap_times
            total_times[is_dnf] = 1e9
            prev_lap_time = np.where(is_dnf, 90.0, lap_times)

            # Atualiza posições
            order = np.argsort(total_times, axis=1)
            for sim_i in range(n_sims):
                for rank, drv_idx in enumerate(order[sim_i]):
                    positions[sim_i, drv_idx] = rank + 1

            if verbose and lap_number % 10 == 0:
                elapsed = time.perf_counter() - t0
                eta = elapsed / lap_number * (total_laps - lap_number)
                print(f"  Volta {lap_number:2d}/{total_laps} | {elapsed:.0f}s | ETA {eta:.0f}s | DNFs: {is_dnf.sum()}")

        # Agregar resultados
        final_order = np.argsort(total_times, axis=1)
        position_counts = {grid[d]["driver"]: np.zeros(n_drivers + 1) for d in range(n_drivers)}

        for sim_i in range(n_sims):
            for rank, drv_idx in enumerate(final_order[sim_i]):
                drv = grid[drv_idx]["driver"]
                if is_dnf[sim_i, drv_idx]:
                    position_counts[drv][0] += 1
                else:
                    position_counts[drv][rank + 1] += 1

        probabilities = {}
        for drv in position_counts:
            probs = {"DNF": float(position_counts[drv][0] / n_sims)}
            for pos in range(1, n_drivers + 1):
                probs[f"P{pos}"] = float(position_counts[drv][pos] / n_sims)
            probs["win"] = probs["P1"]
            probs["podium"] = sum(probs.get(f"P{p}", 0) for p in range(1, 4))
            probs["top6"] = sum(probs.get(f"P{p}", 0) for p in range(1, 7))
            probs["top10"] = sum(probs.get(f"P{p}", 0) for p in range(1, 11))
            probs["points"] = probs["top10"]
            probabilities[drv] = probs

        total_elapsed = time.perf_counter() - t0
        if verbose:
            print(f"\n[OK] Concluído em {total_elapsed/60:.1f} min ({n_sims/total_elapsed:.1f} sims/s)")

        return {
            "gp": gp, "n_simulations": n_sims, "total_laps": total_laps,
            "sc_probability": sc_prob, "probabilities": probabilities,
            "dnf_counts": {grid[d]["driver"]: int(is_dnf[:, d].sum()) for d in range(n_drivers)},
        }


def format_results(results):
    lines = [f"\n{'='*70}", f"SIMULAÇÃO MONTE CARLO — {results['gp']}", f"{'='*70}",
             f"Simulações: {results['n_simulations']:,}", f"Voltas: {results['total_laps']}",
             f"P(Safety Car): {results['sc_probability']*100:.1f}%",
             f"\n{'Driver':>5s}  {'Win':>6s}  {'Podium':>7s}  {'Top6':>6s}  {'Top10':>6s}  {'DNF':>5s}",
             "-" * 50]
    probs = results["probabilities"]
    for drv in sorted(probs.keys(), key=lambda d: probs[d]["win"], reverse=True):
        p = probs[drv]
        lines.append(f"{drv:>5s}  {p['win']*100:5.1f}%  {p['podium']*100:6.1f}%  "
                      f"{p['top6']*100:5.1f}%  {p['top10']*100:5.1f}%  {p['DNF']*100:4.1f}%")
    return "\n".join(lines)


def main():
    # Teste 1: VER em pole (deveria ganhar com ~30-35%, não 44%)
    grid_ver_pole = [
        {"driver": "VER", "team": "Red Bull Racing", "grid_pos": 1, "quali_pos": 1, "gap_to_pole_ms": 0},
        {"driver": "NOR", "team": "McLaren", "grid_pos": 2, "quali_pos": 2, "gap_to_pole_ms": 100},
        {"driver": "LEC", "team": "Ferrari", "grid_pos": 3, "quali_pos": 3, "gap_to_pole_ms": 150},
        {"driver": "SAI", "team": "Ferrari", "grid_pos": 4, "quali_pos": 4, "gap_to_pole_ms": 200},
        {"driver": "PIA", "team": "McLaren", "grid_pos": 5, "quali_pos": 5, "gap_to_pole_ms": 300},
        {"driver": "RUS", "team": "Mercedes", "grid_pos": 6, "quali_pos": 6, "gap_to_pole_ms": 400},
        {"driver": "HAM", "team": "Mercedes", "grid_pos": 7, "quali_pos": 7, "gap_to_pole_ms": 500},
        {"driver": "ALO", "team": "Aston Martin", "grid_pos": 8, "quali_pos": 8, "gap_to_pole_ms": 700},
        {"driver": "PER", "team": "Red Bull Racing", "grid_pos": 9, "quali_pos": 9, "gap_to_pole_ms": 800},
        {"driver": "STR", "team": "Aston Martin", "grid_pos": 10, "quali_pos": 10, "gap_to_pole_ms": 1000},
    ]

    # Teste 2: NOR em pole, VER em P5 (NOR deveria ter mais chance)
    grid_nor_pole = [
        {"driver": "NOR", "team": "McLaren", "grid_pos": 1, "quali_pos": 1, "gap_to_pole_ms": 0},
        {"driver": "LEC", "team": "Ferrari", "grid_pos": 2, "quali_pos": 2, "gap_to_pole_ms": 80},
        {"driver": "SAI", "team": "Ferrari", "grid_pos": 3, "quali_pos": 3, "gap_to_pole_ms": 120},
        {"driver": "PIA", "team": "McLaren", "grid_pos": 4, "quali_pos": 4, "gap_to_pole_ms": 200},
        {"driver": "VER", "team": "Red Bull Racing", "grid_pos": 5, "quali_pos": 5, "gap_to_pole_ms": 350},
        {"driver": "RUS", "team": "Mercedes", "grid_pos": 6, "quali_pos": 6, "gap_to_pole_ms": 500},
        {"driver": "HAM", "team": "Mercedes", "grid_pos": 7, "quali_pos": 7, "gap_to_pole_ms": 600},
        {"driver": "ALO", "team": "Aston Martin", "grid_pos": 8, "quali_pos": 8, "gap_to_pole_ms": 800},
        {"driver": "PER", "team": "Red Bull Racing", "grid_pos": 9, "quali_pos": 9, "gap_to_pole_ms": 900},
        {"driver": "STR", "team": "Aston Martin", "grid_pos": 10, "quali_pos": 10, "gap_to_pole_ms": 1200},
    ]

    print("Carregando modelos...")
    sim = RaceSimulator()

    print("\n" + "=" * 70)
    print("TESTE 1: VER em pole (gap competitivo com NOR)")
    print("=" * 70)
    r1 = sim.simulate(gp="Australian Grand Prix", grid=grid_ver_pole,
                       n_simulations=10_000, total_laps=58, seed=42)
    print(format_results(r1))

    print("\n" + "=" * 70)
    print("TESTE 2: NOR em pole, VER em P5 (NOR deveria liderar)")
    print("=" * 70)
    r2 = sim.simulate(gp="Australian Grand Prix", grid=grid_nor_pole,
                       n_simulations=10_000, total_laps=58, seed=42)
    print(format_results(r2))

    # Salva
    output_path = MODELS_DIR / "simulation_demo_results.json"
    with open(output_path, "w") as f:
        json.dump({"test1_ver_pole": r1["probabilities"], "test2_nor_pole": r2["probabilities"]}, f, indent=2)
    print(f"\n[OK] Resultados salvos em {output_path}")


if __name__ == "__main__":
    main()