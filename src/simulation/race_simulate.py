"""
Simulador Monte Carlo de Corrida de F1 — v2 OTIMIZADO.

SPEEDUP vs v1:
    v1: 1 predict por (sim × lap × driver) = 5.8M chamadas → 13 horas
    v2: 1 predict por lap com batch de (n_sims × n_drivers) → ~58 chamadas
    Estimativa: 5-15 min pra 10k simulações (100-500x speedup)

COMO FUNCIONA:
    - Features fixas (target encoding, grid_pos, etc.) pré-computadas UMA VEZ
    - Estado da corrida em arrays numpy (n_sims × n_drivers)
    - A cada volta, monta batch DataFrame (n_sims * n_drivers linhas) e
      chama model.predict() UMA VEZ
    - DNF e SC amostrados vetorialmente via numpy

MESMA API do v1 — drop-in replacement.
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


class RaceSimulator:
    def __init__(self, models_dir: Path | str | None = None):
        models = Path(models_dir) if models_dir else MODELS_DIR

        self.lap_model = joblib.load(models / "global_model_v3.pkl")

        with open(models / "global_feature_columns_v3.json", "r") as f:
            self.feat_meta = json.load(f)
        self.all_features = self.feat_meta["all_features_in_order"]
        self.n_features = len(self.all_features)
        # Mapa feature_name -> index na array
        self.feat_idx = {f: i for i, f in enumerate(self.all_features)}

        self.encoder = joblib.load(models / "global_target_encoder_v3.pkl")

        dnf_data = joblib.load(models / "dnf_model.pkl")
        self.dnf_model = dnf_data["model"]
        self.dnf_median = dnf_data["median_fill"]

        self.sc_model = joblib.load(models / "sc_model.pkl")
        self.tyre_model = joblib.load(models / "tyre_deg_model.pkl")

        # Pré-computa coeficientes de degradação como arrays
        self._deg_coefs = {}
        self._deg_intercepts = {}
        for comp_name, comp_enc in COMPOUND_MAP.items():
            if comp_name in self.tyre_model["coefficients"]:
                c = self.tyre_model["coefficients"][comp_name]
                self._deg_coefs[comp_enc] = c["coef_per_lap"]
                self._deg_intercepts[comp_enc] = c["intercept"]
            else:
                # Fallback: MEDIUM
                c = self.tyre_model["coefficients"].get("MEDIUM", {"coef_per_lap": -0.03, "intercept": 0.0})
                self._deg_coefs[comp_enc] = c["coef_per_lap"]
                self._deg_intercepts[comp_enc] = c["intercept"]

    def _precompute_fixed_features(self, grid: list[dict], gp: str) -> dict:
        """
        Calcula UMA VEZ tudo que não muda entre voltas/simulações:
        target encoding, grid_position, etc.
        Retorna dict com arrays prontos pra broadcast.
        """
        n_drivers = len(grid)

        # Target encoding (1 call por piloto, total n_drivers calls)
        driver_te = np.zeros(n_drivers)
        team_te = np.zeros(n_drivers)
        gp_te_val = 0.0

        for i, entry in enumerate(grid):
            enc_row = pd.DataFrame([{"Driver": entry["driver"], "Team": entry["team"], "gp": gp}])
            te = self.encoder.transform(enc_row).iloc[0]
            driver_te[i] = te["Driver_te"]
            team_te[i] = te["Team_te"]
            gp_te_val = te["gp_te"]

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
            "driver_te": driver_te,
            "team_te": team_te,
            "gp_te": gp_te_val,
            "dnf_probs": dnf_probs,
            "grid_pos": grid_pos,
            "quali_pos": quali_pos,
            "gap_to_pole": gap_to_pole,
            "avg_res": avg_res,
        }

    def _build_batch_features(
        self,
        fixed: dict,
        n_sims: int,
        n_drivers: int,
        lap_number: int,
        total_laps: int,
        # Tudo abaixo: shape (n_sims, n_drivers)
        positions: np.ndarray,
        tyre_life: np.ndarray,
        stint: np.ndarray,
        compound_enc: np.ndarray,
        is_sc: np.ndarray,       # shape (n_sims,) broadcast pra (n_sims, n_drivers)
        prev_lap_time: np.ndarray,
    ) -> pd.DataFrame:
        """
        Monta DataFrame batch de (n_sims * n_drivers) linhas × n_features colunas.
        Isso é a peça central da otimização: UMA chamada predict() por volta.
        """
        N = n_sims * n_drivers  # total de linhas no batch
        lap_pct = lap_number / total_laps

        # Flatten: (n_sims, n_drivers) -> (N,)
        pos_flat = positions.ravel()
        tl_flat = tyre_life.ravel()
        stint_flat = stint.ravel()
        comp_flat = compound_enc.ravel()
        prev_lt_flat = prev_lap_time.ravel()

        # is_sc broadcast: (n_sims,) -> (n_sims, n_drivers) -> (N,)
        is_sc_broad = np.broadcast_to(is_sc[:, None], (n_sims, n_drivers)).ravel()

        # Features fixas broadcast: (n_drivers,) -> (n_sims, n_drivers) -> (N,)
        def bc(arr_1d):
            return np.broadcast_to(arr_1d[None, :], (n_sims, n_drivers)).ravel()

        driver_te = bc(fixed["driver_te"])
        team_te = bc(fixed["team_te"])
        grid_pos = bc(fixed["grid_pos"])
        quali_pos = bc(fixed["quali_pos"])
        gap_pole = bc(fixed["gap_to_pole"])
        avg_res = bc(fixed["avg_res"])

        # Montar array 2D (N, n_features) de uma vez
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
        data[:, fi["quali_position"]] = quali_pos
        data[:, fi["grid_position"]] = grid_pos
        data[:, fi["gap_to_pole_ms"]] = gap_pole
        data[:, fi["avg_residual_last_3_races"]] = avg_res

        # Prev lap time features
        data[:, fi["lap_time_prev"]] = prev_lt_flat
        data[:, fi["lap_time_mean_3_prev"]] = prev_lt_flat
        data[:, fi["lap_time_delta_prev"]] = 0.0
        data[:, fi["lap_time_std_5_prev"]] = 0.3

        # Telemetria prev (defaults fixos — mesma simplificação do v1)
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

        # Target encoding
        data[:, fi["Driver_te"]] = driver_te
        data[:, fi["Team_te"]] = team_te
        data[:, fi["gp_te"]] = fixed["gp_te"]

        return pd.DataFrame(data, columns=self.all_features)

    def simulate(
        self,
        gp: str,
        grid: list[dict],
        n_simulations: int = 10_000,
        total_laps: int = DEFAULT_RACE_LAPS,
        strategy: list[dict] | None = None,
        seed: int = 42,
        verbose: bool = True,
    ) -> dict:
        if strategy is None:
            strategy = DEFAULT_STRATEGY

        rng = np.random.default_rng(seed)
        n_drivers = len(grid)
        n_sims = n_simulations

        t0 = time.perf_counter()
        if verbose:
            print(f"[INFO] Pré-computando features fixas pra {n_drivers} pilotos...")

        fixed = self._precompute_fixed_features(grid, gp)
        sc_prob = self.sc_model["gp_probs"].get(gp, self.sc_model["global_mean"])

        # Pré-computar compound encoding por stint da estratégia
        strat_compound_enc = [COMPOUND_MAP.get(s["compound"].upper(), 1) for s in strategy]
        strat_laps = [s["laps"] for s in strategy]

        # ---- PRÉ-AMOSTRAGEM VETORIAL ----
        # SC: (n_sims,) — quais simulações tem SC
        has_sc = rng.random(n_sims) < sc_prob
        sc_start = rng.integers(5, max(total_laps - 5, 6), size=n_sims)
        sc_duration = rng.integers(3, 7, size=n_sims)

        # DNF: pré-amostra pra todas as voltas de uma vez
        # Shape: (total_laps, n_sims, n_drivers)
        dnf_per_lap = fixed["dnf_probs"][None, :] / total_laps  # (1, n_drivers)
        dnf_rolls = rng.random((total_laps, n_sims, n_drivers))  # (laps, sims, drivers)

        # Ruído: pré-amostra também
        noise_all = np.zeros((total_laps, n_sims, n_drivers))
        for lap in range(total_laps):
            lap_pct = (lap + 1) / total_laps
            noise_std = NOISE_BASE + NOISE_SLOPE * lap_pct
            noise_all[lap] = rng.normal(0, noise_std, size=(n_sims, n_drivers))

        # ---- ESTADO ----
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

        # ---- LOOP DE VOLTAS (único loop restante) ----
        for lap_idx in range(total_laps):
            lap_number = lap_idx + 1

            # SC nesta volta? (n_sims,)
            is_sc_lap = (has_sc & (sc_start <= lap_number) & (lap_number < sc_start + sc_duration)).astype(np.float64)

            # DNF nesta volta
            new_dnf = (dnf_rolls[lap_idx] < dnf_per_lap) & (~is_dnf)
            is_dnf |= new_dnf

            # Tyre management
            tyre_life += 1
            stint_laps_remaining -= 1

            # Pit stops: onde stint_laps_remaining <= 0 e ainda tem stint disponível
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

            # ---- BATCH PREDICT ----
            batch_df = self._build_batch_features(
                fixed=fixed,
                n_sims=n_sims,
                n_drivers=n_drivers,
                lap_number=lap_number,
                total_laps=total_laps,
                positions=positions,
                tyre_life=tyre_life,
                stint=stint_idx.astype(np.float64),
                compound_enc=compound_enc,
                is_sc=is_sc_lap,
                prev_lap_time=prev_lap_time,
            )

            # UMA chamada predict pra todas as sims × drivers
            residuals = self.lap_model.predict(batch_df)  # (N,)
            residuals = residuals.reshape(n_sims, n_drivers)

            # Degradação vetorizada
            deg = np.zeros_like(tyre_life)
            for comp_enc_val, coef in self._deg_coefs.items():
                mask = (compound_enc == comp_enc_val)
                if mask.any():
                    deg[mask] = coef * tyre_life[mask] + self._deg_intercepts[comp_enc_val]

            # Lap time
            base_lap = 90.0
            lap_times = np.where(
                is_sc_lap[:, None] == 1,
                base_lap + 30.0,
                base_lap + residuals + deg + noise_all[lap_idx],
            )

            # DNF: piloto com DNF não acumula tempo (fica infinito)
            lap_times[is_dnf] = 0.0

            total_times += lap_times
            total_times[is_dnf] = 1e9  # garante que DNFs ficam por último

            prev_lap_time = np.where(is_dnf, 90.0, lap_times)

            # Atualiza posições por tempo acumulado
            # argsort por simulação: posição 1 = menor tempo
            order = np.argsort(total_times, axis=1)  # (n_sims, n_drivers)
            for sim_i in range(n_sims):
                for rank, drv_idx in enumerate(order[sim_i]):
                    positions[sim_i, drv_idx] = rank + 1

            if verbose and lap_number % 10 == 0:
                elapsed = time.perf_counter() - t0
                eta = elapsed / lap_number * (total_laps - lap_number)
                print(
                    f"  Volta {lap_number:2d}/{total_laps} | "
                    f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s | "
                    f"DNFs até agora: {is_dnf.sum()}"
                )

        # ---- AGREGAR RESULTADOS ----
        # Posição final de cada piloto em cada simulação
        final_order = np.argsort(total_times, axis=1)
        position_counts = {grid[d]["driver"]: np.zeros(n_drivers + 1) for d in range(n_drivers)}

        for sim_i in range(n_sims):
            for rank, drv_idx in enumerate(final_order[sim_i]):
                drv = grid[drv_idx]["driver"]
                if is_dnf[sim_i, drv_idx]:
                    position_counts[drv][0] += 1
                else:
                    position_counts[drv][rank + 1] += 1

        # Probabilidades
        probabilities = {}
        for drv in position_counts:
            probs = {}
            probs["DNF"] = float(position_counts[drv][0] / n_sims)
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
            print(
                f"\n[OK] Concluído em {total_elapsed/60:.1f} min "
                f"({n_sims/total_elapsed:.1f} sims/s)"
            )

        return {
            "gp": gp,
            "n_simulations": n_sims,
            "total_laps": total_laps,
            "sc_probability": sc_prob,
            "probabilities": probabilities,
            "dnf_counts": {grid[d]["driver"]: int(is_dnf[:, d].sum()) for d in range(n_drivers)},
        }


def format_results(results: dict) -> str:
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"SIMULAÇÃO MONTE CARLO — {results['gp']}")
    lines.append(f"{'='*70}")
    lines.append(f"Simulações: {results['n_simulations']:,}")
    lines.append(f"Voltas: {results['total_laps']}")
    lines.append(f"P(Safety Car): {results['sc_probability']*100:.1f}%")
    lines.append(f"\n{'Driver':>5s}  {'Win':>6s}  {'Podium':>7s}  {'Top6':>6s}  {'Top10':>6s}  {'DNF':>5s}")
    lines.append("-" * 45)
    probs = results["probabilities"]
    for drv in sorted(probs.keys(), key=lambda d: probs[d]["win"], reverse=True):
        p = probs[drv]
        lines.append(
            f"{drv:>5s}  {p['win']*100:5.1f}%  {p['podium']*100:6.1f}%  "
            f"{p['top6']*100:5.1f}%  {p['top10']*100:5.1f}%  {p['DNF']*100:4.1f}%"
        )
    return "\n".join(lines)


def main():
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

    output_path = MODELS_DIR / "simulation_demo_results.json"
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