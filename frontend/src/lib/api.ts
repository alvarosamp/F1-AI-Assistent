export interface ReferenceData {
  gps: string[];
  gp_laps: Record<string, number>;
  drivers: Record<string, string>;
  team_colors: Record<string, string>;
}

export interface DriverProbabilities {
  DNF: number;
  win: number;
  podium: number;
  top6: number;
  top10: number;
  points: number;
  [posKey: string]: number;
}

export interface SimulateResponse {
  gp: string;
  total_laps: number;
  n_simulations: number;
  sc_probability: number;
  probabilities: Record<string, DriverProbabilities>;
}

export interface ReliabilityBin {
  bin_lo: number;
  bin_hi: number;
  n: number;
  pred_mean: number;
  real_rate: number;
  gap: number;
}

export interface MarketCalibration {
  brier_model: number;
  brier_baseline: number;
  improvement_pct: number;
  ece: number;
  base_rate: number;
  reliability: ReliabilityBin[];
}

export type CalibrationReport = Record<string, MarketCalibration>;

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface RmseByGp {
  gp: string;
  rmse: number;
  mae: number;
  n: number;
}

export interface ModelAnalysis {
  metrics: {
    rmse: number;
    r2: number;
    mae: number;
    gain_vs_trivial_pct: number;
  };
  feature_importance: FeatureImportance[];
  rmse_by_gp: RmseByGp[];
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`/api${path}`);
  if (!res.ok) throw new Error(`GET /api${path} falhou: ${res.status}`);
  return res.json();
}

export function fetchReference() {
  return getJSON<ReferenceData>('/reference');
}

export function fetchCalibration() {
  return getJSON<CalibrationReport>('/calibration');
}

export function fetchModelAnalysis() {
  return getJSON<ModelAnalysis>('/model-analysis');
}

export async function runSimulation(params: {
  gp: string;
  drivers: string[];
  n_simulations: number;
}): Promise<SimulateResponse> {
  const res = await fetch('/api/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`POST /api/simulate falhou: ${res.status}`);
  return res.json();
}
