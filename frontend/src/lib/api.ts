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

export interface SeasonAdaptation {
  profile: string;
  historical_weight: number;
  pace_sensitivity: number;
  recovery_boost: number;
  favorite_shrink: number;
  longshot_floor: number;
  dnf_base_blend: number;
}

export interface CurrentFormSource {
  session: string;
  year: number;
  n_drivers: number;
  used_static_fallback_for: string[];
}

export interface SimulateResponse {
  gp: string;
  total_laps: number;
  n_simulations: number;
  sc_probability: number;
  probabilities: Record<string, DriverProbabilities>;
  probabilities_raw: Record<string, DriverProbabilities> | null;
  season_adaptation: SeasonAdaptation | null;
  current_form_source: CurrentFormSource | null;
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

export interface TelemetrySample {
  Distance: number;
  Time?: number;
  Speed?: number;
  Throttle?: number;
  Brake?: number;
  brake_pressure_proxy?: number;
  RPM?: number;
  nGear?: number;
  DRS?: number;
  X?: number;
  Y?: number;
  DriverAhead?: string;
  DistanceToDriverAhead?: number;
  dirty_air_score?: number;
  steering_proxy?: number;
  lateral_change?: number;
  accel_proxy?: number;
}

export interface DriverTelemetry {
  driver: string;
  lap_number: number;
  lap_time_s: number | null;
  compound: string | null;
  team?: string | null;
  summary: Record<string, number>;
  samples: TelemetrySample[];
}

export interface ComparisonSample {
  Distance: number;
  speed_a: number | null;
  speed_b: number | null;
  speed_delta: number | null;
  time_delta: number | null;
  throttle_a: number | null;
  throttle_b: number | null;
  brake_a: number | null;
  brake_b: number | null;
  dirty_air_a: number | null;
  dirty_air_b: number | null;
}

export interface EngineeringAnalysis {
  faster_driver: string | null;
  gap_s: number | null;
  max_speed_delta: ComparisonSample | null;
  dirty_air_delta_pct: number;
  brake_delta_pct: number;
  full_throttle_delta_pct: number;
  findings: string[];
}

export interface EngineeringCopilot {
  mode: string;
  summary: string;
  recommendation: string;
  risk_flags: string[];
}

export interface EngineeringComparison {
  year: number;
  gp: string;
  session: string;
  event_name: string;
  driver_a: DriverTelemetry;
  driver_b: DriverTelemetry;
  summary_delta: Record<string, number>;
  comparison_samples: ComparisonSample[];
  analysis: EngineeringAnalysis;
  copilot: EngineeringCopilot;
}

export interface DriftMonitor {
  source: string;
  report: Record<string, unknown>;
  markdown: string;
}

export interface SessionsCatalog {
  years: number[];
  session_codes: string[];
}

export interface EngineeringSessionsForYear {
  year: number;
  gps: string[];
  session_codes: string[];
}

export interface EngineeringDriverRow {
  driver: string;
  best_lap_s: number | null;
  best_lap_number: number | null;
  laps_completed: number;
  compounds: string[];
}

export interface EngineeringSummary {
  year: number;
  gp: string;
  session: string;
  event_name: string;
  drivers: EngineeringDriverRow[];
}

export interface ReplayStanding {
  driver: string;
  position: number | null;
  last_lap_s: number | null;
  compound: string | null;
}

export interface ReplayStartResponse {
  replay_id: string;
  total_laps: number;
  event_name: string;
}

export interface ReplayState {
  lap_number: number;
  total_laps: number;
  finished: boolean;
  standings: ReplayStanding[];
}

export interface LiveStatus {
  connected: boolean;
  session_active: boolean;
}

export interface LiveState {
  available: boolean;
  error: string | null;
  lap_number?: number;
  standings: ReplayStanding[];
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`/api${path}`);
  if (!res.ok) throw new Error(`GET /api${path} falhou: ${res.status}`);
  return res.json();
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST /api${path} falhou: ${res.status}`);
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
  return postJSON<SimulateResponse>('/simulate', params);
}

// ---- Engenharia ----

export function fetchEngineeringYears() {
  return getJSON<SessionsCatalog>('/engineering/sessions');
}

export function fetchEngineeringGps(year: number) {
  return getJSON<EngineeringSessionsForYear>(`/engineering/sessions?year=${year}`);
}

export function fetchEngineeringSummary(year: number, gp: string, session: string) {
  const q = new URLSearchParams({ year: String(year), gp, session });
  return getJSON<EngineeringSummary>(`/engineering/summary?${q}`);
}

export function fetchEngineeringTelemetry(year: number, gp: string, session: string, driver: string, lap?: number) {
  const q = new URLSearchParams({ year: String(year), gp, session, driver });
  if (lap != null) q.set('lap', String(lap));
  return getJSON<DriverTelemetry>(`/engineering/telemetry?${q}`);
}

export function fetchEngineeringComparison(params: {
  year: number;
  gp: string;
  session: string;
  driver_a: string;
  driver_b: string;
  lap_a?: number | null;
  lap_b?: number | null;
}) {
  const q = new URLSearchParams({
    year: String(params.year),
    gp: params.gp,
    session: params.session,
    driver_a: params.driver_a,
    driver_b: params.driver_b,
  });
  if (params.lap_a != null) q.set('lap_a', String(params.lap_a));
  if (params.lap_b != null) q.set('lap_b', String(params.lap_b));
  return getJSON<EngineeringComparison>(`/engineering/compare?${q}`);
}

export function fetchEngineeringDrift(year: number) {
  return getJSON<DriftMonitor>(`/engineering/drift?year=${year}`);
}

// ---- Replay ----

export function fetchReplaySessions() {
  return getJSON<SessionsCatalog>('/replay/sessions');
}

export function startReplay(params: { year: number; gp: string; session: string }) {
  return postJSON<ReplayStartResponse>('/replay/start', params);
}

export function fetchReplayState(replayId: string) {
  return getJSON<ReplayState>(`/replay/${replayId}/state`);
}

export function fetchReplayTelemetry(replayId: string, driver: string) {
  return getJSON<DriverTelemetry>(`/replay/${replayId}/telemetry/${driver}`);
}

// ---- Ao vivo ----

export function connectLive(params: { year: number; gp: string; session: string }) {
  return postJSON<{ connecting: boolean }>('/live/connect', params);
}

export function fetchLiveStatus() {
  return getJSON<LiveStatus>('/live/status');
}

export function fetchLiveState() {
  return getJSON<LiveState>('/live/state');
}

export function fetchLiveTelemetry(driver: string) {
  return getJSON<DriverTelemetry>(`/live/telemetry/${driver}`);
}

// ---- Engenheiro de corrida (IA local) ----

export interface IdealLapSector {
  sector: number;
  time_s: number;
  lap_number: number;
}

export interface IdealLap {
  available: boolean;
  driver: string;
  reason?: string;
  ideal_lap_s?: number;
  actual_best_lap_s?: number | null;
  actual_best_lap_number?: number | null;
  gap_to_ideal_s?: number | null;
  sectors?: IdealLapSector[];
}

export interface DriverAhead {
  driver_ahead: string | null;
  position: number;
  is_leader: boolean;
  gap_s?: number | null;
}

export interface StrategyRecommendation {
  driver: string;
  lap_number: number;
  compound: string | null;
  tyre_life: number;
  lap_time_mean_3_s: number | null;
  lap_time_delta_s: number | null;
  pit_probability: number | null;
  pace_trend_s_per_lap: number | null;
  recommended_action: 'pit_now' | 'pit_soon' | 'stay_out';
}

export interface RaceEngineerComparison {
  driver_ahead: string;
  comparison_samples: ComparisonSample[];
  analysis: EngineeringAnalysis;
}

export interface RaceEngineerAnalysis {
  year: number;
  gp: string;
  session: string;
  driver: string;
  lap_number: number;
  ideal_lap: IdealLap;
  ahead: DriverAhead | null;
  strategy: StrategyRecommendation;
  comparison_vs_ahead: RaceEngineerComparison | null;
}

export interface RaceEngineerBriefing {
  mode: 'huggingface' | 'deterministic_fallback';
  model: string | null;
  text: string;
  error?: string;
}

export interface RaceEngineerBriefingResponse extends RaceEngineerAnalysis {
  briefing: RaceEngineerBriefing;
}

export function fetchRaceEngineerAnalysis(year: number, gp: string, session: string, driver: string, lap?: number) {
  const q = new URLSearchParams({ year: String(year), gp, session, driver });
  if (lap != null) q.set('lap', String(lap));
  return getJSON<RaceEngineerAnalysis>(`/race-engineer/analysis?${q}`);
}

export function fetchRaceEngineerBriefing(params: { year: number; gp: string; session: string; driver: string; lap?: number | null }) {
  return postJSON<RaceEngineerBriefingResponse>('/race-engineer/briefing', params);
}
