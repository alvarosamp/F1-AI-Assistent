import { useState } from 'react';
import {
  fetchRaceEngineerAnalysis,
  fetchRaceEngineerBriefing,
  type RaceEngineerAnalysis,
  type RaceEngineerBriefing,
} from '../lib/api';

function fmtLapTime(s: number | null | undefined) {
  if (s == null) return '-';
  const m = Math.floor(s / 60);
  const rem = (s % 60).toFixed(3);
  return `${m}:${rem.padStart(6, '0')}`;
}

const ACTION_LABELS: Record<string, string> = {
  pit_now: 'Boxes agora',
  pit_soon: 'Preparar boxes',
  stay_out: 'Seguir na pista',
};

const ACTION_COLORS: Record<string, string> = {
  pit_now: 'bg-red-500/20 text-red-300 border-red-500/40',
  pit_soon: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40',
  stay_out: 'bg-green-500/20 text-green-300 border-green-500/40',
};

interface Props {
  year: number;
  gp: string;
  session: string;
  driver: string;
  lap: number | null;
}

export function RaceEngineerPanel({ year, gp, session, driver, lap }: Props) {
  const [analysis, setAnalysis] = useState<RaceEngineerAnalysis | null>(null);
  const [briefing, setBriefing] = useState<RaceEngineerBriefing | null>(null);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [loadingBriefing, setLoadingBriefing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadAnalysis() {
    if (!driver) return;
    setLoadingAnalysis(true);
    setError(null);
    setAnalysis(null);
    setBriefing(null);
    try {
      const data = await fetchRaceEngineerAnalysis(year, gp, session, driver, lap ?? undefined);
      setAnalysis(data);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoadingAnalysis(false);
    }
  }

  async function askBriefing() {
    if (!driver) return;
    setLoadingBriefing(true);
    try {
      const data = await fetchRaceEngineerBriefing({ year, gp, session, driver, lap });
      setAnalysis(data);
      setBriefing(data.briefing);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoadingBriefing(false);
    }
  }

  return (
    <div className="card p-5 space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-bold uppercase text-text-muted">Engenheiro de corrida (IA local)</h3>
          <p className="text-xs text-text-faint mt-0.5">
            Volta ideal, comparação com o carro da frente e recomendação de pit/pneu para {driver || 'o piloto selecionado'}.
          </p>
        </div>
        <button
          onClick={loadAnalysis}
          disabled={loadingAnalysis || !driver}
          className="bg-white text-black hover:bg-text-muted disabled:opacity-50 font-bold px-4 py-2 rounded-md text-sm whitespace-nowrap"
        >
          {loadingAnalysis ? 'Analisando...' : 'Analisar meu piloto'}
        </button>
      </div>

      {error && <p className="text-yellow-400 text-sm">{error}</p>}

      {analysis && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Metric label="Volta ideal" value={analysis.ideal_lap.available ? fmtLapTime(analysis.ideal_lap.ideal_lap_s) : '-'} sub="soma dos melhores setores" />
            <Metric
              label="Gap p/ ideal"
              value={analysis.ideal_lap.gap_to_ideal_s != null ? `${analysis.ideal_lap.gap_to_ideal_s.toFixed(3)}s` : '-'}
              sub={`melhor volta: ${fmtLapTime(analysis.ideal_lap.actual_best_lap_s)}`}
            />
            <Metric
              label="Piloto à frente"
              value={analysis.ahead?.is_leader ? 'Líder' : analysis.ahead?.driver_ahead ?? '-'}
              sub={analysis.ahead?.gap_s != null ? `${analysis.ahead.gap_s.toFixed(1)}s à frente` : undefined}
            />
            <Metric
              label="Pneu"
              value={`${analysis.strategy.compound ?? '-'} · ${analysis.strategy.tyre_life.toFixed(0)}v`}
              sub={analysis.strategy.pit_probability != null ? `P(pit)=${(analysis.strategy.pit_probability * 100).toFixed(0)}%` : undefined}
            />
          </div>

          {analysis.ideal_lap.available && analysis.ideal_lap.sectors && (
            <div className="grid grid-cols-3 gap-3">
              {analysis.ideal_lap.sectors.map((s) => (
                <div key={s.sector} className="card p-3 text-center">
                  <div className="text-[10px] uppercase text-text-muted">Setor {s.sector}</div>
                  <div className="text-lg font-bold">{s.time_s.toFixed(3)}s</div>
                  <div className="text-[10px] text-text-faint">volta {s.lap_number}</div>
                </div>
              ))}
            </div>
          )}

          <div className={`inline-flex items-center gap-2 text-xs font-semibold border rounded-full px-3 py-1.5 ${ACTION_COLORS[analysis.strategy.recommended_action] ?? ''}`}>
            {ACTION_LABELS[analysis.strategy.recommended_action] ?? analysis.strategy.recommended_action}
          </div>

          {analysis.comparison_vs_ahead && (
            <div className="card p-4">
              <h4 className="text-xs font-bold uppercase text-text-muted mb-2">
                Vs. {analysis.comparison_vs_ahead.driver_ahead} (carro da frente)
              </h4>
              <ul className="text-sm space-y-1 text-text-muted">
                {analysis.comparison_vs_ahead.analysis.findings.map((f, i) => (
                  <li key={i}>• {f}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="flex items-center gap-3">
            <button
              onClick={askBriefing}
              disabled={loadingBriefing}
              className="bg-accent hover:bg-accent-dim disabled:opacity-50 text-white font-bold px-4 py-2 rounded-md text-sm"
            >
              {loadingBriefing ? 'Gerando briefing...' : 'Pedir briefing da IA'}
            </button>
            {briefing && briefing.mode === 'deterministic_fallback' && (
              <span className="text-[11px] text-text-faint">
                Modelo local indisponível — resposta gerada por regras.
              </span>
            )}
            {briefing && briefing.mode === 'huggingface' && (
              <span className="text-[11px] text-text-faint">via {briefing.model}</span>
            )}
          </div>

          {briefing && (
            <div className="card p-4 bg-accent/5 border-accent/20">
              <p className="text-sm leading-6">{briefing.text}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card p-3">
      <div className="text-[10px] uppercase font-semibold text-text-muted">{label}</div>
      <div className="text-lg font-extrabold mt-0.5 truncate">{value}</div>
      {sub && <div className="text-[10px] text-text-faint mt-0.5 truncate">{sub}</div>}
    </div>
  );
}
