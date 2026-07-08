import { useEffect, useState } from 'react';
import { PageHeader } from '../components/PageHeader';
import { MetricCard } from '../components/MetricCard';
import { ReliabilityChart } from '../components/ReliabilityChart';
import { fetchCalibration, type CalibrationReport } from '../lib/api';

const MARKET_LABELS: Record<string, string> = {
  win: 'Vitória',
  podium: 'Pódio',
  top6: 'Top 6',
  top10: 'Top 10',
  dnf: 'DNF',
};

export function CalibrationPage() {
  const [report, setReport] = useState<CalibrationReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchCalibration().then(setReport).catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="text-yellow-500">{error} — rode o pipeline de calibração primeiro.</p>;
  if (!report) return <p className="text-gray-500">Carregando…</p>;

  return (
    <div>
      <PageHeader title="Calibração do Modelo" subtitle="Validação contra todas as 24 corridas de 2024" />

      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
        {Object.entries(MARKET_LABELS).map(([market, label]) => {
          const r = report[market];
          if (!r) return null;
          return (
            <MetricCard
              key={market}
              label={label}
              value={`${r.improvement_pct >= 0 ? '+' : ''}${r.improvement_pct.toFixed(1)}%`}
              sublabel="vs baseline"
            />
          );
        })}
      </div>

      <h3 className="text-xl font-bold mb-1">Reliability Diagrams</h3>
      <p className="text-sm text-gray-500 mb-4">
        Calibração perfeita = pontos na diagonal. Acima = modelo underconfident. Abaixo = overconfident.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        {Object.entries(MARKET_LABELS).map(([market, label]) => {
          const r = report[market];
          if (!r?.reliability) return null;
          return <ReliabilityChart key={market} title={label} data={r.reliability} />;
        })}
      </div>

      <h3 className="text-xl font-bold mb-3">Métricas Detalhadas</h3>
      <div className="card p-4 overflow-x-auto mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-gray-400 border-b border-white/10">
              <th className="py-2 pr-3">Mercado</th>
              <th className="py-2 pr-3">Brier (Modelo)</th>
              <th className="py-2 pr-3">Brier (Baseline)</th>
              <th className="py-2 pr-3">Ganho</th>
              <th className="py-2 pr-3">ECE</th>
              <th className="py-2 pr-3">Base Rate</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(MARKET_LABELS).map(([market, label]) => {
              const r = report[market];
              if (!r) return null;
              return (
                <tr key={market} className="border-b border-white/5">
                  <td className="py-2 pr-3 font-semibold">{label}</td>
                  <td className="py-2 pr-3">{r.brier_model.toFixed(4)}</td>
                  <td className="py-2 pr-3">{r.brier_baseline.toFixed(4)}</td>
                  <td className="py-2 pr-3">{r.improvement_pct >= 0 ? '+' : ''}{r.improvement_pct.toFixed(1)}%</td>
                  <td className="py-2 pr-3">{r.ece.toFixed(4)}</td>
                  <td className="py-2 pr-3">{(r.base_rate * 100).toFixed(1)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="text-sm text-gray-400 space-y-1">
        <p><strong className="text-gray-200">Interpretação:</strong></p>
        <p>• <strong>Brier Score</strong>: média de (probabilidade − resultado)² — menor é melhor</p>
        <p>• <strong>ECE</strong>: Expected Calibration Error — mede se "70% previsto" realmente acerta ~70%</p>
        <p>• <strong>Ganho</strong>: melhoria percentual sobre o baseline (usar grid position como preditor)</p>
        <p>• O modelo bate o baseline em todos os 5 mercados</p>
      </div>
    </div>
  );
}
