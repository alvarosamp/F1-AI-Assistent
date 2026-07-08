import { useEffect, useState } from 'react';
import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { PageHeader } from '../components/PageHeader';
import { fetchModelAnalysis, type ModelAnalysis } from '../lib/api';

export function ModelAnalysisPage() {
  const [data, setData] = useState<ModelAnalysis | null>(null);

  useEffect(() => {
    fetchModelAnalysis().then(setData);
  }, []);

  if (!data) return <p className="text-gray-500">Carregando…</p>;

  const impData = [...data.feature_importance].reverse();
  const rmseData = data.rmse_by_gp;

  return (
    <div>
      <PageHeader title="Análise do Modelo v3" subtitle="Feature importance, performance por GP e piloto" />

      {impData.length > 0 && (
        <div className="rounded-2xl border border-white/10 bg-white/5 p-4 mb-8">
          <h3 className="text-lg font-bold mb-3">📊 Feature Importance (Top 20)</h3>
          <ResponsiveContainer width="100%" height={550}>
            <BarChart data={impData} layout="vertical" margin={{ left: 40, right: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" horizontal={false} />
              <XAxis type="number" stroke="#999" tick={{ fill: '#ccc', fontSize: 11 }} />
              <YAxis type="category" dataKey="feature" stroke="#999" tick={{ fill: '#ccc', fontSize: 11 }} width={160} />
              <Tooltip
                contentStyle={{ background: '#1a1a2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                formatter={(v) => Number(v).toFixed(3)}
              />
              <Bar dataKey="importance" fill="#E10600" radius={[0, 4, 4, 0]}>
                <LabelList dataKey="importance" position="right" formatter={(v) => Number(v).toFixed(3)} fill="#fff" fontSize={11} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <h3 className="text-xl font-bold mb-3">Métricas Walk-Forward (treina 2022+2023 → testa 2024)</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <MetricBox label="RMSE" value={data.metrics.rmse.toString()} />
        <MetricBox label="R²" value={data.metrics.r2.toString()} />
        <MetricBox label="MAE" value={data.metrics.mae.toString()} />
        <MetricBox label="Ganho vs Trivial" value={`${data.metrics.gain_vs_trivial_pct}%`} />
      </div>

      {rmseData.length > 0 && (
        <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
          <h3 className="text-lg font-bold mb-3">🌍 RMSE por Grand Prix (Ordenado)</h3>
          <ResponsiveContainer width="100%" height={450}>
            <BarChart data={rmseData} margin={{ bottom: 90 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" vertical={false} />
              <XAxis dataKey="gp" stroke="#999" tick={{ fill: '#ccc', fontSize: 10 }} angle={-45} textAnchor="end" interval={0} />
              <YAxis stroke="#999" tick={{ fill: '#ccc', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#1a1a2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                formatter={(v) => `${Number(v).toFixed(2)}s`}
              />
              <Bar dataKey="rmse" fill="#3671C6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function MetricBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-center">
      <div className="text-2xl font-extrabold">{value}</div>
      <div className="text-xs text-gray-400 uppercase tracking-wider mt-1">{label}</div>
    </div>
  );
}
