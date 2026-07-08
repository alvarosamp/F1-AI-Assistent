import { CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Scatter, ComposedChart, Tooltip, XAxis, YAxis } from 'recharts';
import type { ReliabilityBin } from '../lib/api';

export function ReliabilityChart({ title, data }: { title: string; data: ReliabilityBin[] }) {
  const points = data.map((b) => ({ x: b.pred_mean, y: b.real_rate, n: b.n }));

  return (
    <div className="card p-4">
      <h4 className="font-semibold mb-2">{title}</h4>
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
          <XAxis type="number" dataKey="x" domain={[0, 1]} tick={{ fill: '#ccc', fontSize: 11 }} name="Previsto" />
          <YAxis type="number" dataKey="y" domain={[0, 1]} tick={{ fill: '#ccc', fontSize: 11 }} name="Real" />
          <Tooltip
            contentStyle={{ background: '#1a1a2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
            formatter={(v) => Number(v).toFixed(3)}
          />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="gray" strokeDasharray="4 4" />
          <Scatter data={points} fill="#E10600" line={{ stroke: '#E10600', strokeWidth: 1 }} />
          <Legend />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
