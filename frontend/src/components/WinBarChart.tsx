import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

interface Props {
  data: { driver: string; winPct: number; color: string }[];
}

export function WinBarChart({ data }: Props) {
  return (
    <div className="card p-4">
      <h3 className="text-lg font-bold mb-2">🏆 Probabilidade de Vitória</h3>
      <ResponsiveContainer width="100%" height={380}>
        <BarChart data={data} margin={{ top: 20, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" vertical={false} />
          <XAxis dataKey="driver" stroke="#999" tick={{ fill: '#ccc', fontSize: 12 }} />
          <YAxis stroke="#999" tick={{ fill: '#ccc', fontSize: 12 }} unit="%" />
          <Tooltip
            contentStyle={{ background: '#1a1a2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
            formatter={(value) => [`${Number(value).toFixed(1)}%`, 'Vitória']}
          />
          <Bar dataKey="winPct" radius={[6, 6, 0, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.color} />
            ))}
            <LabelList
              dataKey="winPct"
              position="top"
              formatter={(v) => (Number(v) > 1 ? `${Number(v).toFixed(1)}%` : '')}
              fill="#fff"
              fontSize={12}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
