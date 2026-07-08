interface Props {
  drivers: string[];
  positions: string[];
  matrix: number[][]; // rows = drivers, cols = positions, values in %
}

function cellColor(pct: number) {
  // Plasma-ish scale: dark purple -> orange -> yellow
  const t = Math.min(1, pct / 60);
  const stops: [number, [number, number, number]][] = [
    [0, [13, 8, 61]],
    [0.33, [126, 3, 168]],
    [0.66, [237, 121, 83]],
    [1, [240, 249, 33]],
  ];
  let lo = stops[0], hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i][0] && t <= stops[i + 1][0]) {
      lo = stops[i];
      hi = stops[i + 1];
      break;
    }
  }
  const span = hi[0] - lo[0] || 1;
  const localT = (t - lo[0]) / span;
  const rgb = lo[1].map((c, i) => Math.round(c + (hi[1][i] - c) * localT));
  return `rgb(${rgb.join(',')})`;
}

export function PositionHeatmap({ drivers, positions, matrix }: Props) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
      <h3 className="text-lg font-bold mb-3">🚦 Distribuição de Posição Final (%)</h3>
      <div className="overflow-x-auto">
        <table className="border-collapse text-xs w-full">
          <thead>
            <tr>
              <th className="p-1 text-left text-gray-400 sticky left-0 bg-[#0d0d14]">Piloto</th>
              {positions.map((p) => (
                <th key={p} className="p-1 text-gray-400 font-medium">{p}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {drivers.map((drv, ri) => (
              <tr key={drv}>
                <td className="p-1 font-semibold sticky left-0 bg-[#0d0d14]">{drv}</td>
                {matrix[ri]?.map((v, ci) => (
                  <td
                    key={ci}
                    className="p-1 text-center min-w-[42px]"
                    style={{ background: cellColor(v), color: v > 30 ? '#1a1a1a' : '#f4f4f6' }}
                  >
                    {v > 0.1 ? v.toFixed(1) : ''}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
