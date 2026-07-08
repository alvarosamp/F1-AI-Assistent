import { useEffect, useMemo, useState } from 'react';
import { PageHeader } from '../components/PageHeader';
import { WinBarChart } from '../components/WinBarChart';
import { PositionHeatmap } from '../components/PositionHeatmap';
import { fetchReference, runSimulation, type ReferenceData, type SimulateResponse } from '../lib/api';

const N_SIM_OPTIONS = [100, 500, 1000, 2000, 5000, 10000];
const DEFAULT_GRID = ['VER', 'SAI', 'LEC', 'NOR', 'PIA', 'RUS', 'HAM', 'ALO', 'STR', 'PER'];

export function SimulationPage() {
  const [ref, setRef] = useState<ReferenceData | null>(null);
  const [gp, setGp] = useState('');
  const [nSims, setNSims] = useState(2000);
  const [selectedDrivers, setSelectedDrivers] = useState<string[]>(DEFAULT_GRID);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SimulateResponse | null>(null);

  useEffect(() => {
    fetchReference().then((data) => {
      setRef(data);
      setGp(data.gps[0]);
    });
  }, []);

  const totalLaps = ref && gp ? ref.gp_laps[gp] ?? 55 : 55;

  const allDrivers = useMemo(() => (ref ? Object.keys(ref.drivers).sort() : []), [ref]);

  function toggleDriver(drv: string) {
    setSelectedDrivers((prev) =>
      prev.includes(drv) ? prev.filter((d) => d !== drv) : prev.length < 20 ? [...prev, drv] : prev
    );
  }

  async function handleSimulate() {
    if (selectedDrivers.length < 2) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runSimulation({ gp, drivers: selectedDrivers, n_simulations: nSims });
      setResult(res);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const rows = useMemo(() => {
    if (!result || !ref) return [];
    return selectedDrivers
      .filter((d) => result.probabilities[d])
      .map((d) => {
        const p = result.probabilities[d];
        return {
          driver: d,
          team: ref.drivers[d] ?? '?',
          win: p.win,
          podium: p.podium,
          top6: p.top6,
          top10: p.top10,
          dnf: p.DNF,
        };
      });
  }, [result, ref, selectedDrivers]);

  const winChartData = useMemo(() => {
    if (!result || !ref) return [];
    return Object.entries(result.probabilities)
      .sort((a, b) => b[1].win - a[1].win)
      .map(([driver, p]) => ({
        driver,
        winPct: p.win * 100,
        color: ref.team_colors[ref.drivers[driver] ?? 'Unknown'] ?? '#666666',
      }));
  }, [result, ref]);

  const heatmap = useMemo(() => {
    if (!result) return null;
    const drivers = selectedDrivers.slice(0, 10).filter((d) => result.probabilities[d]);
    const positions = selectedDrivers.map((_, i) => `P${i + 1}`);
    const matrix = drivers.map((d) => {
      const p = result.probabilities[d];
      return positions.map((_, i) => (p[`P${i + 1}`] ?? 0) * 100);
    });
    return { drivers, positions, matrix };
  }, [result, selectedDrivers]);

  return (
    <div>
      <PageHeader title="Simulação Monte Carlo" subtitle="Selecione o GP e o grid de largada para simular a corrida" />

      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">
        <div className="rounded-2xl border border-white/10 bg-white/5 p-5 h-fit space-y-4">
          <h3 className="font-bold text-lg">⚙️ Parâmetros</h3>

          <div>
            <label className="text-sm text-gray-400 block mb-1">Grand Prix</label>
            <select
              className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-sm"
              value={gp}
              onChange={(e) => setGp(e.target.value)}
            >
              {ref?.gps.map((g) => (
                <option key={g} value={g}>{g}</option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">🏁 Voltas: {totalLaps}</p>
          </div>

          <div>
            <label className="text-sm text-gray-400 block mb-1">Simulações</label>
            <select
              className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-sm"
              value={nSims}
              onChange={(e) => setNSims(Number(e.target.value))}
            >
              {N_SIM_OPTIONS.map((n) => (
                <option key={n} value={n}>{n.toLocaleString()}</option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">Mais simulações = mais preciso, mais lento (~{(nSims / 880).toFixed(0)}s)</p>
          </div>

          <hr className="border-white/10" />

          <div>
            <label className="text-sm text-gray-400 block mb-2">🏎️ Grid de Largada ({selectedDrivers.length}/20)</label>
            <div className="max-h-64 overflow-y-auto flex flex-wrap gap-2">
              {allDrivers.map((drv) => {
                const active = selectedDrivers.includes(drv);
                return (
                  <button
                    key={drv}
                    onClick={() => toggleDriver(drv)}
                    className={`px-2.5 py-1 rounded-md text-xs font-semibold border transition-colors ${
                      active
                        ? 'bg-[#E10600] border-[#E10600] text-white'
                        : 'bg-black/20 border-white/10 text-gray-400 hover:border-white/30'
                    }`}
                  >
                    {active ? selectedDrivers.indexOf(drv) + 1 : ''} {drv}
                  </button>
                );
              })}
            </div>
            <p className="text-xs text-gray-500 mt-1">Ordem do clique = ordem de largada.</p>
          </div>

          <button
            onClick={handleSimulate}
            disabled={loading || selectedDrivers.length < 2 || !gp}
            className="w-full bg-[#E10600] hover:bg-[#c00500] disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold py-2.5 rounded-lg transition-colors"
          >
            {loading ? 'Simulando...' : '🏁 Simular Corrida'}
          </button>
          {selectedDrivers.length < 2 && (
            <p className="text-yellow-500 text-xs">Selecione pelo menos 2 pilotos.</p>
          )}
          {error && <p className="text-red-500 text-xs">{error}</p>}
        </div>

        <div className="space-y-6">
          {result && (
            <>
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4 overflow-x-auto">
                <h3 className="text-lg font-bold mb-3">Resultados — {result.gp}</h3>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-gray-400 border-b border-white/10">
                      <th className="py-2 pr-3">Piloto</th>
                      <th className="py-2 pr-3">Equipe</th>
                      <th className="py-2 pr-3">Vitória</th>
                      <th className="py-2 pr-3">Pódio</th>
                      <th className="py-2 pr-3">Top 6</th>
                      <th className="py-2 pr-3">Top 10</th>
                      <th className="py-2 pr-3">DNF</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r) => (
                      <tr key={r.driver} className="border-b border-white/5">
                        <td className="py-2 pr-3 font-semibold">{r.driver}</td>
                        <td className="py-2 pr-3 text-gray-400">{r.team}</td>
                        <td className="py-2 pr-3">{(r.win * 100).toFixed(1)}%</td>
                        <td className="py-2 pr-3">{(r.podium * 100).toFixed(1)}%</td>
                        <td className="py-2 pr-3">{(r.top6 * 100).toFixed(1)}%</td>
                        <td className="py-2 pr-3">{(r.top10 * 100).toFixed(1)}%</td>
                        <td className="py-2 pr-3">{(r.dnf * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <WinBarChart data={winChartData} />

              {heatmap && <PositionHeatmap {...heatmap} />}

              <p className="text-xs text-gray-500">
                SC probability: {(result.sc_probability * 100).toFixed(1)}% | Simulações: {result.n_simulations.toLocaleString()}
              </p>
            </>
          )}

          {!result && !loading && (
            <div className="rounded-2xl border border-dashed border-white/10 p-10 text-center text-gray-500">
              Configure o grid e clique em "Simular Corrida" para ver os resultados.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
