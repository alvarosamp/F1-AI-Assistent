import { useEffect, useMemo, useState } from 'react';
import { PageHeader } from '../components/PageHeader';
import { TelemetryPanel } from '../components/TelemetryPanel';
import { RaceEngineerPanel } from '../components/RaceEngineerPanel';
import {
  fetchEngineeringComparison,
  fetchEngineeringDrift,
  fetchEngineeringGps,
  fetchEngineeringSummary,
  fetchEngineeringTelemetry,
  fetchEngineeringYears,
  type DriverTelemetry,
  type DriftMonitor,
  type EngineeringComparison,
  type EngineeringDriverRow,
  type EngineeringSummary,
} from '../lib/api';

const SESSION_LABELS: Record<string, string> = {
  FP1: 'Treino Livre 1',
  FP2: 'Treino Livre 2',
  FP3: 'Treino Livre 3',
  Q: 'Classificacao',
  R: 'Corrida',
};

function fmtLapTime(s: number | null) {
  if (s == null) return '-';
  const m = Math.floor(s / 60);
  const rem = (s % 60).toFixed(3);
  return `${m}:${rem.padStart(6, '0')}`;
}

export function EngineeringPage() {
  const [years, setYears] = useState<number[]>([]);
  const [sessionCodes, setSessionCodes] = useState<string[]>([]);
  const [year, setYear] = useState<number | null>(null);
  const [gps, setGps] = useState<string[]>([]);
  const [gp, setGp] = useState('');
  const [sessionCode, setSessionCode] = useState('Q');

  const [summary, setSummary] = useState<EngineeringSummary | null>(null);
  const [selectedDriver, setSelectedDriver] = useState('');
  const [compareDriver, setCompareDriver] = useState('');
  const [selectedLap, setSelectedLap] = useState<number | null>(null);
  const [compareLap, setCompareLap] = useState<number | null>(null);

  const [telemetry, setTelemetry] = useState<DriverTelemetry | null>(null);
  const [comparison, setComparison] = useState<EngineeringComparison | null>(null);
  const [drift, setDrift] = useState<DriftMonitor | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchEngineeringYears().then((data) => {
      setYears(data.years);
      setSessionCodes(data.session_codes);
      setYear(data.years[data.years.length - 1]);
    }).catch((e) => setError((e as Error).message));
  }, []);

  useEffect(() => {
    if (year == null) return;
    fetchEngineeringGps(year).then((data) => {
      setGps(data.gps);
      setGp(data.gps[0] ?? '');
    }).catch((e) => setError((e as Error).message));
    fetchEngineeringDrift(year).then(setDrift).catch(() => setDrift(null));
  }, [year]);

  const driverRows = useMemo(() => summary?.drivers ?? [], [summary]);
  async function loadSummary() {
    if (year == null || !gp) return;
    setLoading(true);
    setError(null);
    setSummary(null);
    setTelemetry(null);
    setComparison(null);
    try {
      const data = await fetchEngineeringSummary(year, gp, sessionCode);
      setSummary(data);
      const first = data.drivers[0];
      const second = data.drivers[1] ?? data.drivers[0];
      setSelectedDriver(first?.driver ?? '');
      setCompareDriver(second?.driver ?? '');
      setSelectedLap(first?.best_lap_number ?? null);
      setCompareLap(second?.best_lap_number ?? null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function loadTelemetryAndAnalysis() {
    if (year == null || !gp || !selectedDriver) return;
    setLoading(true);
    setError(null);
    setComparison(null);
    try {
      const tel = await fetchEngineeringTelemetry(year, gp, sessionCode, selectedDriver, selectedLap ?? undefined);
      setTelemetry(tel);
      if (compareDriver && compareDriver !== selectedDriver) {
        const comp = await fetchEngineeringComparison({
          year,
          gp,
          session: sessionCode,
          driver_a: selectedDriver,
          driver_b: compareDriver,
          lap_a: selectedLap,
          lap_b: compareLap,
        });
        setComparison(comp);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  function pickDriver(row: EngineeringDriverRow, slot: 'main' | 'compare') {
    if (slot === 'main') {
      setSelectedDriver(row.driver);
      setSelectedLap(row.best_lap_number);
    } else {
      setCompareDriver(row.driver);
      setCompareLap(row.best_lap_number);
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader title="Race Engineer Copilot" subtitle="Telemetria FastF1, comparacao entre pilotos, sinais derivados e drift da temporada" />

      <section className="card p-5 grid md:grid-cols-[1fr_1fr_0.7fr_auto] gap-4 items-end">
        <Field label="Ano">
          <select className="control" value={year ?? ''} onChange={(e) => setYear(Number(e.target.value))}>
            {years.map((y) => <option key={y} value={y}>{y}</option>)}
          </select>
        </Field>
        <Field label="Grand Prix">
          <select className="control" value={gp} onChange={(e) => setGp(e.target.value)}>
            {gps.map((g) => <option key={g} value={g}>{g}</option>)}
          </select>
        </Field>
        <Field label="Sessao">
          <select className="control" value={sessionCode} onChange={(e) => setSessionCode(e.target.value)}>
            {sessionCodes.map((c) => <option key={c} value={c}>{SESSION_LABELS[c] ?? c}</option>)}
          </select>
        </Field>
        <button onClick={loadSummary} disabled={loading || !gp} className="bg-accent hover:bg-accent-dim disabled:opacity-50 text-white font-bold px-5 py-2 rounded-md transition-colors">
          {loading ? 'Carregando...' : 'Carregar'}
        </button>
      </section>

      {summary && (
        <section className="grid xl:grid-cols-[1fr_0.82fr] gap-5">
          <div className="card p-4 overflow-x-auto">
            <div className="flex items-center justify-between gap-3 mb-3">
              <h2 className="text-lg font-bold">{summary.event_name} - {SESSION_LABELS[summary.session] ?? summary.session}</h2>
              <button onClick={loadTelemetryAndAnalysis} disabled={loading || !selectedDriver} className="bg-white text-black hover:bg-text-muted disabled:opacity-50 font-bold px-4 py-2 rounded-md text-sm">
                Analisar telemetria
              </button>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-text-muted border-b border-white/10">
                  <th className="py-2 pr-3">#</th>
                  <th className="py-2 pr-3">Piloto</th>
                  <th className="py-2 pr-3">Melhor volta</th>
                  <th className="py-2 pr-3">Volta</th>
                  <th className="py-2 pr-3">Compostos</th>
                  <th className="py-2 pr-3">Selecao</th>
                </tr>
              </thead>
              <tbody>
                {driverRows.map((d, i) => (
                  <tr key={d.driver} className="border-b border-white/5">
                    <td className="py-2 pr-3 text-text-faint">{i + 1}</td>
                    <td className="py-2 pr-3 font-semibold">{d.driver}</td>
                    <td className="py-2 pr-3">{fmtLapTime(d.best_lap_s)}</td>
                    <td className="py-2 pr-3">{d.best_lap_number ?? '-'}</td>
                    <td className="py-2 pr-3 text-text-muted">{d.compounds.join(', ') || '-'}</td>
                    <td className="py-2 pr-3">
                      <div className="flex gap-2">
                        <button className={`tiny-btn ${selectedDriver === d.driver ? 'active' : ''}`} onClick={() => pickDriver(d, 'main')}>A</button>
                        <button className={`tiny-btn ${compareDriver === d.driver ? 'active' : ''}`} onClick={() => pickDriver(d, 'compare')}>B</button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="space-y-5">
            <div className="card p-4 grid grid-cols-2 gap-3">
              <Field label="Piloto A">
                <DriverSelect value={selectedDriver} onChange={(driver) => {
                  const row = driverRows.find((d) => d.driver === driver);
                  if (row) pickDriver(row, 'main');
                }} rows={driverRows} />
              </Field>
              <Field label="Volta A">
                <input className="control" type="number" value={selectedLap ?? ''} onChange={(e) => setSelectedLap(e.target.value ? Number(e.target.value) : null)} />
              </Field>
              <Field label="Piloto B">
                <DriverSelect value={compareDriver} onChange={(driver) => {
                  const row = driverRows.find((d) => d.driver === driver);
                  if (row) pickDriver(row, 'compare');
                }} rows={driverRows} />
              </Field>
              <Field label="Volta B">
                <input className="control" type="number" value={compareLap ?? ''} onChange={(e) => setCompareLap(e.target.value ? Number(e.target.value) : null)} />
              </Field>
            </div>

            <DriftCard drift={drift} />
          </div>
        </section>
      )}

      {summary && selectedDriver && year != null && (
        <RaceEngineerPanel year={year} gp={gp} session={sessionCode} driver={selectedDriver} lap={selectedLap} />
      )}

      {error && <div className="card p-4 text-yellow-400 text-sm">{error}</div>}
      <TelemetryPanel telemetry={telemetry} comparison={comparison} loading={loading && !!summary} error={null} />
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs text-text-muted block mb-1 uppercase font-semibold">{label}</span>
      {children}
    </label>
  );
}

function DriverSelect({ value, onChange, rows }: { value: string; onChange: (value: string) => void; rows: EngineeringDriverRow[] }) {
  return (
    <select className="control" value={value} onChange={(e) => onChange(e.target.value)}>
      {rows.map((d) => <option key={d.driver} value={d.driver}>{d.driver}</option>)}
    </select>
  );
}

function DriftCard({ drift }: { drift: DriftMonitor | null }) {
  if (!drift) {
    return <div className="card p-4 text-sm text-text-muted">Monitor de drift ainda nao carregado para esta temporada.</div>;
  }
  const report = drift.report as Record<string, unknown>;
  const headline = typeof report.status === 'string' ? report.status : drift.source;
  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold uppercase text-text-muted">Monitor de drift</h3>
        <span className="text-xs rounded bg-white/5 border border-white/10 px-2 py-1">{drift.source}</span>
      </div>
      <div className="text-xl font-extrabold">{headline}</div>
      <p className="text-sm text-text-muted leading-6 mt-3 whitespace-pre-line line-clamp-6">
        {drift.markdown.replace(/^#.*$/m, '').trim().slice(0, 700) || 'Relatorio de drift disponivel para leitura via API.'}
      </p>
    </div>
  );
}
