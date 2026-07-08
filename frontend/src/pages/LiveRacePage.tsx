import { useEffect, useState } from 'react';
import { PageHeader } from '../components/PageHeader';
import { TelemetryPanel } from '../components/TelemetryPanel';
import {
  connectLive,
  fetchLiveState,
  fetchLiveStatus,
  fetchLiveTelemetry,
  fetchReference,
  fetchReplaySessions,
  fetchReplayState,
  fetchReplayTelemetry,
  startReplay,
  type DriverTelemetry,
  type LiveState,
  type LiveStatus,
  type ReplayState,
} from '../lib/api';

type Mode = 'live' | 'replay';

function fmtLapTime(s: number | null) {
  if (s == null) return '—';
  const m = Math.floor(s / 60);
  const rem = (s % 60).toFixed(3);
  return `${m}:${rem.padStart(6, '0')}`;
}

export function LiveRacePage() {
  const [mode, setMode] = useState<Mode>('replay');

  // Replay setup
  const [year, setYear] = useState(2024);
  const [gps, setGps] = useState<string[]>([]);
  const [gp, setGp] = useState('');
  const [sessionCode, setSessionCode] = useState('R');
  const [replayId, setReplayId] = useState<string | null>(null);
  const [replayState, setReplayState] = useState<ReplayState | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  // Live
  const [liveStatus, setLiveStatus] = useState<LiveStatus | null>(null);
  const [liveState, setLiveState] = useState<LiveState | null>(null);
  const [connecting, setConnecting] = useState(false);

  const [telemetry, setTelemetry] = useState<DriverTelemetry | null>(null);
  const [telemetryLoading, setTelemetryLoading] = useState(false);
  const [telemetryError, setTelemetryError] = useState<string | null>(null);
  const [panelOpen, setPanelOpen] = useState(false);
  const [selectedDriver, setSelectedDriver] = useState<string | null>(null);

  useEffect(() => {
    fetchReplaySessions().then((data) => {
      if (data.years.length) setYear(data.years[data.years.length - 1]);
    });
    fetchReference().then((data) => {
      setGps(data.gps);
      setGp(data.gps[0] ?? '');
    });
  }, []);

  // Poll replay state
  useEffect(() => {
    if (mode !== 'replay' || !replayId) return;
    const tick = () => fetchReplayState(replayId).then(setReplayState).catch(() => {});
    tick();
    const id = window.setInterval(tick, 2500);
    return () => window.clearInterval(id);
  }, [mode, replayId]);

  // Poll live status/state
  useEffect(() => {
    if (mode !== 'live') return;
    const tick = () => {
      fetchLiveStatus().then(setLiveStatus).catch(() => {});
      fetchLiveState().then(setLiveState).catch(() => setLiveState(null));
    };
    tick();
    const id = window.setInterval(tick, 3000);
    return () => window.clearInterval(id);
  }, [mode]);

  // Refresh selected driver's telemetry while panel is open (live/replay progresses)
  useEffect(() => {
    if (!panelOpen || !selectedDriver) return;
    loadTelemetry(selectedDriver);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replayState?.lap_number, liveState?.lap_number]);

  async function handleStartReplay() {
    setStarting(true);
    setStartError(null);
    setReplayState(null);
    try {
      const res = await startReplay({ year, gp, session: sessionCode });
      setReplayId(res.replay_id);
    } catch (e) {
      setStartError((e as Error).message);
    } finally {
      setStarting(false);
    }
  }

  async function handleConnectLive() {
    setConnecting(true);
    try {
      await connectLive({ year, gp, session: sessionCode });
    } finally {
      setConnecting(false);
    }
  }

  async function loadTelemetry(driver: string) {
    setSelectedDriver(driver);
    setPanelOpen(true);
    setTelemetryLoading(true);
    setTelemetryError(null);
    try {
      const data = mode === 'replay' && replayId
        ? await fetchReplayTelemetry(replayId, driver)
        : await fetchLiveTelemetry(driver);
      setTelemetry(data);
    } catch (e) {
      setTelemetryError((e as Error).message);
    } finally {
      setTelemetryLoading(false);
    }
  }

  const standings = mode === 'replay' ? replayState?.standings : liveState?.standings;

  return (
    <div>
      <PageHeader title="Corrida ao Vivo" subtitle="Ao vivo (sessão real de F1) ou replay de uma corrida já disputada" />

      <div className="flex gap-2 mb-6">
        <ModeButton active={mode === 'replay'} onClick={() => setMode('replay')}>Replay</ModeButton>
        <ModeButton active={mode === 'live'} onClick={() => setMode('live')}>Ao Vivo</ModeButton>
      </div>

      <div className="card p-5 mb-6 flex flex-wrap items-end gap-4">
        <Field label="Ano">
          <select className="bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-sm" value={year} onChange={(e) => setYear(Number(e.target.value))}>
            {[2022, 2023, 2024, 2025, 2026].map((y) => <option key={y} value={y}>{y}</option>)}
          </select>
        </Field>
        <Field label="Grand Prix">
          <select className="bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-sm min-w-[220px]" value={gp} onChange={(e) => setGp(e.target.value)}>
            {gps.map((g) => <option key={g} value={g}>{g}</option>)}
          </select>
        </Field>
        <Field label="Sessão">
          <select className="bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-sm" value={sessionCode} onChange={(e) => setSessionCode(e.target.value)}>
            <option value="R">Corrida</option>
            <option value="Q">Classificação</option>
          </select>
        </Field>

        {mode === 'replay' ? (
          <button onClick={handleStartReplay} disabled={starting} className="bg-accent hover:bg-accent-dim disabled:opacity-50 text-white font-bold px-5 py-2 rounded-lg">
            {starting ? 'Carregando…' : 'Iniciar Replay'}
          </button>
        ) : (
          <button onClick={handleConnectLive} disabled={connecting} className="bg-accent hover:bg-accent-dim disabled:opacity-50 text-white font-bold px-5 py-2 rounded-lg">
            {connecting ? 'Conectando…' : 'Conectar ao Vivo'}
          </button>
        )}
      </div>

      {startError && <p className="text-yellow-500 text-sm mb-4">{startError}</p>}

      {mode === 'live' && liveStatus && !liveStatus.session_active && (
        <p className="text-yellow-500 text-sm mb-4">
          Nenhuma sessão de F1 ao vivo no momento — a conexão fica esperando. Use o modo Replay para ver uma corrida real já disputada.
        </p>
      )}
      {mode === 'live' && liveState && !liveState.available && liveState.error && (
        <p className="text-text-muted text-sm mb-4">{liveState.error}</p>
      )}

      {mode === 'replay' && replayState && (
        <p className="text-text-muted text-sm mb-3">
          Volta {replayState.lap_number} / {replayState.total_laps} {replayState.finished && '· corrida encerrada'}
        </p>
      )}

      {standings && standings.length > 0 && (
        <div className="card p-4 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-text-muted border-b border-white/10">
                <th className="py-2 pr-3">Pos</th>
                <th className="py-2 pr-3">Piloto</th>
                <th className="py-2 pr-3">Última volta</th>
                <th className="py-2 pr-3">Pneu</th>
              </tr>
            </thead>
            <tbody>
              {standings.map((s) => (
                <tr
                  key={s.driver}
                  className="border-b border-white/5 cursor-pointer hover:bg-white/[0.03]"
                  onClick={() => loadTelemetry(s.driver)}
                >
                  <td className="py-2 pr-3 font-semibold">{s.position ?? '—'}</td>
                  <td className="py-2 pr-3">{s.driver}</td>
                  <td className="py-2 pr-3">{fmtLapTime(s.last_lap_s)}</td>
                  <td className="py-2 pr-3 text-text-muted">{s.compound ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-text-faint mt-3">Clique num piloto para ver a telemetria real da volta atual.</p>
        </div>
      )}

      {!standings?.length && (
        <div className="rounded-2xl border border-dashed border-white/10 p-10 text-center text-text-faint">
          {mode === 'replay' ? 'Inicie um replay para ver a classificação avançar volta a volta.' : 'Conecte para receber dados quando houver uma sessão de F1 real acontecendo.'}
        </div>
      )}

      {panelOpen && (
        <TelemetryPanel telemetry={telemetry} loading={telemetryLoading} error={telemetryError} onClose={() => setPanelOpen(false)} />
      )}
    </div>
  );
}

function ModeButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 rounded-lg text-sm font-semibold transition-colors ${
        active ? 'bg-accent text-white' : 'bg-white/5 text-text-muted hover:bg-white/10'
      }`}
    >
      {children}
    </button>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="text-xs text-text-muted block mb-1">{label}</label>
      {children}
    </div>
  );
}
