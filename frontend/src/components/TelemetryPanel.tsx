import {
  Area,
  AreaChart,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { DriverTelemetry, EngineeringComparison, TelemetrySample } from '../lib/api';

interface Props {
  telemetry: DriverTelemetry | null;
  comparison?: EngineeringComparison | null;
  loading: boolean;
  error: string | null;
  onClose?: () => void;
}

const tooltipStyle = {
  background: '#12121a',
  border: '1px solid rgba(255,255,255,0.12)',
  borderRadius: 8,
  fontSize: 12,
};

function fmt(value: number | null | undefined, digits = 1) {
  if (value == null || Number.isNaN(value)) return '-';
  return value.toFixed(digits);
}

function fmtLapTime(s: number | null) {
  if (s == null) return '-';
  const m = Math.floor(s / 60);
  const rem = (s % 60).toFixed(3);
  return `${m}:${rem.padStart(6, '0')}`;
}

export function TelemetryPanel({ telemetry, comparison, loading, error, onClose }: Props) {
  if (loading) return <div className="card p-5 text-text-muted">Carregando telemetria real e calculando sinais...</div>;
  if (error) return <div className="card p-5 text-yellow-400">{error}</div>;
  if (!telemetry) return <div className="card p-5 text-text-muted">Carregue uma sessao para abrir o painel de engenharia.</div>;

  return (
    <div className="space-y-5">
      {onClose && (
        <div className="flex justify-end">
          <button onClick={onClose} className="text-sm text-text-muted hover:text-white">
            Fechar
          </button>
        </div>
      )}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <TelemetryMetric label="Piloto" value={telemetry.driver} sub={telemetry.team ?? telemetry.compound ?? undefined} />
        <TelemetryMetric label="Volta" value={String(telemetry.lap_number)} sub={fmtLapTime(telemetry.lap_time_s)} />
        <TelemetryMetric label="Vel. max" value={`${fmt(telemetry.summary.max_speed)} km/h`} sub="Speed" />
        <TelemetryMetric label="Ar sujo" value={`${fmt(telemetry.summary.dirty_air_pct)}%`} sub="DistanceToDriverAhead" />
      </div>

      <div className="grid xl:grid-cols-[1.4fr_0.8fr] gap-5">
        <div className="space-y-3">
          <SignalChart title="Speed / Throttle / Brake pressure proxy" data={telemetry.samples} lines={[
            ['Speed', '#e10600'],
            ['Throttle', '#2dd4bf'],
            ['brake_pressure_proxy', '#f59e0b'],
          ]} />
          <SignalChart title="RPM / Gear / DRS" data={telemetry.samples} lines={[
            ['RPM', '#60a5fa'],
            ['nGear', '#f97316'],
            ['DRS', '#22c55e'],
          ]} />
          <SignalChart title="Steering proxy / lateral change / acceleration proxy" data={telemetry.samples} lines={[
            ['steering_proxy', '#a78bfa'],
            ['lateral_change', '#f472b6'],
            ['accel_proxy', '#38bdf8'],
          ]} />
          <DirtyAirChart data={telemetry.samples} />
        </div>
        <div className="space-y-5">
          <TrackMap data={telemetry.samples} />
          <SummaryTable telemetry={telemetry} />
        </div>
      </div>

      {comparison && (
        <div className="grid xl:grid-cols-[1.25fr_0.75fr] gap-5">
          <div className="card p-4">
            <h3 className="text-sm font-bold uppercase text-text-muted mb-3">
              Comparacao {comparison.driver_a.driver} x {comparison.driver_b.driver}
            </h3>
            <ResponsiveContainer width="100%" height={260}>
              <ComposedChart data={comparison.comparison_samples} margin={{ top: 8, right: 12, left: -14, bottom: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="Distance" tick={{ fill: '#8b8b96', fontSize: 10 }} />
                <YAxis tick={{ fill: '#8b8b96', fontSize: 10 }} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="speed_a" name={`${comparison.driver_a.driver} speed`} stroke="#e10600" dot={false} strokeWidth={1.4} isAnimationActive={false} />
                <Line type="monotone" dataKey="speed_b" name={`${comparison.driver_b.driver} speed`} stroke="#2dd4bf" dot={false} strokeWidth={1.4} isAnimationActive={false} />
                <Line type="monotone" dataKey="time_delta" name="delta tempo" stroke="#f59e0b" dot={false} strokeWidth={1.2} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <CopilotPanel comparison={comparison} />
        </div>
      )}
    </div>
  );
}

function TelemetryMetric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card p-4">
      <div className="text-[10px] uppercase font-semibold text-text-muted">{label}</div>
      <div className="text-xl font-extrabold mt-1 truncate">{value}</div>
      {sub && <div className="text-xs text-text-faint mt-1 truncate">{sub}</div>}
    </div>
  );
}

function SignalChart({ title, data, lines }: { title: string; data: TelemetrySample[]; lines: [keyof TelemetrySample, string][] }) {
  return (
    <div className="card p-4">
      <h3 className="text-xs font-bold uppercase text-text-muted mb-2">{title}</h3>
      <ResponsiveContainer width="100%" height={155}>
        <LineChart data={data} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" />
          <XAxis dataKey="Distance" tick={{ fill: '#8b8b96', fontSize: 10 }} />
          <YAxis tick={{ fill: '#8b8b96', fontSize: 10 }} />
          <Tooltip contentStyle={tooltipStyle} />
          {lines.map(([key, color]) => (
            <Line key={String(key)} type="monotone" dataKey={String(key)} stroke={color} dot={false} strokeWidth={1.25} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function DirtyAirChart({ data }: { data: TelemetrySample[] }) {
  return (
    <div className="card p-4">
      <h3 className="text-xs font-bold uppercase text-text-muted mb-2">Dirty air / carro a frente</h3>
      <ResponsiveContainer width="100%" height={135}>
        <AreaChart data={data} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
          <XAxis dataKey="Distance" tick={{ fill: '#8b8b96', fontSize: 10 }} />
          <YAxis tick={{ fill: '#8b8b96', fontSize: 10 }} />
          <Tooltip contentStyle={tooltipStyle} />
          <Area type="monotone" dataKey="dirty_air_score" stroke="#ef4444" fill="#ef4444" fillOpacity={0.22} isAnimationActive={false} />
          <Line type="monotone" dataKey="DistanceToDriverAhead" stroke="#60a5fa" dot={false} strokeWidth={1.1} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function TrackMap({ data }: { data: TelemetrySample[] }) {
  const points = data.filter((d) => d.X != null && d.Y != null).map((d) => ({ x: d.X, y: d.Y, speed: d.Speed }));
  return (
    <div className="card p-4">
      <h3 className="text-xs font-bold uppercase text-text-muted mb-2">Mapa da volta</h3>
      <ResponsiveContainer width="100%" height={260}>
        <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
          <XAxis dataKey="x" type="number" hide domain={['dataMin', 'dataMax']} />
          <YAxis dataKey="y" type="number" hide domain={['dataMin', 'dataMax']} />
          <Tooltip contentStyle={tooltipStyle} cursor={{ stroke: 'rgba(255,255,255,0.15)' }} />
          <Scatter data={points} fill="#e10600" line={{ stroke: '#e10600', strokeWidth: 1 }} isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function SummaryTable({ telemetry }: { telemetry: DriverTelemetry }) {
  const rows = [
    ['Avg speed', telemetry.summary.avg_speed, 'km/h'],
    ['Full throttle', telemetry.summary.full_throttle_pct, '%'],
    ['Brake usage', telemetry.summary.brake_pct, '%'],
    ['DRS usage', telemetry.summary.drs_pct, '%'],
    ['Cornering intensity', telemetry.summary.cornering_intensity, 'proxy'],
    ['Avg distance ahead', telemetry.summary.avg_distance_to_car_ahead, 'm'],
  ];
  return (
    <div className="card p-4">
      <h3 className="text-xs font-bold uppercase text-text-muted mb-3">Engenharia dos sinais</h3>
      <div className="space-y-2">
        {rows.map(([label, value, unit]) => (
          <div key={String(label)} className="flex items-center justify-between gap-3 border-b border-white/5 pb-2 text-sm">
            <span className="text-text-muted">{label}</span>
            <span className="font-semibold">{fmt(value as number | undefined, 2)} {unit}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CopilotPanel({ comparison }: { comparison: EngineeringComparison }) {
  return (
    <div className="card p-4">
      <h3 className="text-sm font-bold uppercase text-text-muted mb-3">Race Engineer Copilot</h3>
      <p className="text-sm leading-6 text-text">{comparison.copilot.summary}</p>
      <p className="text-sm leading-6 text-text-muted mt-3">{comparison.copilot.recommendation}</p>
      <div className="mt-4 space-y-2">
        {comparison.analysis.findings.map((finding) => (
          <div key={finding} className="rounded-md border border-white/10 bg-white/[0.03] px-3 py-2 text-sm">
            {finding}
          </div>
        ))}
      </div>
      <div className="flex flex-wrap gap-2 mt-4">
        {comparison.copilot.risk_flags.length === 0 ? (
          <span className="text-xs text-text-faint">Sem alerta forte nesta comparacao.</span>
        ) : (
          comparison.copilot.risk_flags.map((flag) => (
            <span key={flag} className="text-xs rounded bg-yellow-500/15 text-yellow-300 border border-yellow-500/20 px-2 py-1">
              {flag}
            </span>
          ))
        )}
      </div>
    </div>
  );
}
