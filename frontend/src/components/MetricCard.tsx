export function MetricCard({ label, value, sublabel }: { label: string; value: string; sublabel?: string }) {
  return (
    <div className="card p-5 text-center transition-colors hover:border-accent/30">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-text-muted">{label}</div>
      <div className="text-3xl font-extrabold my-2 text-text">{value}</div>
      {sublabel && <div className="text-[11px] font-semibold uppercase tracking-wider text-text-faint">{sublabel}</div>}
    </div>
  );
}
