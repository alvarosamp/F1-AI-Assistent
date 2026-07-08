export function MetricCard({ label, value, sublabel }: { label: string; value: string; sublabel?: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur p-6 text-center transition-transform hover:-translate-y-1 hover:border-[#E10600]/40">
      <div className="text-xs font-semibold uppercase tracking-wider text-gray-400">{label}</div>
      <div className="text-3xl font-extrabold my-2 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
        {value}
      </div>
      {sublabel && <div className="text-xs font-semibold uppercase tracking-wider text-gray-500">{sublabel}</div>}
    </div>
  );
}
