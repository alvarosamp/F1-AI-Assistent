import { NavLink } from 'react-router-dom';

const NAV_ITEMS = [
  { to: '/', label: 'Ao Vivo', icon: '🔴' },
  { to: '/previsoes', label: 'Previsões', icon: '🏁' },
  { to: '/engenharia', label: 'Engenharia', icon: '🧠' },
  { to: '/analise', label: 'Modelo', icon: '📈' },
  { to: '/calibracao', label: 'Calibração', icon: '📊' },
  { to: '/sobre', label: 'Sobre', icon: 'ℹ️' },
];

export function Sidebar() {
  return (
    <aside className="w-60 shrink-0 border-r border-white/[0.08] bg-surface/60 p-4 hidden md:flex md:flex-col">
      <div className="flex items-center gap-2 px-2 mb-8 mt-2">
        <span className="text-2xl">🏎️</span>
        <div>
          <p className="text-sm font-extrabold tracking-tight leading-none">F1 AI</p>
          <p className="text-[11px] text-text-faint leading-none mt-0.5">Race Insights</p>
        </div>
      </div>
      <nav className="flex flex-col gap-1">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-accent/15 text-white border border-accent/30'
                  : 'text-text-muted border border-transparent hover:bg-white/[0.04] hover:text-white'
              }`
            }
          >
            <span className="text-base leading-none">{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>
      <div className="mt-auto px-2 pt-6 text-[11px] text-text-faint leading-relaxed">
        Sistema de decision support.
        <br />
        Não é aconselhamento de apostas.
      </div>
    </aside>
  );
}
