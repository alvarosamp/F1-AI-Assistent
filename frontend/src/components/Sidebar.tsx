import { NavLink } from 'react-router-dom';

const NAV_ITEMS = [
  { to: '/', label: '🏁 Simulação de Corrida' },
  { to: '/calibracao', label: '📊 Calibração' },
  { to: '/analise', label: '🧠 Análise do Modelo' },
  { to: '/sobre', label: 'ℹ️ Sobre o Projeto' },
];

export function Sidebar() {
  return (
    <aside className="w-64 shrink-0 border-r border-white/10 p-5 hidden md:block">
      <h1 className="text-xl font-extrabold tracking-tight mb-6">🏎️ F1 AI Insights</h1>
      <nav className="flex flex-col gap-1">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-[#E10600]/20 text-white border border-[#E10600]/40'
                  : 'text-gray-400 hover:bg-white/5 hover:text-white'
              }`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
