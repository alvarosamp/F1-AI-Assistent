import { useLocation } from 'react-router-dom';

const TITLES: Record<string, string> = {
  '/': 'Corrida ao Vivo',
  '/previsoes': 'Previsões',
  '/engenharia': 'Engenharia',
  '/analise': 'Análise do Modelo',
  '/calibracao': 'Calibração',
  '/sobre': 'Sobre o Projeto',
};

export function Topbar() {
  const { pathname } = useLocation();
  const title = TITLES[pathname] ?? 'F1 AI Race Insights';

  return (
    <header className="h-14 shrink-0 border-b border-white/[0.08] flex items-center px-6 md:px-10 sticky top-0 bg-bg/80 backdrop-blur z-10">
      <span className="text-sm font-semibold text-text-muted">{title}</span>
    </header>
  );
}
