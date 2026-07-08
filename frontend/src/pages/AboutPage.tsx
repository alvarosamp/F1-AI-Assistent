import { PageHeader } from '../components/PageHeader';

export function AboutPage() {
  return (
    <div className="prose prose-invert max-w-none">
      <PageHeader title="F1 AI Race Insights" subtitle="Sistema de decision support para corridas de Fórmula 1" />

      <div className="space-y-6 text-gray-300 text-sm leading-relaxed">
        <section>
          <h3 className="text-lg font-bold text-white mb-2">Arquitetura</h3>
          <p>O sistema é composto por 4 modelos que alimentam um simulador Monte Carlo:</p>
          <ol className="list-decimal list-inside space-y-2 mt-2">
            <li><strong>Modelo de Lap Time</strong> (XGBoost, RMSE 0.72s walk-forward) — 49 features incluindo telemetria, weather, race control, Ergast. Target encoding CV-safe para Driver/Team/GP. Validação walk-forward: treina 2022+2023, testa 2024.</li>
            <li><strong>Modelo de DNF</strong> (Logistic Regression, Brier 0.087) — Features históricas anti-leakage (expanding mean + shift). Calibrado nativamente (ECE 0.014).</li>
            <li><strong>Modelo de Safety Car</strong> (Beta-Binomial bayesiano) — Estimativa por pista com smoothing e pesos temporais.</li>
            <li><strong>Modelo de Degradação de Pneu</strong> (Regressão linear por compound) — Coeficientes medidos empiricamente em 59k voltas.</li>
          </ol>
        </section>

        <section>
          <h3 className="text-lg font-bold text-white mb-2">Simulador Monte Carlo</h3>
          <p>
            O simulador roda a corrida <strong>10.000 vezes</strong> em ~12 segundos, amostrando eventos
            aleatórios (DNF, Safety Car) a cada simulação. O resultado é uma <strong>distribuição de
            probabilidade</strong> por piloto por posição final.
          </p>
        </section>

        <section>
          <h3 className="text-lg font-bold text-white mb-2">Calibração</h3>
          <p>
            Validado contra todas as 24 corridas de 2024. O modelo <strong>bate o baseline trivial (grid
            position) em todos os 5 mercados</strong>: Win (+4%), Podium (+20%), Top 6 (+35%), Top 10 (+12%),
            DNF (+1.4%).
          </p>
        </section>

        <section>
          <h3 className="text-lg font-bold text-white mb-2">Decisões Técnicas Importantes</h3>
          <ul className="list-disc list-inside space-y-1">
            <li><strong>Anti-leakage</strong>: 10+ testes pytest provando que nenhuma feature vaza info do futuro</li>
            <li><strong>Target encoding</strong>: substituiu label encoding que estava piorando o modelo</li>
            <li><strong>Walk-forward</strong>: validação temporal honesta (treina no passado, testa no futuro)</li>
            <li><strong>Reliability diagrams</strong>: prova formal que probabilidades são calibradas</li>
            <li><strong>Baseline comparison</strong>: todo modelo é comparado contra "chutar a média"</li>
          </ul>
        </section>

        <section>
          <h3 className="text-lg font-bold text-white mb-2">Stack Tecnológico</h3>
          <p>FastF1 · XGBoost · scikit-learn · Optuna · MLflow · NumPy · Pandas · FastAPI · React · Recharts</p>
        </section>

        <section>
          <h3 className="text-lg font-bold text-white mb-2">Dados</h3>
          <ul className="list-disc list-inside space-y-1">
            <li><strong>96.598 voltas brutas</strong> coletadas via FastF1 (2022+2023+2024)</li>
            <li><strong>59.362 voltas de race</strong> após filtragem (IQR + residual)</li>
            <li><strong>78 features</strong> por volta</li>
            <li><strong>Weather, race control, Ergast</strong> integrados por volta</li>
          </ul>
        </section>

        <hr className="border-white/10" />
        <p className="text-xs text-gray-500 italic">
          Projeto desenvolvido como sistema de decision support. Não é aconselhamento de apostas.
        </p>
      </div>
    </div>
  );
}
