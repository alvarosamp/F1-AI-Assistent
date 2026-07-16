/// <reference types="cypress" />

describe('ModelAnalysisPage', () => {
  it('mostra "Carregando…" antes da resposta da API', () => {
    cy.intercept('GET', '/api/model-analysis', (req) => {
      req.reply({ delay: 300, fixture: 'model-analysis.json' });
    }).as('getAnalysisSlow');

    cy.visit('/#/analise');
    cy.contains('Carregando…').should('be.visible');
    cy.wait('@getAnalysisSlow');
  });

  describe('com dados carregados', () => {
    beforeEach(() => {
      cy.intercept('GET', '/api/model-analysis', { fixture: 'model-analysis.json' }).as('getAnalysis');
      cy.visit('/#/analise');
      cy.wait('@getAnalysis');
    });

    it('mostra o gráfico de feature importance com o título correto', () => {
      cy.contains('📊 Feature Importance (Top 20)').should('be.visible');
    });

    it('mostra as 4 métricas walk-forward', () => {
      cy.contains('0.72').should('be.visible'); // RMSE
      cy.contains('0.81').should('be.visible'); // R²
      cy.contains('0.51').should('be.visible'); // MAE
      cy.contains('34.5%').should('be.visible'); // Ganho vs trivial
      cy.contains('RMSE').should('be.visible');
      cy.contains('Ganho vs Trivial').should('be.visible');
    });

    it('mostra o gráfico de RMSE por Grand Prix', () => {
      cy.contains('🌍 RMSE por Grand Prix (Ordenado)').should('be.visible');
    });
  });

  it('não renderiza os gráficos quando as listas vêm vazias', () => {
    cy.intercept('GET', '/api/model-analysis', {
      body: {
        metrics: { rmse: 0.9, r2: 0.7, mae: 0.6, gain_vs_trivial_pct: 10.0 },
        feature_importance: [],
        rmse_by_gp: [],
      },
    }).as('getEmptyAnalysis');

    cy.visit('/#/analise');
    cy.wait('@getEmptyAnalysis');

    cy.contains('📊 Feature Importance (Top 20)').should('not.exist');
    cy.contains('🌍 RMSE por Grand Prix (Ordenado)').should('not.exist');
    // As métricas walk-forward continuam visíveis mesmo sem os gráficos
    cy.contains('Ganho vs Trivial').should('be.visible');
  });
});
