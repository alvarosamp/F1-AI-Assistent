/// <reference types="cypress" />

describe('AboutPage', () => {
  beforeEach(() => {
    cy.visit('/#/sobre');
  });

  it('mostra o título e subtítulo do projeto', () => {
    cy.contains('F1 AI Race Insights').should('be.visible');
    cy.contains('Sistema de decision support para corridas de Fórmula 1').should('be.visible');
  });

  it('mostra todas as seções principais', () => {
    ['Arquitetura', 'Simulador Monte Carlo', 'Calibração', 'Decisões Técnicas Importantes', 'Stack Tecnológico', 'Dados'].forEach((heading) => {
      cy.get('h3').contains(heading).should('be.visible');
    });
  });

  it('descreve os 4 modelos do pipeline na seção de arquitetura', () => {
    cy.contains('Modelo de Lap Time').should('be.visible');
    cy.contains('Modelo de DNF').should('be.visible');
    cy.contains('Modelo de Safety Car').should('be.visible');
    cy.contains('Modelo de Degradação de Pneu').should('be.visible');
  });

  it('mostra os números-chave do simulador e da base de dados', () => {
    cy.contains('10.000 vezes').should('be.visible');
    cy.contains('96.598 voltas brutas').should('be.visible');
    cy.contains('59.362 voltas de race').should('be.visible');
    cy.contains('78 features').should('be.visible');
  });

  it('mostra o resultado de calibração contra as 24 corridas de 2024', () => {
    cy.contains('bate o baseline trivial').should('be.visible');
    cy.contains('Win (+4%)').should('be.visible');
    cy.contains('Podium (+20%)').should('be.visible');
  });

  it('mostra o stack tecnológico completo', () => {
    cy.contains('FastF1').should('be.visible');
    cy.contains('XGBoost').should('be.visible');
    cy.contains('FastAPI').should('be.visible');
    cy.contains('React').should('be.visible');
    cy.contains('Recharts').should('be.visible');
  });

  it('mostra o aviso legal no rodapé', () => {
    cy.contains('Não é aconselhamento de apostas.').should('be.visible');
  });

  it('não faz nenhuma chamada de API (página 100% estática)', () => {
    let apiCalled = false;
    cy.intercept('/api/**', () => { apiCalled = true; });
    cy.visit('/#/sobre');
    cy.contains('F1 AI Race Insights').should('be.visible').then(() => {
      expect(apiCalled).to.be.false;
    });
  });
});
