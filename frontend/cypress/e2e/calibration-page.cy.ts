/// <reference types="cypress" />

describe('CalibrationPage', () => {
  it('mostra "Carregando…" antes da resposta da API', () => {
    cy.intercept('GET', '/api/calibration', (req) => {
      req.reply({ delay: 300, fixture: 'calibration-report.json' });
    }).as('getCalibrationSlow');

    cy.visit('/#/calibracao');
    cy.contains('Carregando…').should('be.visible');
    cy.wait('@getCalibrationSlow');
  });

  it('mostra mensagem de erro se o pipeline de calibração não tiver rodado', () => {
    cy.intercept('GET', '/api/calibration', { statusCode: 500, body: {} }).as('getCalibrationError');
    cy.visit('/#/calibracao');
    cy.wait('@getCalibrationError');

    cy.contains('rode o pipeline de calibração primeiro.').should('be.visible');
  });

  describe('com dados carregados', () => {
    beforeEach(() => {
      cy.intercept('GET', '/api/calibration', { fixture: 'calibration-report.json' }).as('getCalibration');
      cy.visit('/#/calibracao');
      cy.wait('@getCalibration');
    });

    it('mostra os 5 cards de métricas com o ganho vs baseline correto', () => {
      cy.contains('Vitória').parents('.card').should('contain', '+4.0%');
      cy.contains('Pódio').parents('.card').should('contain', '+20.0%');
      cy.contains('Top 6').parents('.card').should('contain', '+35.0%');
      cy.contains('Top 10').parents('.card').should('contain', '+12.0%');
      cy.contains('DNF').parents('.card').should('contain', '+1.4%');
    });

    it('renderiza um reliability diagram por mercado', () => {
      cy.contains('Reliability Diagrams').should('be.visible');
      cy.get('h4').contains('Vitória').should('be.visible');
      cy.get('h4').contains('Pódio').should('be.visible');
      cy.get('h4').contains('Top 6').should('be.visible');
      cy.get('h4').contains('Top 10').should('be.visible');
      cy.get('h4').contains('DNF').should('be.visible');
    });

    it('mostra a tabela de métricas detalhadas com os valores corretos', () => {
      cy.get('table').contains('td', 'Vitória').parents('tr').within(() => {
        cy.get('td').eq(1).should('contain', '0.0812'); // brier modelo
        cy.get('td').eq(2).should('contain', '0.0846'); // brier baseline
        cy.get('td').eq(3).should('contain', '+4.0%');
        cy.get('td').eq(4).should('contain', '0.0110'); // ECE
        cy.get('td').eq(5).should('contain', '5.0%'); // base rate
      });
    });

    it('mostra a nota de interpretação', () => {
      cy.contains('Interpretação:').should('be.visible');
      cy.contains('Brier Score').should('be.visible');
      cy.contains('Expected Calibration Error').should('be.visible');
      cy.contains('O modelo bate o baseline em todos os 5 mercados').should('be.visible');
    });
  });
});
