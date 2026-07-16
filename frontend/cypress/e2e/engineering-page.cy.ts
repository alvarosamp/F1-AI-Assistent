/// <reference types="cypress" />

/**
 * fetchEngineeringYears() -> GET /api/engineering/sessions
 * fetchEngineeringGps(year) -> GET /api/engineering/sessions?year=YYYY
 * Ambos batem no mesmo pathname, então usamos um único intercept com
 * handler que decide a fixture pela presença de "year=" na query string.
 */
function interceptSessions() {
  return cy.intercept('GET', '/api/engineering/sessions*', (req) => {
    if (req.url.includes('year=')) {
      req.reply({ fixture: 'engineering-gps.json' });
    } else {
      req.reply({ fixture: 'engineering-years.json' });
    }
  }).as('getSessions');
}

describe('EngineeringPage', () => {
  beforeEach(() => {
    interceptSessions();
    cy.intercept('GET', '/api/engineering/drift?year=*', { fixture: 'engineering-drift.json' }).as('getDrift');

    cy.visit('/#/engenharia');
    // 1a chamada: anos; 2a chamada (disparada pelo useEffect[year]): gps do ano selecionado
    cy.wait(['@getSessions', '@getSessions', '@getDrift']);
  });

  it('carrega anos, sessões e GPs disponíveis, e o monitor de drift', () => {
    cy.get('select').eq(0).find('option').should('have.length', 3);
    cy.get('select').eq(1).find('option').should('have.length', 2);
    cy.get('select').eq(2).find('option').should('have.length', 5);

    cy.contains('Monitor de drift').should('be.visible');
    cy.contains('estavel').should('be.visible');
    cy.contains('drift_monitor_2024').should('be.visible');
  });

  it('mostra placeholder de drift quando ainda não carregado', () => {
    cy.intercept('GET', '/api/engineering/drift?year=*', { statusCode: 500, body: {} });
    interceptSessions();
    cy.visit('/#/engenharia');
    cy.wait(['@getSessions', '@getSessions']);
    cy.contains('Monitor de drift ainda nao carregado para esta temporada.').should('be.visible');
  });

  it('carrega o sumário da sessão ao clicar em "Carregar"', () => {
    cy.intercept('GET', '/api/engineering/summary?*', { fixture: 'engineering-summary.json' }).as('getSummary');

    cy.contains('button', 'Carregar').click();
    cy.wait('@getSummary');

    cy.contains('Bahrain Grand Prix - Classificacao').should('be.visible');
    cy.contains('td', 'VER').should('be.visible');
    cy.contains('td', 'LEC').should('be.visible');
    cy.contains('td', 'NOR').should('be.visible');
  });

  it('pré-seleciona o piloto A (1º da tabela) após carregar o sumário', () => {
    cy.intercept('GET', '/api/engineering/summary?*', { fixture: 'engineering-summary.json' }).as('getSummary');
    cy.contains('button', 'Carregar').click();
    cy.wait('@getSummary');

    cy.contains('button', 'A').first().should('have.class', 'active');
  });

  it('carrega telemetria e comparação ao clicar em "Analisar telemetria"', () => {
    cy.intercept('GET', '/api/engineering/summary?*', { fixture: 'engineering-summary.json' }).as('getSummary');
    cy.intercept('GET', '/api/engineering/telemetry?*', { fixture: 'engineering-telemetry-ver.json' }).as('getTelemetry');
    cy.intercept('GET', '/api/engineering/compare?*', { fixture: 'engineering-comparison.json' }).as('getComparison');

    cy.contains('button', 'Carregar').click();
    cy.wait('@getSummary');

    cy.contains('button', 'Analisar telemetria').click();
    cy.wait(['@getTelemetry', '@getComparison']);

    cy.get('.card').contains('Piloto').closest('.card').should('contain', 'VER');
    cy.contains('Comparacao VER x LEC').should('be.visible');
    cy.contains('Race Engineer Copilot').should('be.visible');
    cy.contains('VER foi mais rapido por 0.24s').should('be.visible');
    cy.contains('gap_pequeno').should('be.visible');
  });

  it('troca o piloto A ao clicar no botão da linha de outro piloto', () => {
    cy.intercept('GET', '/api/engineering/summary?*', { fixture: 'engineering-summary.json' }).as('getSummary');
    cy.contains('button', 'Carregar').click();
    cy.wait('@getSummary');

    cy.contains('td', 'NOR').parents('tr').within(() => {
      cy.contains('button', 'A').click();
      cy.contains('button', 'A').should('have.class', 'active');
    });
  });

  it('exibe mensagem de erro se o carregamento do sumário falhar', () => {
    cy.intercept('GET', '/api/engineering/summary?*', { statusCode: 500, body: {} }).as('getSummaryError');
    cy.contains('button', 'Carregar').click();
    cy.wait('@getSummaryError');

    cy.contains('GET /api/engineering/summary').should('be.visible');
    cy.contains('falhou: 500').should('be.visible');
  });

  it('desabilita "Carregar" quando nenhum GP está disponível para o ano', () => {
    cy.intercept('GET', '/api/engineering/sessions*', (req) => {
      if (req.url.includes('year=')) {
        req.reply({ body: { year: 2024, gps: [], session_codes: ['FP1', 'FP2', 'FP3', 'Q', 'R'] } });
      } else {
        req.reply({ fixture: 'engineering-years.json' });
      }
    }).as('getSessionsEmpty');

    cy.visit('/#/engenharia');
    cy.wait(['@getSessionsEmpty', '@getSessionsEmpty']);

    cy.contains('button', 'Carregar').should('be.disabled');
  });
});
