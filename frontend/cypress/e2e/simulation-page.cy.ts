/// <reference types="cypress" />

describe('SimulationPage', () => {
  beforeEach(() => {
    cy.intercept('GET', '/api/reference', { fixture: 'reference.json' }).as('getReference');
    cy.visit('/#/previsoes');
    cy.wait('@getReference');
  });

  it('carrega o formulário com o GP e o número de voltas corretos', () => {
    cy.get('select').eq(0).find('option').should('have.length', 3);
    cy.get('select').eq(0).find('option').first().should('have.text', 'Bahrain Grand Prix');
    cy.contains('🏁 Voltas: 57').should('be.visible');
  });

  it('inicia com o grid padrão de 10 pilotos selecionado', () => {
    cy.contains('🏎️ Grid de Largada (10/20)').should('be.visible');
    cy.contains('button', 'VER').should('have.class', 'bg-[#E10600]');
    cy.contains('button', 'BOT').should('not.have.class', 'bg-[#E10600]');
  });

  it('desabilita "Simular Corrida" quando restam menos de 2 pilotos', () => {
    // Remove 9 dos 10 pilotos do grid padrão, deixando só 1
    ['SAI', 'LEC', 'NOR', 'PIA', 'RUS', 'HAM', 'ALO', 'STR', 'PER'].forEach((d) => {
      cy.contains('button', d).click();
    });
    cy.contains('🏎️ Grid de Largada (1/20)').should('be.visible');
    cy.contains('Selecione pelo menos 2 pilotos.').should('be.visible');
    cy.contains('button', '🏁 Simular Corrida').should('be.disabled');
  });

  it('permite adicionar um piloto fora do grid padrão (toggle de seleção)', () => {
    cy.contains('button', 'BOT').should('not.have.class', 'bg-[#E10600]');
    cy.contains('button', 'BOT').click();
    cy.contains('🏎️ Grid de Largada (11/20)').should('be.visible');
    cy.contains('button', 'BOT').should('have.class', 'bg-[#E10600]');

    cy.contains('button', 'BOT').click();
    cy.contains('🏎️ Grid de Largada (10/20)').should('be.visible');
  });

  it('roda a simulação e mostra tabela, gráfico de vitórias e heatmap de posição', () => {
    cy.intercept('POST', '/api/simulate', { fixture: 'simulate-response.json' }).as('simulate');

    cy.contains('button', '🏁 Simular Corrida').click();
    cy.wait('@simulate');

    cy.contains('Resultados — Bahrain Grand Prix').should('be.visible');
    cy.get('table').contains('td', 'VER').parents('tr').within(() => {
      cy.get('td').eq(2).should('contain', '32.0%'); // win
      cy.get('td').eq(3).should('contain', '58.0%'); // podium
    });

    cy.contains('🏆 Probabilidade de Vitória').should('be.visible');
    cy.contains('🚦 Distribuição de Posição Final (%)').should('be.visible');
    cy.contains('SC probability: 42.0%').should('be.visible');
    cy.contains('Simulações: 2,000').should('be.visible');
  });

  it('não mostra o banner de ajuste de temporada quando season_adaptation é nulo', () => {
    cy.intercept('POST', '/api/simulate', { fixture: 'simulate-response.json' }).as('simulate');
    cy.contains('button', '🏁 Simular Corrida').click();
    cy.wait('@simulate');

    cy.contains('Ajuste de temporada aplicado').should('not.exist');
  });

  it('mostra o banner de ajuste de temporada quando season_adaptation está presente', () => {
    cy.intercept('POST', '/api/simulate', { fixture: 'simulate-response-adaptation.json' }).as('simulateAdapted');
    cy.contains('button', '🏁 Simular Corrida').click();
    cy.wait('@simulateAdapted');

    cy.contains('Ajuste de temporada aplicado').should('be.visible');
    cy.contains('regulation_shift_2026').should('be.visible');
  });

  it('exibe mensagem de erro se a simulação falhar', () => {
    cy.intercept('POST', '/api/simulate', { statusCode: 500, body: {} }).as('simulateError');
    cy.contains('button', '🏁 Simular Corrida').click();
    cy.wait('@simulateError');

    cy.contains('POST /api/simulate falhou: 500').should('be.visible');
  });

  it('mostra o botão em estado de carregamento durante a simulação', () => {
    cy.intercept('POST', '/api/simulate', (req) => {
      req.reply({ delay: 500, fixture: 'simulate-response.json' });
    }).as('simulateSlow');

    cy.contains('button', '🏁 Simular Corrida').click();
    cy.contains('button', 'Simulando...').should('be.disabled');
    cy.wait('@simulateSlow');
    cy.contains('button', '🏁 Simular Corrida').should('be.visible');
  });

  it('mostra o placeholder antes de rodar qualquer simulação', () => {
    cy.contains('Configure o grid e clique em "Simular Corrida" para ver os resultados.').should('be.visible');
  });

  it('permite trocar de GP e reflete o novo número de voltas', () => {
    cy.get('select').eq(0).select('Saudi Arabian Grand Prix');
    cy.contains('🏁 Voltas: 50').should('be.visible');
  });
});
