/// <reference types="cypress" />

/**
 * Testes da LiveRacePage (rota "/").
 *
 * Baseado em frontend/src/lib/api.ts:
 * todos os requests saem como `/api${path}` (mesma origem), então os
 * intercepts abaixo usam paths relativos, que o Cypress casa contra o
 * pathname da requisição.
 */

describe('LiveRacePage', () => {
  beforeEach(() => {
    cy.intercept('GET', '/api/reference', { fixture: 'reference.json' }).as('getReference');
    cy.intercept('GET', '/api/replay/sessions', { fixture: 'replay-sessions.json' }).as('getReplaySessions');
  });

  it('carrega a página com o título e os dois modos disponíveis', () => {
    cy.visit('/');
    cy.wait(['@getReference', '@getReplaySessions']);

    cy.contains('Corrida ao Vivo').should('be.visible');
    cy.contains('button', 'Replay').should('be.visible');
    cy.contains('button', 'Ao Vivo').should('be.visible');

    cy.contains('Inicie um replay para ver a classificação avançar volta a volta.').should('be.visible');
  });

  it('preenche o formulário de setup com os GPs recebidos da API', () => {
    cy.visit('/');
    cy.wait(['@getReference', '@getReplaySessions']);

    cy.get('select').eq(1).find('option').should('have.length', 3);
    cy.get('select').eq(1).find('option').first().should('have.text', 'Bahrain Grand Prix');
  });

  describe('Modo Replay', () => {
    it('inicia um replay e exibe a classificação após o estado carregar', () => {
      cy.intercept('POST', '/api/replay/start', {
        statusCode: 200,
        body: { replay_id: 'replay-123', total_laps: 57, event_name: 'Bahrain GP' },
      }).as('startReplay');

      cy.intercept('GET', '/api/replay/replay-123/state', { fixture: 'replay-state.json' }).as('getReplayState');

      cy.visit('/');
      cy.wait(['@getReference', '@getReplaySessions']);

      cy.contains('button', 'Iniciar Replay').click();
      cy.wait('@startReplay');
      cy.wait('@getReplayState');

      cy.contains('Volta 12 / 57').should('be.visible');
      cy.contains('td', 'VER').should('be.visible');
      cy.contains('td', 'NOR').should('be.visible');
      cy.contains('td', 'LEC').should('be.visible');

      cy.contains('Inicie um replay para ver a classificação avançar volta a volta.').should('not.exist');
    });

    it('mostra estado de carregamento no botão enquanto o replay inicia', () => {
      cy.intercept('POST', '/api/replay/start', (req) => {
        req.reply({
          delay: 500,
          statusCode: 200,
          body: { replay_id: 'replay-123', total_laps: 57, event_name: 'Bahrain GP' },
        });
      }).as('startReplaySlow');
      cy.intercept('GET', '/api/replay/replay-123/state', { fixture: 'replay-state.json' });

      cy.visit('/');
      cy.wait(['@getReference', '@getReplaySessions']);

      cy.contains('button', 'Iniciar Replay').click();
      cy.contains('button', 'Carregando…').should('be.disabled');
      cy.wait('@startReplaySlow');
      cy.contains('button', 'Iniciar Replay').should('be.visible');
    });

    it('exibe mensagem de erro se o replay falhar ao iniciar', () => {
      cy.intercept('POST', '/api/replay/start', { statusCode: 500, body: {} }).as('startReplayError');

      cy.visit('/');
      cy.wait(['@getReference', '@getReplaySessions']);

      cy.contains('button', 'Iniciar Replay').click();
      cy.wait('@startReplayError');

      // getJSON lança: `POST /api${path} falhou: ${status}`
      cy.contains('POST /api/replay/start falhou: 500').should('be.visible');
    });

    it('abre o painel de telemetria ao clicar em um piloto da classificação', () => {
      cy.intercept('POST', '/api/replay/start', {
        statusCode: 200,
        body: { replay_id: 'replay-123', total_laps: 57, event_name: 'Bahrain GP' },
      }).as('startReplay');
      cy.intercept('GET', '/api/replay/replay-123/state', { fixture: 'replay-state.json' }).as('getReplayState');
      cy.intercept('GET', '/api/replay/replay-123/telemetry/VER', { fixture: 'telemetry-ver.json' }).as('getTelemetry');

      cy.visit('/');
      cy.wait(['@getReference', '@getReplaySessions']);
      cy.contains('button', 'Iniciar Replay').click();
      cy.wait(['@startReplay', '@getReplayState']);

      cy.contains('td', 'VER').click();

      // Estado de loading do painel (TelemetryPanel)
      cy.contains('Carregando telemetria real e calculando sinais...').should('be.visible');

      cy.wait('@getTelemetry');

      // Métricas do topo do painel: card "Piloto" com valor VER
      cy.get('.card').contains('Piloto').closest('.card').should('contain', 'VER');
      cy.get('.card').contains('Volta').closest('.card').should('contain', '12');
      cy.get('.card').contains('Vel. max').closest('.card').should('contain', '318.4');

      // Gráficos renderizados
      cy.contains('Speed / Throttle / Brake pressure proxy').should('be.visible');
      cy.contains('Mapa da volta').should('be.visible');
      cy.contains('Engenharia dos sinais').should('be.visible');
    });

    it('fecha o painel de telemetria ao clicar em "Fechar"', () => {
      cy.intercept('POST', '/api/replay/start', {
        statusCode: 200,
        body: { replay_id: 'replay-123', total_laps: 57, event_name: 'Bahrain GP' },
      }).as('startReplay');
      cy.intercept('GET', '/api/replay/replay-123/state', { fixture: 'replay-state.json' }).as('getReplayState');
      cy.intercept('GET', '/api/replay/replay-123/telemetry/VER', { fixture: 'telemetry-ver.json' }).as('getTelemetry');

      cy.visit('/');
      cy.wait(['@getReference', '@getReplaySessions']);
      cy.contains('button', 'Iniciar Replay').click();
      cy.wait(['@startReplay', '@getReplayState']);

      cy.contains('td', 'VER').click();
      cy.wait('@getTelemetry');
      cy.contains('Engenharia dos sinais').should('be.visible');

      cy.contains('button', 'Fechar').click();
      cy.contains('Engenharia dos sinais').should('not.exist');
    });

    it('mostra mensagem de erro no painel se a telemetria falhar', () => {
      cy.intercept('POST', '/api/replay/start', {
        statusCode: 200,
        body: { replay_id: 'replay-123', total_laps: 57, event_name: 'Bahrain GP' },
      }).as('startReplay');
      cy.intercept('GET', '/api/replay/replay-123/state', { fixture: 'replay-state.json' }).as('getReplayState');
      cy.intercept('GET', '/api/replay/replay-123/telemetry/VER', { statusCode: 500, body: {} }).as('getTelemetryError');

      cy.visit('/');
      cy.wait(['@getReference', '@getReplaySessions']);
      cy.contains('button', 'Iniciar Replay').click();
      cy.wait(['@startReplay', '@getReplayState']);

      cy.contains('td', 'VER').click();
      cy.wait('@getTelemetryError');

      cy.contains('GET /api/replay/replay-123/telemetry/VER falhou: 500').should('be.visible');
    });
  });

  describe('Modo Ao Vivo', () => {
    it('alterna para o modo Ao Vivo e mostra aviso quando não há sessão ativa', () => {
      cy.intercept('GET', '/api/live/status', {
        statusCode: 200,
        body: { connected: false, session_active: false },
      }).as('getLiveStatus');
      cy.intercept('GET', '/api/live/state', {
        statusCode: 200,
        body: { available: false, error: 'Nenhum dado disponível no momento.', standings: [] },
      }).as('getLiveState');

      cy.visit('/');
      cy.wait(['@getReference', '@getReplaySessions']);

      cy.contains('button', 'Ao Vivo').click();
      cy.wait(['@getLiveStatus', '@getLiveState']);

      cy.contains('Nenhuma sessão de F1 ao vivo no momento').should('be.visible');
      cy.contains('Conecte para receber dados quando houver uma sessão de F1 real acontecendo.').should('be.visible');
    });

    it('conecta ao vivo e exibe a classificação quando há sessão ativa', () => {
      cy.intercept('GET', '/api/live/status', {
        statusCode: 200,
        body: { connected: false, session_active: true },
      }).as('getLiveStatus');

      cy.intercept('POST', '/api/live/connect', { statusCode: 200, body: { connecting: true } }).as('connectLive');

      cy.intercept('GET', '/api/live/state', {
        statusCode: 200,
        body: {
          available: true,
          error: null,
          lap_number: 5,
          standings: [{ driver: 'HAM', position: 1, last_lap_s: 90.512, compound: 'SOFT' }],
        },
      }).as('getLiveState');

      cy.visit('/');
      cy.wait(['@getReference', '@getReplaySessions']);

      cy.contains('button', 'Ao Vivo').click();
      cy.wait(['@getLiveStatus', '@getLiveState']);

      cy.contains('button', 'Conectar ao Vivo').click();
      cy.wait('@connectLive');

      cy.contains('td', 'HAM').should('be.visible');
    });

    it('mostra estado de "Conectando…" enquanto a conexão ao vivo está em andamento', () => {
      cy.intercept('GET', '/api/live/status', { statusCode: 200, body: { connected: false, session_active: true } });
      cy.intercept('GET', '/api/live/state', { statusCode: 200, body: { available: false, error: null, standings: [] } });
      cy.intercept('POST', '/api/live/connect', (req) => {
        req.reply({ delay: 500, statusCode: 200, body: { connecting: true } });
      }).as('connectLiveSlow');

      cy.visit('/');
      cy.wait(['@getReference', '@getReplaySessions']);
      cy.contains('button', 'Ao Vivo').click();

      cy.contains('button', 'Conectar ao Vivo').click();
      cy.contains('button', 'Conectando…').should('be.disabled');
      cy.wait('@connectLiveSlow');
      cy.contains('button', 'Conectar ao Vivo').should('be.visible');
    });

    it('abre o painel de telemetria de um piloto no modo ao vivo', () => {
      cy.intercept('GET', '/api/live/status', { statusCode: 200, body: { connected: true, session_active: true } });
      cy.intercept('GET', '/api/live/state', {
        statusCode: 200,
        body: {
          available: true,
          error: null,
          lap_number: 5,
          standings: [{ driver: 'HAM', position: 1, last_lap_s: 90.512, compound: 'SOFT' }],
        },
      }).as('getLiveState');
      cy.intercept('GET', '/api/live/telemetry/HAM', { fixture: 'telemetry-ham.json' }).as('getLiveTelemetry');

      cy.visit('/');
      cy.wait(['@getReference', '@getReplaySessions']);
      cy.contains('button', 'Ao Vivo').click();
      cy.wait('@getLiveState');

      cy.contains('td', 'HAM').click();
      cy.wait('@getLiveTelemetry');

      cy.get('.card').contains('Piloto').closest('.card').should('contain', 'HAM');
    });
  });

  it('permite trocar de volta para o modo Replay a partir do modo Ao Vivo', () => {
    cy.intercept('GET', '/api/live/status', { statusCode: 200, body: { connected: false, session_active: false } });
    cy.intercept('GET', '/api/live/state', { statusCode: 200, body: { available: false, error: null, standings: [] } });

    cy.visit('/');
    cy.wait(['@getReference', '@getReplaySessions']);

    cy.contains('button', 'Ao Vivo').click();
    cy.contains('button', 'Conectar ao Vivo').should('be.visible');

    cy.contains('button', 'Replay').click();
    cy.contains('button', 'Iniciar Replay').should('be.visible');
  });
});
