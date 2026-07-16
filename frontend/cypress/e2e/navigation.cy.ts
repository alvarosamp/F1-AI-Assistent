/// <reference types="cypress" />

/**
 * Teste de regressão de navegação — cobre Sidebar + Topbar + roteamento
 * (HashRouter) para todas as rotas declaradas em App.tsx.
 *
 * Não faz mock de API: o objetivo aqui é garantir que a navegação em si
 * não quebra (sem crash de render) e que título/estado ativo do menu
 * ficam corretos. Cada página pode ter seus próprios testes de conteúdo
 * em specs dedicados (ex: live-race-page.cy.ts).
 */

const ROUTES: { path: string; hash: string; navLabel: string; title: string }[] = [
  { path: '/', hash: '#/', navLabel: 'Ao Vivo', title: 'Corrida ao Vivo' },
  { path: '/previsoes', hash: '#/previsoes', navLabel: 'Previsões', title: 'Previsões' },
  { path: '/engenharia', hash: '#/engenharia', navLabel: 'Engenharia', title: 'Engenharia' },
  { path: '/analise', hash: '#/analise', navLabel: 'Modelo', title: 'Análise do Modelo' },
  { path: '/calibracao', hash: '#/calibracao', navLabel: 'Calibração', title: 'Calibração' },
  { path: '/sobre', hash: '#/sobre', navLabel: 'Sobre', title: 'Sobre o Projeto' },
];

describe('Navegação e layout global', () => {
  it('carrega a home ("/") por padrão com o item "Ao Vivo" ativo', () => {
    cy.visit('/');
    cy.location('hash').should('eq', '#/');
    cy.contains('header span', 'Corrida ao Vivo').should('be.visible');
    cy.get('nav a').contains('Ao Vivo').should('have.class', 'text-white');
  });

  ROUTES.forEach(({ hash, navLabel, title }) => {
    it(`navega para "${navLabel}" pela Sidebar e atualiza o Topbar para "${title}"`, () => {
      cy.visit('/');
      cy.get('nav a').contains(navLabel).click();

      cy.location('hash').should('eq', hash);
      cy.contains('header span', title).should('be.visible');

      // O link ativo deve ter as classes de destaque aplicadas via NavLink isActive
      cy.get('nav a').contains(navLabel).should('have.class', 'text-white');
    });
  });

  it('acessa cada rota diretamente pela URL (deep link) sem quebrar', () => {
    ROUTES.forEach(({ path, title }) => {
      cy.visit(`/#${path}`);
      cy.contains('header span', title).should('be.visible');
      // Garante que não sobrou nenhuma tela de erro genérica do React
      cy.get('body').should('not.contain.text', 'Cannot read properties');
      cy.get('body').should('not.contain.text', 'Uncaught');
    });
  });

  it('mostra apenas um item de navegação ativo por vez', () => {
    cy.visit('/#/previsoes');
    cy.get('nav a.text-white').should('have.length', 1);
    cy.get('nav a.text-white').should('contain.text', 'Previsões');
  });

  it('exibe o aviso de "decision support" no rodapé da Sidebar em todas as páginas', () => {
    ROUTES.forEach(({ path }) => {
      cy.visit(`/#${path}`);
      cy.contains('Sistema de decision support.').should('be.visible');
      cy.contains('Não é aconselhamento de apostas.').should('be.visible');
    });
  });

  it('usa o título genérico do Topbar para uma rota desconhecida (fallback)', () => {
    cy.visit('/#/rota-que-nao-existe', { failOnStatusCode: false });
    cy.contains('header span', 'F1 AI Race Insights').should('be.visible');
  });
});
