/// <reference types="cypress" />

Cypress.Commands.add("openHome", () => {
  cy.visit("/");
});

declare global {
  namespace Cypress {
    interface Chainable {
      openHome(): Chainable<void>;
    }
  }
}

export {};