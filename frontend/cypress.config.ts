import { defineConfig } from "cypress";
import mochawesome from "cypress-mochawesome-reporter/plugin";

export default defineConfig({
  video: true,

  screenshotsFolder: "cypress/screenshots",

  videosFolder: "cypress/videos",

  downloadsFolder: "cypress/downloads",

  viewportWidth: 1440,

  viewportHeight: 900,

  defaultCommandTimeout: 10000,

  pageLoadTimeout: 30000,

  retries: {
    runMode: 2,
    openMode: 0,
  },

  reporter: "cypress-mochawesome-reporter",

  reporterOptions: {
    reportDir: "cypress/reports",
    overwrite: false,
    html: true,
    json: true,
    embeddedScreenshots: true,
    inlineAssets: true,
    charts: true,
  },

  e2e: {
    baseUrl: "http://localhost:5173",

    specPattern: "cypress/e2e/**/*.cy.{js,jsx,ts,tsx}",

    supportFile: "cypress/support/e2e.ts",

    setupNodeEvents(on, config) {
      mochawesome(on);

      return config;
    },
  },
});