import { HashRouter, Route, Routes } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { Topbar } from './components/Topbar';
import { LiveRacePage } from './pages/LiveRacePage';
import { SimulationPage } from './pages/SimulationPage';
import { EngineeringPage } from './pages/EngineeringPage';
import { CalibrationPage } from './pages/CalibrationPage';
import { ModelAnalysisPage } from './pages/ModelAnalysisPage';
import { AboutPage } from './pages/AboutPage';

export default function App() {
  return (
    <HashRouter>
      <div className="flex min-h-screen">
        <Sidebar />
        <div className="flex-1 flex flex-col min-w-0">
          <Topbar />
          <main className="flex-1 p-6 md:p-10 max-w-7xl mx-auto w-full">
            <Routes>
              <Route path="/" element={<LiveRacePage />} />
              <Route path="/previsoes" element={<SimulationPage />} />
              <Route path="/engenharia" element={<EngineeringPage />} />
              <Route path="/analise" element={<ModelAnalysisPage />} />
              <Route path="/calibracao" element={<CalibrationPage />} />
              <Route path="/sobre" element={<AboutPage />} />
            </Routes>
          </main>
        </div>
      </div>
    </HashRouter>
  );
}
