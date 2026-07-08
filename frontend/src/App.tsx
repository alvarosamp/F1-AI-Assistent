import { HashRouter, Route, Routes } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { SimulationPage } from './pages/SimulationPage';
import { CalibrationPage } from './pages/CalibrationPage';
import { ModelAnalysisPage } from './pages/ModelAnalysisPage';
import { AboutPage } from './pages/AboutPage';

export default function App() {
  return (
    <HashRouter>
      <div className="flex min-h-screen">
        <Sidebar />
        <main className="flex-1 p-6 md:p-10 max-w-7xl mx-auto w-full">
          <Routes>
            <Route path="/" element={<SimulationPage />} />
            <Route path="/calibracao" element={<CalibrationPage />} />
            <Route path="/analise" element={<ModelAnalysisPage />} />
            <Route path="/sobre" element={<AboutPage />} />
          </Routes>
        </main>
      </div>
    </HashRouter>
  );
}
