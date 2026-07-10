import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, HashRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import Dashboard from "./pages/Dashboard";
import Logs from "./pages/Logs";
import TrajectoryAnalysis from "./pages/TrajectoryAnalysis";
import Support from "./pages/Support";
import Download from "./pages/Download";
import Admin from "./pages/Admin";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const AppRoutes = () => (
  <Routes>
    <Route path="/" element={<Index />} />
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/logs" element={<Logs />} />
    <Route path="/trajectory" element={<TrajectoryAnalysis />} />
    <Route path="/support" element={<Support />} />
    <Route path="/download" element={<Download />} />
    <Route path="/admin" element={<Admin />} />
    <Route path="/admin-desktop" element={<Admin />} />
    <Route path="/admin-desctop" element={<Admin />} />
    <Route path="*" element={<NotFound />} />
  </Routes>
);

const App = () => {
  const isElectron = typeof window !== 'undefined' && window.location?.protocol === 'file:';
  const Router = isElectron ? HashRouter : BrowserRouter;
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <AppRoutes />
        </Router>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
