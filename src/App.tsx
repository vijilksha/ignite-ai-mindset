import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ProgressProvider } from "@/hooks/useProgress";
import { AppSidebar } from "@/components/AppSidebar";
import Dashboard from "./pages/Dashboard";
import ModulePage from "./pages/ModulePage";
import LessonPage from "./pages/LessonPage";
import RoadmapPage from "./pages/RoadmapPage";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <ProgressProvider>
        <BrowserRouter>
          <div className="flex min-h-screen w-full">
            <AppSidebar />
            <main className="flex-1 min-w-0">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/module/:moduleId" element={<ModulePage />} />
                <Route path="/module/:moduleId/lesson/:lessonId" element={<LessonPage />} />
                <Route path="/roadmap" element={<RoadmapPage />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </main>
          </div>
        </BrowserRouter>
      </ProgressProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
