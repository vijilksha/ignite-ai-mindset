import { NavLink } from "@/components/NavLink";
import { useLocation } from "react-router-dom";
import { modules } from "@/data/modules";
import { useProgress } from "@/hooks/useProgress";
import { Brain, Menu, X, MapPin } from "lucide-react";
import { useState } from "react";

export function AppSidebar() {
  const location = useLocation();
  const { getModuleProgress } = useProgress();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <>
      {/* Mobile toggle */}
      <button
        onClick={() => setMobileOpen(!mobileOpen)}
        className="fixed top-4 left-4 z-50 lg:hidden p-2 rounded-lg bg-primary text-primary-foreground shadow-lg"
        aria-label="Toggle menu"
      >
        {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </button>

      {/* Overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-30 bg-foreground/20 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed top-0 left-0 z-40 h-screen w-72 bg-sidebar text-sidebar-foreground border-r border-sidebar-border flex flex-col transition-transform duration-300 lg:translate-x-0 lg:static lg:z-auto ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Logo */}
        <NavLink
          to="/"
          className="flex items-center gap-3 px-6 py-5 border-b border-sidebar-border hover:bg-sidebar-accent/50 transition-colors"
          activeClassName=""
          onClick={() => setMobileOpen(false)}
        >
          <div className="w-9 h-9 rounded-lg gradient-accent flex items-center justify-center">
            <Brain className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-sidebar-accent-foreground">AI Engineer</h1>
            <p className="text-xs text-sidebar-foreground/60">Learning Path</p>
          </div>
        </NavLink>

        {/* Module list */}
        <nav className="flex-1 overflow-y-auto py-3 px-3 space-y-0.5">
          {modules.map((mod) => {
            const progress = getModuleProgress(mod.lessons.map((l) => l.id));
            const isActive = location.pathname === `/module/${mod.id}`;

            return (
              <NavLink
                key={mod.id}
                to={`/module/${mod.id}`}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all group hover:bg-sidebar-accent/50 ${
                  isActive ? "bg-sidebar-accent text-sidebar-accent-foreground" : ""
                }`}
                activeClassName="bg-sidebar-accent text-sidebar-accent-foreground"
                onClick={() => setMobileOpen(false)}
              >
                <div className="flex items-center justify-center w-7 h-7 rounded-md text-xs font-semibold shrink-0"
                  style={{ backgroundColor: `${mod.color}20`, color: mod.color }}
                >
                  {mod.number}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="truncate font-medium leading-tight">{mod.title}</p>
                  {progress > 0 && (
                    <div className="mt-1 h-1 w-full rounded-full bg-sidebar-border">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{ width: `${progress}%`, backgroundColor: mod.color }}
                      />
                    </div>
                  )}
                </div>
              </NavLink>
            );
          })}
        </nav>

        {/* Roadmap link */}
        <div className="px-3 pb-2">
          <NavLink
            to="/roadmap"
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all hover:bg-sidebar-accent/50 ${
              location.pathname === "/roadmap" ? "bg-sidebar-accent text-sidebar-accent-foreground" : ""
            }`}
            activeClassName="bg-sidebar-accent text-sidebar-accent-foreground"
            onClick={() => setMobileOpen(false)}
          >
            <div className="flex items-center justify-center w-7 h-7 rounded-md text-xs shrink-0 bg-accent/20 text-accent">
              <MapPin className="h-3.5 w-3.5" />
            </div>
            <p className="truncate font-medium leading-tight">12-Month Roadmap</p>
          </NavLink>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-sidebar-border">
          <p className="text-xs text-sidebar-foreground/50">
            Built for senior engineers transitioning to AI
          </p>
        </div>
      </aside>
    </>
  );
}
