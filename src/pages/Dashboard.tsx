import { modules } from "@/data/modules";
import { useProgress } from "@/hooks/useProgress";
import { Link } from "react-router-dom";
import { ArrowRight, BookOpen, CheckCircle2, Clock, Target } from "lucide-react";

const Dashboard = () => {
  const { getModuleProgress } = useProgress();

  const totalLessons = modules.reduce((acc, m) => acc + m.lessons.length, 0);
  const completedTotal = modules.reduce((acc, m) => {
    const ids = m.lessons.map((l) => l.id);
    return acc + ids.filter((id) => getModuleProgress([id]) === 100).length;
  }, 0);
  const overallProgress = Math.round((completedTotal / totalLessons) * 100);

  return (
    <div className="min-h-screen">
      {/* Hero */}
      <header className="px-6 pt-16 pb-10 lg:pt-10 lg:px-10">
        <div className="max-w-4xl">
          <p className="text-sm font-semibold text-accent uppercase tracking-wider mb-2">
            AI Learning Path
          </p>
          <h1 className="text-3xl lg:text-4xl font-bold text-foreground leading-tight mb-3">
            From Senior Engineer
            <br />
            <span className="text-accent">to AI Engineer</span>
          </h1>
          <p className="text-muted-foreground max-w-xl text-base">
            A structured, engineering-first approach to AI — built for corporate trainers 
            with Java Full Stack and Automation Testing experience.
          </p>
        </div>

        {/* Stats */}
        <div className="flex flex-wrap gap-4 mt-8">
          {[
            { icon: BookOpen, label: "Modules", value: modules.length },
            { icon: Target, label: "Lessons", value: totalLessons },
            { icon: CheckCircle2, label: "Completed", value: completedTotal },
            { icon: Clock, label: "Progress", value: `${overallProgress}%` },
          ].map((stat) => (
            <div
              key={stat.label}
              className="flex items-center gap-3 px-4 py-3 rounded-xl bg-card border border-border"
            >
              <stat.icon className="h-4 w-4 text-accent" />
              <div>
                <p className="text-xs text-muted-foreground">{stat.label}</p>
                <p className="text-lg font-bold text-foreground">{stat.value}</p>
              </div>
            </div>
          ))}
        </div>
      </header>

      {/* Modules Grid */}
      <section className="px-6 lg:px-10 pb-16">
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {modules.map((mod, i) => {
            const progress = getModuleProgress(mod.lessons.map((l) => l.id));
            const Icon = mod.icon;

            return (
              <Link
                key={mod.id}
                to={`/module/${mod.id}`}
                className="group relative rounded-xl bg-card border border-border p-5 hover:border-accent/40 hover:shadow-lg hover:shadow-accent/5 transition-all duration-300"
                style={{ animationDelay: `${i * 60}ms` }}
              >
                <div className="flex items-start justify-between mb-3">
                  <div
                    className="w-10 h-10 rounded-lg flex items-center justify-center"
                    style={{ backgroundColor: `${mod.color}15`, color: mod.color }}
                  >
                    <Icon className="h-5 w-5" />
                  </div>
                  <span className="text-xs font-mono text-muted-foreground">
                    {mod.lessons.length} lessons
                  </span>
                </div>

                <h3 className="font-semibold text-foreground mb-1 group-hover:text-accent transition-colors">
                  {mod.title}
                </h3>
                <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                  {mod.subtitle}
                </p>

                {/* Progress bar */}
                <div className="flex items-center gap-3">
                  <div className="flex-1 h-1.5 rounded-full bg-muted">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{ width: `${progress}%`, backgroundColor: mod.color }}
                    />
                  </div>
                  <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:text-accent group-hover:translate-x-0.5 transition-all" />
                </div>
              </Link>
            );
          })}
        </div>
      </section>
    </div>
  );
};

export default Dashboard;
