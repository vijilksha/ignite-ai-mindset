import { useParams, Link, useNavigate } from "react-router-dom";
import { modules, getLessonTypeLabel, getLessonTypeColor } from "@/data/modules";
import { useProgress } from "@/hooks/useProgress";
import {
  ArrowLeft,
  CheckCircle2,
  Circle,
  Clock,
  HelpCircle,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { useState } from "react";

const ModulePage = () => {
  const { moduleId } = useParams();
  const mod = modules.find((m) => m.id === moduleId);
  const { isCompleted, toggleLesson, getModuleProgress } = useProgress();
  const [showQuestions, setShowQuestions] = useState(false);

  if (!mod) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-muted-foreground mb-4">Module not found</p>
          <Link to="/" className="text-accent hover:underline">← Back to Dashboard</Link>
        </div>
      </div>
    );
  }

  const progress = getModuleProgress(mod.lessons.map((l) => l.id));
  const Icon = mod.icon;

  return (
    <div className="min-h-screen max-w-3xl mx-auto px-6 py-16 lg:py-10">
      {/* Back link */}
      <Link
        to="/"
        className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-8"
      >
        <ArrowLeft className="h-4 w-4" />
        All Modules
      </Link>

      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-4">
          <div
            className="w-12 h-12 rounded-xl flex items-center justify-center"
            style={{ backgroundColor: `${mod.color}15`, color: mod.color }}
          >
            <Icon className="h-6 w-6" />
          </div>
          <div>
            <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider">
              Module {mod.number}
            </p>
            <h1 className="text-2xl font-bold text-foreground">{mod.title}</h1>
          </div>
        </div>
        <p className="text-muted-foreground">{mod.description}</p>

        {/* Progress bar */}
        <div className="mt-4 flex items-center gap-3">
          <div className="flex-1 h-2 rounded-full bg-muted">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{ width: `${progress}%`, backgroundColor: mod.color }}
            />
          </div>
          <span className="text-sm font-semibold text-foreground">{progress}%</span>
        </div>
      </div>

      {/* Lessons */}
      <div className="space-y-3 mb-10">
        {mod.lessons.map((lesson) => {
          const completed = isCompleted(lesson.id);

          return (
            <Link
              key={lesson.id}
              to={`/module/${mod.id}/lesson/${lesson.id}`}
              className={`w-full text-left flex items-start gap-4 p-4 rounded-xl border transition-all duration-200 ${
                completed
                  ? "bg-success/5 border-success/20"
                  : "bg-card border-border hover:border-accent/30 hover:shadow-sm"
              }`}
            >
              <div className="mt-0.5 shrink-0">
                {completed ? (
                  <CheckCircle2 className="h-5 w-5 text-success" />
                ) : (
                  <Circle className="h-5 w-5 text-muted-foreground/40" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h3
                    className={`font-medium text-sm ${
                      completed ? "text-success line-through" : "text-foreground"
                    }`}
                  >
                    {lesson.title}
                  </h3>
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${getLessonTypeColor(
                      lesson.type
                    )}`}
                  >
                    {getLessonTypeLabel(lesson.type)}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground">{lesson.description}</p>
              </div>
              <div className="flex items-center gap-1 text-xs text-muted-foreground shrink-0">
                <Clock className="h-3 w-3" />
                {lesson.duration}
              </div>
            </Link>
          );
        })}
      </div>

      {/* Interview Questions */}
      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <button
          onClick={() => setShowQuestions(!showQuestions)}
          className="w-full flex items-center justify-between px-5 py-4 hover:bg-muted/50 transition-colors"
        >
          <div className="flex items-center gap-2">
            <HelpCircle className="h-4 w-4 text-accent" />
            <span className="font-semibold text-sm text-foreground">
              Interview Questions ({mod.interviewQuestions.length})
            </span>
          </div>
          {showQuestions ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </button>
        {showQuestions && (
          <div className="px-5 pb-5 space-y-3">
            {mod.interviewQuestions.map((q, i) => (
              <div
                key={i}
                className="flex items-start gap-3 text-sm text-foreground/90 p-3 rounded-lg bg-muted/50"
              >
                <span className="text-accent font-mono text-xs mt-0.5">Q{i + 1}</span>
                <p>{q}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ModulePage;
