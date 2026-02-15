import { useParams, Link } from "react-router-dom";
import { modules, getLessonTypeLabel, getLessonTypeColor } from "@/data/modules";
import { lessonContents } from "@/data/lessonContents";
import { useProgress } from "@/hooks/useProgress";
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  Clock,
  Lightbulb,
  BookOpen,
  Target,
  Code,
  GraduationCap,
  ChevronRight,
} from "lucide-react";

const LessonPage = () => {
  const { moduleId, lessonId } = useParams();
  const mod = modules.find((m) => m.id === moduleId);
  const { isCompleted, toggleLesson } = useProgress();

  if (!mod) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Link to="/" className="text-accent hover:underline">← Back to Dashboard</Link>
      </div>
    );
  }

  const lesson = mod.lessons.find((l) => l.id === lessonId);
  const content = lessonId ? lessonContents[lessonId] : undefined;
  const lessonIndex = mod.lessons.findIndex((l) => l.id === lessonId);
  const prevLesson = lessonIndex > 0 ? mod.lessons[lessonIndex - 1] : null;
  const nextLesson = lessonIndex < mod.lessons.length - 1 ? mod.lessons[lessonIndex + 1] : null;

  if (!lesson || !content) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-muted-foreground mb-4">Lesson not found</p>
          <Link to={`/module/${moduleId}`} className="text-accent hover:underline">← Back to Module</Link>
        </div>
      </div>
    );
  }

  const completed = isCompleted(lesson.id);

  return (
    <div className="min-h-screen max-w-3xl mx-auto px-6 py-16 lg:py-10">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-2 text-sm text-muted-foreground mb-8 flex-wrap">
        <Link to="/" className="hover:text-foreground transition-colors">Dashboard</Link>
        <ChevronRight className="h-3 w-3" />
        <Link to={`/module/${mod.id}`} className="hover:text-foreground transition-colors">{mod.title}</Link>
        <ChevronRight className="h-3 w-3" />
        <span className="text-foreground">{lesson.title}</span>
      </nav>

      {/* Header */}
      <div className="mb-10">
        <div className="flex items-center gap-2 mb-3">
          <span className={`text-[11px] px-2.5 py-1 rounded-full font-medium ${getLessonTypeColor(lesson.type)}`}>
            {getLessonTypeLabel(lesson.type)}
          </span>
          <span className="flex items-center gap-1 text-xs text-muted-foreground">
            <Clock className="h-3 w-3" />
            {lesson.duration}
          </span>
        </div>
        <h1 className="text-2xl lg:text-3xl font-bold text-foreground mb-2">{lesson.title}</h1>
        <p className="text-muted-foreground">{lesson.description}</p>
      </div>

      {/* Core Concept */}
      <Section icon={Lightbulb} title="Core Concept" color="text-accent">
        <p className="text-foreground/90 leading-relaxed">{content.concept}</p>
      </Section>

      {/* Why It Matters */}
      <Section icon={Target} title="Why It Matters" color="text-info">
        <p className="text-foreground/90 leading-relaxed">{content.whyItMatters}</p>
      </Section>

      {/* Real-World Example */}
      <Section icon={BookOpen} title={`Real-World Example: ${content.realWorldExample.title}`} color="text-success">
        <div className="space-y-4">
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Scenario</h4>
            <p className="text-foreground/90 leading-relaxed">{content.realWorldExample.scenario}</p>
          </div>
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Implementation</h4>
            <p className="text-foreground/90 leading-relaxed">{content.realWorldExample.implementation}</p>
          </div>
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Outcome</h4>
            <p className="text-foreground/90 leading-relaxed font-medium">{content.realWorldExample.outcome}</p>
          </div>
        </div>
      </Section>

      {/* Code Snippet */}
      {content.codeSnippet && (
        <Section icon={Code} title="Code Perspective" color="text-accent">
          <pre className="bg-primary text-primary-foreground rounded-lg p-4 overflow-x-auto text-sm leading-relaxed font-mono mb-3">
            <code>{content.codeSnippet.code}</code>
          </pre>
          <p className="text-sm text-muted-foreground italic">{content.codeSnippet.explanation}</p>
        </Section>
      )}

      {/* Case Study */}
      <Section icon={Target} title={`Practice Case Study: ${content.caseStudy.title}`} color="text-warning">
        <div className="space-y-4">
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Background</h4>
            <p className="text-foreground/90 leading-relaxed">{content.caseStudy.background}</p>
          </div>
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Challenge</h4>
            <p className="text-foreground/90 leading-relaxed">{content.caseStudy.challenge}</p>
          </div>
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Approach</h4>
            <ol className="space-y-1.5">
              {content.caseStudy.approach.map((step, i) => (
                <li key={i} className="flex items-start gap-2 text-foreground/90">
                  <span className="text-accent font-mono text-xs mt-1 shrink-0">{i + 1}.</span>
                  <span>{step}</span>
                </li>
              ))}
            </ol>
          </div>
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Result</h4>
            <p className="text-foreground/90 leading-relaxed font-medium">{content.caseStudy.result}</p>
          </div>
          <div className="bg-accent/10 border border-accent/20 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-accent-foreground mb-1">📋 Your Task</h4>
            <p className="text-sm text-foreground/90">{content.caseStudy.yourTask}</p>
          </div>
        </div>
      </Section>

      {/* Key Takeaways */}
      <Section icon={CheckCircle2} title="Key Takeaways" color="text-success">
        <ul className="space-y-2">
          {content.keyTakeaways.map((takeaway, i) => (
            <li key={i} className="flex items-start gap-2 text-foreground/90">
              <CheckCircle2 className="h-4 w-4 text-success shrink-0 mt-0.5" />
              <span>{takeaway}</span>
            </li>
          ))}
        </ul>
      </Section>

      {/* Trainer Tip */}
      {content.trainerTip && (
        <Section icon={GraduationCap} title="Trainer Tip" color="text-accent">
          <div className="bg-accent/10 border border-accent/20 rounded-lg p-4">
            <p className="text-sm text-foreground/90 italic">💡 {content.trainerTip}</p>
          </div>
        </Section>
      )}

      {/* Mark Complete */}
      <div className="mt-10 mb-8">
        <button
          onClick={() => toggleLesson(lesson.id)}
          className={`w-full py-3 px-4 rounded-xl font-medium text-sm transition-all ${
            completed
              ? "bg-success/10 text-success border border-success/20 hover:bg-success/20"
              : "gradient-accent text-accent-foreground hover:opacity-90"
          }`}
        >
          {completed ? "✓ Completed — Click to Unmark" : "Mark as Complete"}
        </button>
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-between gap-4 pt-6 border-t border-border">
        {prevLesson ? (
          <Link
            to={`/module/${mod.id}/lesson/${prevLesson.id}`}
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            <span className="hidden sm:inline">{prevLesson.title}</span>
            <span className="sm:hidden">Previous</span>
          </Link>
        ) : (
          <div />
        )}
        {nextLesson ? (
          <Link
            to={`/module/${mod.id}/lesson/${nextLesson.id}`}
            className="flex items-center gap-2 text-sm text-accent hover:text-accent/80 transition-colors font-medium"
          >
            <span className="hidden sm:inline">{nextLesson.title}</span>
            <span className="sm:hidden">Next</span>
            <ArrowRight className="h-4 w-4" />
          </Link>
        ) : (
          <Link
            to={`/module/${mod.id}`}
            className="flex items-center gap-2 text-sm text-accent hover:text-accent/80 transition-colors font-medium"
          >
            Back to Module
            <ArrowRight className="h-4 w-4" />
          </Link>
        )}
      </div>
    </div>
  );
};

function Section({
  icon: Icon,
  title,
  color,
  children,
}: {
  icon: React.ElementType;
  title: string;
  color: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mb-8">
      <div className="flex items-center gap-2 mb-3">
        <Icon className={`h-4 w-4 ${color}`} />
        <h2 className="font-semibold text-foreground">{title}</h2>
      </div>
      <div className="pl-6">{children}</div>
    </section>
  );
}

export default LessonPage;
