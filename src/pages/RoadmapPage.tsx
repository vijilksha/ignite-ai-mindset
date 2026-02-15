import { Link } from "react-router-dom";
import {
  ArrowLeft,
  Calendar,
  Target,
  Rocket,
  CheckCircle2,
  BookOpen,
  Code,
  GraduationCap,
  Briefcase,
  TrendingUp,
  Lightbulb,
  Zap,
  Award,
  Star,
  MapPin,
} from "lucide-react";

const phases = [
  {
    month: "Month 1–2",
    title: "Foundation & AI Mindset",
    icon: Lightbulb,
    color: "hsl(36 90% 55%)",
    focus: "Build AI mental models using your Java/Testing background",
    modules: ["Module 1: AI Mindset for Engineers", "Module 2: AI & ML Foundations"],
    weeklyPlan: [
      { week: "Week 1", tasks: ["Complete Module 1 lessons", "Map 10 Java skills → AI equivalents", "Set up Python environment (Anaconda/Colab)"] },
      { week: "Week 2", tasks: ["ML vs DL vs GenAI deep dive", "Supervised vs Unsupervised with K-Means revision", "Practice: classify 5 business problems as ML/DL/GenAI"] },
      { week: "Week 3", tasks: ["Features, Labels & Overfitting", "Hands-on: train your first model (scikit-learn)", "Compare model training to CI/CD pipeline"] },
      { week: "Week 4", tasks: ["Model Training Pipeline end-to-end", "Build: simple loan prediction model", "Interview prep: explain ML pipeline in Java terms"] },
      { week: "Week 5–6", tasks: ["Python for ML essentials (NumPy, Pandas, Matplotlib)", "Revisit K-Means with business datasets", "Complete all Module 1 & 2 interview questions"] },
      { week: "Week 7–8", tasks: ["Mini project: customer segmentation with K-Means", "Write 2 LinkedIn posts about your AI journey", "Peer review: explain ML to a non-technical colleague"] },
    ],
    deliverables: [
      "Skill mapping document (Java → AI equivalents)",
      "First ML model (loan/fraud prediction)",
      "K-Means business segmentation project",
      "2 LinkedIn articles published",
    ],
    skills: ["Python basics", "scikit-learn", "ML fundamentals", "Data preprocessing", "K-Means clustering"],
    interviewReady: [
      "Explain AI vs traditional programming with examples",
      "ML vs DL vs GenAI — when to use which",
      "What is overfitting? How does it relate to flaky tests?",
      "Walk through an ML training pipeline",
    ],
  },
  {
    month: "Month 3–4",
    title: "Data Thinking & Hands-on ML",
    icon: Code,
    color: "hsl(152 60% 42%)",
    focus: "Master data engineering and build production-ready ML models",
    modules: ["Module 3: Data Thinking", "Module 4: Hands-on ML"],
    weeklyPlan: [
      { week: "Week 9", tasks: ["Data quality fundamentals — audit a real dataset", "Structured vs Unstructured data pipelines", "Practice: clean a messy CSV dataset"] },
      { week: "Week 10", tasks: ["Feature engineering deep dive", "Build features from raw data (Uber ride time example)", "Apply feature engineering to test flakiness prediction"] },
      { week: "Week 11", tasks: ["Image data preprocessing (leverage your background!)", "K-Means → CNN progression for image classification", "Hands-on: transfer learning with ResNet"] },
      { week: "Week 12", tasks: ["Model lifecycle management & MLOps basics", "Set up MLflow for experiment tracking", "Compare model versioning to Git versioning"] },
      { week: "Week 13–14", tasks: ["Evaluation metrics mastery (precision, recall, F1)", "Build: fraud detection model with proper metrics", "Practice: choose right metric for 5 different scenarios"] },
      { week: "Week 15–16", tasks: ["Capstone: Image classification project (product defects)", "Deploy model as REST API (Spring Boot + DJL)", "Write up project for portfolio with architecture diagram"] },
    ],
    deliverables: [
      "Feature engineering notebook",
      "Image classification model (transfer learning)",
      "Deployed ML API (Spring Boot)",
      "MLflow experiment tracking setup",
      "Portfolio write-up with architecture diagrams",
    ],
    skills: ["Feature engineering", "Image preprocessing", "Transfer learning", "MLOps basics", "Model evaluation", "DJL (Deep Java Library)"],
    interviewReady: [
      "How do you handle missing data in production?",
      "Explain feature engineering with a real example",
      "Precision vs Recall — when do you optimize which?",
      "Design an MLOps pipeline for continuous model improvement",
    ],
  },
  {
    month: "Month 5–6",
    title: "Generative AI & LLM Mastery",
    icon: Zap,
    color: "hsl(210 80% 55%)",
    focus: "Master LLMs, prompt engineering, RAG architecture, and embeddings",
    modules: ["Module 5: Generative AI & LLMs"],
    weeklyPlan: [
      { week: "Week 17", tasks: ["How LLMs work — Transformer architecture simplified", "Token economics: understand pricing and optimization", "Hands-on: OpenAI API / HuggingFace API calls from Java"] },
      { week: "Week 18", tasks: ["Prompt engineering mastery — 3-version methodology", "Build: prompt template library for your domain", "Practice: generate test cases with engineered prompts"] },
      { week: "Week 19", tasks: ["Embeddings & vector databases deep dive", "Set up pgvector or Pinecone", "Build: semantic search for your test documentation"] },
      { week: "Week 20", tasks: ["RAG architecture end-to-end", "Build: RAG pipeline with document chunking + retrieval", "Test: compare RAG answers vs vanilla LLM answers"] },
      { week: "Week 21–22", tasks: ["Hallucination management strategies", "Build guardrails: confidence scoring, fact-checking", "Compare with flaky test management frameworks"] },
      { week: "Week 23–24", tasks: ["Capstone: Enterprise RAG chatbot (your test documentation)", "Full pipeline: ingest → embed → retrieve → generate → validate", "Deploy with Spring Boot WebSocket endpoint"] },
    ],
    deliverables: [
      "Prompt engineering template library",
      "Semantic search system with vector DB",
      "RAG chatbot for documentation Q&A",
      "Hallucination management framework document",
      "Deployed chatbot with Spring Boot backend",
    ],
    skills: ["LLM APIs (OpenAI, HuggingFace)", "Prompt engineering", "Embeddings", "Vector databases", "RAG architecture", "Guardrails & validation"],
    interviewReady: [
      "How do Transformers and attention mechanisms work?",
      "Design a RAG system for enterprise knowledge management",
      "How do you handle AI hallucinations in production?",
      "Compare embeddings vs keyword search — when to use which?",
    ],
  },
  {
    month: "Month 7–8",
    title: "AI in Java Full Stack & Testing",
    icon: Rocket,
    color: "hsl(280 65% 55%)",
    focus: "Integrate AI into your Java stack and revolutionize test automation",
    modules: ["Module 6: AI in Java Full Stack", "Module 7: AI for Test Automation"],
    weeklyPlan: [
      { week: "Week 25–26", tasks: ["AI system architecture patterns for enterprise", "Build: Spring Boot + AI API microservice", "Implement circuit breaker & retry patterns for AI calls"] },
      { week: "Week 27–28", tasks: ["Microservices decomposition for AI systems", "Error handling for non-deterministic AI outputs", "Build: multi-model architecture (ML + LLM combined)"] },
      { week: "Week 29–30", tasks: ["AI-generated test cases — production-quality pipeline", "Self-healing test locators with Healenium", "Build: AI test case generator from user stories"] },
      { week: "Week 31–32", tasks: ["Validating AI outputs — multi-layered testing strategy", "Testing AI systems themselves (model quality assurance)", "Build: AI validation framework for your projects"] },
    ],
    deliverables: [
      "Spring Boot AI microservice (production-ready)",
      "AI test case generator integrated with Jira",
      "Self-healing test suite POC",
      "AI output validation framework",
      "Architecture decision record (ADR) for AI integration",
    ],
    skills: ["Spring Boot + AI APIs", "AI error handling", "AI test generation", "Self-healing locators", "AI system testing", "Circuit breaker patterns"],
    interviewReady: [
      "Architect a Spring Boot app that uses multiple AI APIs",
      "How is error handling different for AI vs traditional services?",
      "What is a self-healing test and how does AI enable it?",
      "How do you validate non-deterministic AI outputs?",
    ],
  },
  {
    month: "Month 9–10",
    title: "Capstone Projects & Portfolio",
    icon: Award,
    color: "hsl(15 80% 55%)",
    focus: "Build impressive portfolio projects and prepare for AI roles",
    modules: ["Module 8: Capstone Projects"],
    weeklyPlan: [
      { week: "Week 33–34", tasks: ["AI Resume Screener — NLP pipeline end-to-end", "Architecture diagram, data flow, bias mitigation", "Deploy and document for portfolio"] },
      { week: "Week 35–36", tasks: ["AI Test Case Generator — LLM-powered tool", "Integration with test frameworks (Selenium/Playwright)", "Collect metrics: time saved, quality comparison"] },
      { week: "Week 37–38", tasks: ["Enterprise AI Chatbot — RAG with domain knowledge", "Full stack: React UI + Spring Boot + pgvector + LLM", "Add guardrails, citations, confidence scoring"] },
      { week: "Week 39–40", tasks: ["Image Defect Detection — computer vision project", "Transfer learning + Grad-CAM explainability", "Production deployment with monitoring dashboard"] },
    ],
    deliverables: [
      "4 portfolio-ready AI projects with documentation",
      "GitHub repos with READMEs, architecture diagrams",
      "Live demos for each project",
      "Video walkthroughs (2-3 min each)",
      "Portfolio website showcasing all projects",
    ],
    skills: ["End-to-end AI project delivery", "NLP pipelines", "Computer vision", "RAG systems", "Full-stack AI applications", "Technical documentation"],
    interviewReady: [
      "Walk through the architecture of your AI chatbot",
      "How did you handle bias in the resume screener?",
      "What metrics did you use to evaluate the test generator?",
      "Explain your defect detection model's decision-making",
    ],
  },
  {
    month: "Month 11–12",
    title: "Trainer Mode & Career Launch",
    icon: GraduationCap,
    color: "hsl(260 60% 55%)",
    focus: "Launch your AI training career and monetize your skills",
    modules: ["Module 9: Trainer Mode", "Module 10: Career & Monetization"],
    weeklyPlan: [
      { week: "Week 41–42", tasks: ["Design your 1-day AI workshop curriculum", "Prepare demo scripts with backup plans", "Practice teaching AI to non-technical audience"] },
      { week: "Week 43–44", tasks: ["Deliver first internal AI training (free)", "Collect testimonials and feedback", "Create executive-level AI pitch deck"] },
      { week: "Week 45–46", tasks: ["Build 'AI for Testers' online course (5 modules)", "Set up on Udemy + personal site", "Launch marketing: LinkedIn articles, YouTube shorts"] },
      { week: "Week 47–48", tasks: ["Finalize AI career path (engineer/trainer/product)", "Get 1 AI certification (AWS ML or Google Cloud AI)", "Update resume, LinkedIn, portfolio — start applying!"] },
    ],
    deliverables: [
      "1-day AI workshop curriculum & materials",
      "First training delivered with testimonials",
      "Online course (5 modules) published",
      "AI certification earned",
      "Updated resume & portfolio for AI roles",
      "3 corporate training proposals sent",
    ],
    skills: ["Curriculum design", "Live demo techniques", "Course creation", "Corporate training delivery", "AI business development", "Personal branding"],
    interviewReady: [
      "How would you explain neural networks to a non-technical audience?",
      "Design a 3-day corporate AI training curriculum",
      "What's your 6-month plan to transition fully into AI?",
      "How would you pitch an AI training program to a CTO?",
    ],
  },
];

const skillMapping = [
  { existing: "Java OOP & Design Patterns", ai: "ML class design, model architecture", icon: "☕" },
  { existing: "Spring Boot & Microservices", ai: "AI API serving, model deployment", icon: "🍃" },
  { existing: "JDBC & Database", ai: "Data pipelines, feature stores", icon: "🗄️" },
  { existing: "JUnit & TestNG", ai: "Model validation, evaluation metrics", icon: "✅" },
  { existing: "Selenium & Playwright", ai: "AI-powered testing, self-healing locators", icon: "🤖" },
  { existing: "CI/CD (Jenkins, GitHub Actions)", ai: "MLOps, model retraining pipelines", icon: "⚙️" },
  { existing: "REST APIs & WebClient", ai: "LLM API integration, AI microservices", icon: "🔗" },
  { existing: "Image Processing & K-Means", ai: "Computer Vision, unsupervised ML", icon: "🖼️" },
  { existing: "Performance Testing", ai: "Model inference optimization, load testing AI", icon: "📊" },
  { existing: "Corporate Training", ai: "AI training delivery, curriculum design", icon: "🎓" },
];

const certifications = [
  { name: "AWS Certified Machine Learning – Specialty", priority: "High", timeline: "Month 11", reason: "Most recognized by enterprise employers" },
  { name: "Google Professional Machine Learning Engineer", priority: "High", timeline: "Month 11", reason: "Strong for cloud-based AI roles" },
  { name: "Microsoft Azure AI Engineer Associate", priority: "Medium", timeline: "Month 12", reason: "Good for .NET/Azure enterprise environments" },
  { name: "DeepLearning.AI TensorFlow Developer", priority: "Medium", timeline: "Month 8", reason: "Validates DL skills, Coursera-based" },
  { name: "LangChain / LlamaIndex Certification", priority: "Low", timeline: "Anytime", reason: "Emerging, good for RAG/LLM specialization" },
];

const RoadmapPage = () => {
  return (
    <div className="min-h-screen max-w-4xl mx-auto px-6 py-16 lg:py-10">
      {/* Back link */}
      <Link
        to="/"
        className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-8"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Dashboard
      </Link>

      {/* Header */}
      <div className="mb-12">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-12 h-12 rounded-xl gradient-accent flex items-center justify-center">
            <MapPin className="h-6 w-6 text-primary" />
          </div>
          <div>
            <p className="text-xs font-mono text-accent uppercase tracking-wider">Your Complete Path</p>
            <h1 className="text-3xl font-bold text-foreground">12-Month AI Learning Roadmap</h1>
          </div>
        </div>
        <p className="text-muted-foreground text-lg">
          A structured, week-by-week plan to transform from Senior Java Engineer → AI Solutions Architect & Corporate AI Trainer.
          Every phase builds on your existing 10+ years of experience.
        </p>
      </div>

      {/* Skill Mapping Section */}
      <section className="mb-14">
        <div className="flex items-center gap-2 mb-4">
          <Target className="h-5 w-5 text-accent" />
          <h2 className="text-xl font-bold text-foreground">Your Skill Mapping: Java → AI</h2>
        </div>
        <p className="text-sm text-muted-foreground mb-4">
          You're NOT starting from zero. Here's how your existing skills directly translate to AI capabilities:
        </p>
        <div className="grid gap-2">
          {skillMapping.map((skill, i) => (
            <div key={i} className="flex items-center gap-3 p-3 rounded-lg border border-border bg-card hover:bg-muted/50 transition-colors">
              <span className="text-xl shrink-0">{skill.icon}</span>
              <div className="flex-1 min-w-0 grid grid-cols-1 sm:grid-cols-2 gap-1">
                <span className="text-sm font-medium text-foreground">{skill.existing}</span>
                <span className="text-sm text-accent">→ {skill.ai}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Timeline */}
      <section className="mb-14">
        <div className="flex items-center gap-2 mb-6">
          <Calendar className="h-5 w-5 text-accent" />
          <h2 className="text-xl font-bold text-foreground">Month-by-Month Learning Plan</h2>
        </div>

        <div className="space-y-10">
          {phases.map((phase, phaseIdx) => {
            const Icon = phase.icon;
            return (
              <div key={phaseIdx} className="relative">
                {/* Phase header */}
                <div className="flex items-start gap-4 mb-5">
                  <div
                    className="w-11 h-11 rounded-xl flex items-center justify-center shrink-0"
                    style={{ backgroundColor: `${phase.color}18`, color: phase.color }}
                  >
                    <Icon className="h-5 w-5" />
                  </div>
                  <div>
                    <p className="text-xs font-mono uppercase tracking-wider" style={{ color: phase.color }}>
                      {phase.month}
                    </p>
                    <h3 className="text-lg font-bold text-foreground">{phase.title}</h3>
                    <p className="text-sm text-muted-foreground">{phase.focus}</p>
                    <div className="flex flex-wrap gap-1.5 mt-2">
                      {phase.modules.map((mod, i) => (
                        <span key={i} className="text-[11px] px-2 py-0.5 rounded-full bg-muted text-muted-foreground font-medium">
                          {mod}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Weekly breakdown */}
                <div className="ml-[3.25rem] space-y-3 mb-5">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Weekly Breakdown</h4>
                  <div className="grid gap-2">
                    {phase.weeklyPlan.map((week, wi) => (
                      <div key={wi} className="p-3 rounded-lg border border-border bg-card">
                        <p className="text-xs font-semibold text-accent mb-1.5">{week.week}</p>
                        <ul className="space-y-1">
                          {week.tasks.map((task, ti) => (
                            <li key={ti} className="flex items-start gap-2 text-sm text-foreground/85">
                              <CheckCircle2 className="h-3.5 w-3.5 text-muted-foreground/50 shrink-0 mt-0.5" />
                              <span>{task}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Deliverables */}
                <div className="ml-[3.25rem] grid sm:grid-cols-2 gap-4 mb-5">
                  <div className="p-4 rounded-lg border border-border bg-card">
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                      📦 Deliverables
                    </h4>
                    <ul className="space-y-1.5">
                      {phase.deliverables.map((d, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-foreground/85">
                          <Star className="h-3.5 w-3.5 text-accent shrink-0 mt-0.5" />
                          <span>{d}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div>
                    <div className="p-4 rounded-lg border border-border bg-card mb-3">
                      <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                        🛠️ Skills Gained
                      </h4>
                      <div className="flex flex-wrap gap-1.5">
                        {phase.skills.map((s, i) => (
                          <span key={i} className="text-[11px] px-2 py-0.5 rounded-full bg-accent/10 text-accent-foreground font-medium">
                            {s}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="p-4 rounded-lg border border-accent/20 bg-accent/5">
                      <h4 className="text-xs font-semibold uppercase tracking-wider text-accent-foreground mb-2">
                        🎯 Interview Ready
                      </h4>
                      <ul className="space-y-1">
                        {phase.interviewReady.map((q, i) => (
                          <li key={i} className="text-xs text-foreground/80">
                            • {q}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>

                {/* Divider */}
                {phaseIdx < phases.length - 1 && (
                  <div className="ml-[3.25rem] border-b border-border" />
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* Certifications */}
      <section className="mb-14">
        <div className="flex items-center gap-2 mb-4">
          <Award className="h-5 w-5 text-accent" />
          <h2 className="text-xl font-bold text-foreground">Recommended Certifications</h2>
        </div>
        <p className="text-sm text-muted-foreground mb-4">
          Skills &gt; Certifications — but strategic certifications help pass ATS filters. Pick 1-2 maximum.
        </p>
        <div className="space-y-2">
          {certifications.map((cert, i) => (
            <div key={i} className="flex items-start gap-4 p-3 rounded-lg border border-border bg-card">
              <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold shrink-0 mt-0.5 ${
                cert.priority === "High" ? "bg-success/10 text-success" :
                cert.priority === "Medium" ? "bg-warning/10 text-warning" :
                "bg-muted text-muted-foreground"
              }`}>
                {cert.priority}
              </span>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground">{cert.name}</p>
                <p className="text-xs text-muted-foreground">{cert.reason}</p>
              </div>
              <span className="text-xs text-muted-foreground shrink-0">{cert.timeline}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Daily Routine */}
      <section className="mb-14">
        <div className="flex items-center gap-2 mb-4">
          <Briefcase className="h-5 w-5 text-accent" />
          <h2 className="text-xl font-bold text-foreground">Recommended Daily Routine</h2>
        </div>
        <p className="text-sm text-muted-foreground mb-4">
          You're working full-time. Here's how to fit AI learning into your schedule:
        </p>
        <div className="grid sm:grid-cols-2 gap-3">
          {[
            { time: "Morning (30 min)", activity: "Read 1 lesson + take notes", icon: BookOpen },
            { time: "Lunch Break (20 min)", activity: "Watch 1 AI video / read 1 article", icon: TrendingUp },
            { time: "Evening (45 min)", activity: "Hands-on coding / project work", icon: Code },
            { time: "Weekend (3-4 hrs)", activity: "Capstone project sprints + content creation", icon: Rocket },
          ].map((slot, i) => (
            <div key={i} className="p-4 rounded-lg border border-border bg-card flex items-start gap-3">
              <slot.icon className="h-4 w-4 text-accent shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-foreground">{slot.time}</p>
                <p className="text-xs text-muted-foreground">{slot.activity}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Success Metrics */}
      <section className="mb-14">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="h-5 w-5 text-accent" />
          <h2 className="text-xl font-bold text-foreground">12-Month Success Metrics</h2>
        </div>
        <div className="p-6 rounded-xl border border-accent/20 bg-accent/5">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
            {[
              { label: "Portfolio Projects", value: "6+", sub: "deployed & documented" },
              { label: "Certifications", value: "1-2", sub: "AWS ML / Google Cloud" },
              { label: "Articles Published", value: "24+", sub: "LinkedIn / blog posts" },
              { label: "Trainings Delivered", value: "3+", sub: "internal + external" },
            ].map((metric, i) => (
              <div key={i} className="text-center">
                <p className="text-2xl font-bold text-accent">{metric.value}</p>
                <p className="text-sm font-medium text-foreground">{metric.label}</p>
                <p className="text-xs text-muted-foreground">{metric.sub}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Final Motivation */}
      <section className="mb-10">
        <div className="p-6 rounded-xl border border-border bg-card text-center">
          <h2 className="text-lg font-bold text-foreground mb-2">🎯 Remember</h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            You have <strong className="text-foreground">10+ years of engineering experience</strong>. You're not starting from zero — 
            you're expanding your toolkit. Companies need people who can <strong className="text-foreground">BUILD AI systems</strong>, 
            not just train models. Your unique combination of Java + Testing + Training + AI makes you 
            <strong className="text-accent"> irreplaceable</strong>.
          </p>
          <div className="mt-4">
            <Link
              to="/"
              className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl gradient-accent text-accent-foreground font-medium text-sm hover:opacity-90 transition-opacity"
            >
              Start Learning Now
              <Rocket className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default RoadmapPage;
