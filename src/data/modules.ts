import {
  Brain,
  Cpu,
  Database,
  Wrench,
  Sparkles,
  Server,
  TestTube,
  Rocket,
  GraduationCap,
  TrendingUp,
  LucideIcon,
} from "lucide-react";

export interface Lesson {
  id: string;
  title: string;
  description: string;
  duration: string;
  type: "concept" | "hands-on" | "interview" | "project";
}

export interface Module {
  id: string;
  number: number;
  title: string;
  subtitle: string;
  description: string;
  icon: LucideIcon;
  color: string;
  lessons: Lesson[];
  interviewQuestions: string[];
}

export const modules: Module[] = [
  {
    id: "ai-mindset",
    number: 1,
    title: "AI Mindset for Engineers",
    subtitle: "Think differently about programming",
    description: "Understand why AI flips traditional programming on its head. Map your Java/testing experience to AI thinking patterns.",
    icon: Brain,
    color: "hsl(36 90% 55%)",
    lessons: [
      { id: "1-1", title: "AI vs Traditional Programming", description: "Rules-based vs learning-based systems — why this changes everything", duration: "15 min", type: "concept" },
      { id: "1-2", title: "Rule-Based vs Learning Systems", description: "From if-else to pattern recognition — a Java dev's perspective", duration: "20 min", type: "concept" },
      { id: "1-3", title: "Enterprise AI Use Cases", description: "Where AI actually delivers ROI in corporate environments", duration: "15 min", type: "concept" },
      { id: "1-4", title: "Your AI Learning Roadmap", description: "Mapping your existing skills to the AI landscape", duration: "10 min", type: "hands-on" },
    ],
    interviewQuestions: [
      "How does AI programming differ from traditional software development?",
      "What are the key considerations when deciding whether to use AI vs rule-based systems?",
      "Name 3 enterprise use cases where AI outperforms traditional approaches.",
    ],
  },
  {
    id: "ml-foundations",
    number: 2,
    title: "AI & ML Foundations",
    subtitle: "The engineering perspective",
    description: "ML vs DL vs GenAI demystified. Supervised vs unsupervised learning explained through testing analogies.",
    icon: Cpu,
    color: "hsl(210 80% 55%)",
    lessons: [
      { id: "2-1", title: "ML vs DL vs GenAI", description: "The hierarchy explained — think of it as Java → Spring → Spring Boot", duration: "20 min", type: "concept" },
      { id: "2-2", title: "Supervised vs Unsupervised Learning", description: "Like test cases with vs without expected results", duration: "20 min", type: "concept" },
      { id: "2-3", title: "Features, Labels & Overfitting", description: "Overfitting = your tests pass but production fails", duration: "15 min", type: "concept" },
      { id: "2-4", title: "Model Training Pipeline", description: "Think CI/CD but for machine learning models", duration: "25 min", type: "hands-on" },
    ],
    interviewQuestions: [
      "Explain the difference between ML, DL, and Generative AI with examples.",
      "What is overfitting and how does it relate to test coverage?",
      "When would you choose unsupervised learning over supervised learning?",
    ],
  },
  {
    id: "data-thinking",
    number: 3,
    title: "Data Thinking",
    subtitle: "Data is the new code",
    description: "Feature engineering, data preprocessing, structured vs unstructured data — all mapped to your engineering background.",
    icon: Database,
    color: "hsl(152 60% 42%)",
    lessons: [
      { id: "3-1", title: "Data as Code", description: "Why data quality matters more than algorithm choice", duration: "15 min", type: "concept" },
      { id: "3-2", title: "Structured vs Unstructured Data", description: "SQL tables vs images, text, audio — different pipelines", duration: "20 min", type: "concept" },
      { id: "3-3", title: "Feature Engineering Basics", description: "Transforming raw data into model-ready features", duration: "25 min", type: "hands-on" },
      { id: "3-4", title: "Image Data & Preprocessing", description: "Building on your image processing background", duration: "20 min", type: "hands-on" },
    ],
    interviewQuestions: [
      "Why is feature engineering important in ML?",
      "How do you handle missing data in a dataset?",
      "What's the difference between data preprocessing for structured vs unstructured data?",
    ],
  },
  {
    id: "hands-on-ml",
    number: 4,
    title: "Hands-on ML",
    subtitle: "Build, train, evaluate",
    description: "K-Means in business scenarios, image classification, model lifecycle, and evaluation metrics you can explain in interviews.",
    icon: Wrench,
    color: "hsl(280 65% 55%)",
    lessons: [
      { id: "4-1", title: "K-Means for Business", description: "Customer segmentation, anomaly detection — your clustering experience upgraded", duration: "25 min", type: "hands-on" },
      { id: "4-2", title: "Image Classification Basics", description: "From K-Means clustering to neural network classifiers", duration: "30 min", type: "hands-on" },
      { id: "4-3", title: "Model Lifecycle Management", description: "Training, validation, deployment — the MLOps pipeline", duration: "20 min", type: "concept" },
      { id: "4-4", title: "Evaluation Metrics", description: "Accuracy, precision, recall, F1 — what to use when", duration: "20 min", type: "concept" },
    ],
    interviewQuestions: [
      "Explain precision vs recall with a real-world example.",
      "How would you deploy and monitor an ML model in production?",
      "What is the bias-variance tradeoff?",
    ],
  },
  {
    id: "genai-llms",
    number: 5,
    title: "Generative AI & LLMs",
    subtitle: "The revolution explained",
    description: "How LLMs work, prompt engineering, embeddings, RAG architecture — hallucinations vs automation flakiness.",
    icon: Sparkles,
    color: "hsl(36 90% 55%)",
    lessons: [
      { id: "5-1", title: "How LLMs Actually Work", description: "Transformers, attention, tokens — the architecture simplified", duration: "25 min", type: "concept" },
      { id: "5-2", title: "Prompt Engineering Mastery", description: "Systematic prompting techniques for reliable outputs", duration: "30 min", type: "hands-on" },
      { id: "5-3", title: "Embeddings & Vector Databases", description: "How machines understand meaning — the search revolution", duration: "20 min", type: "concept" },
      { id: "5-4", title: "RAG Architecture", description: "Retrieval-Augmented Generation — the enterprise pattern", duration: "25 min", type: "hands-on" },
      { id: "5-5", title: "Hallucinations vs Flaky Tests", description: "Same problem, different domain — how to handle both", duration: "15 min", type: "concept" },
    ],
    interviewQuestions: [
      "What is RAG and why is it important for enterprise AI?",
      "How do embeddings capture semantic meaning?",
      "Compare AI hallucinations with flaky test failures.",
    ],
  },
  {
    id: "ai-java",
    number: 6,
    title: "AI in Java Full Stack",
    subtitle: "Your stack, supercharged",
    description: "AI system architecture with Spring Boot, microservices patterns for AI, error handling for non-deterministic systems.",
    icon: Server,
    color: "hsl(200 70% 45%)",
    lessons: [
      { id: "6-1", title: "AI System Architecture", description: "Design patterns for AI-powered enterprise applications", duration: "25 min", type: "concept" },
      { id: "6-2", title: "Spring Boot + AI APIs", description: "Integrating OpenAI, HuggingFace APIs into Spring services", duration: "30 min", type: "hands-on" },
      { id: "6-3", title: "Microservices for AI", description: "Service decomposition for ML inference, data pipelines", duration: "20 min", type: "concept" },
      { id: "6-4", title: "Error Handling for AI", description: "Handling non-deterministic outputs, retries, fallbacks", duration: "20 min", type: "hands-on" },
    ],
    interviewQuestions: [
      "How would you architect a Spring Boot application that uses AI APIs?",
      "What are the key differences in error handling for AI vs traditional services?",
      "Design a microservices architecture for an AI-powered recommendation system.",
    ],
  },
  {
    id: "ai-testing",
    number: 7,
    title: "AI for Test Automation",
    subtitle: "Testing meets intelligence",
    description: "AI-generated test cases, self-healing locators, validating AI outputs, and testing AI systems themselves.",
    icon: TestTube,
    color: "hsl(340 65% 50%)",
    lessons: [
      { id: "7-1", title: "AI-Generated Test Cases", description: "Using LLMs to generate comprehensive test suites", duration: "25 min", type: "hands-on" },
      { id: "7-2", title: "Self-Healing Tests", description: "AI-powered locator strategies that adapt to UI changes", duration: "25 min", type: "hands-on" },
      { id: "7-3", title: "Validating AI Outputs", description: "How do you test something non-deterministic?", duration: "20 min", type: "concept" },
      { id: "7-4", title: "Testing AI Systems", description: "Quality assurance for ML models and AI pipelines", duration: "20 min", type: "concept" },
    ],
    interviewQuestions: [
      "How do you test AI-powered features when outputs are non-deterministic?",
      "What is a self-healing test and how does AI enable it?",
      "How would you validate an LLM-based feature in production?",
    ],
  },
  {
    id: "capstone",
    number: 8,
    title: "Capstone Projects",
    subtitle: "Build your portfolio",
    description: "AI Resume Screener, Test Case Generator, Chatbot, Image Defect Detection — with architecture, data flow, and interview points.",
    icon: Rocket,
    color: "hsl(15 80% 55%)",
    lessons: [
      { id: "8-1", title: "AI Resume Screener", description: "NLP-based resume parsing and ranking system", duration: "45 min", type: "project" },
      { id: "8-2", title: "AI Test Case Generator", description: "LLM-powered test generation from requirements", duration: "45 min", type: "project" },
      { id: "8-3", title: "Enterprise AI Chatbot", description: "RAG-based chatbot with domain knowledge", duration: "45 min", type: "project" },
      { id: "8-4", title: "Image Defect Detection", description: "Computer vision for quality assurance", duration: "45 min", type: "project" },
    ],
    interviewQuestions: [
      "Walk me through the architecture of an AI-powered chatbot.",
      "How would you handle edge cases in an AI resume screening system?",
      "What metrics would you use to evaluate an AI test case generator?",
    ],
  },
  {
    id: "trainer-mode",
    number: 9,
    title: "Trainer Mode",
    subtitle: "Teach AI to others",
    description: "How to teach AI to freshers, managers, and corporate teams. Demo strategies, curriculum design, engagement techniques.",
    icon: GraduationCap,
    color: "hsl(260 60% 55%)",
    lessons: [
      { id: "9-1", title: "Teaching AI to Freshers", description: "Simplify without dumbing down — the trainer's art", duration: "20 min", type: "concept" },
      { id: "9-2", title: "AI for Managers & Execs", description: "Business value, ROI, risk — the executive language", duration: "20 min", type: "concept" },
      { id: "9-3", title: "Demo Strategies", description: "Live demos that wow — preparation and recovery tactics", duration: "15 min", type: "hands-on" },
      { id: "9-4", title: "Curriculum Design for AI", description: "Structuring multi-day AI training programs", duration: "20 min", type: "hands-on" },
    ],
    interviewQuestions: [
      "How would you explain neural networks to a non-technical audience?",
      "Design a 3-day corporate AI training curriculum.",
      "How do you handle skepticism about AI in training sessions?",
    ],
  },
  {
    id: "career",
    number: 10,
    title: "Career & Monetization",
    subtitle: "Turn skills into income",
    description: "AI courses, corporate training, freelancing, certifications vs real skills — your career acceleration plan.",
    icon: TrendingUp,
    color: "hsl(152 60% 42%)",
    lessons: [
      { id: "10-1", title: "AI Career Paths", description: "Engineer, trainer, consultant — which fits you?", duration: "15 min", type: "concept" },
      { id: "10-2", title: "Building AI Courses", description: "Create and sell AI training content", duration: "20 min", type: "hands-on" },
      { id: "10-3", title: "Corporate AI Training Business", description: "Pricing, proposals, delivery — the business side", duration: "20 min", type: "concept" },
      { id: "10-4", title: "Certifications vs Skills", description: "What actually matters in the AI job market", duration: "15 min", type: "concept" },
    ],
    interviewQuestions: [
      "What AI certifications are most valued by employers?",
      "How would you pitch an AI training program to a corporate client?",
      "What's your 6-month plan to transition into AI?",
    ],
  },
];

export const getLessonTypeLabel = (type: Lesson["type"]) => {
  switch (type) {
    case "concept": return "Concept";
    case "hands-on": return "Hands-on";
    case "interview": return "Interview Prep";
    case "project": return "Project";
  }
};

export const getLessonTypeColor = (type: Lesson["type"]) => {
  switch (type) {
    case "concept": return "bg-info/10 text-info";
    case "hands-on": return "bg-success/10 text-success";
    case "interview": return "bg-warning/10 text-warning";
    case "project": return "bg-accent/10 text-accent-foreground";
  }
};
