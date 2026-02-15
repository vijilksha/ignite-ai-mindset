export interface LessonContent {
  id: string;
  concept: string;
  whyItMatters: string;
  realWorldExample: {
    title: string;
    scenario: string;
    implementation: string;
    outcome: string;
  };
  caseStudy: {
    title: string;
    background: string;
    challenge: string;
    approach: string[];
    result: string;
    yourTask: string;
  };
  keyTakeaways: string[];
  trainerTip?: string;
  codeSnippet?: {
    language: string;
    code: string;
    explanation: string;
  };
}

export const lessonContents: Record<string, LessonContent> = {
  // ============ MODULE 1: AI Mindset for Engineers ============
  "1-1": {
    id: "1-1",
    concept: "In traditional programming, YOU write the rules: if (age > 18) → allow. In AI, you provide examples (data) and the system LEARNS the rules itself. Think of it like this — traditional programming is writing a Selenium script with explicit locators, while AI is like training a model to find elements by learning patterns from the page structure.",
    whyItMatters: "As a Java developer, you've spent years writing deterministic logic. AI flips this — the program writes its own logic from data. Understanding this paradigm shift is the first step to thinking like an AI engineer.",
    realWorldExample: {
      title: "Spam Filter at a Banking Corporation",
      scenario: "A large bank receives 10,000+ emails daily. Traditional approach: developers write 500+ rules like 'if subject contains URGENT and sender not in whitelist → spam'. Problem: spammers evolve faster than rules.",
      implementation: "AI approach: Feed 100,000 labeled emails (spam/not-spam) to a model. The model learns patterns humans never thought of — like specific combinations of fonts, sending times, and header metadata that correlate with spam.",
      outcome: "Rule-based system caught 78% of spam. ML model caught 99.2%. More importantly, the ML model adapted to new spam patterns without code changes — it just needed new training data."
    },
    caseStudy: {
      title: "Infosys – Automating Invoice Processing",
      background: "Infosys handled 2M+ invoices/year for a telecom client. Traditional OCR + rules failed on 35% of invoices due to format variations.",
      challenge: "Invoices came in 200+ formats from different vendors. Writing rules for each was unsustainable.",
      approach: [
        "Collected 50,000 labeled invoices across all formats",
        "Trained a document understanding model (layout + text)",
        "Model learned to identify key fields regardless of format",
        "Deployed as a microservice in the existing Java pipeline"
      ],
      result: "Processing accuracy jumped from 65% to 94%. Manual review reduced by 80%. The system learned new formats with just 50 examples instead of weeks of rule-writing.",
      yourTask: "Think about a process in your current project that relies on complex if-else logic. Could it benefit from a learning-based approach? Write down: (1) What rules exist today, (2) What data could replace those rules, (3) What would the training data look like."
    },
    keyTakeaways: [
      "Traditional: Human writes rules → Program executes",
      "AI: Human provides data → Program learns rules",
      "AI excels when rules are too complex, too many, or constantly changing",
      "Your Java/testing skills are assets — AI needs good engineering around it"
    ],
    trainerTip: "When teaching this to freshers, use the 'sorting hat' analogy from Harry Potter — the hat learned from thousands of students, it doesn't have if-else for each student."
  },
  "1-2": {
    id: "1-2",
    concept: "Rule-based systems follow explicit logic trees — like your JUnit test assertions. Learning systems find patterns in data — like how Netflix figures out what you'll watch next without anyone coding 'if user watched X, recommend Y' for millions of combinations.",
    whyItMatters: "Knowing WHEN to use rule-based vs learning systems is crucial for architecture decisions. Not everything needs AI, and using AI where rules suffice wastes resources.",
    realWorldExample: {
      title: "E-commerce Price Engine at Flipkart",
      scenario: "Flipkart needed to set prices for 150M+ products. Simple products (commodities) used rule-based pricing: cost + margin + competitor_price. But fashion items had complex pricing influenced by trends, season, inventory, and Instagram buzz.",
      implementation: "Hybrid approach: Rule-based engine for 80% of products (clear formulas). ML model for fashion/trending categories that considers 40+ signals including social media trends, weather forecasts, and competitor stock levels.",
      outcome: "Fashion category revenue increased 23% with ML pricing. Rule-based products stayed on deterministic pricing — reliable and auditable."
    },
    caseStudy: {
      title: "TCS – Customer Support Routing",
      background: "A TCS client (insurance company) routed 50,000 daily calls using a decision tree: press 1 for claims, 2 for billing, etc. 40% of calls were misrouted.",
      challenge: "Customers described problems in natural language that didn't fit neat categories. 'My car was hit and I need to know about my premium' touches claims, billing, AND policy.",
      approach: [
        "Phase 1: Kept rule-based routing for simple, clear requests",
        "Phase 2: Added NLP model for ambiguous queries",
        "Phase 3: Model classified intent + urgency + department",
        "Phase 4: Continuous learning from agent corrections"
      ],
      result: "Misrouting dropped from 40% to 8%. Average handle time reduced by 3 minutes. Key lesson: hybrid approach worked better than pure AI.",
      yourTask: "List 5 features in your current application. For each, decide: Rule-based or AI? Justify with: (1) Is the logic static or evolving? (2) How many edge cases exist? (3) Is the data available to train a model?"
    },
    keyTakeaways: [
      "Use rules when logic is clear, auditable, and doesn't change often",
      "Use AI when patterns are complex, data is abundant, and rules can't keep up",
      "Hybrid approaches often win in enterprise — rules + AI together",
      "Think of it as: Selenium (rules) vs Playwright's auto-wait (learned behavior)"
    ],
    trainerTip: "Draw a 2x2 matrix on the whiteboard: Complexity (low/high) × Change frequency (low/high). Rules go in low-low, AI in high-high, hybrid in the rest."
  },
  "1-3": {
    id: "1-3",
    concept: "Enterprise AI isn't about replacing humans — it's about augmenting decisions, automating repetitive cognitive tasks, and extracting insights from data that's too massive for manual analysis. Think of it as adding 'intelligence middleware' to your existing enterprise architecture.",
    whyItMatters: "As a trainer, you need to answer: 'Where exactly should my company invest in AI?' This lesson gives you a framework to evaluate AI use cases by ROI, feasibility, and risk.",
    realWorldExample: {
      title: "Wipro – Predictive Maintenance for Manufacturing",
      scenario: "A Wipro manufacturing client had 2,000 CNC machines. Unplanned downtime cost ₹15 lakhs per hour. Maintenance was calendar-based (every 30 days), which meant either too early (wasting parts) or too late (breakdowns).",
      implementation: "IoT sensors collected vibration, temperature, and sound data every second. An ML model trained on 2 years of historical breakdown data predicted failures 48-72 hours in advance. Integrated with the existing SAP PM module via REST API.",
      outcome: "Unplanned downtime reduced 73%. Maintenance costs dropped 40%. ROI achieved in 8 months."
    },
    caseStudy: {
      title: "Healthcare Chain – Patient No-Show Prediction",
      background: "A hospital chain had a 25% no-show rate for outpatient appointments, costing ₹2Cr annually in lost revenue and idle doctor time.",
      challenge: "Overbooking caused chaos. Under-booking wasted resources. They needed to predict which patients would actually show up.",
      approach: [
        "Collected 3 years of appointment data (200K records)",
        "Features: day of week, lead time, patient history, weather, distance",
        "Trained gradient boosting model to predict no-show probability",
        "Integrated prediction into appointment booking Spring Boot API",
        "Auto-overbooking for high no-show probability slots"
      ],
      result: "No-show impact reduced by 45%. Doctor utilization increased 20%. The model ran as a simple REST endpoint — no complex AI infrastructure needed.",
      yourTask: "Pick your current industry/domain. Identify 3 AI use cases that could save time or money. For each: (1) What data exists? (2) What decision does it improve? (3) What's the estimated ROI? (4) What could go wrong?"
    },
    keyTakeaways: [
      "Enterprise AI use cases: prediction, classification, recommendation, anomaly detection, NLP",
      "Start with high-data, low-risk use cases for quick wins",
      "AI is middleware — it fits INTO existing architecture, not replaces it",
      "ROI framework: (Time saved × Cost per hour) - (AI development + maintenance cost)"
    ],
    trainerTip: "Always start corporate training with 3 industry-specific AI success stories. Executives care about ROI numbers, not algorithms."
  },
  "1-4": {
    id: "1-4",
    concept: "Your existing skills map directly to AI roles. Java → ML system engineering. Testing → AI validation. Full stack → end-to-end AI applications. This isn't starting over — it's expanding your toolkit.",
    whyItMatters: "Many experienced developers feel like beginners when entering AI. This roadmap shows how your 10+ years of experience accelerates your AI journey compared to someone starting fresh.",
    realWorldExample: {
      title: "Senior Java Developer's 6-Month AI Transition",
      scenario: "Rajesh, a 12-year Java architect at HCL, wanted to move into AI. Instead of starting with math, he mapped his skills: Spring Boot → ML API serving, JUnit → Model validation, Jenkins CI/CD → MLOps pipelines, Design patterns → ML system patterns.",
      implementation: "Month 1-2: Python basics + ML fundamentals (mapped to Java equivalents). Month 3-4: Built AI features in existing Java projects using APIs. Month 5-6: Led an AI proof-of-concept for a client project.",
      outcome: "Got promoted to 'AI Solutions Architect' without a data science degree. Key insight: companies need people who can BUILD AI systems, not just train models."
    },
    caseStudy: {
      title: "Your Personal AI Roadmap Exercise",
      background: "You have 10+ years of Java Full Stack, 2+ years of automation testing, and image processing + K-Means experience.",
      challenge: "How do you leverage ALL of this to become an AI engineer in the shortest time possible?",
      approach: [
        "Map Java skills: OOP → ML class design, Spring → AI API serving, JDBC → Data pipelines",
        "Map Testing skills: Test strategies → Model validation, Selenium → AI-powered testing",
        "Map Image Processing: OpenCV basics → Computer Vision, K-Means → Unsupervised ML",
        "Identify gaps: Deep Learning theory, Prompt Engineering, MLOps",
        "Create a 90-day sprint plan with weekly deliverables"
      ],
      result: "Your unique combination of full-stack + testing + image processing is rare in the AI market. Most AI engineers lack production engineering and testing skills.",
      yourTask: "Create YOUR roadmap: (1) List your top 10 technical skills, (2) Map each to an AI equivalent, (3) Identify 3 gaps to fill, (4) Set a 30-60-90 day goal. Share this with a mentor or peer for feedback."
    },
    keyTakeaways: [
      "Your engineering background is an ADVANTAGE, not a limitation",
      "AI engineering needs: 30% ML knowledge + 70% software engineering",
      "Companies pay more for 'ML Engineers who can code' than 'Data Scientists who can't deploy'",
      "Start building AI features in your current projects — don't wait to 'fully learn' first"
    ],
    trainerTip: "In corporate training, always start with a skill-mapping exercise. It builds confidence and shows trainees they're not starting from zero."
  },

  // ============ MODULE 2: AI & ML Foundations ============
  "2-1": {
    id: "2-1",
    concept: "Think of ML, DL, and GenAI as layers — like Java → Spring → Spring Boot. ML is the foundation (learning from data). Deep Learning is ML with neural networks (many layers of processing). GenAI is DL that creates new content (text, images, code). Each layer adds power but also complexity.",
    whyItMatters: "In interviews, you'll be asked to distinguish these. In architecture decisions, knowing which level you need prevents over-engineering (using GPT-4 when a simple regression works).",
    realWorldExample: {
      title: "Amazon's Product Stack Uses All Three",
      scenario: "Amazon uses ML for demand forecasting (regression models predict how many units to stock). DL for product image search (CNNs understand visual similarity). GenAI for product descriptions (LLMs write compelling copy for millions of products).",
      implementation: "ML: XGBoost model trained on 5 years of sales data → predicts demand. DL: ResNet model → finds visually similar products from photos. GenAI: Fine-tuned GPT → generates SEO-optimized descriptions from product attributes.",
      outcome: "Each level solves a different problem at a different cost. ML model runs for $0.001/prediction. DL costs $0.01/image. GenAI costs $0.03/description. Choosing the right level = cost optimization."
    },
    caseStudy: {
      title: "Retail Chain – Choosing the Right AI Level",
      background: "A retail chain wanted to 'add AI'. The CTO asked for recommendations across three problems: inventory prediction, visual merchandising, and customer chatbot.",
      challenge: "The team defaulted to 'use GPT for everything' — wasting budget on over-engineered solutions.",
      approach: [
        "Inventory: Simple ML (gradient boosting) — structured data, clear target variable",
        "Visual merchandising: DL (CNN) — image understanding needed, pre-trained models available",
        "Customer chatbot: GenAI (LLM + RAG) — natural language, creative responses needed",
        "Cost analysis: ML=$2K/month, DL=$8K/month, GenAI=$15K/month"
      ],
      result: "Saved 60% on AI infrastructure by right-sizing solutions. ML handled 70% of use cases. Key lesson: GenAI is powerful but expensive — use it only when needed.",
      yourTask: "Take 5 AI use cases from your company/industry. Classify each as ML, DL, or GenAI. Justify: What type of data? What type of output? What's the cost sensitivity?"
    },
    keyTakeaways: [
      "ML = learning from structured data (tables, numbers) → predictions",
      "DL = learning from unstructured data (images, audio, text) → complex patterns",
      "GenAI = creating new content → text, images, code, video",
      "Always start with the simplest level that solves the problem"
    ],
    codeSnippet: {
      language: "java",
      code: `// Think of it like Java abstraction layers:
// ML = Writing JDBC code directly
// DL = Using Hibernate (more abstraction, more power)
// GenAI = Using Spring Data JPA (highest abstraction, magic underneath)

// ML equivalent in Java thinking:
double predict(double[] features) {
    // Simple: weighted sum of features
    return weights[0]*features[0] + weights[1]*features[1] + bias;
}

// DL equivalent: layers of processing
double predict(double[] input) {
    double[] layer1 = applyWeights(input, weightsL1);  // 100s of neurons
    double[] layer2 = applyWeights(layer1, weightsL2);  // deeper patterns
    return applyWeights(layer2, weightsL3)[0];           // final prediction
}`,
      explanation: "ML is like writing raw JDBC — you control everything. DL is like Hibernate — more abstraction, handles complexity. GenAI is like Spring Data JPA — maximum abstraction, you just describe what you want."
    },
    trainerTip: "Use a Russian nesting doll visual: GenAI contains DL, which contains ML. Each outer layer adds capability and cost."
  },
  "2-2": {
    id: "2-2",
    concept: "Supervised learning = you have test cases WITH expected results (labeled data). Like JUnit tests where you know the expected output. Unsupervised learning = you have data but NO expected results. Like exploratory testing — you're looking for patterns you didn't know existed.",
    whyItMatters: "This determines what kind of AI project you can build. Have labels? Supervised. No labels? Unsupervised. This is the first question you ask when scoping an AI project.",
    realWorldExample: {
      title: "Bank Fraud Detection — Both Types Working Together",
      scenario: "HDFC Bank processes 10M transactions/day. Supervised model: trained on 5 years of known fraud cases (labeled: fraud/not-fraud). Catches known fraud patterns. Unsupervised model: finds unusual transaction clusters that don't match ANY known pattern — potentially new types of fraud.",
      implementation: "Supervised (Random Forest): 99.1% accurate on known fraud types. Unsupervised (Isolation Forest): discovered 3 new fraud patterns in Q1 that the supervised model missed. Combined: both models run in parallel on every transaction.",
      outcome: "Fraud detection improved 34% by combining both approaches. Like having regression tests (supervised) AND exploratory tests (unsupervised) running together."
    },
    caseStudy: {
      title: "E-commerce Customer Segmentation at Myntra",
      background: "Myntra wanted to personalize marketing for 50M users but didn't know how many customer types existed.",
      challenge: "No predefined labels — you can't label 50M users manually. Classic unsupervised learning problem.",
      approach: [
        "Collected behavioral data: browsing patterns, purchase history, return rates",
        "Applied K-Means clustering (your existing knowledge!) with K=2 to K=20",
        "Elbow method identified K=6 natural segments",
        "Named segments: Budget Browsers, Brand Loyalists, Trend Chasers, etc.",
        "Then used these segments as LABELS for supervised models to classify new users"
      ],
      result: "Campaign conversion rate increased 40%. Marketing spend efficiency improved 25%. Unsupervised → Supervised pipeline is a common enterprise pattern.",
      yourTask: "Using your K-Means knowledge: (1) What features would you use to segment users of your current application? (2) How would you determine the right K value? (3) How would you validate the segments make business sense?"
    },
    keyTakeaways: [
      "Supervised = labeled data → prediction (like test cases with assertions)",
      "Unsupervised = unlabeled data → pattern discovery (like exploratory testing)",
      "Semi-supervised: small labeled set + large unlabeled set → best of both",
      "Your K-Means experience IS unsupervised learning — you're already ahead!"
    ],
    trainerTip: "Ask trainees: 'If I give you 1000 photos with no labels, can you group similar ones?' That's unsupervised. 'Now if I tell you cat/dog labels for 100 of them?' That's supervised."
  },
  "2-3": {
    id: "2-3",
    concept: "Features = input variables (like method parameters). Labels = expected output (like return values). Overfitting = your model memorizes training data but fails on new data. It's EXACTLY like when your test suite passes perfectly in CI but the app crashes in production — your tests were too specific to test data.",
    whyItMatters: "Overfitting is the #1 pitfall in ML. If you understand it through the testing lens, you'll build better models AND better test suites.",
    realWorldExample: {
      title: "Resume Screening Model That Failed in Production",
      scenario: "A hiring platform built a resume screening model. Training accuracy: 95%. Production accuracy: 62%. Why? The model learned that resumes from specific colleges (in the training data) meant 'good candidate'. It memorized college names instead of learning actual skills.",
      implementation: "Fix: Removed college name as a feature. Added: years of experience, skill keywords, project descriptions. Used cross-validation (5-fold) instead of single train/test split. Added regularization to prevent memorization.",
      outcome: "Production accuracy rose to 84%. The model now generalized to new colleges and backgrounds. Like fixing flaky tests by removing environment-specific hardcoded values."
    },
    caseStudy: {
      title: "Automation Testing Parallel – Flaky Tests = Overfit Models",
      background: "Your Selenium tests pass on your machine but fail in CI. Your ML model scores 98% on training data but 60% on new data. Same root cause: too tightly coupled to specific conditions.",
      challenge: "Identify the parallels and solutions across both domains.",
      approach: [
        "Flaky test: hardcoded sleep(5000) → Overfit: model memorized timestamps",
        "Flaky test: specific screen resolution → Overfit: model trained on narrow data",
        "Fix tests: use explicit waits, parameterize → Fix model: regularization, cross-validation",
        "Fix tests: run on multiple browsers → Fix model: train on diverse data",
        "Fix tests: separate test data from assertions → Fix model: separate train/test sets"
      ],
      result: "Both problems share one solution: make your system generalize instead of memorize. In testing, we call it 'robust tests'. In ML, we call it 'generalization'.",
      yourTask: "Take a model or test suite: (1) Identify 3 ways it could be 'overfitting' to specific conditions, (2) Propose fixes using cross-validation/regularization concepts, (3) How would you monitor for overfitting in production?"
    },
    keyTakeaways: [
      "Features = inputs to your model (choose carefully, garbage in = garbage out)",
      "Labels = what you're predicting (must be clean and consistent)",
      "Overfitting = memorization, not learning (like hardcoded test data)",
      "Cross-validation = running tests on multiple 'environments' for the model"
    ],
    codeSnippet: {
      language: "java",
      code: `// Testing analogy for overfitting:

// OVERFIT TEST (too specific - like an overfit model):
@Test
void testLogin() {
    driver.findElement(By.id("user-input-v2-2024")).sendKeys("admin");
    // ^ Hardcoded to specific ID version — breaks when UI changes
}

// GENERALIZED TEST (like a well-trained model):
@Test 
void testLogin() {
    driver.findElement(By.cssSelector("[data-testid='username']")).sendKeys("admin");
    // ^ Generalizes across UI versions — robust like a good model
}

// ML equivalent:
// Overfit: model.train(sameData, epochs=10000)  // memorizes
// Good:   model.train(diverseData, epochs=100, regularization=0.01)  // learns`,
      explanation: "Just like robust test locators generalize across UI changes, well-trained models generalize across new data. Both require the same mindset: don't be too specific."
    },
    trainerTip: "Show a graph of training accuracy going UP while validation accuracy goes DOWN — that V-shape is the 'aha moment' for overfitting."
  },
  "2-4": {
    id: "2-4",
    concept: "The ML training pipeline is your CI/CD pipeline for models. Collect data → Clean data → Feature engineering → Train → Validate → Deploy → Monitor. Just like: Code → Build → Test → Stage → Deploy → Monitor. Same engineering discipline, different artifact.",
    whyItMatters: "Understanding the pipeline means you can architect ML systems using your DevOps experience. MLOps IS DevOps for machine learning.",
    realWorldExample: {
      title: "Swiggy's Real-Time Delivery Time Prediction Pipeline",
      scenario: "Swiggy predicts delivery time for every order. The pipeline: ingest real-time data (traffic, weather, restaurant prep time, rider location) → feature engineering → model inference → display ETA → collect actual delivery time → retrain weekly.",
      implementation: "Data pipeline: Kafka streams → Spark processing → Feature store. Training: Weekly retraining on last 4 weeks of data. Deployment: Blue-green deployment of models (like microservices). Monitoring: Track prediction accuracy daily, auto-alert if accuracy drops below 85%.",
      outcome: "ETA accuracy improved from ±15 min to ±5 min. Pipeline runs like clockwork — same reliability standards as their Java microservices."
    },
    caseStudy: {
      title: "Building Your First ML Pipeline — Loan Approval System",
      background: "A bank wants to automate loan approval decisions currently made by 50 underwriters manually reviewing applications.",
      challenge: "Build an end-to-end pipeline that a Java developer can understand and maintain.",
      approach: [
        "Data Collection: Extract 5 years of loan data from Oracle DB (JDBC skills!)",
        "Data Cleaning: Handle nulls, outliers, inconsistent formats (like test data setup)",
        "Feature Engineering: Create new features — debt-to-income ratio, payment history score",
        "Training: Split 80/20, train Random Forest, cross-validate",
        "Evaluation: Accuracy, precision, recall, F1 — like test coverage metrics",
        "Deployment: Expose as REST API (Spring Boot — your comfort zone!)",
        "Monitoring: Log predictions, track accuracy weekly, retrain monthly"
      ],
      result: "The pipeline mirrors what you already know. The new parts are: feature engineering, model selection, and continuous monitoring. Everything else is familiar.",
      yourTask: "Design a CI/CD pipeline diagram for an ML model. Include: data versioning, model versioning, automated testing, staging environment, production deployment, rollback strategy."
    },
    keyTakeaways: [
      "ML Pipeline = Data CI/CD — same principles, different artifacts",
      "Key stages: Collect → Clean → Engineer → Train → Validate → Deploy → Monitor",
      "Model versioning is like code versioning — always be able to rollback",
      "Monitoring is crucial — models degrade over time (concept drift)"
    ],
    trainerTip: "Draw the ML pipeline NEXT to a CI/CD pipeline — the parallels are stunning. This is the fastest way for engineers to 'get it'."
  },

  // ============ MODULE 3: Data Thinking ============
  "3-1": {
    id: "3-1",
    concept: "In traditional development, code is king. In AI, data is king. A mediocre algorithm with great data beats a brilliant algorithm with bad data. Think of data as your 'source code' for AI — if the source is buggy, no compiler (algorithm) can fix it.",
    whyItMatters: "80% of ML project time goes into data preparation. Developers who understand data quality ship better models faster.",
    realWorldExample: {
      title: "Google's Lesson — 'More Data Beats Better Algorithms'",
      scenario: "Google's translation team had a sophisticated linguistic model with expert-crafted grammar rules. A competing team used a simple statistical model but fed it 10x more training data (billions of web pages).",
      implementation: "The simple model with more data won decisively. Google then rebuilt Translate around this principle — invest in data pipelines, not just algorithms.",
      outcome: "Google Translate improved from 40% accuracy to 85%+ by focusing on data quantity and quality rather than algorithm complexity."
    },
    caseStudy: {
      title: "Insurance Claims — Data Quality Nightmare",
      background: "An insurance company wanted to predict claim fraud. They had 10 years of claims data — sounds great, right?",
      challenge: "Data issues: 30% missing agent codes, inconsistent date formats (DD/MM vs MM/DD), duplicate records, label inconsistency ('fraud', 'FRAUD', 'suspicious', 'flagged' all meant different things to different adjusters).",
      approach: [
        "Data audit: profiled every column for completeness, consistency, accuracy",
        "Standardization: unified labels to binary (fraud/legitimate)",
        "Deduplication: used fuzzy matching on policyholder name + claim date",
        "Missing data: imputed agent codes using geographic region patterns",
        "Validation: cross-referenced with external fraud databases"
      ],
      result: "Data cleaning took 3 months. Model training took 2 weeks. After cleaning, even a simple logistic regression achieved 87% accuracy — the 'magic' was in the data.",
      yourTask: "Audit a dataset (or database) you work with: (1) What % of fields have missing values? (2) Are there inconsistent formats? (3) Are there duplicate records? (4) How confident are you in the labels/categories?"
    },
    keyTakeaways: [
      "Data quality > algorithm sophistication — always",
      "80% of ML work is data preparation (like 80% of testing is setup)",
      "Bad data = bad model, no matter how fancy the algorithm",
      "Data profiling is the ML equivalent of code review"
    ],
    trainerTip: "Show real messy data in class — Excel sheets with missing values, typos, duplicates. Ask: 'Would you trust a model trained on this?' The reaction teaches the lesson."
  },
  "3-2": {
    id: "3-2",
    concept: "Structured data = database tables (SQL-queryable). Unstructured data = images, text, audio, video (needs AI to parse). Like comparing a well-defined REST API response (JSON) with a raw email body — both contain information, but you need different tools to extract it.",
    whyItMatters: "85% of enterprise data is unstructured. If you only know structured data ML, you're missing the biggest opportunity.",
    realWorldExample: {
      title: "Hospital Records — Structured Meets Unstructured",
      scenario: "A hospital has structured data (patient age, blood pressure, lab results in tables) AND unstructured data (doctor's handwritten notes, X-ray images, discharge summaries as free text).",
      implementation: "Structured pipeline: SQL → Pandas DataFrame → XGBoost model. Unstructured pipeline: OCR on notes → NLP extraction → Image CNN for X-rays → combine all features. The structured model predicted readmission at 72% accuracy. Adding unstructured data (especially doctor notes sentiment) pushed it to 89%.",
      outcome: "The unstructured data contained insights that structured fields missed — like a doctor writing 'patient seems confused about medication schedule' which strongly predicted readmission."
    },
    caseStudy: {
      title: "Legal Contract Analysis at Deloitte",
      background: "Deloitte's legal team reviewed 10,000 contracts/year for risk clauses. Each contract was 50-200 pages of unstructured text.",
      challenge: "Extracting structured information (party names, dates, liability caps, termination clauses) from unstructured legal documents.",
      approach: [
        "Classified documents: structured metadata (date, parties) + unstructured (clauses)",
        "NLP model for clause extraction — trained on 5,000 annotated contracts",
        "Named Entity Recognition for party names, dates, amounts",
        "Classification model for risk level (high/medium/low)",
        "Output: structured JSON from each contract — queryable, searchable"
      ],
      result: "Review time reduced from 4 hours to 20 minutes per contract. 92% extraction accuracy. Lawyers focused on edge cases instead of routine extraction.",
      yourTask: "In your domain, identify: (1) 3 sources of structured data, (2) 3 sources of unstructured data, (3) How combining them could create better predictions than either alone."
    },
    keyTakeaways: [
      "Structured data: tables, numbers, categories → traditional ML works well",
      "Unstructured data: text, images, audio → needs DL/NLP/CV",
      "Most real-world problems need BOTH — multimodal approaches",
      "Your image processing experience is directly applicable to unstructured visual data"
    ],
    trainerTip: "Bring two data sources to training: a clean CSV and a folder of scanned documents. Ask: 'How would you predict customer churn using BOTH?'"
  },
  "3-3": {
    id: "3-3",
    concept: "Feature engineering = transforming raw data into meaningful inputs for your model. Like refactoring code — the logic is the same, but the structure makes it faster and more effective. Raw data is your 'spaghetti code'; engineered features are your 'clean architecture'.",
    whyItMatters: "Good feature engineering can improve model performance more than changing algorithms. It's where domain expertise (your Java/testing knowledge) becomes a superpower.",
    realWorldExample: {
      title: "Uber's Ride Time Prediction",
      scenario: "Raw data: pickup lat/lng, dropoff lat/lng, timestamp. Simple model with raw features: ±12 minutes accuracy. After feature engineering: ±4 minutes accuracy.",
      implementation: "Engineered features: straight-line distance → actual road distance. Timestamp → hour_of_day, day_of_week, is_holiday, is_rush_hour. Lat/lng → neighborhood_id, airport_zone. Historical: avg_speed_this_route_this_hour, surge_multiplier. Weather API integration: is_raining, temperature.",
      outcome: "Feature engineering tripled the model's useful features from 4 to 15. Accuracy improved 3x. The algorithm (gradient boosting) didn't change — only the features did."
    },
    caseStudy: {
      title: "Feature Engineering for Test Flakiness Prediction",
      background: "Your test suite has 5,000 automated tests. Some are flaky. Can you predict which tests will be flaky BEFORE running them?",
      challenge: "Raw data: test name, test code, test results history. Not very useful as-is.",
      approach: [
        "From test code: lines_of_code, num_assertions, uses_sleep, uses_network_call, has_date_dependency",
        "From history: pass_rate_last_30_runs, flip_count, avg_duration, duration_variance",
        "From context: time_since_last_code_change, num_dependencies, UI_vs_API_test",
        "Interaction features: sleep_count × network_calls (compound risk)",
        "Temporal: day_of_week effect (some tests fail more on Mondays after deployments)"
      ],
      result: "Model predicted flaky tests with 85% accuracy. The features your testing experience suggested (sleep usage, network calls) were the top predictors. Domain expertise = feature engineering superpower.",
      yourTask: "For a problem you care about, start with raw data columns and engineer 5 new features. For each: (1) What raw columns does it use? (2) What transformation did you apply? (3) Why does it capture information the raw columns miss?"
    },
    keyTakeaways: [
      "Feature engineering = extracting signal from noise in your data",
      "Domain expertise is MORE valuable than algorithm knowledge here",
      "Common techniques: binning, encoding, interaction features, temporal features",
      "Your testing/Java experience gives you intuition for good features"
    ],
    codeSnippet: {
      language: "java",
      code: `// Feature Engineering = Data Refactoring

// RAW DATA (messy, not useful directly):
class RawOrder {
    LocalDateTime orderTime;
    double latitude, longitude;
    String productId;
}

// ENGINEERED FEATURES (clean, meaningful):
class OrderFeatures {
    int hourOfDay;           // from orderTime
    boolean isWeekend;       // from orderTime  
    boolean isRushHour;      // hourOfDay in [8-10, 17-19]
    String neighborhood;     // from lat/lng geocoding
    double avgOrderValue;    // from historical productId data
    int daysSinceLastOrder;  // from customer history
    
    // Interaction feature
    boolean rushHourWeekday; // isRushHour && !isWeekend
}`,
      explanation: "Just like refactoring a God class into clean, focused classes — feature engineering transforms raw data into structured, meaningful model inputs."
    },
    trainerTip: "Give trainees a raw CSV and ask them to create 5 new columns using only domain knowledge. No ML needed — this teaches the most impactful ML skill."
  },
  "3-4": {
    id: "3-4",
    concept: "Image data is just a matrix of numbers (pixels). Your K-Means experience with image processing directly maps to computer vision preprocessing. Images need cleaning too — resizing, normalization, augmentation — just like data cleaning for structured data.",
    whyItMatters: "With your image processing background, you already understand the fundamentals. Computer vision is just adding ML on top of what you already know.",
    realWorldExample: {
      title: "Quality Control in Manufacturing — Image-Based Defect Detection",
      scenario: "A PCB manufacturer inspected 50,000 boards/day visually. Human inspectors missed 5-8% of defects due to fatigue. Each missed defect cost $200 in returns.",
      implementation: "Camera captures high-res images of each PCB. Preprocessing: resize to 224x224, normalize pixel values to 0-1, apply augmentation (rotation, brightness variation). CNN model classifies: defective/non-defective. Grad-CAM highlights WHICH region the model flagged.",
      outcome: "Defect detection rate: 99.2% (up from 92%). Inspection speed: 0.3 seconds per board (vs 15 seconds for humans). Your image processing knowledge makes this architecture intuitive."
    },
    caseStudy: {
      title: "From K-Means to CNN — Your Image Processing Journey",
      background: "You've used K-Means for image segmentation — clustering pixels by color similarity. Now extend that to classification.",
      challenge: "Transition from 'grouping pixels' (unsupervised) to 'understanding images' (supervised).",
      approach: [
        "K-Means on images: cluster pixels → segment foreground/background (you know this!)",
        "Feature extraction: histogram of colors, edge detection, texture patterns",
        "Traditional ML: Extract features manually → feed to SVM/Random Forest",
        "CNN approach: Let the neural network LEARN features automatically",
        "Transfer learning: Use pre-trained model (ResNet) — only train final layer"
      ],
      result: "Transfer learning is the shortcut — instead of training from scratch, use models pre-trained on millions of images. Like using a well-tested library instead of writing from scratch.",
      yourTask: "Take 50 images of two categories (e.g., good vs defective products, or any two types). (1) Apply K-Means to segment them. (2) Extract 3 visual features. (3) Design how a CNN would automate this feature extraction."
    },
    keyTakeaways: [
      "Images = matrices of pixel values (your linear algebra works here)",
      "Preprocessing: resize, normalize, augment — like data cleaning for tables",
      "K-Means → feature extraction → CNN — a natural progression",
      "Transfer learning = using pre-trained models — fast and effective"
    ],
    trainerTip: "Show pixel values of a small image (8x8) as a spreadsheet. When trainees see that an image IS just numbers, the fear of computer vision disappears."
  },

  // ============ MODULE 4: Hands-on ML ============
  "4-1": {
    id: "4-1",
    concept: "You already know K-Means from image processing. Now apply it to business data. Same algorithm, different data — instead of clustering pixels by color, you're clustering customers by behavior, products by similarity, or transactions by patterns.",
    whyItMatters: "K-Means is often the first unsupervised algorithm in production. It's simple, scalable, and explains well to stakeholders. Your existing knowledge is directly deployable.",
    realWorldExample: {
      title: "Reliance Retail — Customer Segmentation for Personalized Marketing",
      scenario: "Reliance Retail has 200M loyalty card members. Marketing was sending the same promotions to everyone — 2% conversion rate. They needed data-driven segmentation.",
      implementation: "Features: monthly_spend, visit_frequency, avg_basket_size, category_preference, recency. Applied K-Means with K=5 (determined by elbow method + business validation). Segments: Budget Regulars, Weekend Splurgers, Brand Hunters, Discount Chasers, Premium Loyalists. Each segment got tailored promotions.",
      outcome: "Conversion rate jumped from 2% to 8.5%. Marketing spend efficiency improved 300%. The K-Means model runs as a batch job every Sunday — simple, reliable, impactful."
    },
    caseStudy: {
      title: "Anomaly Detection in Server Logs Using K-Means",
      background: "Your application generates 5M log entries/day. Normal logs cluster together. Anomalous logs (attacks, failures) are outliers that don't fit any cluster.",
      challenge: "Detect unusual patterns without labeled 'attack' data — unsupervised anomaly detection.",
      approach: [
        "Feature extraction: requests_per_minute, unique_IPs, error_rate, payload_size, time_of_day",
        "Apply K-Means with K=3: normal_traffic, high_traffic, suspicious",
        "Points far from all cluster centers = anomalies",
        "Threshold: distance > 2 standard deviations from nearest centroid",
        "Alert system: flag anomalies for security team review"
      ],
      result: "Detected 3 DDoS attempts and 7 scanning attacks in the first month — all missed by rule-based monitoring. K-Means found patterns that static thresholds couldn't capture.",
      yourTask: "Apply K-Means to your application's data: (1) Choose 3-5 numerical features, (2) Determine optimal K using elbow method, (3) Interpret each cluster — what does it represent? (4) Identify outlier points — are they interesting?"
    },
    keyTakeaways: [
      "K-Means is versatile: customer segmentation, anomaly detection, image compression",
      "Elbow method + domain knowledge = right K value",
      "Outlier detection: points far from all centroids are anomalies",
      "Production deployment: batch processing, weekly retraining, simple API"
    ],
    trainerTip: "Live demo: cluster the audience by age and experience using K-Means on a whiteboard. Physical demonstration > PowerPoint every time."
  },
  "4-2": {
    id: "4-2",
    concept: "Image classification = teaching a computer to say 'this image is a cat' or 'this PCB is defective'. You go from K-Means (grouping pixels) to CNNs (understanding content). Think of it as evolving from XPath locators (pattern matching) to visual AI testing (understanding UI semantics).",
    whyItMatters: "Computer vision is one of the most deployable AI skills. From quality control to medical imaging to document processing — the demand is massive.",
    realWorldExample: {
      title: "Tata Steel — Weld Defect Classification",
      scenario: "Tata Steel's quality team visually inspected X-ray images of welds. 200 images/day, 6 defect types. Human accuracy: 85%. Needed: automated, consistent classification.",
      implementation: "Dataset: 10,000 labeled X-ray images (6 categories + 'good'). Model: ResNet-50 (pre-trained on ImageNet, fine-tuned on weld images). Training: Only retrained the last 3 layers — transfer learning. Deployment: REST API endpoint that accepts image → returns defect type + confidence score.",
      outcome: "Model accuracy: 94%. Processing time: 0.5 seconds vs 3 minutes per image. Deployed as a Spring Boot microservice — familiar architecture for the Java team."
    },
    caseStudy: {
      title: "Building an Image Classifier — Step by Step",
      background: "You want to classify product images as 'authentic' or 'counterfeit' for an e-commerce platform.",
      challenge: "Build a production-ready image classification pipeline using your engineering skills.",
      approach: [
        "Data collection: 5,000 authentic + 5,000 counterfeit product images",
        "Preprocessing: resize to 224x224, normalize, augment (flip, rotate, brightness)",
        "Model selection: ResNet-50 with transfer learning (don't train from scratch!)",
        "Training: Freeze base layers, train classifier head, use learning rate scheduling",
        "Evaluation: confusion matrix, precision per class, recall per class",
        "Deployment: Export to ONNX → serve via Spring Boot with DJL (Deep Java Library)"
      ],
      result: "Transfer learning achieves 90%+ accuracy with just 10K images. Without it, you'd need 100K+ images and weeks of training.",
      yourTask: "Design (don't build yet) an image classification system for YOUR domain: (1) What would you classify? (2) How would you collect training data? (3) What pre-trained model would you use? (4) How would you serve it in production?"
    },
    keyTakeaways: [
      "Transfer learning = use pre-trained models, retrain only the final layers",
      "You don't need millions of images — 1,000-10,000 per class often suffices",
      "CNNs automatically learn features that you'd manually extract in traditional image processing",
      "Deep Java Library (DJL) lets you deploy CV models in Java — your comfort zone"
    ],
    trainerTip: "Show trainees what CNN layers 'see' — edge detection in early layers, textures in middle, objects in deep layers. It's like watching a baby learn to see."
  },
  "4-3": {
    id: "4-3",
    concept: "Model lifecycle = the continuous process of training, deploying, monitoring, and retraining models. Like your application lifecycle: develop → test → deploy → monitor → hotfix → redeploy. Models aren't 'done' when deployed — they degrade over time (concept drift).",
    whyItMatters: "Most ML tutorials end at 'model.fit()'. Production ML requires lifecycle management — this is where your engineering skills become essential.",
    realWorldExample: {
      title: "Zomato's Restaurant Rating Prediction — Lifecycle in Action",
      scenario: "Zomato predicts restaurant ratings for new restaurants. Model trained in 2022 on pre-COVID dining patterns. By 2023, prediction accuracy dropped 15% because dining habits changed (more delivery, different cuisine preferences).",
      implementation: "Implemented MLOps pipeline: Daily data ingestion → Weekly accuracy monitoring → Monthly retraining trigger (if accuracy < 80%) → A/B testing new model vs old → Gradual rollout (10% → 50% → 100%). Versioning: MLflow tracks every model version with its training data snapshot.",
      outcome: "Automated retraining restored accuracy within 2 weeks of detecting drift. Without lifecycle management, the model would have silently degraded for months."
    },
    caseStudy: {
      title: "MLOps for a Credit Scoring Model",
      background: "A bank deployed a credit scoring model 18 months ago. Initial accuracy: 88%. Current accuracy: 71%. Nobody noticed because there was no monitoring.",
      challenge: "Design an MLOps pipeline that prevents this silent degradation.",
      approach: [
        "Version control: DVC for data versioning, MLflow for model versioning",
        "Monitoring: track accuracy, precision, recall weekly against a holdout set",
        "Drift detection: compare incoming data distribution to training data distribution",
        "Automated alerts: Slack notification when accuracy drops below threshold",
        "Retraining pipeline: Jenkins job triggered by alert → retrain → validate → deploy",
        "Rollback: keep last 3 model versions, auto-rollback if new model performs worse"
      ],
      result: "MLOps pipeline catches degradation within 1 week instead of 18 months. Retraining is automated and tested — like CI/CD for models.",
      yourTask: "Design an MLOps pipeline for any ML model: (1) What metrics would you monitor? (2) What triggers retraining? (3) How do you test a new model before deploying? (4) How do you rollback?"
    },
    keyTakeaways: [
      "Models degrade over time — concept drift is inevitable",
      "MLOps = DevOps for ML: version, monitor, retrain, deploy",
      "Your CI/CD experience directly applies to ML pipelines",
      "Key tools: MLflow (versioning), DVC (data versioning), Airflow (orchestration)"
    ],
    trainerTip: "Ask: 'Who has a CI/CD pipeline?' (everyone raises hand). 'Who has a model retraining pipeline?' (silence). That's the gap you're filling."
  },
  "4-4": {
    id: "4-4",
    concept: "Evaluation metrics tell you HOW your model is failing. Accuracy alone is misleading — like saying 'all tests passed' when you only have 5 tests. Precision = 'when the model says YES, how often is it right?' Recall = 'of all actual YES cases, how many did the model find?'",
    whyItMatters: "In interviews, 'which metric did you choose and why?' is a guaranteed question. In production, choosing the wrong metric can be catastrophic.",
    realWorldExample: {
      title: "Medical Diagnosis — Where Wrong Metric = Wrong Priority",
      scenario: "Cancer screening model: 99% accuracy sounds great. But if only 1% of patients have cancer, a model that always says 'no cancer' gets 99% accuracy while missing EVERY cancer case. Accuracy is useless here.",
      implementation: "Correct metric: Recall (sensitivity) — 'of all cancer patients, how many did we catch?' Target: 99%+ recall even if precision drops to 70% (some false alarms are OK, missing cancer is NOT). Trade-off managed via threshold tuning on the ROC curve.",
      outcome: "By optimizing recall over accuracy, the model caught 99.5% of actual cancer cases. The 30% false positive rate led to extra tests but saved lives."
    },
    caseStudy: {
      title: "Choosing Metrics for a Fraud Detection System",
      background: "A payment processor handles 1M transactions/day. 0.1% are fraudulent (1,000 fraud transactions among 999,000 legitimate ones).",
      challenge: "Which metric matters? Accuracy? Precision? Recall? It depends on the business cost.",
      approach: [
        "Accuracy = 99.9% if you never flag fraud → useless metric",
        "Precision = of flagged transactions, how many are actually fraud? (high precision = fewer false alarms = less customer friction)",
        "Recall = of actual fraud, how many did we catch? (high recall = catch more fraud = less financial loss)",
        "Business cost: each missed fraud = $500 average loss, each false alarm = $2 customer support cost",
        "Optimize for: recall first (catch fraud), then improve precision (reduce false alarms)"
      ],
      result: "F1 score balanced both concerns. At 95% recall and 80% precision: caught 950 of 1,000 frauds (saving $475K) with 237 false alarms (costing $474). Net benefit: $474,526/day.",
      yourTask: "For your ML project: (1) What's the cost of a false positive? (2) What's the cost of a false negative? (3) Which metric should you optimize? (4) What's the acceptable trade-off?"
    },
    keyTakeaways: [
      "Accuracy is misleading for imbalanced datasets (most real-world problems)",
      "Precision = quality of positive predictions (minimize false alarms)",
      "Recall = completeness of positive predictions (minimize missed cases)",
      "F1 = harmonic mean of precision and recall (balanced metric)",
      "Always choose metrics based on BUSINESS COST, not mathematical convenience"
    ],
    codeSnippet: {
      language: "java",
      code: `// Testing analogy for ML metrics:

// ACCURACY = "% tests passed" (misleading if tests are trivial)
// PRECISION = "When a test fails, is it a real bug?" (low = too many false alarms)
// RECALL = "Did tests catch all the bugs?" (low = bugs in production)
// F1 = Balance between precision and recall

// In your automation testing world:
// High Precision, Low Recall = "Tests that fail are real bugs, but many bugs slip through"
// Low Precision, High Recall = "Catches every bug, but also flags non-bugs (flaky tests)"
// Goal: Maximize both = reliable test suite that catches real bugs consistently`,
      explanation: "Think of precision as 'test reliability' (no false failures) and recall as 'test coverage' (no missed bugs). You optimize both in testing — do the same for ML models."
    },
    trainerTip: "Use the spam filter analogy: Precision = 'legit emails in spam folder?' (annoying). Recall = 'spam in inbox?' (dangerous). Which is worse? Context determines the answer."
  },

  // ============ MODULE 5: Generative AI & LLMs ============
  "5-1": {
    id: "5-1",
    concept: "LLMs are giant pattern-matching machines trained on the entire internet. They don't 'understand' language — they predict the next word based on statistical patterns from trillions of training tokens. Think of it like autocomplete on steroids — but instead of completing one word, it completes entire paragraphs.",
    whyItMatters: "Understanding HOW LLMs work helps you use them better, debug issues, and explain capabilities/limitations to stakeholders and trainees.",
    realWorldExample: {
      title: "How ChatGPT Generates Code",
      scenario: "When you ask ChatGPT to write a Java method, it's not 'thinking about Java'. It's predicting: given the pattern 'write a Java method that sorts...', what tokens most likely follow? It learned these patterns from millions of GitHub repos, Stack Overflow answers, and documentation.",
      implementation: "Architecture simplified: Input text → tokenize into numbers → pass through 96 layers of 'attention' → each layer asks 'which previous words matter for predicting the next word?' → output probability distribution over 100K possible next tokens → pick the most likely one → repeat.",
      outcome: "It works because language patterns are statistically predictable. It fails when: the pattern is rare in training data, the task requires actual reasoning, or the context exceeds the model's 'memory' (context window)."
    },
    caseStudy: {
      title: "Understanding Transformer Architecture Through Java Analogy",
      background: "The Transformer architecture (2017) revolutionized AI. It's the 'T' in GPT.",
      challenge: "Understand attention mechanism — the core innovation — without heavy math.",
      approach: [
        "Self-attention = like a HashMap lookup: for each word, find which other words are 'relevant'",
        "Query-Key-Value = like a search: Query='what am I looking for?', Key='what do I contain?', Value='what info do I provide?'",
        "Multiple heads = like running multiple regex patterns in parallel — each captures different relationships",
        "Layer stacking = like middleware chain in Spring — each layer adds understanding",
        "Position encoding = like array indices — tells the model word ORDER matters"
      ],
      result: "The Transformer is essentially a very sophisticated pattern-matching pipeline. Your experience with request processing pipelines in Java gives you the mental model.",
      yourTask: "Explain in your own words: (1) Why does attention mechanism work better than processing words one-by-one? (2) What's the 'context window' and why does it limit LLMs? (3) Why can LLMs generate creative text if they're 'just predicting next words'?"
    },
    keyTakeaways: [
      "LLMs predict next tokens, they don't 'understand' — but the results are remarkably useful",
      "Attention mechanism = dynamic relevance scoring between all words",
      "More parameters = more patterns learned = better predictions (but more expensive)",
      "Context window = maximum 'working memory' — a key limitation to design around"
    ],
    trainerTip: "Don't start with math. Start with: 'Complete this sentence: The cat sat on the ___'. That's what LLMs do, just at massive scale with incredible context awareness."
  },
  "5-2": {
    id: "5-2",
    concept: "Prompt engineering = designing inputs to get reliable outputs from LLMs. It's NOT just 'asking nicely'. It's a systematic engineering discipline — like writing good test specifications. The quality of your prompt determines the quality of the output, just like the quality of your test case determines the quality of your testing.",
    whyItMatters: "This is the highest-ROI AI skill right now. No model training needed, no infrastructure — just better prompts = better results immediately.",
    realWorldExample: {
      title: "Enterprise Document Summarization — Prompt Evolution",
      scenario: "A consulting firm needed to summarize 50-page client reports into 1-page executive briefs. Three prompt iterations showed dramatic improvement.",
      implementation: "V1 (bad): 'Summarize this document' → Got generic, missed key insights. V2 (better): 'Summarize this consulting report for a C-suite executive. Focus on: financial impact, risks, and recommended actions. Use bullet points. Max 300 words.' → Much better structure. V3 (best): 'You are a senior management consultant at McKinsey. Given this client report, write an executive brief that a CEO would read before a board meeting. Structure: 1) Key Findings (3 bullets), 2) Financial Impact (with numbers), 3) Top 3 Risks, 4) Recommended Actions with timeline. Tone: confident, data-driven. Max 300 words.' → Production-quality output.",
      outcome: "V3 reduced human editing time from 2 hours to 15 minutes per report. The prompt became a reusable 'template' across all client engagements."
    },
    caseStudy: {
      title: "Prompt Engineering for Test Case Generation",
      background: "You want an LLM to generate Selenium test cases from user stories. Naive prompting gives generic, non-runnable code.",
      challenge: "Engineer prompts that generate production-quality, runnable test cases.",
      approach: [
        "Role: 'You are a senior SDET with 10 years of Selenium experience'",
        "Context: Provide page object model structure, naming conventions, project framework",
        "Few-shot: Include 2-3 examples of YOUR test style as references",
        "Constraints: 'Use Page Object Model. Use explicit waits only. Follow our naming: test_[Feature]_[Scenario]_[ExpectedResult]'",
        "Output format: 'Return: test method, page objects needed, test data required'",
        "Chain of thought: 'First identify test scenarios, then for each scenario identify steps, then write code'"
      ],
      result: "With engineered prompts, 70% of generated test cases were usable with minor edits (vs 10% with naive prompts). Saved 3 hours/sprint on test creation.",
      yourTask: "Write 3 versions of a prompt for YOUR use case — bad, better, best. Test each version with the same input. Measure: (1) Output quality (1-5), (2) Consistency across 5 runs, (3) Time to usable result."
    },
    keyTakeaways: [
      "Prompt engineering follows patterns: Role + Context + Task + Format + Constraints",
      "Few-shot examples are the most powerful technique — show, don't just tell",
      "Chain-of-thought prompting improves reasoning tasks significantly",
      "Treat prompts as code — version control them, test them, iterate",
      "Temperature: 0 for deterministic tasks, 0.7 for creative tasks"
    ],
    trainerTip: "Live workshop: give everyone the SAME task but let them write their own prompts. Compare outputs. The variation teaches prompt engineering better than any lecture."
  },
  "5-3": {
    id: "5-3",
    concept: "Embeddings convert text/images into numerical vectors that capture MEANING. 'King' and 'Queen' are close in embedding space because they share semantic properties. It's like converting Java objects to hashCodes — but instead of uniqueness, embeddings capture similarity. Vector databases store and search these embeddings efficiently.",
    whyItMatters: "Embeddings power semantic search, recommendations, and RAG. Understanding them is essential for building any AI application that needs to 'find similar things'.",
    realWorldExample: {
      title: "Jio's Semantic Product Search",
      scenario: "JioMart's keyword search: 'cold drink' returned results only for items with those exact words. Missed: 'soft drink', 'soda', 'carbonated beverage', 'Coca-Cola'. Users got frustrated and left.",
      implementation: "Solution: Embed all product descriptions into vectors using a sentence transformer model. When user searches 'cold drink', embed the query → find nearest vectors in the product embedding space. 'Coca-Cola', 'Pepsi', 'Mountain Dew' are all close to 'cold drink' in embedding space even though they don't contain those words.",
      outcome: "Search-to-purchase conversion increased 35%. Zero results dropped from 12% to 2%. The embedding model understood meaning, not just keywords."
    },
    caseStudy: {
      title: "Building a Similar Test Case Finder",
      background: "Your test suite has 5,000 test cases. When writing a new test, you want to find similar existing tests to avoid duplication and reuse patterns.",
      challenge: "Keyword search fails: 'login test' doesn't find 'authentication validation' or 'sign-in verification' even though they're semantically similar.",
      approach: [
        "Embed all test case names + descriptions using a pre-trained model",
        "Store embeddings in a vector database (Pinecone, Weaviate, or pgvector)",
        "When writing new test: embed the description → find top 5 nearest neighbors",
        "Add metadata filtering: same module, same test type, same page",
        "Similarity threshold: cosine similarity > 0.85 = 'very similar'",
        "Alert: 'This test might duplicate TestCase#3421 (91% similar)'"
      ],
      result: "Reduced duplicate test cases by 20%. New testers found relevant existing tests instantly. Onboarding time for new SDETs reduced by 30%.",
      yourTask: "Choose a text dataset you have access to (test cases, code comments, documentation). (1) What would you embed? (2) What search queries would you run? (3) How would you measure if the results are semantically relevant?"
    },
    keyTakeaways: [
      "Embeddings = numerical representations that capture meaning/similarity",
      "Similar items are close in vector space (cosine similarity)",
      "Vector databases = specialized storage for embedding search",
      "Use cases: semantic search, deduplication, recommendations, RAG"
    ],
    trainerTip: "Show a 2D plot of word embeddings: king-queen, man-woman form parallel vectors. When trainees see that 'king - man + woman ≈ queen', minds are blown."
  },
  "5-4": {
    id: "5-4",
    concept: "RAG (Retrieval-Augmented Generation) = give the LLM access to YOUR data without retraining it. Like dependency injection for AI — instead of hardcoding knowledge into the model, you inject relevant context at runtime. This is THE enterprise AI pattern for 2024-2025.",
    whyItMatters: "RAG solves the biggest enterprise AI problem: 'How do I use ChatGPT with MY company's private data?' Without RAG, LLMs only know public information up to their training cutoff.",
    realWorldExample: {
      title: "Cognizant — Internal Knowledge Base Chatbot",
      scenario: "Cognizant has 300,000 employees. HR policies, project documentation, and technical guides spread across 50+ systems. Employees spent 45 min/day searching for information.",
      implementation: "RAG pipeline: 1) Chunked 100K documents into 2M segments. 2) Embedded each chunk into vectors (stored in Pinecone). 3) When employee asks a question: embed the question → find top 5 relevant chunks → inject them into the LLM prompt as context → LLM generates answer citing sources.",
      outcome: "Information search time dropped from 45 to 5 min/day. 300K employees × 40 min saved × 250 workdays = massive productivity gain. Cost: $50K/month (API + infrastructure). ROI: 100x."
    },
    caseStudy: {
      title: "Building a RAG System for Your Test Documentation",
      background: "Your QA team has 500 pages of test strategy documents, bug reports, and automation framework guides. New joiners take 3 months to find relevant information.",
      challenge: "Build a chatbot that answers questions about YOUR specific testing documentation.",
      approach: [
        "Document ingestion: Parse all QA docs → chunk into 500-token segments",
        "Embedding: Convert each chunk to vector using sentence-transformers",
        "Storage: Store in pgvector (PostgreSQL extension — familiar territory!)",
        "Retrieval: On question → embed query → find top 5 similar chunks → rank by relevance",
        "Generation: Send to LLM with prompt: 'Answer based on these documents. Cite sources.'",
        "Guardrails: If no relevant chunks found, say 'I don't have information about this'"
      ],
      result: "New joiner onboarding reduced from 3 months to 6 weeks. 80% of routine questions answered without senior team involvement.",
      yourTask: "Design a RAG system for your domain: (1) What documents would you index? (2) What's the ideal chunk size? (3) How many chunks would you retrieve per query? (4) What guardrails would you add?"
    },
    keyTakeaways: [
      "RAG = Retrieve relevant context + Augment the prompt + Generate answer",
      "Solves: hallucination (model answers from YOUR data), freshness, privacy",
      "Architecture: Document store → Embeddings → Vector DB → Retrieval → LLM → Response",
      "This is the #1 pattern for enterprise AI applications right now"
    ],
    codeSnippet: {
      language: "java",
      code: `// RAG = Dependency Injection for AI

// WITHOUT RAG (like hardcoded dependencies):
String answer = llm.ask("What is our refund policy?");
// LLM: "I don't know your company's refund policy" ❌

// WITH RAG (like injected dependencies):
List<String> relevantDocs = vectorDB.search("refund policy", topK=5);
String context = String.join("\\n", relevantDocs);
String prompt = """
  Based on these company documents:
  %s
  
  Answer the question: What is our refund policy?
  Only use information from the documents above.
  """.formatted(context);
String answer = llm.ask(prompt);
// LLM: "Per our policy document v3.2, refunds are..." ✅`,
      explanation: "RAG 'injects' relevant knowledge into the LLM at query time, just like Spring injects dependencies at runtime. The LLM doesn't need to be retrained — it just needs the right context."
    },
    trainerTip: "Demo RAG live: upload a PDF → ask questions about it → show cited sources. The 'wow factor' of an AI answering questions about a document it just saw is unbeatable."
  },
  "5-5": {
    id: "5-5",
    concept: "AI hallucinations = the model confidently generates incorrect information. Like a confident but unreliable witness. Flaky tests = tests that pass/fail inconsistently without code changes. Same root cause: non-deterministic systems producing unreliable outputs. Same solution philosophy: detection, isolation, and guardrails.",
    whyItMatters: "Both are unsolved problems. Your experience handling flaky tests gives you frameworks for handling AI hallucinations — a critical enterprise concern.",
    realWorldExample: {
      title: "Legal AI Disaster — Hallucinated Case Citations",
      scenario: "A US law firm used ChatGPT to research case law. The AI cited 6 court cases that sounded real but never existed. The lawyer submitted these to court. Judge discovered the fabrication — lawyer was sanctioned.",
      implementation: "Fix implemented: RAG pipeline with verified case law database. Every citation cross-referenced against the actual legal database. If citation not found → flagged for human review. Confidence scoring: answers backed by 3+ real sources = high confidence, otherwise = needs review.",
      outcome: "Zero hallucinated citations after implementing verification. Like adding assertion verification to your test results instead of trusting raw output."
    },
    caseStudy: {
      title: "Handling Non-Determinism — AI Hallucinations vs Flaky Tests",
      background: "Your test suite has 5% flaky tests. Your AI chatbot has 10% hallucination rate. Both undermine trust. Both need systematic solutions.",
      challenge: "Apply your flaky test mitigation strategies to AI hallucination management.",
      approach: [
        "Flaky test: retry logic → AI: regenerate with higher temperature, compare outputs",
        "Flaky test: quarantine flaky tests → AI: flag low-confidence answers for human review",
        "Flaky test: root cause analysis → AI: analyze which topics cause hallucinations",
        "Flaky test: stable environment → AI: RAG (ground in verified data)",
        "Flaky test: deterministic assertions → AI: fact-checking against knowledge base",
        "Flaky test: monitoring dashboard → AI: hallucination rate tracking over time"
      ],
      result: "The mental model is identical: detect unreliable outputs → isolate them → add guardrails → monitor. Your testing discipline is your AI quality advantage.",
      yourTask: "Design a 'hallucination management framework': (1) How do you detect hallucinations? (2) How do you isolate unreliable responses? (3) What guardrails prevent hallucinations from reaching users? (4) How do you measure improvement?"
    },
    keyTakeaways: [
      "Hallucinations and flaky tests share root cause: non-deterministic systems",
      "Detection: cross-reference outputs against verified sources",
      "Prevention: RAG, constrained outputs, confidence scoring",
      "Mitigation: human-in-the-loop, retry with comparison, graceful fallbacks",
      "Your QA mindset is exactly what AI systems need"
    ],
    trainerTip: "Ask: 'How do you handle flaky tests?' Then say: 'Now apply the same thinking to AI.' The parallel clicks instantly for testers."
  },

  // ============ MODULE 6: AI in Java Full Stack ============
  "6-1": {
    id: "6-1",
    concept: "AI system architecture adds 'intelligence layers' to your existing enterprise stack. Think: your Java microservices stay the same, but now some services call AI APIs for predictions, classifications, or generation instead of hardcoded business rules.",
    whyItMatters: "You don't need to rebuild your architecture for AI. You need to know WHERE to plug AI in and HOW to handle its unique characteristics (latency, non-determinism, cost).",
    realWorldExample: {
      title: "Razorpay — AI Layer in Payment Architecture",
      scenario: "Razorpay's existing architecture: API Gateway → Payment Service → Bank Integration. They added an AI fraud detection layer WITHOUT changing the core flow.",
      implementation: "Architecture: Transaction comes in → Payment Service calls Fraud Detection Service (async) → Fraud Service runs ML model → Returns risk score in <100ms → Payment Service checks: score > 0.8 → block, 0.5-0.8 → additional verification, <0.5 → proceed. The AI service is just another microservice with standard REST APIs.",
      outcome: "Fraud reduced by 60%. Architecture change was minimal — just adding one more service call in the existing pipeline. Key: AI as a SERVICE, not a replacement."
    },
    caseStudy: {
      title: "Designing an AI-Powered Enterprise Application",
      background: "An HR platform wants to add: AI resume screening, chatbot for employee queries, and sentiment analysis of feedback forms.",
      challenge: "Architect the system so AI components are modular, scalable, and replaceable.",
      approach: [
        "Keep existing HR services unchanged (SOLID principle: Open for extension)",
        "AI Gateway Service: routes AI requests to appropriate model services",
        "Resume Screening Service: REST API → takes PDF → returns structured evaluation",
        "Chatbot Service: WebSocket → RAG-based Q&A → streams response",
        "Sentiment Service: REST API → takes text → returns sentiment + confidence",
        "Circuit breaker: if AI service is down, fallback to manual process",
        "Cost tracking: meter every AI API call for budget management"
      ],
      result: "Each AI capability is an independent microservice. Can swap models without touching the core HR system. Familiar patterns: API Gateway, Circuit Breaker, Service Registry.",
      yourTask: "Take your current application's architecture diagram. Identify 3 points where AI could add value. For each: (1) What type of AI? (2) How does it integrate? (3) What's the fallback if AI is unavailable?"
    },
    keyTakeaways: [
      "AI fits INTO existing architecture — it doesn't replace it",
      "AI components should be microservices: independent, scalable, replaceable",
      "Key concerns: latency (AI is slow), cost (per-call pricing), non-determinism",
      "Design patterns: Circuit Breaker, Bulkhead, Fallback — all apply to AI services"
    ],
    trainerTip: "Draw the existing architecture. Then add AI services in RED. Show that 90% of the system stays the same — this removes the 'we need to rebuild everything' fear."
  },
  "6-2": {
    id: "6-2",
    concept: "Integrating AI APIs into Spring Boot is like integrating any external API — RestTemplate/WebClient, DTOs, error handling. The difference: AI responses are non-deterministic, may be slow (2-10 seconds), and cost money per call. Your API integration skills transfer directly.",
    whyItMatters: "This is where your Java skills become your AI superpower. Most AI engineers can train models but can't build production-grade APIs around them. You can.",
    realWorldExample: {
      title: "Building an AI-Powered Code Review Service",
      scenario: "A development team built a Spring Boot service that reviews pull requests using OpenAI's API. On every PR webhook, it analyzes the code diff and posts review comments.",
      implementation: "Spring Boot app → receives GitHub webhook → extracts code diff → sends to OpenAI with prompt template → parses structured response → posts comments back to GitHub. Retry logic (exponential backoff for rate limits), response caching (same code pattern → cached review), cost tracking (log token usage per request).",
      outcome: "Automated reviews caught 30% of issues before human reviewers. Average review time reduced by 45 minutes. Monthly API cost: $200 for a 20-developer team."
    },
    caseStudy: {
      title: "Spring Boot + OpenAI — Production-Ready Integration",
      background: "You're building a customer support API that uses AI to draft responses to customer emails.",
      challenge: "Build it production-grade: handle failures, manage costs, ensure response quality.",
      approach: [
        "Service layer: CustomerSupportAIService with clean interface (not coupled to OpenAI)",
        "Prompt management: templates stored in configuration, not hardcoded",
        "Error handling: timeout (10s), rate limit (retry with backoff), invalid response (fallback template)",
        "Caching: cache responses for similar queries (embedding similarity > 0.95)",
        "Cost management: track tokens per request, daily budget cap, alert at 80%",
        "Quality: confidence scoring, human review for low-confidence responses"
      ],
      result: "Production-ready AI integration using patterns you already know. The AI-specific additions: prompt management, token tracking, confidence scoring.",
      yourTask: "Build a simple Spring Boot service that calls an AI API: (1) Clean service abstraction (interface-based), (2) Retry logic with exponential backoff, (3) Response validation, (4) Token/cost tracking, (5) Graceful fallback."
    },
    keyTakeaways: [
      "AI API integration = REST API integration + AI-specific concerns",
      "Always add: retry logic, timeout handling, cost tracking, response validation",
      "Abstract the AI provider behind an interface — switch from OpenAI to Gemini easily",
      "Cache aggressively — same/similar inputs often produce same outputs"
    ],
    codeSnippet: {
      language: "java",
      code: `@Service
public class AIService {
    
    @Value("\${ai.api.key}")
    private String apiKey;
    
    @Retryable(maxAttempts = 3, backoff = @Backoff(delay = 1000, multiplier = 2))
    public AIResponse analyze(String input) {
        // Familiar Spring patterns!
        var request = new AIRequest(input, "gpt-4", 0.3);
        
        try {
            var response = webClient.post()
                .uri("/v1/chat/completions")
                .header("Authorization", "Bearer " + apiKey)
                .bodyValue(request)
                .retrieve()
                .bodyToMono(AIResponse.class)
                .timeout(Duration.ofSeconds(10))
                .block();
            
            trackUsage(response.getTokensUsed()); // Cost tracking
            return response;
        } catch (WebClientResponseException.TooManyRequests e) {
            throw new RetryableException("Rate limited"); // Triggers retry
        }
    }
}`,
      explanation: "This is standard Spring Boot code with AI-specific additions: retry for rate limits, timeout for slow AI responses, and token usage tracking. Your Spring expertise makes this natural."
    },
    trainerTip: "Code this live in class using Spring Initializr. When Java developers see familiar @Service, @Retryable, WebClient patterns, AI integration stops being scary."
  },
  "6-3": {
    id: "6-3",
    concept: "Microservices for AI follows the same decomposition principles as regular microservices — but with AI-specific considerations: GPU resource allocation, model versioning, A/B testing, and data pipeline orchestration.",
    whyItMatters: "Enterprise AI systems are complex — multiple models, data pipelines, and inference services. Microservices patterns you already know keep this manageable.",
    realWorldExample: {
      title: "Flipkart's AI Microservices Ecosystem",
      scenario: "Flipkart's product page uses 5+ AI microservices simultaneously: recommendation engine, price optimization, review summarization, visual search, and chatbot.",
      implementation: "Each AI capability is a separate service: Recommendation Service (collaborative filtering model), Price Service (regression model), Review Service (LLM summarization), Visual Search (CNN model), Chatbot (RAG-based LLM). All communicate via Kafka events and REST APIs. Each has independent scaling — visual search needs GPUs, recommendations run on CPU.",
      outcome: "Independent deployment, scaling, and model updates. When they upgraded the recommendation model, zero impact on other services. Standard microservice patterns applied to AI."
    },
    caseStudy: {
      title: "Decomposing a Monolithic AI Application",
      background: "A startup built an 'AI-powered recruitment platform' as a monolith — resume parsing, job matching, interview scheduling, and candidate scoring all in one service.",
      challenge: "It's slow, hard to update, and any model update requires full redeployment.",
      approach: [
        "Resume Parser Service: NLP model → structured JSON (scales independently)",
        "Job Matcher Service: embedding similarity model (GPU-intensive, auto-scale)",
        "Interview Scheduler Service: rule-based optimization (no AI needed!)",
        "Candidate Scorer Service: gradient boosting model (lightweight, CPU)",
        "API Gateway: routes requests, handles auth, rate limiting",
        "Event Bus (Kafka): async communication between services",
        "Model Registry: version and deploy models independently"
      ],
      result: "Deployment frequency: monthly → daily per service. Model updates: no downtime, A/B tested. Resource optimization: 40% cost reduction by right-sizing each service.",
      yourTask: "Take an AI-heavy application idea. (1) Identify 4-5 services, (2) Classify each as AI/non-AI, (3) Define communication patterns (sync/async), (4) Identify which services need GPUs vs CPUs."
    },
    keyTakeaways: [
      "Same microservice principles apply: single responsibility, loose coupling",
      "AI-specific: GPU vs CPU resource planning, model versioning, A/B testing",
      "Not everything needs AI — keep rule-based services simple",
      "Event-driven architecture works well for AI pipelines (async processing)"
    ],
    trainerTip: "Use the restaurant analogy: kitchen (ML training), counter (API serving), delivery (edge deployment). Each is a 'service' that scales independently."
  },
  "6-4": {
    id: "6-4",
    concept: "AI systems are non-deterministic — same input can give different outputs. This changes EVERYTHING about error handling. You can't use assertEquals() on AI output. Instead, you need range checks, semantic validation, confidence scoring, and graceful degradation.",
    whyItMatters: "This is where most AI projects fail in production. Engineers apply deterministic error handling patterns to non-deterministic systems. Your testing discipline helps you design better error handling.",
    realWorldExample: {
      title: "AI Customer Service Bot — Handling the Unexpected",
      scenario: "A telecom company's AI bot occasionally generated responses like: recommending competitor products, making up discount codes, or providing technically accurate but legally problematic advice.",
      implementation: "Guardrails implemented: 1) Output validation: regex check for competitor names, discount code format validation. 2) Confidence threshold: <70% confidence → route to human. 3) Topic boundary: classify response topic → reject if outside allowed topics. 4) PII detection: scan response for personal data leakage. 5) Rate limiting: max 3 AI failures → circuit breaker → full human handoff.",
      outcome: "Incident rate dropped from 5% to 0.3%. The guardrails caught issues that the model itself couldn't prevent."
    },
    caseStudy: {
      title: "Building Resilient AI Error Handling in Java",
      background: "Your Spring Boot application calls an LLM to generate product descriptions. In production, you see: timeouts, garbled responses, off-topic content, and cost spikes.",
      challenge: "Design an error handling strategy that keeps the system reliable despite AI unpredictability.",
      approach: [
        "Timeout: 10-second hard limit, fallback to template-based description",
        "Garbled output: JSON schema validation on response, retry once if invalid",
        "Off-topic: keyword/topic classifier on output, reject if confidence < 0.8",
        "Cost spike: per-request token limit, daily budget cap, alert at 80%",
        "Circuit breaker: 5 failures in 1 minute → open circuit → use cached/template responses",
        "Logging: log full prompt + response for every request → debugging + fine-tuning"
      ],
      result: "System availability: 99.9% despite AI provider issues. Graceful degradation: users always get a response, just not always AI-generated.",
      yourTask: "For an AI feature you're building: (1) List 5 things that could go wrong, (2) Design a fallback for each, (3) Define metrics to monitor health, (4) Create an incident response playbook."
    },
    keyTakeaways: [
      "AI errors are different: not just 500 errors, but semantically wrong 200 responses",
      "Validate outputs, not just status codes — content validation is crucial",
      "Circuit breaker pattern is essential for AI service calls",
      "Always have a non-AI fallback — template, cache, or human handoff",
      "Log everything — prompt + response pairs are your debugging goldmine"
    ],
    codeSnippet: {
      language: "java",
      code: `// AI-specific error handling in Spring Boot:

@Service
public class AIProductService {
    
    @CircuitBreaker(name = "aiService", fallbackMethod = "fallbackDescription")
    public ProductDescription generate(Product product) {
        var aiResponse = aiClient.call(buildPrompt(product));
        
        // AI-specific validation (NOT just null check!)
        if (aiResponse.getConfidence() < 0.7) {
            log.warn("Low confidence AI response for product: {}", product.getId());
            return fallbackDescription(product, new LowConfidenceException());
        }
        
        if (containsCompetitorMention(aiResponse.getText())) {
            log.error("AI mentioned competitor for product: {}", product.getId());
            return fallbackDescription(product, new ContentViolationException());
        }
        
        return aiResponse;
    }
    
    // Graceful fallback - always have a plan B
    private ProductDescription fallbackDescription(Product product, Throwable t) {
        return templateEngine.generate(product); // Rule-based fallback
    }
}`,
      explanation: "Notice the AI-specific validations: confidence scoring, content checking, competitor mention detection. This goes beyond standard HTTP error handling — you're validating MEANING, not just status."
    },
    trainerTip: "Ask developers: 'What does a 200 OK response mean?' (Success). 'What if the response is 200 OK but contains wrong information?' That's the AI error handling challenge."
  },

  // ============ MODULE 7: AI for Test Automation ============
  "7-1": {
    id: "7-1",
    concept: "LLMs can generate test cases from requirements, user stories, or even production code. But naive prompting gives generic tests. The key is structured prompting with your testing expertise — providing context about the test framework, conventions, and quality standards.",
    whyItMatters: "AI-generated test cases save 30-50% of test creation time when done right. Combined with your testing expertise for validation, it's a force multiplier.",
    realWorldExample: {
      title: "Accenture — AI-Powered Test Generation at Scale",
      scenario: "Accenture's QA team for a banking client generated 2,000+ test cases from 500 user stories. Manual creation: 4 weeks. AI-assisted: 1 week (including review and refinement).",
      implementation: "Pipeline: User story → structured prompt with test framework context → LLM generates test scenarios → automated validation (does it compile? does it follow naming conventions?) → human review for business logic → final test suite. Template: 'Given [user story], generate test cases covering: happy path, edge cases, negative tests, boundary values. Framework: TestNG, Page Object Model. Format: test method name, steps, expected results, test data.'",
      outcome: "3x faster test creation. 85% of AI-generated tests passed human review without changes. The remaining 15% needed minor business logic corrections."
    },
    caseStudy: {
      title: "Building an AI Test Case Generator for Your Project",
      background: "You want to integrate AI test generation into your sprint workflow. Currently, writing test cases from user stories takes 40% of QA sprint time.",
      challenge: "Generate test cases that match YOUR project's framework, conventions, and quality standards.",
      approach: [
        "Input: user story + acceptance criteria + related page objects",
        "Prompt includes: test framework (Selenium/Playwright), naming convention, project patterns",
        "Few-shot examples: include 3 of YOUR best test cases as references",
        "Generate: happy path, negative, boundary, edge case, accessibility tests",
        "Validate: compile check, naming convention check, assertion presence check",
        "Review: human reviews business logic, adds domain-specific assertions"
      ],
      result: "AI handles the boilerplate (setup, teardown, navigation). Humans focus on business logic validation. Sweet spot: AI as a 'junior SDET' that you review.",
      yourTask: "Take one user story from your current sprint. (1) Write a detailed prompt for test generation, (2) Generate tests using an LLM, (3) Score each test: correct/partially correct/incorrect, (4) Calculate time saved vs manual writing."
    },
    keyTakeaways: [
      "AI test generation works best with: structured prompts, few-shot examples, framework context",
      "AI generates boilerplate well, but business logic needs human review",
      "Always validate: compilation, convention compliance, assertion presence",
      "Treat AI as a junior SDET — it generates, you review and refine"
    ],
    trainerTip: "Live demo: take a user story from the audience, generate tests live, review together. The 'wow' moment + critical review teaches both capabilities and limitations."
  },
  "7-2": {
    id: "7-2",
    concept: "Self-healing tests use AI to automatically fix broken locators when the UI changes. Instead of hardcoded selectors that break on every UI update, AI learns multiple ways to identify elements and adapts when one way breaks. Like having a smart GPS that finds alternate routes when roads are blocked.",
    whyItMatters: "Flaky locators are the #1 cause of test maintenance costs. Self-healing reduces maintenance by 60-80%, freeing your team for meaningful testing.",
    realWorldExample: {
      title: "Healenium — Open Source Self-Healing in Action",
      scenario: "A fintech company had 3,000 Selenium tests. After every sprint, 15-20% of tests broke due to UI changes (renamed IDs, changed class names, restructured DOM). 2 SDETs spent 3 days/sprint fixing locators.",
      implementation: "Integrated Healenium: stores all historical locator-element pairs. When a locator fails, Healenium: 1) Finds the element using alternative strategies (ML-based DOM similarity), 2) Heals the locator in real-time, 3) Logs the healing for review, 4) Optionally auto-updates the test code. Uses ML model trained on DOM trees to find 'closest match' when exact locator fails.",
      outcome: "Locator-related failures reduced by 75%. SDET maintenance time dropped from 3 days to 4 hours/sprint. Test suite reliability improved from 85% to 97%."
    },
    caseStudy: {
      title: "Designing a Self-Healing Locator Strategy",
      background: "Your Page Object Model has 500 elements across 50 pages. Every release breaks 10-15% of locators.",
      challenge: "Design a system that finds elements even when primary locators break.",
      approach: [
        "Store multiple locator strategies per element: ID, CSS, XPath, text, ARIA label",
        "Record element 'fingerprint': tag, position, surrounding elements, visual appearance",
        "On locator failure: try alternatives in priority order",
        "If all fail: use ML model to find 'most similar' element in current DOM",
        "ML features: tag type, text content, position, parent hierarchy, CSS classes",
        "Auto-heal: update locator, log the change, alert for review",
        "Dashboard: track heal rate, most frequently healed elements"
      ],
      result: "Self-healing is not magic — it's systematic fallback + ML similarity matching. Your understanding of DOM structure and locator strategies makes you the ideal architect for this.",
      yourTask: "Take 5 critical page elements in your test suite. For each: (1) List 4 different locator strategies, (2) Which is most likely to break? (3) Design a 'fingerprint' that identifies it regardless of locator changes."
    },
    keyTakeaways: [
      "Self-healing = multiple locator strategies + ML-based fallback",
      "Not magic — it's systematic: try alternatives → ML similarity → alert → auto-update",
      "Tools: Healenium (open source), Testim, Mabl, Functionize",
      "Your locator strategy expertise is the foundation for building self-healing systems"
    ],
    trainerTip: "Demo: intentionally break a locator, show the test failing, then enable self-healing and show it passing. Then show the heal log. It's like watching a magic trick with the explanation."
  },
  "7-3": {
    id: "7-3",
    concept: "Testing AI outputs is fundamentally different from testing traditional software. Traditional: assert(expected == actual). AI: assert(output is reasonable, relevant, safe, and consistent). You're testing for QUALITY RANGES, not exact values. Like evaluating an essay vs grading a math test.",
    whyItMatters: "As AI features become standard in enterprise apps, SDETs who can validate AI outputs are in massive demand. Your testing expertise + AI understanding = unique skillset.",
    realWorldExample: {
      title: "Testing GPT-4 Powered Customer Support at Zendesk",
      scenario: "Zendesk integrated AI-generated response suggestions. Testing challenge: every response is different, even for the same input. Traditional assertions are impossible.",
      implementation: "Multi-layered validation: 1) Structural: response is valid JSON, within length limits, proper formatting. 2) Relevance: cosine similarity between response embedding and expected topic > 0.8. 3) Safety: no PII, no competitor mentions, no promises outside policy. 4) Consistency: run same input 10 times, check variance is within acceptable range. 5) Human eval: sample 5% for manual quality review.",
      outcome: "Built a validation pipeline that catches 95% of problematic responses automatically. Remaining 5% caught by human review. Like having unit tests + integration tests + manual testing for AI."
    },
    caseStudy: {
      title: "Building a Test Strategy for an AI Feature",
      background: "Your application has a new AI feature: automatic bug report summarization. It takes verbose bug reports and generates concise 2-line summaries.",
      challenge: "How do you test something where the 'correct' answer isn't deterministic?",
      approach: [
        "Golden dataset: 100 bug reports with human-written summaries (reference, not exact match)",
        "Structural tests: summary length (20-50 words), proper grammar, no code snippets",
        "Semantic tests: embedding similarity between AI summary and human summary > 0.75",
        "Information retention: key entities (bug type, severity, component) must appear",
        "Consistency: same bug report → 10 runs → variance in meaning < 10%",
        "Regression: compare new model version outputs vs previous version",
        "Load testing: 100 concurrent requests → response time < 5 seconds"
      ],
      result: "Multi-layered testing catches different failure modes: structural (formatting), semantic (meaning), safety (content), and performance (speed).",
      yourTask: "For an AI feature you're testing or building: (1) Define 5 validation criteria (beyond 'is it correct?'), (2) For each, design an automated check, (3) What % should pass to consider the feature 'healthy'?"
    },
    keyTakeaways: [
      "AI testing = structural + semantic + safety + consistency + performance",
      "Use embeddings for semantic similarity instead of exact string matching",
      "Golden datasets with human references (not exact answers) are essential",
      "Test distributions, not individual outputs — is the OVERALL quality acceptable?",
      "Your test strategy skills apply perfectly — just with different assertion types"
    ],
    trainerTip: "Ask testers: 'How would you test a translation feature?' They'll struggle with exact matching. Then introduce semantic similarity — it clicks."
  },
  "7-4": {
    id: "7-4",
    concept: "Testing AI systems = testing the ML pipeline end-to-end. Not just the model output, but: data quality, feature engineering, training process, deployment, monitoring, and feedback loops. Like testing a CI/CD pipeline — every stage needs its own tests.",
    whyItMatters: "ML systems have unique failure modes: data drift, model degradation, training bugs, feature store inconsistencies. Your pipeline testing experience is directly applicable.",
    realWorldExample: {
      title: "Uber's ML Testing Framework",
      scenario: "Uber tests ML systems at multiple levels: unit (individual functions), integration (pipeline stages), system (end-to-end), and A/B (production impact).",
      implementation: "Unit: test feature engineering functions with known inputs/outputs. Integration: test data pipeline → feature store → model → prediction service flow. System: test with production-like data, validate predictions against historical ground truth. A/B: deploy new model to 5% of traffic, compare metrics against baseline.",
      outcome: "Caught a data pipeline bug that would have corrupted 2 weeks of features. Traditional software testing wouldn't have caught this — ML-specific tests did."
    },
    caseStudy: {
      title: "ML Testing Pyramid — Your Test Strategy for AI",
      background: "You're QA lead for a team building an AI-powered pricing engine. You need a comprehensive test strategy.",
      challenge: "Design tests for every layer of the ML pipeline, not just the model output.",
      approach: [
        "Data tests: schema validation, null checks, distribution drift, freshness",
        "Feature tests: feature engineering functions with unit tests (known I/O)",
        "Model tests: accuracy on holdout set > threshold, performance regression",
        "Integration tests: full pipeline (data → features → model → API → response)",
        "API tests: response format, latency, error handling, rate limiting",
        "Fairness tests: model performance equal across demographic groups",
        "Monitoring tests: alerting works when accuracy drops below threshold"
      ],
      result: "ML testing pyramid: Data tests (base, most) → Feature tests → Model tests → Integration → API → Monitoring (top, fewest). Like the traditional testing pyramid but for AI.",
      yourTask: "Design an ML testing pyramid for your AI project: (1) List tests for each layer, (2) Estimate effort per layer, (3) Determine automation feasibility per test, (4) Create a dashboard spec for monitoring."
    },
    keyTakeaways: [
      "ML testing pyramid: data → features → model → integration → API → monitoring",
      "Data quality tests are the FOUNDATION — most ML bugs are data bugs",
      "Model tests need baselines — compare against previous versions",
      "Fairness testing is non-optional — biased models are legal/ethical risks",
      "Your testing strategy skills make you invaluable in AI teams"
    ],
    trainerTip: "Show the ML testing pyramid next to the traditional testing pyramid. SDETs immediately see the parallels and feel confident applying their skills."
  },

  // ============ MODULE 8: Capstone Projects ============
  "8-1": {
    id: "8-1",
    concept: "An AI Resume Screener automates the initial filtering of resumes — extracting skills, experience, and qualifications, then matching against job requirements. It combines NLP (text understanding), classification (qualify/disqualify), and scoring (ranking candidates).",
    whyItMatters: "This is one of the most common AI project interview topics. It covers end-to-end ML: data collection, feature engineering, model training, deployment, and ethical considerations.",
    realWorldExample: {
      title: "Unilever's AI Hiring Pipeline",
      scenario: "Unilever receives 1.8M applications/year. Human screening: 3 months per hiring cycle. AI screening: initial filter in minutes, shortlisting in hours.",
      implementation: "Pipeline: Resume PDF → OCR/parsing → NLP entity extraction (skills, experience, education) → Feature vector creation → Match score against job description → Rank candidates → Top 20% to human review. De-biasing: removed name, gender, age, photo from feature set. Regular fairness audits across demographics.",
      outcome: "Screening time reduced 75%. Diversity of shortlisted candidates improved 16% (AI removed unconscious bias that human screeners had). Critical: human always makes the final decision."
    },
    caseStudy: {
      title: "Build: AI Resume Screener — Architecture & Implementation",
      background: "Design and implement a resume screening system for a tech company hiring Java developers.",
      challenge: "Parse varied resume formats, extract relevant info, score against job requirements, handle bias.",
      approach: [
        "Data: Collect 1,000 resumes + job descriptions (anonymized from job boards)",
        "Parsing: PDF → text extraction → NLP for entities (skills, years, certifications)",
        "Feature Engineering: skill_match_score, experience_years, relevant_projects_count",
        "Matching: Embed resume + job description → cosine similarity for semantic match",
        "Scoring: weighted combination of skill match (40%), experience (30%), project relevance (30%)",
        "API: Spring Boot REST endpoint: POST /screen → returns ranked candidates with scores",
        "Bias mitigation: remove demographic features, test fairness across groups"
      ],
      result: "Complete project with architecture diagram, data flow, API design, and ethical considerations — exactly what interviewers want to see.",
      yourTask: "Build this project step by step: (1) Design the architecture diagram, (2) Define the API contract, (3) List features you'd extract from resumes, (4) How would you evaluate the system's fairness? (5) What are the failure modes and fallbacks?"
    },
    keyTakeaways: [
      "End-to-end ML project: data → parse → extract → match → score → deploy",
      "Combine NLP + embeddings + traditional ML for robust matching",
      "Bias mitigation is non-optional — legal and ethical requirement",
      "Always keep human in the loop for final decisions",
      "This project demonstrates: system design, ML application, ethics, production thinking"
    ],
    trainerTip: "In interviews, always mention bias mitigation PROACTIVELY. It shows maturity and awareness that distinguishes senior engineers from juniors."
  },
  "8-2": {
    id: "8-2",
    concept: "An AI Test Case Generator takes requirements or code as input and produces comprehensive test cases. It combines code understanding (parsing), requirement analysis (NLP), and test strategy (domain knowledge encoded in prompts).",
    whyItMatters: "This project directly leverages your testing expertise + AI skills. It's highly demonstrable in interviews and immediately useful in your work.",
    realWorldExample: {
      title: "Microsoft's IntelliTest — AI-Driven Test Generation",
      scenario: "Microsoft's IntelliTest analyzes C# code and automatically generates unit tests that achieve high code coverage. It uses symbolic execution + AI to find interesting test inputs.",
      implementation: "Code analysis → identify all execution paths → generate inputs that trigger each path → create assertions based on observed behavior → produce runnable test files. For Java, similar tools exist: EvoSuite (evolutionary test generation), Diffblue Cover (AI-based test generation).",
      outcome: "Achieves 80%+ code coverage automatically. Developers review and refine generated tests. Best use: regression test generation for legacy codebases."
    },
    caseStudy: {
      title: "Build: AI Test Case Generator for User Stories",
      background: "Your team receives user stories and manually creates test cases. This takes 40% of sprint QA capacity.",
      challenge: "Automate initial test case generation while maintaining your quality standards.",
      approach: [
        "Input: user story + acceptance criteria + UI wireframe description",
        "Prompt: Include your test framework, naming conventions, 3 example test cases",
        "Generation: happy path, negative, boundary, edge case, accessibility, performance",
        "Validation: compile check, naming convention check, coverage analysis",
        "Output: structured test scenarios with steps, expected results, test data",
        "Integration: Slack bot that generates tests from Jira ticket on demand",
        "Feedback loop: track which generated tests are modified → improve prompts"
      ],
      result: "Generates first draft in 30 seconds vs 2 hours manually. Human review takes 30 minutes. Net time saving: 60% on test creation.",
      yourTask: "Build a test case generator: (1) Choose your framework (Selenium/Playwright/Cypress), (2) Create a prompt template with your conventions, (3) Generate tests for 5 user stories, (4) Measure quality: what % are production-ready without edits?"
    },
    keyTakeaways: [
      "AI test generation = requirements understanding + test strategy + code generation",
      "Your testing expertise makes the prompt engineering significantly better",
      "Always validate generated tests: compilation, coverage, business logic",
      "Feedback loop: track edits to improve generation over time"
    ],
    trainerTip: "This project is perfect for your corporate training portfolio. Demo it live — audience engagement is guaranteed when they see tests generated from their own user stories."
  },
  "8-3": {
    id: "8-3",
    concept: "An enterprise AI chatbot uses RAG to answer questions from company documents. Unlike generic chatbots, it's grounded in YOUR data, cites sources, and knows when to say 'I don't know'. Architecture: document ingestion → embedding → vector search → LLM generation → response validation.",
    whyItMatters: "RAG chatbots are the #1 enterprise AI use case. Building one demonstrates full-stack AI skills: data engineering, NLP, system design, and production concerns.",
    realWorldExample: {
      title: "Walmart's Internal Knowledge Assistant",
      scenario: "Walmart's 2.1M employees needed quick access to HR policies, store procedures, and product information spread across thousands of documents.",
      implementation: "RAG architecture: 50K documents chunked into 500-token segments → embedded with ada-002 → stored in Pinecone. Query flow: employee question → embed → find top 5 chunks → construct prompt with context → GPT-4 generates answer → cite sources → display with confidence score. Guardrails: topic boundary (HR, procedures, products only), PII filter, escalation to human for low confidence.",
      outcome: "80% of routine queries answered instantly. HR ticket volume reduced 40%. Employee satisfaction with internal search improved 3x."
    },
    caseStudy: {
      title: "Build: AI Chatbot for Your Test Documentation",
      background: "Your QA team has 500+ pages of documentation: test strategies, framework guides, best practices, bug report templates. New joiners spend months learning the system.",
      challenge: "Build a chatbot that answers questions about YOUR specific documentation and says 'I don't know' when it should.",
      approach: [
        "Document ingestion: parse all docs → chunk into 500-token overlapping segments",
        "Embedding: sentence-transformers model → vector per chunk",
        "Vector DB: pgvector in PostgreSQL (your familiar SQL world!)",
        "Retrieval: embed question → cosine similarity search → top 5 chunks",
        "Generation: LLM prompt with context + instruction to cite sources",
        "Guardrails: if no chunk similarity > 0.7 → 'I don't have this information'",
        "API: Spring Boot WebSocket endpoint for real-time chat",
        "UI: React chat interface with source citations"
      ],
      result: "Full-stack AI project: database, backend, frontend, ML — using your entire skill set. Perfect for portfolio and interviews.",
      yourTask: "Design this system end-to-end: (1) Architecture diagram, (2) Data flow from document upload to user response, (3) How would you handle a question the system can't answer? (4) How would you measure accuracy?"
    },
    keyTakeaways: [
      "RAG chatbot = Document Pipeline + Vector Search + LLM Generation + Guardrails",
      "Chunking strategy matters: too small = no context, too large = noise",
      "Always implement 'I don't know' — worse than no answer is a wrong answer",
      "This project demonstrates: full-stack, AI, data engineering, production thinking"
    ],
    trainerTip: "This is the BEST live demo project. Upload a document, ask questions, show cited sources. Every audience member immediately sees the value."
  },
  "8-4": {
    id: "8-4",
    concept: "Image-based defect detection uses computer vision to automatically identify defects in products, components, or infrastructure. It combines your image processing knowledge (preprocessing, feature extraction) with deep learning (CNN classification) for production-grade quality control.",
    whyItMatters: "Computer vision in manufacturing/quality control has massive ROI. Your image processing + K-Means background gives you a unique head start.",
    realWorldExample: {
      title: "Tesla's Visual Inspection Pipeline",
      scenario: "Tesla inspects every panel, weld, and paint surface using high-speed cameras on the production line. Human inspectors missed 3-5% of defects and slowed the line.",
      implementation: "Camera captures → image preprocessing (resize, normalize, enhance contrast) → CNN model classifies: pass/fail → for failures, second model identifies defect TYPE (scratch, dent, bubble, misalignment) → Grad-CAM highlights exact defect location → alert to quality team with annotated image.",
      outcome: "Defect detection: 99.5% (up from 95%). Inspection speed: 0.2 seconds/image. Continuous learning: new defect types added with just 100 labeled examples using transfer learning."
    },
    caseStudy: {
      title: "Build: PCB Defect Detection System",
      background: "A PCB manufacturer inspects boards visually. Defects: missing component, misaligned component, solder bridge, scratched trace.",
      challenge: "Build an end-to-end visual inspection system that integrates with the production line.",
      approach: [
        "Data: 1,000 images per defect type + 5,000 'good' images (collect from production line cameras)",
        "Preprocessing: resize 224x224, normalize, augment (rotation, brightness, flip)",
        "Model: ResNet-50 pretrained on ImageNet → transfer learning → 5-class classifier",
        "Training: freeze base layers, train classifier head, learning rate scheduling",
        "Explainability: Grad-CAM to highlight WHERE the model sees the defect",
        "Deployment: ONNX model served via REST API, triggered by camera capture",
        "Integration: Spring Boot service receives image → calls model → returns verdict + annotated image",
        "Monitoring: track accuracy daily, alert on drift, weekly human review of borderline cases"
      ],
      result: "From K-Means image segmentation → CNN defect classification is a natural progression. Your image processing foundation makes this achievable.",
      yourTask: "Design this system: (1) What images would you collect? (2) How would you handle class imbalance (fewer defect images)? (3) How would you explain model decisions to non-technical quality managers? (4) What's the acceptable false positive/negative rate?"
    },
    keyTakeaways: [
      "Your K-Means + image processing experience = CV head start",
      "Transfer learning makes this feasible with 1K-10K images (not millions)",
      "Explainability (Grad-CAM) is essential — 'why did you flag this?' must be answerable",
      "Production pipeline: camera → preprocess → inference → alert → log → retrain"
    ],
    trainerTip: "Show K-Means segmentation → feature extraction → CNN classification as a progression story. Trainees see their existing knowledge leading to cutting-edge CV."
  },

  // ============ MODULE 9: Trainer Mode ============
  "9-1": {
    id: "9-1",
    concept: "Teaching AI to freshers requires translating complex concepts into relatable analogies WITHOUT losing technical accuracy. Your job: make it simple enough to understand, deep enough to be useful, and exciting enough to inspire curiosity.",
    whyItMatters: "As a corporate trainer, your ability to teach AI determines your market value. Freshers are the largest training audience — mastering this is your competitive advantage.",
    realWorldExample: {
      title: "Google's Internal AI Literacy Program",
      scenario: "Google trained 10,000+ non-technical employees in 'AI basics' using: no math, no code, just analogies, demos, and interactive exercises.",
      implementation: "Curriculum: 1) 'What is AI?' — using the 'teaching a child vs writing instructions' analogy. 2) ML types — 'supervised = studying with answer key, unsupervised = grouping songs into playlists'. 3) Hands-on — Teachable Machine (browser-based ML). 4) Ethics — real bias examples that freshers can relate to.",
      outcome: "96% of attendees reported 'confident understanding of AI basics'. Key: zero math, maximum interaction, relatable examples."
    },
    caseStudy: {
      title: "Designing a 1-Day AI Workshop for Fresh Graduates",
      background: "Your company hired 50 fresh graduates. You need to give them AI literacy in one day so they can contribute to AI projects.",
      challenge: "Balance breadth (cover key concepts) with depth (actually useful understanding) in 8 hours.",
      approach: [
        "Hour 1-2: What is AI? Live demos (ChatGPT, image generation, translation)",
        "Hour 3-4: ML basics with Teachable Machine (hands-on, no code)",
        "Hour 5: How our company uses AI (internal case studies)",
        "Hour 6-7: Hands-on with prompting — solve real work tasks with AI",
        "Hour 8: Career paths in AI + Q&A",
        "Throughout: polls, quizzes, pair exercises (no lectures > 20 min)",
        "Takeaway: cheat sheet + 3 AI tools they can use tomorrow at work"
      ],
      result: "Fresh graduates leave with: conceptual understanding, hands-on experience, and 3 immediately applicable AI tools.",
      yourTask: "Design YOUR 1-day AI workshop: (1) Define 5 learning objectives, (2) Plan the agenda with time blocks, (3) Create 3 hands-on exercises, (4) Design a feedback form with 5 questions."
    },
    keyTakeaways: [
      "Analogies > definitions — make it relatable before making it technical",
      "Hands-on within the first hour — don't front-load theory",
      "Max 20-minute lecture blocks — attention spans are real",
      "Give tools they can use TOMORROW — immediate ROI = engaged learners"
    ],
    trainerTip: "Start with a live AI demo that makes them laugh or gasp. First impression sets the tone for the entire training. I recommend: AI-generated avatars of attendees."
  },
  "9-2": {
    id: "9-2",
    concept: "Teaching AI to managers/executives means speaking their language: ROI, risk, competitive advantage, and strategic impact. They don't care about algorithms — they care about business outcomes. Your job: translate AI capabilities into business opportunities and risks.",
    whyItMatters: "Managers decide AI budgets. If you can't convince them with business language, your AI projects won't get funded. This skill makes you invaluable.",
    realWorldExample: {
      title: "McKinsey's AI Executive Briefing Format",
      scenario: "McKinsey presents AI strategy to C-suite in a specific format: Problem → Opportunity size ($) → AI solution (1 sentence) → Implementation timeline → ROI → Risks → Recommendation.",
      implementation: "Example pitch: 'Your customer service department handles 100K calls/month at $8/call ($800K/month). AI chatbot can handle 60% of routine queries at $0.10/call. Investment: $150K. Monthly savings: $475K. ROI: 3 months. Risk: customer satisfaction — mitigated by human escalation. Recommendation: 90-day pilot with 10% of traffic.'",
      outcome: "This format gets executive buy-in because it speaks money, not technology. Every number is verifiable, every risk has mitigation."
    },
    caseStudy: {
      title: "Preparing an AI Strategy Presentation for Your CTO",
      background: "Your CTO asked: 'Where should we invest in AI this year?' You need to present 3 AI opportunities with business cases.",
      challenge: "Speak technology accurately while keeping it accessible and decision-oriented for executives.",
      approach: [
        "Opportunity 1: AI test generation (QA efficiency) — ROI calculation with team hours saved",
        "Opportunity 2: Customer support chatbot — cost per ticket before/after AI",
        "Opportunity 3: Predictive maintenance — downtime cost × reduction percentage",
        "For each: problem (in business terms), solution (1 paragraph), investment needed, timeline, ROI, risks",
        "Visual: one-page summary per opportunity, executive summary on first slide",
        "Demo: 3-minute live demo per opportunity — show, don't tell"
      ],
      result: "Executives make decisions based on: ROI, risk, timeline. Your job is to provide clear data for these three dimensions.",
      yourTask: "Create a 1-page AI investment proposal: (1) Business problem in dollar terms, (2) AI solution in 1 paragraph, (3) Investment required, (4) Expected ROI with timeline, (5) Top 3 risks with mitigations."
    },
    keyTakeaways: [
      "Executives care about: ROI, risk, timeline, competitive advantage",
      "Quantify EVERYTHING — vague benefits don't get funded",
      "One-page summaries > 50-slide decks for decision-makers",
      "Always include a demo — 3 minutes of seeing beats 30 minutes of explaining"
    ],
    trainerTip: "Practice the 'elevator pitch': explain your AI project's value in 30 seconds using only business terms. If you can't, you're not ready for the executive audience."
  },
  "9-3": {
    id: "9-3",
    concept: "Great AI demos combine preparation (what to show), storytelling (why it matters), and recovery tactics (when things go wrong — and they will, because AI is non-deterministic). A good demo creates a 'wow moment' followed by 'I can see how we'd use this'.",
    whyItMatters: "Demos sell AI initiatives. A great demo gets budget approval. A failed demo kills projects. Your demo skills directly impact your career and your team's AI adoption.",
    realWorldExample: {
      title: "The Perfect AI Demo Structure",
      scenario: "A trainer demoing an AI-powered document analysis tool to a pharma company's leadership team.",
      implementation: "Structure: 1) START with the problem — 'Your team reviews 500 regulatory documents/month, spending 4 hours each'. 2) SHOW the solution — upload a real document, watch AI extract key information in 30 seconds. 3) VALIDATE — compare AI extraction with manual extraction (pre-prepared). 4) SCALE — 'Now imagine this across all 500 documents'. 5) RISK — 'Here's what it can't do yet, and how we handle that'. Recovery prep: pre-loaded backup results, secondary demo environment, offline video recording of working demo.",
      outcome: "Demo led to $2M project approval. Key: used THEIR actual documents (with permission), showed both capabilities AND limitations."
    },
    caseStudy: {
      title: "Planning Your AI Demo for Corporate Training",
      background: "You're giving a 2-hour AI training to 30 corporate employees. You want 3 live demos that build excitement and understanding.",
      challenge: "Make demos reliable, relevant to the audience, and progressively impressive.",
      approach: [
        "Demo 1 (Easy, 100% reliable): Text summarization — paste a long company email, get a 3-line summary",
        "Demo 2 (Medium, 90% reliable): Code generation — describe a function, watch AI write it",
        "Demo 3 (Impressive, 80% reliable): Image analysis — upload a product photo, AI describes defects",
        "Recovery plan for each: pre-recorded video backup, second prompt ready, graceful 'this shows real AI behavior'",
        "Audience participation: let attendees suggest inputs (builds trust in authenticity)",
        "Close each demo with: 'How could you use this in YOUR work?'"
      ],
      result: "Progressive difficulty keeps engagement high. Recovery plans prevent embarrassment. Audience participation creates buy-in.",
      yourTask: "Plan 3 demos for YOUR training audience: (1) What AI capability? (2) What example input? (3) What's the expected output? (4) What could go wrong? (5) What's your recovery plan?"
    },
    keyTakeaways: [
      "Demo structure: Problem → Solution → Validate → Scale → Risk",
      "Always use audience-relevant examples (their industry, their documents)",
      "Have backup plans — AI WILL misbehave during demos",
      "When AI fails during demo: 'This is actually a great example of why we need guardrails!'",
      "End every demo with: 'How would YOU use this?'"
    ],
    trainerTip: "Secret weapon: pre-test your demo inputs 5 times before the session. Replace any input that gave inconsistent results. Audiences don't know you optimized the inputs."
  },
  "9-4": {
    id: "9-4",
    concept: "Curriculum design for AI training requires balancing theory with practice, pacing with depth, and technical accuracy with audience accessibility. The key: modular design with progressive complexity — like building a well-architected system.",
    whyItMatters: "A well-structured curriculum is a reusable PRODUCT. Build it once, deliver it to multiple clients. This is the foundation of your training business.",
    realWorldExample: {
      title: "Stanford's Executive AI Program Structure",
      scenario: "Stanford's 3-day executive AI program trains C-suite leaders who have zero technical background but need to make AI investment decisions.",
      implementation: "Day 1: AI Landscape (what's possible, what's hype, what's real). Day 2: AI Strategy (where to invest, build vs buy, talent needs). Day 3: AI Implementation (pilot projects, governance, ethics, scaling). Each day: 50% lecture, 30% case study discussion, 20% hands-on exercises. Assessment: each participant creates a 1-page AI strategy for their department.",
      outcome: "Alumni network of 500+ executives driving AI adoption. Curriculum updated quarterly based on feedback. Revenue: $15K per participant."
    },
    caseStudy: {
      title: "Design: 5-Day Corporate AI Training Curriculum",
      background: "A large IT company asked you to train 100 employees (mix of developers, testers, project managers) on AI over 5 days.",
      challenge: "Create a curriculum that serves all three audiences without boring developers or losing managers.",
      approach: [
        "Day 1 (All): AI Fundamentals — shared vocabulary, live demos, industry examples",
        "Day 2 (All): ML & Data Thinking — hands-on with Teachable Machine + data exercises",
        "Day 3 (Split): Developers → AI in code, LLM APIs | Testers → AI testing | PMs → AI project management",
        "Day 4 (Split): Developers → Build AI features | Testers → AI test generation | PMs → AI business cases",
        "Day 5 (All): Capstone presentations — each group presents an AI solution they designed",
        "Assessment: team capstone project + individual quiz + peer evaluation",
        "Materials: slides, handouts, reference links, recorded demos"
      ],
      result: "Split-track on Days 3-4 lets you go deep for each audience. Shared Days 1-2 and 5 build team alignment. Capstone creates deliverables management can see.",
      yourTask: "Design your own training curriculum: (1) Define audience and duration, (2) Create daily agenda with time blocks, (3) Balance theory/practice per day, (4) Design assessment method, (5) Create feedback collection plan."
    },
    keyTakeaways: [
      "Modular design: shared foundation + specialized tracks = serves diverse audiences",
      "50-30-20 rule: 50% interactive lecture, 30% case studies, 20% hands-on",
      "Capstone projects create tangible outcomes that justify training investment",
      "Update curriculum quarterly — AI moves fast",
      "Curriculum is a PRODUCT — invest in its design like you'd invest in code architecture"
    ],
    trainerTip: "Create a 'curriculum as code' document: versioned, modular, with clear interfaces between modules. Same engineering principles you teach, applied to your training design."
  },

  // ============ MODULE 10: Career & Monetization ============
  "10-1": {
    id: "10-1",
    concept: "Three AI career paths for experienced engineers: 1) AI/ML Engineer — build AI systems. 2) AI Trainer/Consultant — teach and advise. 3) AI Product Builder — create AI-powered products. Each leverages different parts of your experience.",
    whyItMatters: "Choosing the right path determines your learning priorities, job search strategy, and income trajectory. Your unique background opens paths that pure ML graduates can't access.",
    realWorldExample: {
      title: "Three Engineers, Three Paths",
      scenario: "Three senior Java developers transitioned to AI careers: Engineer A became an ML Engineer at Google (builds recommendation systems). Engineer B became an AI Trainer at Cognizant (conducts corporate AI training). Engineer C built an AI SaaS product (automated test generation tool).",
      implementation: "Engineer A: focused on Python, ML algorithms, system design. Took 12 months. Now earns 2x previous salary. Engineer B: focused on teaching skills, curriculum design, industry case studies. Took 6 months. Now earns 1.5x through training fees. Engineer C: focused on product design, AI APIs, marketing. Took 18 months. Now earns variable but potentially highest through product revenue.",
      outcome: "All three leveraged their Java experience differently. None started from zero. The 'right' path depends on: risk tolerance, personality (building vs teaching vs selling), and timeline."
    },
    caseStudy: {
      title: "Choosing Your AI Career Path",
      background: "You have: 10+ years Java, testing experience, image processing background, and now AI knowledge from this course.",
      challenge: "Which path maximizes your strengths and aligns with your goals?",
      approach: [
        "Path 1 — ML Engineer: leverage Java + testing for MLOps, model deployment, AI testing",
        "Path 2 — AI Trainer: leverage teaching experience + technical depth for premium corporate training",
        "Path 3 — AI Product Builder: build AI-powered testing tools using your domain expertise",
        "Evaluate: income potential, risk level, lifestyle fit, timeline to profitability",
        "Hybrid approach: start with training (quick income) → build products (long-term value)",
        "Portfolio: showcase projects from this course, client testimonials, published content"
      ],
      result: "Your unique combination (Java + Testing + Training + AI) makes Path 2 immediately viable and Path 3 a strong medium-term goal.",
      yourTask: "Create your career decision matrix: (1) Rate each path on 5 criteria (income, risk, enjoyment, market demand, leverage of skills) on 1-10 scale, (2) Total the scores, (3) Research 3 people on each path, (4) Set a 30-day action plan for your top choice."
    },
    keyTakeaways: [
      "Engineer → builds AI systems (high salary, requires deep ML skills)",
      "Trainer → teaches AI (quick start, leverages communication skills, scalable)",
      "Product Builder → creates AI products (highest potential, highest risk)",
      "Your background uniquely positions you for training + product building",
      "Start with the path that generates income fastest, then expand"
    ],
    trainerTip: "In your training, share these paths with trainees. Many don't know training is a career option. You're expanding their worldview AND creating future peers."
  },
  "10-2": {
    id: "10-2",
    concept: "Building AI courses is a product development exercise. You need: target audience, learning objectives, content creation, platform selection, marketing, and continuous improvement. Your engineering mindset helps you build systematic, high-quality educational products.",
    whyItMatters: "AI courses are high-margin products. Create once, sell repeatedly. Your unique perspective (enterprise engineer teaching AI) is underrepresented and in high demand.",
    realWorldExample: {
      title: "AI Course Creator Success Story",
      scenario: "An Indian Java developer created an 'AI for Java Developers' course on Udemy. No fancy production — just screen recording + slides + practical exercises.",
      implementation: "Course structure: 10 modules, 40 lessons, 15 hours total. Price: ₹499 (intro) to ₹3,499 (full). Marketing: LinkedIn posts about AI+Java, free YouTube shorts, tech blog articles. Differentiator: every AI concept explained with Java equivalents — no Python-first approach.",
      outcome: "2,000 students in 6 months. Revenue: ₹15 lakhs. Time investment: 3 months to create, 2 hours/week to maintain. Key: niche positioning (AI for Java devs) not broad (AI for everyone)."
    },
    caseStudy: {
      title: "Creating Your First AI Course Product",
      background: "You want to create an online course: 'AI for Automation Testers' — targeting SDETs who want to add AI to their skillset.",
      challenge: "Design, create, and launch a course that generates passive income.",
      approach: [
        "Audience: SDETs with 2+ years of Selenium/Playwright experience",
        "USP: 'Learn AI through testing analogies — no data science degree needed'",
        "Modules: AI basics for testers, AI test generation, self-healing tests, testing AI systems, career",
        "Format: 15-minute video lessons + coding exercises + quizzes + capstone project",
        "Platform: Udemy (reach) + own website (margins) + corporate licensing (B2B revenue)",
        "Marketing: LinkedIn articles, testing community talks, free intro module",
        "Pricing: ₹999 individual, ₹25,000 corporate license (per team of 10)"
      ],
      result: "Course creation: 2-3 months part-time. Revenue potential: ₹5-20 lakhs/year from individual + corporate sales.",
      yourTask: "Design YOUR course: (1) Target audience (be specific), (2) 8-10 module titles, (3) Pricing strategy (individual + corporate), (4) Marketing plan (3 channels), (5) First lesson outline."
    },
    keyTakeaways: [
      "Niche courses outperform broad ones — 'AI for X' beats 'AI basics'",
      "Your unique perspective IS the value — don't copy generic AI courses",
      "Start with free content (blog, YouTube) → build audience → launch paid course",
      "Corporate licensing = higher revenue per sale than individual purchases",
      "Course = product. Apply product development principles: MVP, feedback, iterate"
    ],
    trainerTip: "Your FIRST course doesn't need to be perfect. Launch with 5 modules, gather feedback, improve. Same advice you give to developers: ship early, iterate often."
  },
  "10-3": {
    id: "10-3",
    concept: "Corporate AI training is a business: pricing, proposals, delivery, and client management. It's consultative selling — you're not selling hours, you're selling OUTCOMES (team capability, project readiness, risk reduction).",
    whyItMatters: "Corporate training commands premium pricing (₹50K-5L per day). With your technical depth + training skills + AI knowledge, you can charge premium rates.",
    realWorldExample: {
      title: "Independent AI Trainer's Business Model",
      scenario: "A former Infosys architect became a freelance AI corporate trainer. Started with one client, now works with 15 companies annually.",
      implementation: "Services: 1-day AI awareness (₹1.5L), 3-day hands-on AI (₹4L), 5-day comprehensive (₹6L), ongoing advisory (₹50K/month). Client acquisition: LinkedIn thought leadership, conference speaking, referrals. Delivery: on-site preferred (builds relationship), virtual for remote teams. Retention: quarterly 'AI update' sessions, alumni Slack community.",
      outcome: "Annual revenue: ₹40-50 lakhs. Works 120 days/year delivering training, rest for preparation, marketing, and learning. Key: premium positioning through thought leadership."
    },
    caseStudy: {
      title: "Building Your Corporate AI Training Business",
      background: "You want to offer corporate AI training as a side business initially, potentially full-time later.",
      challenge: "Price your services, create a proposal template, and find your first client.",
      approach: [
        "Define offerings: 1-day workshop, 3-day bootcamp, custom programs",
        "Pricing: research market rates, position at 80% of big-firm prices with better personalization",
        "Proposal template: client needs analysis, learning objectives, agenda, facilitator bio, investment",
        "First client strategy: approach your OWN company, then network referrals",
        "Delivery: custom case studies per client industry, hands-on exercises, post-training support",
        "Differentiation: you've BUILT AI systems, not just taught theory — real-world credibility"
      ],
      result: "Start within your company (low risk), build testimonials, then expand to external clients. Your employer becomes your first reference.",
      yourTask: "Create: (1) A 1-page training brochure with 3 offerings, (2) A proposal template for a 3-day AI training, (3) A pricing sheet with justification, (4) A list of 10 potential first clients, (5) Your LinkedIn summary as an AI trainer."
    },
    keyTakeaways: [
      "Corporate training is selling OUTCOMES, not hours — price accordingly",
      "Custom case studies per client industry = premium pricing justification",
      "Start with your employer → referrals → independent clients",
      "Thought leadership (LinkedIn, talks, blogs) = client acquisition pipeline",
      "Premium positioning: 'AI trainer who has built enterprise AI systems' > 'AI trainer'"
    ],
    trainerTip: "Your first 3 trainings should be discounted or free — collect testimonials, video clips, and feedback. These assets are worth more than early revenue."
  },
  "10-4": {
    id: "10-4",
    concept: "In AI careers, skills matter more than certifications — but the right certifications can open doors. The strategy: build skills through projects, validate with select certifications, and showcase through content and portfolio.",
    whyItMatters: "Don't waste time collecting certificates. Invest in projects and content that demonstrate capability. Certifications are supplements, not substitutes for skill.",
    realWorldExample: {
      title: "What Hiring Managers Actually Look For",
      scenario: "A survey of 50 AI hiring managers revealed: 80% prioritize GitHub projects over certifications. 70% value blog posts/talks about AI. 60% consider certifications useful for initial screening only. 90% say interview performance trumps all credentials.",
      implementation: "Optimal strategy: 1-2 top certifications (AWS ML Specialty, Google Professional ML) for resume screening. 3-5 portfolio projects for interview discussions. Active LinkedIn/blog for thought leadership. Open source contributions for credibility. Conference talks for network building.",
      outcome: "The candidate who gets hired: has a certification for the ATS filter + projects to discuss in interviews + content that establishes expertise."
    },
    caseStudy: {
      title: "Your 6-Month AI Career Acceleration Plan",
      background: "You've completed this course. Now you need to translate learning into career advancement.",
      challenge: "Design a 6-month plan that maximizes career impact with limited time (you're still working full-time).",
      approach: [
        "Month 1: Complete 1 capstone project + get 1 certification (AWS ML or Google Cloud AI)",
        "Month 2: Start writing LinkedIn articles (1/week) about AI from an engineer's perspective",
        "Month 3: Build and deploy 1 AI project using your Java skills (Spring Boot + AI API)",
        "Month 4: Give 1 talk (internal brownbag or local meetup) about AI for developers",
        "Month 5: Conduct first AI training (internal team, free) → collect testimonials",
        "Month 6: Update resume, apply to AI roles OR pitch corporate training services",
        "Throughout: 30 minutes/day learning, 2 hours/weekend building"
      ],
      result: "In 6 months: 1 certification, 2 projects, 12 articles, 1 talk, 1 training delivery. This portfolio outweighs any number of certificates alone.",
      yourTask: "Create YOUR 6-month plan: (1) Monthly goals, (2) Weekly time allocation, (3) Accountability partner, (4) Success metrics for each month, (5) Fallback plan if you fall behind."
    },
    keyTakeaways: [
      "Skills > Certifications — but 1-2 strategic certifications help pass ATS filters",
      "Projects > courses — build and showcase, don't just learn",
      "Content creation (blog, talks) = compound interest for your career",
      "Your unique background (Java + Testing + AI) is a NICHE — own it",
      "Consistency beats intensity — 30 min/day > 8 hours on weekends"
    ],
    trainerTip: "Tell trainees: 'The best certification is a deployed project that solves a real problem.' Then show them YOUR projects as proof."
  },
};
