import { useState } from "react";
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
  ChevronDown,
  ChevronUp,
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
      { week: "Week 1", tasks: ["Complete Module 1 lessons", "Map 10 Java skills → AI equivalents", "Set up Python environment (Anaconda/Colab)"],
        content: "**Goal:** Understand the fundamental paradigm shift from traditional programming to AI.\n\n**Key Concept — AI vs Traditional Programming:**\nIn traditional Java, YOU write rules: `if (age > 18) allow()`. In AI, you provide data and the system LEARNS rules. Think of it as: Selenium with explicit locators (rules) vs AI that learns to find elements from page patterns (learning).\n\n**Hands-on Exercise:**\n1. Install Anaconda or open Google Colab\n2. Write a simple Python script that mirrors a Java class you know well\n3. Create a document mapping your top 10 Java skills to AI equivalents:\n   - `Spring Boot → ML API serving`\n   - `JUnit assertions → Model evaluation metrics`\n   - `CI/CD pipeline → ML training pipeline`\n\n**Real-World Example:** Infosys automated invoice processing — rule-based OCR failed on 35% of invoices (200+ formats). An ML model trained on 50K labeled invoices achieved 94% accuracy and learned new formats with just 50 examples.\n\n**Trainer Note:** When teaching this to corporate teams, use the 'spam filter' analogy — 500 hand-written rules caught 78% of spam, while an ML model trained on 100K emails caught 99.2%." },
      { week: "Week 2", tasks: ["ML vs DL vs GenAI deep dive", "Supervised vs Unsupervised with K-Means revision", "Practice: classify 5 business problems as ML/DL/GenAI"],
        content: "**Goal:** Master the AI hierarchy and know when to use each level.\n\n**The Java Analogy:**\n- **ML** = Writing raw JDBC — you control everything, simple, direct\n- **DL** = Using Hibernate — more abstraction, handles complexity\n- **GenAI** = Using Spring Data JPA — maximum abstraction, you describe what you want\n\n**Supervised vs Unsupervised (Testing Analogy):**\n- Supervised = JUnit tests WITH expected results (labeled data)\n- Unsupervised = Exploratory testing — finding patterns you didn't know existed\n- Your K-Means experience IS unsupervised learning!\n\n**Practice Exercise:**\nClassify these 5 problems:\n1. Predict which customers will churn → **ML** (structured data, clear labels)\n2. Find product photos similar to an uploaded image → **DL** (image understanding)\n3. Generate product descriptions from attributes → **GenAI** (creative text generation)\n4. Group customers by behavior (no predefined groups) → **ML/Unsupervised** (K-Means!)\n5. Build a chatbot that answers HR questions → **GenAI** (RAG + LLM)\n\n**Cost Comparison:** ML: $0.001/prediction | DL: $0.01/image | GenAI: $0.03/generation. Always start with the simplest level that solves the problem." },
      { week: "Week 3", tasks: ["Features, Labels & Overfitting", "Hands-on: train your first model (scikit-learn)", "Compare model training to CI/CD pipeline"],
        content: "**Goal:** Understand the core ML vocabulary through testing analogies.\n\n**Key Concepts:**\n- **Features** = input variables (like method parameters)\n- **Labels** = expected output (like return values)\n- **Overfitting** = model memorizes training data but fails on new data\n\n**The Testing Parallel:**\nOverfitting is EXACTLY like when your Selenium tests pass in CI but crash in production. Your tests were too specific to test data — hardcoded locators, specific screen resolutions, sleep timers.\n\n```java\n// OVERFIT TEST (like an overfit model):\ndriver.findElement(By.id(\"user-input-v2-2024\")).sendKeys(\"admin\");\n// ^ Breaks when UI changes\n\n// GENERALIZED TEST (like a well-trained model):\ndriver.findElement(By.cssSelector(\"[data-testid='username']\")).sendKeys(\"admin\");\n// ^ Works across UI versions\n```\n\n**Hands-on: Train Your First Model**\n1. Open Google Colab\n2. Load the Iris dataset (`from sklearn.datasets import load_iris`)\n3. Split: 80% training, 20% testing\n4. Train a Decision Tree classifier\n5. Evaluate accuracy on test set\n6. Intentionally overfit: set `max_depth=None, min_samples_leaf=1` — watch training accuracy hit 100% but test accuracy drop\n\n**Trainer Tip:** Show a graph where training accuracy goes UP while validation accuracy goes DOWN. That V-shape is the 'aha moment' for overfitting." },
      { week: "Week 4", tasks: ["Model Training Pipeline end-to-end", "Build: simple loan prediction model", "Interview prep: explain ML pipeline in Java terms"],
        content: "**Goal:** Build your first complete ML pipeline — this is CI/CD for models.\n\n**ML Pipeline = CI/CD Pipeline:**\n| ML Pipeline | CI/CD Pipeline |\n|---|---|\n| Collect data | Write code |\n| Clean data | Code review |\n| Feature engineering | Build |\n| Train model | Run tests |\n| Validate | Stage |\n| Deploy | Deploy |\n| Monitor | Monitor |\n\n**Loan Prediction Project:**\n1. **Data:** Download loan dataset from Kaggle (50K records)\n2. **Clean:** Handle nulls, encode categories (like test data setup)\n3. **Features:** debt-to-income ratio, employment length, loan amount\n4. **Train:** Random Forest with 5-fold cross-validation\n5. **Evaluate:** Accuracy, precision, recall (which matters more for loans?)\n6. **Deploy concept:** `model.predict(new_application)` returns approve/reject + confidence\n\n**Interview Prep:**\nQ: *Walk me through an ML training pipeline.*\nA: 'It mirrors CI/CD: data collection (code), preprocessing (review), feature engineering (build), training with cross-validation (test on multiple environments), evaluation against holdout set (staging), deployment as REST API (production), monitoring for concept drift (APM). I'd version data with DVC and models with MLflow — same discipline as Git versioning.'\n\n**Key Insight:** The pipeline is 80% engineering (your strength) and 20% ML-specific." },
      { week: "Week 5–6", tasks: ["Python for ML essentials (NumPy, Pandas, Matplotlib)", "Revisit K-Means with business datasets", "Complete all Module 1 & 2 interview questions"],
        content: "**Goal:** Get comfortable with Python ML tools and apply K-Means to business problems.\n\n**Python for Java Developers — Quick Map:**\n- `ArrayList` → `list` / `numpy.array`\n- `HashMap` → `dict` / `pandas.DataFrame`\n- `Stream API` → `pandas` operations (filter, map, reduce)\n- `System.out.println` → `print()` / `matplotlib.pyplot`\n\n**Essential Libraries:**\n```python\nimport numpy as np       # Like Java arrays, but math-optimized\nimport pandas as pd      # Like ResultSet/DataFrame, but powerful\nimport matplotlib.pyplot as plt  # Like JFreeChart, but simpler\nfrom sklearn.cluster import KMeans  # Your old friend!\n```\n\n**K-Means Business Exercise:**\nUsing a retail customer dataset:\n1. Features: monthly_spend, visit_frequency, avg_basket_size\n2. Apply K-Means with K=2 to K=10\n3. Use Elbow Method to find optimal K\n4. Name your segments: 'Budget Browsers', 'Premium Loyalists', etc.\n5. Recommend different marketing strategies per segment\n\n**Real-World Impact:** Reliance Retail used K-Means with K=5 on 200M loyalty members. Campaign conversion jumped from 2% → 8.5%. Marketing efficiency improved 300%.\n\n**Interview Questions to Master:**\n1. How does AI programming differ from traditional development?\n2. When would you choose unsupervised over supervised learning?\n3. What is overfitting and how do you prevent it?\n4. Explain the ML training pipeline using CI/CD analogies." },
      { week: "Week 7–8", tasks: ["Mini project: customer segmentation with K-Means", "Write 2 LinkedIn posts about your AI journey", "Peer review: explain ML to a non-technical colleague"],
        content: "**Goal:** Complete your first portfolio project and start building your AI brand.\n\n**Customer Segmentation Project (Full):**\n1. **Dataset:** Use an e-commerce dataset (Kaggle: Online Retail II)\n2. **RFM Analysis:** Calculate Recency, Frequency, Monetary per customer\n3. **Preprocessing:** Scale features, handle outliers\n4. **K-Means:** Find optimal clusters, interpret each\n5. **Visualization:** Plot clusters in 2D using PCA\n6. **Business Impact:** Write recommendations per segment\n7. **Document:** README with architecture, results, business insights\n\n**LinkedIn Article Ideas:**\n- Article 1: 'From Java Developer to AI Engineer — My First Month'\n  - Share your skill mapping, first model, key insights\n  - Include the testing-overfitting analogy (people love this)\n- Article 2: 'K-Means Isn't Just for Images — How I Used It for Customer Segmentation'\n  - Share your project results (anonymized)\n  - Include code snippets and visualizations\n\n**Peer Review Exercise:**\nExplain these to a non-technical colleague:\n1. What is AI? (Use the 'teaching a child' analogy)\n2. What is ML? (Use the 'Netflix recommendations' example)\n3. What did your K-Means project find? (Use business language, not tech)\n\nIf they understand, you're ready to be a trainer. If not, simplify further.\n\n**Milestone Check:** You should now have: Python environment set up, 1 ML model built, 1 K-Means project done, 2 articles written." },
    ],
    deliverables: ["Skill mapping document (Java → AI equivalents)", "First ML model (loan/fraud prediction)", "K-Means business segmentation project", "2 LinkedIn articles published"],
    skills: ["Python basics", "scikit-learn", "ML fundamentals", "Data preprocessing", "K-Means clustering"],
    interviewReady: ["Explain AI vs traditional programming with examples", "ML vs DL vs GenAI — when to use which", "What is overfitting? How does it relate to flaky tests?", "Walk through an ML training pipeline"],
  },
  {
    month: "Month 3–4",
    title: "Data Thinking & Hands-on ML",
    icon: Code,
    color: "hsl(152 60% 42%)",
    focus: "Master data engineering and build production-ready ML models",
    modules: ["Module 3: Data Thinking", "Module 4: Hands-on ML"],
    weeklyPlan: [
      { week: "Week 9", tasks: ["Data quality fundamentals — audit a real dataset", "Structured vs Unstructured data pipelines", "Practice: clean a messy CSV dataset"],
        content: "**Goal:** Understand why data quality matters more than algorithm choice.\n\n**The Golden Rule:** A mediocre algorithm with great data beats a brilliant algorithm with bad data. Google's translation team proved this — a simple statistical model with 10x more data crushed a sophisticated linguistic model.\n\n**Data Audit Exercise:**\nTake any CSV dataset and check:\n1. **Completeness:** What % of fields have missing values?\n2. **Consistency:** Are dates DD/MM or MM/DD? Mixed formats?\n3. **Accuracy:** Are there obvious errors (age=999, negative prices)?\n4. **Duplicates:** Fuzzy matching on key fields\n5. **Labels:** If categorical, are labels consistent? ('fraud', 'FRAUD', 'suspicious' = mess)\n\n**Real-World Case:** An insurance company had 10 years of claims data. Sounds great, but: 30% missing agent codes, inconsistent date formats, duplicate records. Data cleaning took 3 months. Model training took 2 weeks. After cleaning, even simple logistic regression hit 87% accuracy.\n\n**Structured vs Unstructured:**\n- Structured (SQL tables) → Pandas → scikit-learn pipeline\n- Unstructured (images, text) → preprocessing → specialized models (CNN, NLP)\n- 85% of enterprise data is unstructured — this is the big opportunity\n\n**Trainer Note:** Show real messy data in training — Excel sheets with missing values, typos, duplicates. Ask: 'Would you trust a model trained on this?' The reaction teaches the lesson instantly." },
      { week: "Week 10", tasks: ["Feature engineering deep dive", "Build features from raw data (Uber ride time example)", "Apply feature engineering to test flakiness prediction"],
        content: "**Goal:** Learn the highest-impact ML skill — transforming raw data into meaningful features.\n\n**Feature Engineering = Data Refactoring:**\n```python\n# RAW DATA (not useful directly):\n# order_time, latitude, longitude, product_id\n\n# ENGINEERED FEATURES (meaningful!):\nhour_of_day = order_time.hour           # From timestamp\nis_weekend = order_time.weekday() >= 5  # From timestamp\nis_rush_hour = hour in [8,9,17,18,19]   # Domain knowledge\nneighborhood = geocode(lat, lng)         # From coordinates\navg_order_value = historical_lookup(product_id)  # From history\n```\n\n**Uber's Result:** Raw 4 features → ±12 min accuracy. After engineering 15 features → ±4 min accuracy. Same algorithm — only features changed!\n\n**Test Flakiness Prediction (Your Domain!):**\nPredict which tests will be flaky BEFORE running them:\n- From code: `lines_of_code, num_assertions, uses_sleep, has_network_call`\n- From history: `pass_rate_last_30_runs, flip_count, duration_variance`\n- From context: `time_since_last_code_change, UI_vs_API_test`\n- Interaction: `sleep_count × network_calls` (compound risk)\n\nModel predicted flaky tests with 85% accuracy. The features YOUR testing experience suggested were the top predictors.\n\n**Key Insight:** Domain expertise > algorithm knowledge for feature engineering. Your 10+ years of experience is your superpower here." },
      { week: "Week 11", tasks: ["Image data preprocessing (leverage your background!)", "K-Means → CNN progression for image classification", "Hands-on: transfer learning with ResNet"],
        content: "**Goal:** Bridge your image processing experience to deep learning.\n\n**Your Progression Path:**\n1. **K-Means on images** (you know this!): cluster pixels → segment foreground/background\n2. **Feature extraction:** histogram of colors, edge detection, texture patterns\n3. **Traditional ML:** extract features manually → feed to SVM/Random Forest\n4. **CNN approach:** neural network LEARNS features automatically\n5. **Transfer learning:** use pre-trained model (ResNet) — only train final layer\n\n**Image Preprocessing Pipeline:**\n```python\nfrom torchvision import transforms\n\ntransform = transforms.Compose([\n    transforms.Resize((224, 224)),       # Standardize size\n    transforms.RandomHorizontalFlip(),    # Augmentation\n    transforms.RandomRotation(10),        # More augmentation\n    transforms.ToTensor(),                # Convert to tensor\n    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet stats\n                         [0.229, 0.224, 0.225])\n])\n```\n\n**Transfer Learning Exercise:**\n1. Load ResNet-50 pre-trained on ImageNet (1.2M images)\n2. Replace final classification layer for YOUR categories\n3. Freeze base layers (keep learned features)\n4. Train only the final layer on your data (even 1000 images works!)\n5. Result: 90%+ accuracy without training from scratch\n\n**Real-World:** Tata Steel used ResNet-50 transfer learning for weld defect classification. 10K training images → 94% accuracy. Deployed as Spring Boot REST API — 0.5 seconds per image.\n\n**Trainer Tip:** Show what CNN layers 'see': edge detection in early layers, textures in middle, full objects in deep layers. It's like watching a baby learn to see." },
      { week: "Week 12", tasks: ["Model lifecycle management & MLOps basics", "Set up MLflow for experiment tracking", "Compare model versioning to Git versioning"],
        content: "**Goal:** Learn why models in production need the same discipline as code in production.\n\n**The Problem — Concept Drift:**\nZomato trained a restaurant rating model in 2022 on pre-COVID dining patterns. By 2023, accuracy dropped 15% because dining habits changed. Without monitoring, nobody noticed for months.\n\n**MLOps = DevOps for ML:**\n| DevOps | MLOps |\n|---|---|\n| Git (code versioning) | DVC (data versioning) + MLflow (model versioning) |\n| Jenkins (CI/CD) | Airflow/Kubeflow (training pipelines) |\n| Docker (deployment) | ONNX/TorchServe (model serving) |\n| Prometheus (monitoring) | Model accuracy dashboards |\n| Rollback (bad deploy) | Model rollback (bad retrain) |\n\n**MLflow Setup Exercise:**\n1. Install MLflow: `pip install mlflow`\n2. Log an experiment: parameters, metrics, model artifact\n3. Compare 3 model runs in the MLflow UI\n4. Register the best model in the Model Registry\n5. Concept: 'Staging' → 'Production' model promotion (like CI/CD!)\n\n**Key Design Pattern:**\n```\nData change → Trigger retraining → Validate → A/B test → Deploy → Monitor\n(Same as: Code change → Build → Test → Stage → Deploy → Monitor)\n```\n\n**Interview Answer Template:**\n'I'd treat the model like a microservice: version control with MLflow, automated retraining triggered by accuracy degradation, A/B testing before full rollout, monitoring dashboards for drift detection, and automated rollback if the new model underperforms.'" },
      { week: "Week 13–14", tasks: ["Evaluation metrics mastery (precision, recall, F1)", "Build: fraud detection model with proper metrics", "Practice: choose right metric for 5 different scenarios"],
        content: "**Goal:** Master evaluation metrics — the #1 interview topic after 'explain overfitting'.\n\n**The Testing Analogy:**\n- **Accuracy** = '% tests passed' (misleading if tests are trivial)\n- **Precision** = 'When a test fails, is it a real bug?' (false alarm rate)\n- **Recall** = 'Did tests catch ALL the bugs?' (miss rate)\n- **F1** = Balance between precision and recall\n\n**When to Optimize What:**\n- **Cancer screening:** Optimize RECALL (missing cancer = catastrophic)\n- **Email spam filter:** Balance both (spam in inbox = annoying, legit in spam = bad)\n- **Fraud detection:** Optimize recall first, then precision\n- **Product recommendations:** Optimize PRECISION (bad recs = annoyed user)\n\n**Fraud Detection Project:**\n1. Dataset: Credit card fraud (Kaggle) — 0.1% fraud rate\n2. Accuracy trap: model that says 'never fraud' gets 99.9% accuracy!\n3. Use Precision/Recall/F1 instead\n4. Build Random Forest with class_weight='balanced'\n5. Plot ROC curve, find optimal threshold\n6. Calculate business impact: caught_fraud × $500 - false_alarms × $2\n\n**Result:** At 95% recall and 80% precision: catch 950/1000 frauds (saving $475K) with 237 false alarms (costing $474). Net benefit: $474,526/day.\n\n**5 Scenario Practice:**\nFor each scenario, decide: Precision or Recall?\n1. Self-driving car obstacle detection → ?\n2. Job application screening → ?\n3. Content moderation → ?\n4. Medical diagnosis → ?\n5. Customer churn prediction → ?" },
      { week: "Week 15–16", tasks: ["Capstone: Image classification project (product defects)", "Deploy model as REST API (Spring Boot + DJL)", "Write up project for portfolio with architecture diagram"],
        content: "**Goal:** Complete a production-grade CV project and deploy it with Java.\n\n**Project: Product Defect Detection**\n\n**Architecture:**\n```\nCamera → Image API → Preprocessing → CNN Model → Classification → Alert System\n                         ↓                  ↓\n                    Spring Boot          DJL (Deep Java Library)\n```\n\n**Step-by-Step:**\n1. **Data:** Collect/download 1000+ product images per category (good/defective)\n2. **Preprocess:** Resize 224×224, normalize, augment (flip, rotate, brightness)\n3. **Model:** ResNet-50 with transfer learning, 2-class output (good/defective)\n4. **Train:** 80/20 split, 20 epochs, learning rate scheduling\n5. **Evaluate:** Confusion matrix, precision per class, Grad-CAM for explainability\n6. **Deploy with DJL (Deep Java Library):**\n```java\n@RestController\npublic class DefectController {\n    private final Predictor<Image, Classifications> predictor;\n    \n    @PostMapping(\"/analyze\")\n    public DefectResult analyze(@RequestBody MultipartFile image) {\n        Image img = ImageFactory.getInstance().fromInputStream(image.getInputStream());\n        Classifications result = predictor.predict(img);\n        return new DefectResult(result.best().getClassName(), result.best().getProbability());\n    }\n}\n```\n\n**Portfolio Write-up Must Include:**\n- Problem statement & business impact\n- Architecture diagram (draw.io or Mermaid)\n- Data pipeline & preprocessing steps\n- Model selection rationale\n- Evaluation metrics with confusion matrix\n- Deployment architecture\n- Lessons learned & future improvements\n\n**Milestone Check:** You now have 2 portfolio projects (K-Means segmentation + Image classification), deployed ML models, and hands-on experience with the full pipeline." },
    ],
    deliverables: ["Feature engineering notebook", "Image classification model (transfer learning)", "Deployed ML API (Spring Boot)", "MLflow experiment tracking setup", "Portfolio write-up with architecture diagrams"],
    skills: ["Feature engineering", "Image preprocessing", "Transfer learning", "MLOps basics", "Model evaluation", "DJL (Deep Java Library)"],
    interviewReady: ["How do you handle missing data in production?", "Explain feature engineering with a real example", "Precision vs Recall — when do you optimize which?", "Design an MLOps pipeline for continuous model improvement"],
  },
  {
    month: "Month 5–6",
    title: "Generative AI & LLM Mastery",
    icon: Zap,
    color: "hsl(210 80% 55%)",
    focus: "Master LLMs, prompt engineering, RAG architecture, and embeddings",
    modules: ["Module 5: Generative AI & LLMs"],
    weeklyPlan: [
      { week: "Week 17", tasks: ["How LLMs work — Transformer architecture simplified", "Token economics: understand pricing and optimization", "Hands-on: OpenAI API / HuggingFace API calls from Java"],
        content: "**Goal:** Understand how LLMs actually work — not the math, but the engineering.\n\n**LLMs in One Sentence:** Giant pattern-matching machines that predict the next word based on trillions of training tokens.\n\n**Transformer Architecture (Java Analogy):**\n- **Self-attention** = like a `HashMap` lookup: for each word, find which other words are 'relevant'\n- **Query-Key-Value** = like search: Query='what am I looking for?', Key='what do I contain?', Value='what info do I provide?'\n- **Multiple heads** = like running multiple regex patterns in parallel\n- **Layer stacking** = like middleware chain in Spring\n\n**Token Economics:**\n- GPT-4: ~$0.03/1K input tokens, ~$0.06/1K output tokens\n- 1 token ≈ 4 characters (English)\n- A 500-word prompt ≈ 375 tokens ≈ $0.01\n- Cost optimization: shorter prompts, caching, smaller models for simple tasks\n\n**Hands-on: First API Call from Java**\n```java\nWebClient client = WebClient.builder()\n    .baseUrl(\"https://api.openai.com\")\n    .defaultHeader(\"Authorization\", \"Bearer \" + apiKey)\n    .build();\n\nvar response = client.post().uri(\"/v1/chat/completions\")\n    .bodyValue(Map.of(\n        \"model\", \"gpt-4\",\n        \"messages\", List.of(Map.of(\"role\", \"user\", \"content\", \"Explain K-Means in one paragraph\"))\n    ))\n    .retrieve().bodyToMono(String.class).block();\n```\n\n**Key Limitation:** Context window = maximum 'working memory'. GPT-4: 128K tokens. Design around this constraint." },
      { week: "Week 18", tasks: ["Prompt engineering mastery — 3-version methodology", "Build: prompt template library for your domain", "Practice: generate test cases with engineered prompts"],
        content: "**Goal:** Treat prompts as engineering artifacts — version, test, and iterate.\n\n**The 3-Version Methodology:**\n\n**V1 (Naive):** `'Summarize this document'`\n→ Generic, misses key points\n\n**V2 (Structured):** `'Summarize this consulting report for a C-suite exec. Focus on: financial impact, risks, actions. Use bullets. Max 300 words.'`\n→ Much better structure\n\n**V3 (Expert):** `'You are a senior McKinsey consultant. Write an executive brief for a CEO board meeting. Structure: 1) Key Findings (3 bullets), 2) Financial Impact (numbers), 3) Top 3 Risks, 4) Actions with timeline. Tone: confident, data-driven. Max 300 words.'`\n→ Production-quality output\n\n**Prompt Engineering Formula:**\n`Role + Context + Task + Format + Constraints + Examples`\n\n**Test Case Generation Prompt:**\n```\nRole: Senior SDET with 10 years Selenium experience\nContext: Page Object Model, TestNG framework, explicit waits only\nExamples: [include 2-3 of YOUR test cases]\nTask: Generate test cases for [user story]\nFormat: test method name, steps, expected results, test data\nConstraints: Follow naming: test_[Feature]_[Scenario]_[Expected]\n```\n\n**Build: Prompt Template Library**\nCreate reusable templates for:\n1. Test case generation from user stories\n2. Code review comments generation\n3. Bug report summarization\n4. API documentation generation\n5. Interview question generation per topic\n\n**Metric:** With engineered prompts, 70% of generated test cases are usable (vs 10% with naive prompts)." },
      { week: "Week 19", tasks: ["Embeddings & vector databases deep dive", "Set up pgvector or Pinecone", "Build: semantic search for your test documentation"],
        content: "**Goal:** Understand how machines capture meaning — the foundation for RAG.\n\n**Embeddings = Semantic HashCodes:**\nLike Java hashCodes, but instead of uniqueness, embeddings capture **similarity**.\n- 'King' and 'Queen' → close in vector space (similar meaning)\n- 'King' - 'Man' + 'Woman' ≈ 'Queen' (vector arithmetic captures relationships!)\n\n**Why This Matters:**\nKeyword search for 'cold drink' misses 'soda', 'Coca-Cola', 'carbonated beverage'. Embedding search finds them all because they're semantically close.\n\n**pgvector Setup (PostgreSQL — your comfort zone!):**\n```sql\nCREATE EXTENSION vector;\n\nCREATE TABLE documents (\n    id SERIAL PRIMARY KEY,\n    content TEXT,\n    embedding vector(384)  -- 384-dimensional vector\n);\n\n-- Semantic search!\nSELECT content, 1 - (embedding <=> query_embedding) AS similarity\nFROM documents\nORDER BY embedding <=> query_embedding\nLIMIT 5;\n```\n\n**Project: Test Documentation Search**\n1. Collect all test case descriptions (5000 tests)\n2. Embed each using `sentence-transformers` model\n3. Store in pgvector\n4. Search: 'login validation' → finds 'authentication test', 'sign-in verification'\n5. Add similarity threshold: cosine > 0.85 = 'very similar'\n6. Alert: 'This test might duplicate TestCase#3421 (91% similar)'\n\n**Result:** Reduced duplicate test cases by 20%. New SDETs find relevant tests instantly." },
      { week: "Week 20", tasks: ["RAG architecture end-to-end", "Build: RAG pipeline with document chunking + retrieval", "Test: compare RAG answers vs vanilla LLM answers"],
        content: "**Goal:** Master THE enterprise AI pattern of 2024-2025.\n\n**RAG = Dependency Injection for AI:**\nInstead of hardcoding knowledge into the model (retraining), you INJECT relevant context at runtime.\n\n```java\n// WITHOUT RAG (hardcoded knowledge):\nString answer = llm.ask(\"What is our refund policy?\");\n// → \"I don't know your policy\" ❌\n\n// WITH RAG (injected context):\nList<String> docs = vectorDB.search(\"refund policy\", topK=5);\nString prompt = \"Based on these documents:\\n\" + docs + \"\\nAnswer: What is our refund policy?\";\nString answer = llm.ask(prompt);\n// → \"Per policy v3.2, refunds within 30 days...\" ✅\n```\n\n**RAG Pipeline (Build This!):**\n1. **Ingest:** Parse documents → chunk into 500-token segments (with 50-token overlap)\n2. **Embed:** Convert each chunk to vector using sentence-transformers\n3. **Store:** Insert into pgvector\n4. **Retrieve:** On question → embed query → find top 5 similar chunks\n5. **Generate:** Send chunks + question to LLM with 'answer from these docs only'\n6. **Validate:** If no chunk similarity > 0.7 → say 'I don't have this information'\n\n**Comparison Test:**\nAsk 10 questions about your documentation:\n- Without RAG: LLM makes up answers (hallucination)\n- With RAG: LLM answers from YOUR data with citations\n\n**Cognizant Case Study:** 300K employees × 40 min/day saved searching = massive ROI. RAG chatbot cost: $50K/month. ROI: 100x." },
      { week: "Week 21–22", tasks: ["Hallucination management strategies", "Build guardrails: confidence scoring, fact-checking", "Compare with flaky test management frameworks"],
        content: "**Goal:** Handle AI's biggest reliability challenge using your testing expertise.\n\n**Hallucinations = Flaky Tests for AI:**\n| Flaky Tests | AI Hallucinations |\n|---|---|\n| Pass/fail inconsistently | Correct/incorrect inconsistently |\n| Hardcoded sleep → timing issues | No grounding → fabricated facts |\n| Retry logic | Regenerate + compare |\n| Quarantine flaky tests | Flag low-confidence answers |\n| Root cause analysis | Analyze which topics hallucinate |\n| Stable environment | RAG (ground in verified data) |\n\n**Guardrails to Build:**\n1. **Confidence scoring:** If retrieval similarity < 0.7 → 'I don't have info on this'\n2. **Fact-checking:** Cross-reference generated answer against source documents\n3. **Consistency check:** Generate 3 times, compare — if answers diverge, flag for review\n4. **Topic boundary:** Classify response topic → reject if outside allowed domain\n5. **PII detection:** Scan response for personal data leakage\n\n**Real-World Disaster:** A US lawyer submitted 6 ChatGPT-generated case citations to court. None existed. Lawyer was sanctioned. Fix: RAG + citation verification.\n\n**Hallucination Rate Dashboard:**\n- Track hallucination rate over time (like flaky test rate)\n- Break down by topic (some topics hallucinate more)\n- Set threshold: >5% hallucination rate → trigger investigation\n- Weekly review of flagged responses\n\n**Trainer Insight:** Ask testers: 'How do you handle flaky tests?' Then say: 'Now apply the same thinking to AI.' The parallel clicks instantly." },
      { week: "Week 23–24", tasks: ["Capstone: Enterprise RAG chatbot (your test documentation)", "Full pipeline: ingest → embed → retrieve → generate → validate", "Deploy with Spring Boot WebSocket endpoint"],
        content: "**Goal:** Build and deploy a production-grade RAG chatbot — your flagship project.\n\n**Full Architecture:**\n```\n[Documents] → Parser → Chunker → Embedder → [pgvector DB]\n                                                    ↓\n[User Question] → Embedder → Vector Search → Top 5 chunks\n                                                    ↓\n                            [LLM] ← Prompt (chunks + question + instructions)\n                                                    ↓\n                            Guardrails → Response + Citations → [User]\n```\n\n**Implementation Checklist:**\n- [ ] Document parser (PDF, Markdown, HTML)\n- [ ] Chunking with overlap (500 tokens, 50 overlap)\n- [ ] Embedding with sentence-transformers\n- [ ] pgvector storage and search\n- [ ] Prompt template with citation instructions\n- [ ] Confidence scoring and 'I don't know' fallback\n- [ ] Spring Boot WebSocket endpoint for real-time chat\n- [ ] React chat UI with source citations\n- [ ] Logging: every query + response + sources\n- [ ] Accuracy testing: 50 known Q&A pairs\n\n**Deploy:**\n- Spring Boot backend with WebSocket\n- React frontend with chat interface\n- PostgreSQL + pgvector for embeddings\n- OpenAI/HuggingFace for LLM calls\n\n**Portfolio Documentation:**\n- Architecture diagram (Mermaid/draw.io)\n- Data flow from upload → response\n- Guardrails explanation\n- Accuracy metrics (test with 50 known Q&A)\n- Cost analysis per query\n- Video demo (3 minutes)\n\n**Milestone Check:** You now have 3 major projects, LLM expertise, and RAG architecture knowledge." },
    ],
    deliverables: ["Prompt engineering template library", "Semantic search system with vector DB", "RAG chatbot for documentation Q&A", "Hallucination management framework document", "Deployed chatbot with Spring Boot backend"],
    skills: ["LLM APIs (OpenAI, HuggingFace)", "Prompt engineering", "Embeddings", "Vector databases", "RAG architecture", "Guardrails & validation"],
    interviewReady: ["How do Transformers and attention mechanisms work?", "Design a RAG system for enterprise knowledge management", "How do you handle AI hallucinations in production?", "Compare embeddings vs keyword search — when to use which?"],
  },
  {
    month: "Month 7–8",
    title: "AI in Java Full Stack & Testing",
    icon: Rocket,
    color: "hsl(280 65% 55%)",
    focus: "Integrate AI into your Java stack and revolutionize test automation",
    modules: ["Module 6: AI in Java Full Stack", "Module 7: AI for Test Automation"],
    weeklyPlan: [
      { week: "Week 25–26", tasks: ["AI system architecture patterns for enterprise", "Build: Spring Boot + AI API microservice", "Implement circuit breaker & retry patterns for AI calls"],
        content: "**Goal:** Add 'intelligence layers' to your existing Java enterprise stack.\n\n**AI System Architecture Pattern:**\n```\n[Client] → [API Gateway] → [Business Service] → [AI Service Layer]\n                                                       ↓\n                                    [AI Provider (OpenAI/HuggingFace)]\n                                                       ↓\n                                    [Response Validation] → [Cache]\n```\n\n**Spring Boot AI Microservice:**\n```java\n@Service\npublic class AIService {\n    @Retryable(value = RetryableException.class, maxAttempts = 3)\n    @CircuitBreaker(name = \"aiService\", fallbackMethod = \"fallback\")\n    public AIResponse analyze(String input) {\n        var response = webClient.post()\n            .uri(\"/v1/chat/completions\")\n            .bodyValue(new AIRequest(input, \"gpt-4\", 0.3))\n            .retrieve()\n            .bodyToMono(AIResponse.class)\n            .timeout(Duration.ofSeconds(10))\n            .block();\n        trackUsage(response.getTokensUsed());\n        return response;\n    }\n    \n    private AIResponse fallback(String input, Throwable t) {\n        return templateEngine.generate(input); // Rule-based fallback\n    }\n}\n```\n\n**Key Patterns:**\n1. **Circuit Breaker:** 5 failures in 1 min → open circuit → use fallback\n2. **Retry with backoff:** Handle rate limits gracefully\n3. **Timeout:** 10-second hard limit (AI can be slow)\n4. **Cost tracking:** Log tokens per request, set daily budget caps\n5. **Response caching:** Same input → cached response (save money)\n\n**Trainer Note:** When Java developers see familiar @Service, @Retryable, WebClient patterns, AI integration stops being scary." },
      { week: "Week 27–28", tasks: ["Microservices decomposition for AI systems", "Error handling for non-deterministic AI outputs", "Build: multi-model architecture (ML + LLM combined)"],
        content: "**Goal:** Apply microservice patterns to AI systems with AI-specific considerations.\n\n**Decomposition Example — AI Recruitment Platform:**\n- **Resume Parser Service:** NLP model → structured JSON (scales independently)\n- **Job Matcher Service:** embedding similarity (GPU-intensive, auto-scale)\n- **Interview Scheduler:** rule-based optimization (NO AI needed!)\n- **Candidate Scorer:** gradient boosting model (lightweight, CPU)\n- **API Gateway:** routes, auth, rate limiting\n- **Event Bus (Kafka):** async communication between services\n\n**AI Error Handling — The Critical Difference:**\nTraditional: validate status codes. AI: validate MEANING.\n```java\nif (aiResponse.getConfidence() < 0.7) {\n    log.warn(\"Low confidence for: {}\", input);\n    return fallback(input);\n}\nif (containsCompetitorMention(aiResponse.getText())) {\n    log.error(\"Content violation\");\n    return fallback(input);\n}\nif (!jsonSchemaValid(aiResponse)) {\n    return retry(input); // Garbled output, try again\n}\n```\n\n**Multi-Model Architecture:**\nCombine ML + LLM for better results:\n1. ML model: classifies customer intent (fast, cheap)\n2. If intent = 'complex query' → route to LLM (powerful, expensive)\n3. If intent = 'simple FAQ' → use template response (free)\n\nResult: 70% requests handled without LLM → 70% cost savings.\n\n**Key Insight:** Not everything needs AI. Keep rule-based services simple. The best architecture uses AI WHERE it adds value, not everywhere." },
      { week: "Week 29–30", tasks: ["AI-generated test cases — production-quality pipeline", "Self-healing test locators with Healenium", "Build: AI test case generator from user stories"],
        content: "**Goal:** Revolutionize your test automation with AI.\n\n**AI Test Generation Pipeline:**\n```\nUser Story → Structured Prompt → LLM → Generated Tests → Validation → Human Review\n                  ↓\n    [Framework context + naming conventions + 3 example tests]\n```\n\n**Accenture Case Study:** 500 user stories → 2000+ test cases. Manual: 4 weeks. AI-assisted: 1 week. 85% of AI-generated tests passed review without changes.\n\n**Self-Healing Tests with Healenium:**\nWhen a locator breaks, Healenium:\n1. Finds element using ML-based DOM similarity\n2. Heals the locator in real-time\n3. Logs the healing for review\n4. Optionally auto-updates test code\n\n**Result:** Locator failures reduced 75%. Maintenance time: 3 days → 4 hours/sprint.\n\n**Build: AI Test Case Generator**\n1. Input: Jira user story + acceptance criteria\n2. Prompt with: test framework, naming convention, 3 example tests\n3. Generate: happy path, negative, boundary, edge case, accessibility\n4. Validate: compile check, naming convention, assertion presence\n5. Output: structured test scenarios\n6. Integration: Slack bot that generates tests from Jira tickets\n\n**Metric to Track:** Time saved per sprint, % of tests needing no edits, coverage improvement.\n\n**Trainer Tip:** Demo this live — take a user story from the audience, generate tests, review together. The 'wow' moment is guaranteed." },
      { week: "Week 31–32", tasks: ["Validating AI outputs — multi-layered testing strategy", "Testing AI systems themselves (model quality assurance)", "Build: AI validation framework for your projects"],
        content: "**Goal:** Master the art of testing non-deterministic systems.\n\n**The Challenge:** You can't use `assertEquals()` on AI output. Same input → different output each time.\n\n**Multi-Layered Validation Strategy:**\n1. **Structural:** Response is valid JSON, within length limits, proper format\n2. **Semantic:** Embedding similarity between response and expected topic > 0.8\n3. **Safety:** No PII, no competitor mentions, no policy violations\n4. **Consistency:** Run same input 10 times, variance within acceptable range\n5. **Human eval:** Sample 5% for manual quality review\n\n**Testing AI Models (Quality Assurance for ML):**\n- **Data tests:** Training data quality, distribution, bias checks\n- **Model tests:** Accuracy on holdout set, performance on edge cases\n- **Integration tests:** API response format, latency, error handling\n- **Regression tests:** New model version vs previous on golden dataset\n- **Fairness tests:** Model performance across demographic groups\n\n**Build: Validation Framework**\n```java\npublic class AIResponseValidator {\n    public ValidationResult validate(AIResponse response) {\n        return ValidationResult.builder()\n            .structural(checkStructure(response))\n            .semantic(checkRelevance(response, expectedTopic))\n            .safety(checkSafety(response))\n            .confidence(response.getConfidence() > 0.7)\n            .build();\n    }\n}\n```\n\n**Golden Rule:** Treat AI validation like a test pyramid: many fast structural checks, fewer semantic checks, rare but thorough human reviews.\n\n**Interview Answer:** 'I test AI outputs in layers: structural validation (format, length), semantic validation (embedding similarity), safety checks (PII, policy), and consistency testing (variance across runs). Plus 5% human review as a safety net.'" },
    ],
    deliverables: ["Spring Boot AI microservice (production-ready)", "AI test case generator integrated with Jira", "Self-healing test suite POC", "AI output validation framework", "Architecture decision record (ADR) for AI integration"],
    skills: ["Spring Boot + AI APIs", "AI error handling", "AI test generation", "Self-healing locators", "AI system testing", "Circuit breaker patterns"],
    interviewReady: ["Architect a Spring Boot app that uses multiple AI APIs", "How is error handling different for AI vs traditional services?", "What is a self-healing test and how does AI enable it?", "How do you validate non-deterministic AI outputs?"],
  },
  {
    month: "Month 9–10",
    title: "Capstone Projects & Portfolio",
    icon: Award,
    color: "hsl(15 80% 55%)",
    focus: "Build impressive portfolio projects and prepare for AI roles",
    modules: ["Module 8: Capstone Projects"],
    weeklyPlan: [
      { week: "Week 33–34", tasks: ["AI Resume Screener — NLP pipeline end-to-end", "Architecture diagram, data flow, bias mitigation", "Deploy and document for portfolio"],
        content: "**Goal:** Build a production-grade AI Resume Screener.\n\n**Architecture:**\n```\n[Resume PDF] → OCR/Parser → NLP Entity Extraction → Feature Vector\n                                                         ↓\n[Job Description] → NLP → Required Skills Vector → Match Score → Rank\n                                                         ↓\n                                              Bias Check → Final Score → API\n```\n\n**Implementation Steps:**\n1. **Parse:** Extract text from PDF/DOCX resumes\n2. **NLP extraction:** Skills, experience years, education, certifications\n3. **Feature vector:** Numerical representation of candidate profile\n4. **Job matching:** Cosine similarity between resume vector and JD vector\n5. **Ranking:** Score 0-100 with breakdown (skills: 40%, experience: 30%, education: 20%, extras: 10%)\n6. **Bias mitigation:** Remove name, gender, age, photo from feature set\n7. **Explainability:** 'Scored 85 because: matched 8/10 required skills, 3 years relevant experience'\n\n**Unilever Case Study:** 1.8M applications/year. AI screening: initial filter in minutes, shortlisting in hours. Time-to-hire reduced 75%. But they removed name/gender/age to prevent bias.\n\n**Critical Interview Point:** ALWAYS mention bias mitigation proactively. It shows maturity that distinguishes senior engineers from juniors.\n\n**Deliverable:** Deployed API + React UI showing ranked candidates with explanations + architecture diagram + ethics documentation." },
      { week: "Week 35–36", tasks: ["AI Test Case Generator — LLM-powered tool", "Integration with test frameworks (Selenium/Playwright)", "Collect metrics: time saved, quality comparison"],
        content: "**Goal:** Build the project that directly showcases your unique Java + Testing + AI combination.\n\n**Architecture:**\n```\n[Jira User Story] → API → Prompt Builder → LLM → Test Cases → Validator → Output\n         ↓                      ↓                                  ↓\n  Acceptance Criteria    [Framework Context]              [Compile Check]\n                         [Naming Conventions]             [Coverage Analysis]\n                         [3 Example Tests]                [Assertion Check]\n```\n\n**What Makes This Special:**\n- You understand BOTH sides: testing expertise + AI capability\n- You can judge quality of generated tests (most AI tools can't)\n- Your prompt engineering is informed by real testing patterns\n\n**Metrics to Collect:**\n1. Time: manual test writing vs AI-generated + review\n2. Quality: % of tests passing human review without edits\n3. Coverage: test scenarios found by AI that humans missed\n4. Consistency: same story → similar quality across runs\n\n**Expected Results:**\n- 60% time reduction on test creation\n- 85% of generated tests usable (with minor edits)\n- AI finds 15% more edge cases than manual brainstorming\n\n**Integration Points:**\n- Slack bot: `/generate-tests JIRA-1234`\n- CI/CD: auto-generate tests for new user stories\n- Dashboard: track generation quality over time\n\n**This is YOUR killer project.** It combines Java, testing, AI, and is immediately useful in any company." },
      { week: "Week 37–38", tasks: ["Enterprise AI Chatbot — RAG with domain knowledge", "Full stack: React UI + Spring Boot + pgvector + LLM", "Add guardrails, citations, confidence scoring"],
        content: "**Goal:** Build the #1 enterprise AI use case — a RAG chatbot.\n\n**Full Stack Architecture:**\n```\n[React Chat UI] ←WebSocket→ [Spring Boot] → [Embedding Service]\n                                    ↓                    ↓\n                              [pgvector DB] ←→ [Document Store]\n                                    ↓\n                              [LLM API] → [Guardrails] → [Response]\n```\n\n**Complete Feature List:**\n- Document upload (PDF, Markdown, HTML)\n- Automatic chunking and embedding\n- Real-time chat via WebSocket\n- Source citations in responses\n- Confidence scoring per response\n- 'I don't know' for out-of-scope questions\n- Chat history and context management\n- Admin dashboard: usage stats, query analytics\n\n**Guardrails Checklist:**\n- [ ] Topic boundary (only answer from uploaded documents)\n- [ ] PII detection (scan responses for personal data)\n- [ ] Confidence threshold (< 0.7 → 'I don't have this info')\n- [ ] Source citation requirement\n- [ ] Response length limit\n- [ ] Rate limiting per user\n\n**Walmart Case Study:** 2.1M employees, 50K documents. 80% of routine queries answered instantly. HR tickets reduced 40%.\n\n**Demo Strategy:** Upload a document → immediately ask questions about it → show cited sources. This 'wow factor' is unbeatable in interviews and training sessions.\n\n**Deliverable:** Deployed full-stack app with demo video (3 min walkthrough)." },
      { week: "Week 39–40", tasks: ["Image Defect Detection — computer vision project", "Transfer learning + Grad-CAM explainability", "Production deployment with monitoring dashboard"],
        content: "**Goal:** Leverage your image processing background for a cutting-edge CV project.\n\n**Architecture:**\n```\n[Camera/Upload] → [Preprocessing] → [CNN Model] → [Classification]\n                        ↓                              ↓\n                  Resize/Normalize            [Grad-CAM Heatmap]\n                  Augmentation                      ↓\n                                        [Defect Type + Location + Confidence]\n                                                    ↓\n                                        [Dashboard] → [Alert System]\n```\n\n**Your Unique Advantage:**\n- K-Means image segmentation → CNN classification = natural progression\n- Image preprocessing knowledge transfers directly\n- You understand pixel-level operations\n\n**Implementation:**\n1. **Data:** 1000+ images per defect type + 5000 'good' images\n2. **Model:** ResNet-50 transfer learning → multi-class classifier\n3. **Explainability:** Grad-CAM highlights WHERE the model sees defects\n4. **Deployment:** ONNX model via Spring Boot REST API\n5. **Monitoring:** Daily accuracy tracking, drift detection\n\n**Tesla Case Study:** 99.5% defect detection (vs 95% human). 0.2 sec/image. New defect types added with just 100 labeled examples.\n\n**Monitoring Dashboard (React):**\n- Total inspections today\n- Defect rate trend (daily/weekly)\n- Model confidence distribution\n- False positive/negative samples for review\n- Accuracy comparison: current vs baseline model\n\n**Portfolio Impact:** This project shows: computer vision, transfer learning, explainable AI, production deployment, and monitoring — all in one.\n\n**Milestone Check:** You now have 4 complete capstone projects covering NLP, LLMs, RAG, and Computer Vision." },
    ],
    deliverables: ["4 portfolio-ready AI projects with documentation", "GitHub repos with READMEs, architecture diagrams", "Live demos for each project", "Video walkthroughs (2-3 min each)", "Portfolio website showcasing all projects"],
    skills: ["End-to-end AI project delivery", "NLP pipelines", "Computer vision", "RAG systems", "Full-stack AI applications", "Technical documentation"],
    interviewReady: ["Walk through the architecture of your AI chatbot", "How did you handle bias in the resume screener?", "What metrics did you use to evaluate the test generator?", "Explain your defect detection model's decision-making"],
  },
  {
    month: "Month 11–12",
    title: "Trainer Mode & Career Launch",
    icon: GraduationCap,
    color: "hsl(260 60% 55%)",
    focus: "Launch your AI training career and monetize your skills",
    modules: ["Module 9: Trainer Mode", "Module 10: Career & Monetization"],
    weeklyPlan: [
      { week: "Week 41–42", tasks: ["Design your 1-day AI workshop curriculum", "Prepare demo scripts with backup plans", "Practice teaching AI to non-technical audience"],
        content: "**Goal:** Create a ready-to-deliver AI workshop for corporate audiences.\n\n**1-Day Workshop Curriculum:**\n| Time | Topic | Format |\n|---|---|---|\n| 9:00-10:30 | What is AI? (with live demos) | Interactive lecture |\n| 10:45-12:00 | ML Basics with Teachable Machine | Hands-on |\n| 12:00-1:00 | Lunch Break | - |\n| 1:00-2:30 | How our industry uses AI (case studies) | Discussion |\n| 2:45-4:00 | Hands-on: Solve work tasks with AI | Workshop |\n| 4:00-5:00 | AI Career Paths + Q&A | Open discussion |\n\n**Demo Preparation (3 demos, progressive difficulty):**\n1. **Easy (100% reliable):** Text summarization — paste a long email, get 3-line summary\n2. **Medium (90% reliable):** Code generation — describe a function, AI writes it\n3. **Impressive (80% reliable):** Image analysis — upload photo, AI describes it\n\n**Recovery Plans for Each:**\n- Pre-recorded video backup\n- Second prompt ready if first fails\n- When AI fails: 'This is actually a great example of why guardrails matter!'\n\n**Teaching Rule:** Max 20-minute lecture blocks. Hands-on within first hour. Give tools they can use TOMORROW.\n\n**Google's Approach:** They trained 10K non-technical employees with zero math, maximum interaction. 96% reported 'confident understanding'. Copy this philosophy." },
      { week: "Week 43–44", tasks: ["Deliver first internal AI training (free)", "Collect testimonials and feedback", "Create executive-level AI pitch deck"],
        content: "**Goal:** Get your first real training experience and testimonials.\n\n**First Training Delivery:**\n1. Approach your manager: 'I'd like to do a 2-hour AI awareness session for the team'\n2. Keep it low-pressure: brown bag lunch, optional attendance\n3. Record it (with permission) for your portfolio\n4. Focus on making it INTERACTIVE, not perfect\n\n**Feedback Collection:**\nSurvey questions:\n1. 'I now understand what AI can do for our work' (1-5)\n2. 'I learned at least one tool I can use tomorrow' (1-5)\n3. 'I would recommend this training to colleagues' (1-5)\n4. 'What was the most valuable part?'\n5. 'What should be added or improved?'\n\n**Executive AI Pitch Deck (6 slides):**\n1. **The Opportunity:** Industry AI adoption stats + competitor examples\n2. **Our Use Cases:** 3 specific AI opportunities with ROI estimates\n3. **Quick Win:** One pilot project, 90 days, measurable outcome\n4. **Investment:** Cost, team, timeline\n5. **Risk Mitigation:** What could go wrong + how we handle it\n6. **Next Steps:** Decision needed, timeline\n\n**McKinsey Format for Each Use Case:**\nProblem → Opportunity size ($) → AI solution (1 sentence) → Timeline → ROI → Risk → Recommendation\n\n**Key Insight:** Executives decide AI budgets. If you can speak their language (ROI, risk, timeline), your AI projects get funded." },
      { week: "Week 45–46", tasks: ["Build 'AI for Testers' online course (5 modules)", "Set up on Udemy + personal site", "Launch marketing: LinkedIn articles, YouTube shorts"],
        content: "**Goal:** Create a passive income product from your unique expertise.\n\n**Course: 'AI for Automation Testers'**\n\n**Module Structure:**\n1. AI Basics for Testers (analogies to testing concepts)\n2. AI-Powered Test Generation (prompts, frameworks, validation)\n3. Self-Healing Tests (Healenium, ML-based locators)\n4. Testing AI Systems (non-deterministic validation)\n5. AI Career for Testers (paths, portfolio, interviews)\n\n**Format per Module:**\n- 3-4 video lessons (15 min each)\n- Hands-on exercise\n- Quiz\n- Downloadable resources (templates, cheat sheets)\n\n**Platform Strategy:**\n- **Udemy:** maximum reach, lower margins (₹499 intro price)\n- **Personal site:** higher margins, build your brand\n- **Corporate licensing:** highest revenue (₹25,000/team of 10)\n\n**Marketing Plan:**\n- Week 1: Publish 3 LinkedIn articles from course content\n- Week 2: Create 5 YouTube shorts (60-sec AI tips for testers)\n- Week 3: Share free intro module as lead magnet\n- Week 4: Launch with early-bird pricing\n\n**Revenue Projections:**\n- Conservative: 500 students × ₹999 = ₹5 lakhs/year\n- Optimistic: 2000 students + 5 corporate licenses = ₹15+ lakhs/year\n\n**Key: Niche positioning.** 'AI for Testers' beats 'AI for Everyone' because it speaks to a specific audience with specific problems." },
      { week: "Week 47–48", tasks: ["Finalize AI career path (engineer/trainer/product)", "Get 1 AI certification (AWS ML or Google Cloud AI)", "Update resume, LinkedIn, portfolio — start applying!"],
        content: "**Goal:** Launch your AI career with all assets in place.\n\n**Career Path Decision Matrix:**\n| Criteria | ML Engineer | AI Trainer | AI Product Builder |\n|---|---|---|---|\n| Income potential | ₹25-50L/year | ₹15-40L/year (training) | Variable (highest ceiling) |\n| Risk | Low (employment) | Medium (freelance) | High (entrepreneurship) |\n| Timeline to income | 3-6 months | 1-3 months | 6-18 months |\n| Leverages your skills | 70% | 90% | 80% |\n| Lifestyle | Corporate | Flexible | Startup grind |\n\n**Certification Strategy:**\nPick ONE:\n- **AWS ML Specialty:** Most recognized by enterprises. Study: 4-6 weeks.\n- **Google Professional ML:** Strong for cloud AI. Study: 4-6 weeks.\n- Remember: certification opens doors, but portfolio closes deals.\n\n**Resume Update:**\n- Headline: 'AI Solutions Architect | 10+ Years Java Full Stack | Corporate AI Trainer'\n- Summary: Focus on AI projects built + business impact\n- Projects section: 4 capstone projects with metrics\n- Skills: Add AI/ML tools alongside Java expertise\n\n**LinkedIn Optimization:**\n- Banner: 'Helping Engineers Transition to AI'\n- Featured: Best AI articles + project demos\n- Activity: 2 posts/week about AI insights\n\n**Final Portfolio:**\n- 6+ projects deployed and documented\n- 24+ articles published\n- 1-2 certifications\n- 3+ trainings delivered with testimonials\n- Active LinkedIn presence\n\n**🎯 You made it.** You're not just an AI engineer — you're an AI engineer who can BUILD systems AND TEACH others. That combination is rare and valuable." },
    ],
    deliverables: ["1-day AI workshop curriculum & materials", "First training delivered with testimonials", "Online course (5 modules) published", "AI certification earned", "Updated resume & portfolio for AI roles", "3 corporate training proposals sent"],
    skills: ["Curriculum design", "Live demo techniques", "Course creation", "Corporate training delivery", "AI business development", "Personal branding"],
    interviewReady: ["How would you explain neural networks to a non-technical audience?", "Design a 3-day corporate AI training curriculum", "What's your 6-month plan to transition fully into AI?", "How would you pitch an AI training program to a CTO?"],
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
  const [expandedWeeks, setExpandedWeeks] = useState<Set<string>>(new Set());

  const toggleWeek = (key: string) => {
    setExpandedWeeks((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

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
                <div className="ml-[3.25rem] space-y-2 mb-5">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Weekly Breakdown — click a week for detailed content</h4>
                  <div className="grid gap-2">
                    {phase.weeklyPlan.map((week, wi) => {
                      const weekKey = `${phaseIdx}-${wi}`;
                      const isExpanded = expandedWeeks.has(weekKey);
                      return (
                        <div key={wi} className="rounded-lg border border-border bg-card overflow-hidden">
                          <button
                            onClick={() => toggleWeek(weekKey)}
                            className="w-full flex items-center justify-between p-3 hover:bg-muted/50 transition-colors text-left"
                          >
                            <div className="flex-1 min-w-0">
                              <p className="text-xs font-semibold text-accent mb-1">{week.week}</p>
                              <ul className="space-y-0.5">
                                {week.tasks.map((task, ti) => (
                                  <li key={ti} className="flex items-start gap-2 text-sm text-foreground/85">
                                    <CheckCircle2 className="h-3.5 w-3.5 text-muted-foreground/50 shrink-0 mt-0.5" />
                                    <span>{task}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                            <div className="shrink-0 ml-3">
                              {isExpanded ? (
                                <ChevronUp className="h-4 w-4 text-muted-foreground" />
                              ) : (
                                <ChevronDown className="h-4 w-4 text-muted-foreground" />
                              )}
                            </div>
                          </button>
                          {isExpanded && week.content && (
                            <div className="px-4 pb-4 border-t border-border">
                              <div className="pt-3 prose-sm text-sm text-foreground/85 leading-relaxed whitespace-pre-line">
                                {week.content.split('\n').map((line, li) => {
                                  if (line.startsWith('**') && line.endsWith('**')) {
                                    return <h4 key={li} className="font-bold text-foreground mt-3 mb-1 text-sm">{line.replace(/\*\*/g, '')}</h4>;
                                  }
                                  if (line.startsWith('**')) {
                                    const parts = line.split('**');
                                    return (
                                      <p key={li} className="mt-2 mb-1">
                                        {parts.map((part, pi) =>
                                          pi % 2 === 1 ? <strong key={pi} className="text-foreground font-semibold">{part}</strong> : <span key={pi}>{part}</span>
                                        )}
                                      </p>
                                    );
                                  }
                                  if (line.startsWith('```')) return null;
                                  if (line.startsWith('|')) {
                                    const cells = line.split('|').filter(c => c.trim()).map(c => c.trim());
                                    if (cells.every(c => c.match(/^-+$/))) return null;
                                    return (
                                      <div key={li} className="flex gap-4 text-xs py-0.5 font-mono">
                                        {cells.map((cell, ci) => (
                                          <span key={ci} className="flex-1">{cell}</span>
                                        ))}
                                      </div>
                                    );
                                  }
                                  if (line.startsWith('- ')) {
                                    return <p key={li} className="ml-3 text-foreground/80">• {line.slice(2)}</p>;
                                  }
                                  if (line.trim() === '') return <div key={li} className="h-1" />;
                                  return <p key={li}>{line}</p>;
                                })}
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
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
