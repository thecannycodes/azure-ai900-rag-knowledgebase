# 🧠 Azure AI Knowledge Base Q&A with Live Web Grounding

> **AI-900 Exam Project** — Hybrid RAG (Retrieval-Augmented Generation) application powered by Azure AI Search, Azure AI Language, and Real-time Web Grounding.

---

## 🏗️ Architecture (Hybrid Agentic RAG)

```
                            User Question
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
        [Web Search: OFF]               [Web Search: ON]
                 │                               │
                 ▼                               │
        ┌──────────────────┐                     │
        │  Azure AI Search │                     │
        └──────────────────┘                     │
                 │                               │
          Found in docs?                         │
          ┌──────┴──────┐                        │
         YES            NO                       │
          │              │                       │
          ▼              └───────────┬───────────┘
    ┌──────────┐                     │
    │ Azure AI │                     ▼
    │ Language │             ┌───────────────┐
    │   (QA)   │             │ Live Internet │
    └──────────┘             │  Web Grounding│
          │                  └───────────────┘
          │                          │
          │                          ▼
          │                  ┌───────────────┐
          │                  │   Azure AI    │
          │                  │ Language (QA) │
          │                  └───────────────┘
          │                          │
          ▼                          ▼
    Document Answer            Web-Grounded Answer
   + Local Citations          + Clickable URLs 🌐
```

---

## ✨ Key Features

1. 📄 **Document RAG (Azure AI Search)**:
   * Upload PDFs, TXT, or Markdown documents (Lecture plans, Score reports, Syllabi, Notes).
   * Full-text semantic search via Azure AI Search.

2. 🧠 **Precision Question Answering (Azure AI Language)**:
   * Uses Azure's Question Answering API (`QuestionAnsweringClient`) to pinpoint the **exact direct answer** (`short_answer`) rather than generic summaries.
   * Smart card formatting for score reports and academic metadata.

3. 🌐 **Live Web Grounding (Internet Search)**:
   * One-click **`🌐 Web Search: ON/OFF`** toggle right in the chat input bar.
   * Auto-fallback: If a question cannot be answered from your uploaded documents, it automatically searches the live internet and grounds the response with clickable web sources.

4. 📊 **Accurate Confidence & Relevance Scores**:
   * Normalized 0–100% confidence/match ratings.
   * Only documents and web sources that *actually contributed* to the answer are cited.

---

## 🎯 AI-900 Exam Domains Demonstrated

| AI-900 Domain | Implementation in Project |
|--------------|---------------------------|
| **Knowledge Mining** | Azure AI Search (indexing, full-text retrieval) |
| **Natural Language Processing (NLP)** | Azure AI Language (Question Answering, entity extraction) |
| **Conversational AI** | Multi-turn chat, intent handling, greetings |
| **Hybrid / Agentic RAG** | Document grounding + Live Web Grounding |
| **Responsible AI** | Source attribution, transparency, verifiable URLs, no hallucination |
