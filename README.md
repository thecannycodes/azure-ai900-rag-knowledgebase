# 🧠 Azure AI Knowledge Base Q&A with Live Web Grounding

> **Microsoft Azure AI Fundamentals (AI-900 / AI-901) Capstone Project**  
> A full-stack Hybrid RAG (Retrieval-Augmented Generation) application powered 100% by genuine Azure AI services (**Azure AI Search** + **Azure AI Language**) with live web grounding and real-time temporal awareness. Zero third-party LLM dependencies, running 100% free under Azure for Students / Free Tier.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Azure AI Search](https://img.shields.io/badge/Azure%20AI-Search%20(Free%20Tier)-0078D4.svg)](https://azure.microsoft.com/en-us/products/ai-services/ai-search)
[![Azure AI Language](https://img.shields.io/badge/Azure%20AI-Language%20(Free%20F0)-5C2D91.svg)](https://azure.microsoft.com/en-us/products/ai-services/ai-language)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

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
      ┌─────────────────────┐                    │
      │   Azure AI Search   │                    │
      │  (Full-Text & BM25) │                    │
      └─────────────────────┘                    │
                 │                               │
           Found in docs?                        │
          ┌──────┴──────┐                        │
         YES            NO                       │
          │              │                       ▼
          │         Return Notice:        ┌───────────────┐
          │      "Not found in document"  │ Live Internet │
          ▼                               │ Web Grounding │
    ┌───────────────┐                     └───────────────┘
    │   Azure AI    │                            │
    │ Language (QA) │                            ▼
    └───────────────┘                     ┌───────────────┐
          │                               │   Azure AI    │
          │                               │ Language (QA) │
          │                               └───────────────┘
          │                                      │
          ▼                                      ▼
    Document Answer                        Web-Grounded Answer
   + Direct Answer Pill                   + Direct Answer Pill
   + Source Citations                     + Clickable Source URLs
```

* **Retrieval (Knowledge Mining)**: **Azure AI Search** (`SearchClient` / `SearchIndexClient`) chunks documents and retrieves the top-ranked passages.
* **Comprehension & Generation (NLP)**: **Azure AI Language** (`QuestionAnsweringClient`) reads retrieved passages to extract the precise named entity or direct answer (`short_answer`) alongside context and neural confidence scores.
* **Live Grounding**: When toggled **ON**, Wikipedia and DuckDuckGo knowledge summaries are grounded through Azure AI Language for up-to-date factual queries.
* **Temporal Awareness**: System clock queries (today's date, day of week, current time) are answered directly via real-time system clock.

---

## ☁️ How to Create the Necessary Azure Resources

You need two free services in your Microsoft Azure Portal. Both qualify for the **Free Tier** and **Azure for Students** credits (no credit card charge).

### 1. Create an Azure Resource Group
1. Sign in to the [Azure Portal](https://portal.azure.com/).
2. In the top search bar, search for **Resource groups** and click **Create**.
3. Choose:
   - **Subscription**: Your Azure subscription (e.g., *Azure for Students*).
   - **Resource Group Name**: `rg-ai900-project` (or any preferred name).
   - **Region**: Choose a region near you (e.g., `East US`, `Central India`, `West Europe`).
4. Click **Review + create** &rarr; **Create**.

---

### 2. Create Azure AI Search (Index & Retrieval)
1. In the Azure Portal search bar, search for **Azure AI Search** and click **Create**.
2. Fill in the details:
   - **Resource Group**: Select `rg-ai900-project`.
   - **Service Name**: Enter a globally unique name (e.g. `ai901-search-yourname`).
   - **Location**: Same region as your resource group (e.g. `East US`).
   - **Pricing Tier**: Click **Change Pricing Tier** and select **Free (F)** *(Free for life: up to 3 indexes, 50 MB storage)*.
3. Click **Review + create** &rarr; **Create**.
4. Once deployment finishes, go to the resource:
   - Under the **Overview** blade, copy the **URL** (e.g., `https://ai901-search-yourname.search.windows.net`). This is your `AZURE_SEARCH_ENDPOINT`.
   - In the left menu under **Settings**, click **Keys** &rarr; copy **Primary admin key**. This is your `AZURE_SEARCH_KEY`.

---

### 3. Create Azure AI Language (Question Answering & NLP)
1. In the Azure Portal search bar, search for **Language service** (or **Azure AI services | Language**) and click **Create**.
2. Under default features, select **Question Answering** (or choose the default Language features).
3. Click **Continue to create your resource**.
4. Fill in the details:
   - **Resource Group**: Select `rg-ai900-project`.
   - **Region**: Same region (e.g. `East US` or `Central India`).
   - **Name**: Enter a unique name (e.g. `ai901-lang-yourname`).
   - **Pricing Tier**: Select **Free F0** *(Free: 5,000 text records/month)*.
   - Check the Responsible AI Notice box.
5. Click **Review + create** &rarr; **Create**.
6. Once deployed, go to the resource:
   - In the left menu under **Resource Management**, click **Keys and Endpoint**.
   - Copy **Endpoint** (e.g., `https://ai901-lang-yourname.cognitiveservices.azure.com/`). This is your `AZURE_LANGUAGE_ENDPOINT`.
   - Copy **Key 1**. This is your `AZURE_LANGUAGE_KEY`.

---

## 🚀 How to Run the Project Locally

### Prerequisites
* **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.14 on macOS/Linux/Windows).
* Active Azure AI Search & Azure AI Language credentials.
* Git installed.

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/thecannycodes/azure-ai900-rag-knowledgebase.git
cd azure-ai900-rag-knowledgebase
```

---

### Step 2: Create and Activate a Python Virtual Environment

* **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

* **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

---

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

### Step 4: Configure Environment Variables
Copy the template `.env.example` file to create your own `.env` file:
```bash
cp .env.example .env
```

Open `.env` in any text editor and paste your Azure endpoints and keys:
```env
# ── Azure AI Search ──────────────────────────────
AZURE_SEARCH_ENDPOINT=https://your-search-name.search.windows.net
AZURE_SEARCH_KEY=your_search_admin_key_here
AZURE_SEARCH_INDEX=ai900-knowledge-base

# ── Azure AI Language ────────────────────────────
AZURE_LANGUAGE_ENDPOINT=https://your-language-name.cognitiveservices.azure.com/
AZURE_LANGUAGE_KEY=your_language_key_here
```

> ⚠️ **Note:** `.env` is listed in `.gitignore` and will never be committed or published.

---

### Step 5: Start the Server
```bash
python server.py
```
*(Alternatively, use `uvicorn server:app --host 127.0.0.1 --port 8000 --reload`)*.

On startup, the server automatically:
1. Connects to Azure AI Search and provisions the search index (`ai900-knowledge-base`) if not already present.
2. Connects to Azure AI Language (`QuestionAnsweringClient`).
3. Indexes the pre-loaded AI-900 study guide (`knowledge_base/azure_ai900.txt`).

---

### Step 6: Open the Web UI
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 💡 Using the Application

1. **Ask Pre-Loaded AI-900 Questions**:
   - *"What are the principles of Responsible AI?"*
   - *"What is the difference between Azure AI Services and Azure Machine Learning?"*
   - *"What is Azure AI Vision used for?"*
2. **Upload Your Own Documents**:
   - Drag & drop or click to upload **PDF, TXT, or Markdown** files (lecture plans, student lab reports, syllabus, score reports).
   - Chunks are uploaded and indexed into Azure AI Search in real time.
3. **Query Specific Academic Documents**:
   - *"What does Experiment 7 is about?"* &rarr; Extracts Aim and Objectives.
   - *"Can you provide me the procedure for experiment 7?"* &rarr; Extracts step-by-step experiment instructions.
   - *"What are the tools required?"* &rarr; Lists software/libraries.
   - *"What was my exam score?"* &rarr; Formats Certiport/Microsoft score reports with pass status and score breakdown.
4. **Live Internet Grounding Toggle**:
   - Click the **`🌐 Web Search: ON/OFF`** button in the chat input bar.
   - When **ON**, queries for general knowledge (e.g., *"Who is Satya Nadella?"*) fetch live web summaries grounded through Azure AI Language with clickable source links.
   - When **OFF**, answers are strictly constrained to your local uploaded documents.
5. **Real-Time Temporal Queries**:
   - Ask *"What is today's date?"* or *"What time is it?"* to receive instantaneous, accurate local date and time from the system clock.

---

## 📁 Project Structure

```
├── .env.example               # Template environment configuration (safe to commit)
├── .gitignore                  # Prevents secrets (.env), venv, and cache from being tracked
├── README.md                   # Project documentation and setup guide
├── requirements.txt            # Python dependencies (FastAPI, Azure SDKs, pypdf)
├── server.py                   # FastAPI backend + Azure AI Search + Azure AI Language
├── knowledge_base/
│   └── azure_ai900.txt         # Pre-loaded Azure AI Fundamentals study guide
└── static/
    └── index.html              # Responsive Microsoft Fluent-styled web interface
```

---

## 🎯 AI-900 Exam Objective Mapping

This project maps directly to the official **Microsoft Azure AI Fundamentals (AI-900)** certification domains:

| AI-900 Objective Domain | How It Is Implemented |
|---|---|
| **Describe Features of Computer Vision & NLP Workloads** | Implementation of Azure AI Language Question Answering over unstructured documents. |
| **Knowledge Mining Workloads on Azure** | Provisioning and indexing documents in Azure AI Search with full-text search. |
| **Conversational AI Workloads on Azure** | Multi-turn conversational interface with intent detection, temporal handling, and RAG. |
| **Responsible AI Principles** | Explicit source attribution, confidence scores, transparent web grounding, and hallucination prevention. |
| **Zero Third-Party LLM Dependency** | 100% compliant with student/academic subscriptions without requiring Azure OpenAI approvals. |

---

## 📄 License
This project is licensed under the MIT License - feel free to use and adapt it for your academic and learning pursuits.
