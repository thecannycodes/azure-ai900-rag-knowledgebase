"""
AI-900 Knowledge Base Q&A — RAG Backend
100% Azure AI — No third-party LLMs

Architecture:
  Indexing  : PDF/TXT  →  chunk  →  Azure AI Search (index)
  Retrieval : Question →  Azure AI Search (semantic search)  → top-k chunks
  Generation: chunks + question → Azure AI Language (Question Answering) → answer
"""

import os
import uuid
import json
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import pypdf

import re

# Azure SDKs
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
)
from azure.ai.textanalytics import TextAnalyticsClient
from azure.ai.language.questionanswering import QuestionAnsweringClient

load_dotenv()

# ---------------------------------------------------------------------------
# Azure Clients
# ---------------------------------------------------------------------------
SEARCH_ENDPOINT  = os.getenv("AZURE_SEARCH_ENDPOINT", "")
SEARCH_KEY       = os.getenv("AZURE_SEARCH_KEY", "")
INDEX_NAME       = os.getenv("AZURE_SEARCH_INDEX", "ai900-knowledge-base")

LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "")
LANGUAGE_KEY      = os.getenv("AZURE_LANGUAGE_KEY", "")

search_credential    = AzureKeyCredential(SEARCH_KEY)
language_credential  = AzureKeyCredential(LANGUAGE_KEY)

index_client  = SearchIndexClient(SEARCH_ENDPOINT, search_credential)
search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, search_credential)
lang_client   = TextAnalyticsClient(LANGUAGE_ENDPOINT, language_credential)
qa_client     = QuestionAnsweringClient(LANGUAGE_ENDPOINT, language_credential)

import urllib.request
import urllib.parse

# Document registry: { doc_id: { name, chunk_count } }
DOCS_FILE = Path("data/documents.json")
DOCS_FILE.parent.mkdir(exist_ok=True)
documents: dict[str, dict] = {}

def load_documents():
    global documents
    if DOCS_FILE.exists():
        try:
            with open(DOCS_FILE, "r", encoding="utf-8") as f:
                documents = json.load(f)
        except Exception:
            documents = {}

def save_documents():
    try:
        with open(DOCS_FILE, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2)
    except Exception as e:
        print(f"Error saving documents registry: {e}")

load_documents()


# ---------------------------------------------------------------------------
# Ensure Azure AI Search index exists
# ---------------------------------------------------------------------------
def ensure_index():
    """Create the search index if it doesn't already exist."""
    existing = [idx.name for idx in index_client.list_indexes()]
    if INDEX_NAME in existing:
        return

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SimpleField(
            name="doc_id",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
            retrievable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
        ),
    ]

    index = SearchIndex(name=INDEX_NAME, fields=fields)
    index_client.create_index(index)
    print(f"✅ Created Azure AI Search index: '{INDEX_NAME}'")


# ---------------------------------------------------------------------------
# Text Processing
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks, step = [], chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF."""
    import io
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ---------------------------------------------------------------------------
# Indexing — Upload chunks to Azure AI Search
# ---------------------------------------------------------------------------
def index_document(text: str, source_name: str) -> str:
    """Chunk text and upload to Azure AI Search. Returns doc_id."""
    doc_id = str(uuid.uuid4())
    chunks = chunk_text(text)

    batch = [
        {
            "id":      str(uuid.uuid4()).replace("-", ""),   # Search requires alphanumeric key
            "doc_id":  doc_id,
            "source":  source_name,
            "content": chunk,
        }
        for chunk in chunks
    ]

    # Upload in batches of 100 (Azure Search limit per request)
    for i in range(0, len(batch), 100):
        search_client.upload_documents(documents=batch[i : i + 100])

    documents[doc_id] = {"name": source_name, "chunk_count": len(chunks)}
    save_documents()
    print(f"✅ Indexed {len(chunks)} chunks from '{source_name}'")
    return doc_id


# ---------------------------------------------------------------------------
# Retrieval — Search Azure AI Search
# ---------------------------------------------------------------------------
def retrieve_chunks(question: str, top_k: int = 5) -> list[dict]:
    """Full-text search Azure AI Search for relevant chunks."""
    search_query = question
    # Enrich query if asking for course/subject metadata
    if any(k in question.lower() for k in ["subject", "course", "paper"]):
        search_query = f"{question} Course Name Subject"

    try:
        results = search_client.search(
            search_text=search_query,
            top=top_k,
            select=["id", "doc_id", "source", "content"],
        )
        return [
            {
                "id":      r["id"],
                "doc_id":  r["doc_id"],
                "source":  r["source"],
                "content": r["content"],
                "score":   r["@search.score"],
            }
            for r in results
        ]
    except Exception as e:
        print(f"Azure Search error: {e}")
        return []


# ---------------------------------------------------------------------------
# Generation — Azure AI Language Question Answering
# ---------------------------------------------------------------------------
def extract_academic_metadata(question: str, chunks: list[dict]) -> Optional[tuple[str, list[dict]]]:
    """Extract syllabus / lecture plan specific fields directly if requested."""
    q_lower = question.lower()

    # 1. Instructor / Coordinator
    if any(k in q_lower for k in ["coordinator", "teacher", "faculty", "instructor", "who teaches", "uploaded by", "approved by"]):
        for c in chunks:
            coord_m = re.search(r'(?:Coordinator Name)\s*[:\-]?\s*([A-Za-z\s]+?)(?=\s*(?:Master|Coordinator|Course|E-|$))', c['content'], re.I)
            if coord_m:
                text = f"🎯 **Subject Coordinator:** **{coord_m.group(1).strip()}**\n\n📌 Source: `[{c['source']}]`"
                return text, [{"source": c["source"], "score": 0.98}]

    # 2. Course Code
    if "code" in q_lower:
        for c in chunks:
            code_m = re.search(r'(?:Course Code)\s*[:\-]?\s*([A-Za-z0-9\-]+)', c['content'], re.I)
            if code_m:
                text = f"🎯 **Course Code:** **{code_m.group(1).strip()}**\n\n📌 Source: `[{c['source']}]`"
                return text, [{"source": c["source"], "score": 0.98}]

    # 3. Subject / Course Name
    if any(k in q_lower for k in ["subject", "course", "paper", "which subject"]):
        for c in chunks:
            m = re.search(r'(?:Course Name|Subject Name)\s*[:\-]?\s*([A-Za-z0-9\s\-]+?)(?=\s*(?:Course Code|Lecture|Credit|Subject Type|\n|$))', c['content'], re.I)
            if m:
                course = m.group(1).strip()
                code_m = re.search(r'(?:Course Code)\s*[:\-]?\s*([A-Za-z0-9\-]+)', c['content'], re.I)
                code_str = f" (Course Code: **{code_m.group(1).strip()}**)" if code_m else ""
                prog_m = re.search(r'Program\s*([^\n\r]+?)(?=\s*(?:Master|Course Name|Subject|$))', c['content'], re.I)
                prog_str = f"\n- **Program:** {prog_m.group(1).strip()}" if prog_m else ""
                text = (
                    f"🎯 **Subject / Course:** **{course}**{code_str}\n"
                    f"{prog_str}\n\n"
                    f"📌 Source: `[{c['source']}]`"
                )
                return text, [{"source": c["source"], "score": 0.98}]

    return None


def extract_score_report(question: str, chunks: list[dict]) -> Optional[tuple[str, list[dict]]]:
    """Extract and format Certiport / Microsoft score reports cleanly when queried."""
    q_lower = question.lower()
    score_keywords = ["score", "pass", "result", "outcome", "marks", "grade", "percentage", "failed", "exam result", "report", "certiport", "candidate", "aryan", "did i"]
    if not any(k in q_lower for k in score_keywords):
        return None

    for c in chunks:
        content = c["content"]
        score_m = re.search(r'Your Score\s+(\d+)', content)
        if ("SCORE REPORT" in content or "FINAL SCORE" in content or "Required Score Your Score" in content) and score_m:
            cand_m = re.search(r'CANDIDATE\s+([A-Za-z\s]+?)(?=\s+India|\s+\d|\s+EXAM)', content)
            exam_m = re.search(r'EXAM\s+([A-Za-z0-9\-]+:\s+[A-Za-z\s]+?)(?=\s+Registration|\s+Exam reference)', content)
            req_m = re.search(r'Required Score\s+(\d+)', content)
            out_m = re.search(r'OUTCOME\s+([A-Za-z]+)', content)
            date_m = re.search(r'Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})', content)
            sections = re.findall(r'([A-Za-z\s]+(?:\([\d–\-%\s]+\))?)\s+(\d+%)', content)

            cand_name = cand_m.group(1).strip() if cand_m else "Candidate"
            exam_name = exam_m.group(1).strip() if exam_m else "Microsoft Azure AI Fundamentals"
            your_score = score_m.group(1).strip()
            pass_score = req_m.group(1).strip() if req_m else "700"
            res_outcome = out_m.group(1).strip().upper() if out_m else "PASS"
            exam_date = date_m.group(1).strip() if date_m else ""

            out = f"🎉 **Your Score:** **{your_score}** / 1000 &nbsp;|&nbsp; **Result:** **{res_outcome}** ✅\n"
            out += f"*(Passing score required: {pass_score})*\n\n"
            out += f"📋 **Exam Details:**\n"
            out += f"- **Candidate:** {cand_name}\n"
            out += f"- **Exam:** {exam_name}\n"
            if exam_date:
                out += f"- **Date:** {exam_date}\n"

            if sections:
                out += f"\n📊 **Section Breakdown:**\n"
                for s_name, s_pct in sections:
                    s_clean = re.sub(r'RESULTS|Required Score|Your Score|SECTION ANALYSIS', '', s_name).strip()
                    if s_clean and len(s_clean) > 3:
                        out += f"- **{s_clean}:** {s_pct}\n"

            return out, [{"source": c["source"], "score": 0.99}]

    return None


def extract_lab_report(question: str, chunks: list[dict]) -> Optional[tuple[str, list[dict]]]:
    """Extract sections from academic lab reports / experiment manuals cleanly."""
    q_lower = question.lower()

    # Find chunk that contains lab experiment indicators
    exp_chunk = None
    for c in chunks:
        content = c["content"]
        if re.search(r'\b(?:Experiment\s+\d+|Aim|Objective|Gibbs Sampling Steps|Steps|Tools Required|Learning Outcomes)\b', content, re.I):
            exp_chunk = c
            break

    if not exp_chunk:
        return None

    content = exp_chunk["content"]
    src = exp_chunk["source"]

    # 1. Procedure / Steps / Algorithm
    if any(k in q_lower for k in ["procedure", "step", "steps", "algorithm", "how to perform", "methodology", "working"]):
        m = re.search(r'(?:(?:Gibbs Sampling\s+)?Steps|Procedure|Algorithm)\s*[:\-]?\s*(.*?)(?=\s*(?:Code|Output|Learning Outcomes|Conclusion|\n\n|$))', content, re.I | re.S)
        if m:
            steps_raw = m.group(1).strip()
            steps_items = [s.strip(' •\t-') for s in re.split(r'•|\d+\.\s*', steps_raw) if s.strip()]
            if steps_items:
                formatted = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps_items))
                ans = f"📋 **Procedure / Steps for Experiment:**\n\n{formatted}\n\n📌 Source: `[{src}]`"
                return ans, [{"source": src, "score": 0.98}]

    # 2. Aim / Objective / What is the experiment about
    if any(k in q_lower for k in ["aim", "objective", "objectives", "about", "goal", "purpose", "what does the experiment", "what is the experiment"]):
        aim_m = re.search(r'\bAim\b\s*[:\-]?\s*(.*?)(?=\s*(?:Objective|Tools Required|Theory|Steps|Procedure|$))', content, re.I | re.S)
        obj_m = re.search(r'\bObjective\b\s*[:\-]?\s*(.*?)(?=\s*(?:Tools Required|Theory|Steps|Procedure|Code|$))', content, re.I | re.S)

        out = ""
        if aim_m and aim_m.group(1).strip():
            out += f"🎯 **Aim:**\n{aim_m.group(1).strip()}\n\n"
        if obj_m and obj_m.group(1).strip():
            obj_raw = obj_m.group(1).strip()
            obj_items = [s.strip(' •\t-') for s in re.split(r'•|\d+\.\s*', obj_raw) if s.strip()]
            if obj_items:
                out += "📌 **Objectives:**\n" + "\n".join(f"- {o}" for o in obj_items) + "\n\n"
        if out:
            out += f"📌 Source: `[{src}]`"
            return out, [{"source": src, "score": 0.98}]

    # 3. Tools Required / Requirements
    if any(k in q_lower for k in ["tool", "tools", "software", "hardware", "requirement", "libraries"]):
        m = re.search(r'(?:Tools Required|Requirements)\s*[:\-]?\s*(.*?)(?=\s*(?:Theory|Aim|Objective|Steps|Code|$))', content, re.I | re.S)
        if m:
            ans = f"🛠️ **Tools Required:**\n\n{m.group(1).strip()}\n\n📌 Source: `[{src}]`"
            return ans, [{"source": src, "score": 0.98}]

    # 4. Learning Outcomes / Conclusion
    if any(k in q_lower for k in ["outcome", "outcomes", "learning outcome", "conclusion", "result"]):
        m = re.search(r'(?:Learning Outcomes|Conclusion)\s*[:\-]?\s*(.*?)(?=\s*(?:Reference|Date|$))', content, re.I | re.S)
        if m:
            raw = m.group(1).strip()
            items = [s.strip(' •\t-') for s in re.split(r'•|\d+\.\s*', raw) if s.strip()]
            if items:
                formatted = "\n".join(f"- {item}" for item in items)
                ans = f"🎓 **Learning Outcomes:**\n\n{formatted}\n\n📌 Source: `[{src}]`"
                return ans, [{"source": src, "score": 0.98}]

    # 5. Code / Implementation
    if any(k in q_lower for k in ["code", "implementation", "program", "python script"]):
        m = re.search(r'(?:Code)\s*[:\-]?\s*(.*?)(?=\s*(?:Output|Learning Outcomes|$))', content, re.I | re.S)
        if m:
            code_text = m.group(1).strip()
            ans = f"💻 **Experiment Code:**\n\n```python\n{code_text}\n```\n\n📌 Source: `[{src}]`"
            return ans, [{"source": src, "score": 0.98}]

    # 6. Student / Experiment Info (UID, Name, Date)
    if any(k in q_lower for k in ["uid", "student", "who performed", "date of performance"]):
        name_m = re.search(r'Student Name:\s*([A-Za-z\s]+?)(?=\s*UID|$)', content)
        uid_m = re.search(r'UID:\s*([A-Za-z0-9]+)', content)
        date_m = re.search(r'Date of Performance:\s*([\d\/\.\-]+)', content)
        out = "👤 **Student & Lab Information:**\n\n"
        if name_m:
            out += f"- **Student Name:** {name_m.group(1).strip()}\n"
        if uid_m:
            out += f"- **UID:** {uid_m.group(1).strip()}\n"
        if date_m:
            out += f"- **Date of Performance:** {date_m.group(1).strip()}\n"
        out += f"\n📌 Source: `[{src}]`"
        return out, [{"source": src, "score": 0.98}]

    return None


def generate_answer(question: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Use Azure AI Language Question Answering (QuestionAnsweringClient)
    to extract precise, direct answers from retrieved chunks.
    Returns: (answer_string, list_of_used_sources_with_normalized_scores)
    """
    if not chunks:
        return (
            "I couldn't find relevant information in the knowledge base. "
            "Please try rephrasing your question or upload additional documents.",
            []
        )

    # 1. Check for Score Report documents
    score_report_res = extract_score_report(question, chunks)
    if score_report_res:
        return score_report_res

    # 2. Check for Lab Report documents (Aim, Procedure, Tools, Code, Outcomes)
    lab_report_res = extract_lab_report(question, chunks)
    if lab_report_res:
        return lab_report_res

    # 3. Check for syllabus/academic metadata (e.g. Course/Subject)
    meta_result = extract_academic_metadata(question, chunks)
    if meta_result:
        return meta_result

    # 4. Azure AI Language Question Answering (QnA)
    best_answer = None
    best_conf = -1.0
    best_source = None
    best_short = ""

    for c in chunks[:4]:
        try:
            resp = qa_client.get_answers_from_text(
                question=question,
                text_documents=[c["content"]],
            )
            for a in resp.answers:
                if a.confidence and a.confidence > best_conf and a.answer:
                    best_conf = a.confidence
                    best_answer = a.answer
                    best_source = c["source"]
                    short = getattr(a, "short_answer", None)
                    best_short = getattr(short, "text", "") if short else ""
        except Exception as e:
            print(f"Azure QA error: {e}")

    if best_answer and best_conf > 0.08:
        response_text = ""

        # Clean raw text from boilerplate
        clean_context = re.sub(r'©.*?(?:holders|reserved|Inc\.|All rights reserved\.?|$)', '', best_answer, flags=re.I)
        clean_context = re.sub(r'\b(?:100\s+200\s+300\s+400\s+500\s+600\s+700\s+800\s+900\s+1000|Uploaded by.*?\d{1,2}:\d{2}:\d{2}\s+(?:AM|PM)|Approved by.*?PM)\b', '', clean_context)

        if best_short and best_short.strip():
            response_text += f"🎯 **Direct Answer:** **{best_short.strip()}**\n\n"

            # Find only the specific sentence that contains best_short if helpful
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', clean_context) if len(s.strip()) > 15]
            surrounding = ""
            for s in sentences:
                if best_short.lower() in s.lower() and len(s) > len(best_short) + 8:
                    surrounding = s
                    break

            if surrounding and surrounding.strip() != best_short.strip() and len(surrounding) < 220:
                response_text += f"📖 **Context:**\n> {surrounding.strip()}\n\n"
        else:
            # If no short answer, take the first 1-2 clean sentences only
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', clean_context) if len(s.strip()) > 15]
            if sentences:
                response_text += f"{' '.join(sentences[:2])}\n\n"
            else:
                response_text += f"{clean_context[:200].strip()}...\n\n"

        norm_score = round(max(0.50, min(0.99, best_conf + 0.35)), 2)
        response_text += f"📌 Source: `[{best_source}]` (Confidence: {int(norm_score * 100)}%)"
        return response_text, [{"source": best_source, "score": norm_score}]

    # When no confident answer is found, do NOT dump a summary of unrelated documents
    sources_names = list({c["source"] for c in chunks})
    return (
        f"I couldn't find an answer to that question in the currently loaded document: **{', '.join(sources_names)}**.\n\n"
        "💡 *Tip: Try asking something specific to this document, or toggle 🌐 **Web Search: ON** to search the live internet.*",
        []
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not SEARCH_KEY or not LANGUAGE_KEY:
        print("⚠️ Warning: AZURE_SEARCH_KEY or AZURE_LANGUAGE_KEY is not set in .env yet.")
        print("Please add your Azure keys to .env to connect to Azure AI services.")
    else:
        try:
            ensure_index()
            kb_path = Path("knowledge_base/azure_ai900.txt")
            if not documents and kb_path.exists():
                text = kb_path.read_text(encoding="utf-8")
                index_document(text, "azure_ai900.txt")
                print("✅ Pre-loaded Azure AI-900 knowledge base into Azure AI Search.")
        except Exception as e:
            print(f"⚠️ Startup note: Could not initialize Azure AI Search index yet: {e}")
    yield

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="AI-900 Knowledge Base Q&A", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/documents")
async def list_documents():
    # Sync with Azure AI Search so all indexed documents appear in sidebar
    try:
        results = search_client.search(search_text="*", select=["doc_id", "source"], top=1000)
        counts = {}
        for r in results:
            d_id = r.get("doc_id", "default")
            src = r.get("source", "document")
            if d_id not in counts:
                counts[d_id] = {"name": src, "chunk_count": 0}
            counts[d_id]["chunk_count"] += 1

        for d_id, info in counts.items():
            documents[d_id] = info
        save_documents()
    except Exception as e:
        print(f"Sync error: {e}")

    return [
        {"doc_id": doc_id, **info}
        for doc_id, info in documents.items()
    ]


def check_temporal_query(question: str) -> Optional[tuple[str, list[dict]]]:
    """Catch real-time date/time/clock questions and return accurate live system date/time."""
    q = question.lower().strip()
    date_patterns = [
        r'\b(?:what(?:\'s|\s+is)?\s+)?(?:today(?:\'s)?\s+date|the\s+date\s+today|date\s+today|current\s+date)\b',
        r'\b(?:what\s+day\s+(?:is\s+it|is\s+today)|what\s+is\s+the\s+day)\b',
        r'\b(?:what\s+is\s+today\b)',
        r'\b(?:today\'s\s+date)\b',
    ]
    time_patterns = [
        r'\b(?:what(?:\'s|\s+is)?\s+)?(?:the\s+time|current\s+time|time\s+now|what\s+time\s+is\s+it)\b',
    ]
    year_patterns = [
        r'\b(?:what(?:\'s|\s+is)?\s+)?(?:the\s+year|current\s+year|which\s+year)\b',
    ]

    is_date = any(re.search(p, q) for p in date_patterns)
    is_time = any(re.search(p, q) for p in time_patterns)
    is_year = any(re.search(p, q) for p in year_patterns)

    if is_date or is_time or is_year:
        now = datetime.now()
        day_name = now.strftime("%A")
        date_formatted = now.strftime("%B %d, %Y")
        time_formatted = now.strftime("%I:%M %p")
        tz_name = now.astimezone().tzname() or "Local Time"

        out = "📅 **Current Date & Time:**\n\n"
        out += f"- **Today's Date:** **{day_name}, {date_formatted}**\n"
        out += f"- **Current Time:** **{time_formatted}** ({tz_name})\n"
        out += f"- **Year:** **{now.year}**\n\n"
        out += "*(Retrieved live from system clock)*"
        return out, [{"source": "System Real-Time Clock", "score": 1.0}]

    return None


def get_web_grounding(query: str) -> tuple[str, list[dict]]:
    """Fetch live web results and ground the answer using Azure AI Language."""
    # 0. Check real-time clock first
    temporal = check_temporal_query(query)
    if temporal:
        return temporal

    snippets = []
    sources = []
    clean_q = re.sub(r'^(?:what(?:\'s|\s+is)?|who(?:\'s|\s+is)?|explain|tell me about|how does|what are|where is)\s+', '', query, flags=re.I).strip("?!. ")

    # 1. DuckDuckGo Instant Answer
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            if raw.strip().startswith("{"):
                d = json.loads(raw)
                if d.get("Abstract"):
                    snippets.append(d["Abstract"])
                    sources.append({"title": d.get("Heading", "DuckDuckGo Knowledge"), "url": d.get("AbstractURL", "https://duckduckgo.com")})
    except Exception:
        pass

    # 2. Wikipedia Search + Article Summary API
    search_terms = [clean_q or query, query]
    seen_titles = set()
    for term in search_terms:
        if len(snippets) >= 2:
            break
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(term)}&utf8=&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "AzureAI900HybridRAG/1.0 (student@azure.edu)"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for item in data.get("query", {}).get("search", [])[:3]:
                    title = item.get("title", "")
                    if not title or title in seen_titles or re.search(r'\b(disambiguation|list of)\b', title, re.I):
                        continue
                    seen_titles.add(title)

                    sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                    sum_req = urllib.request.Request(sum_url, headers={"User-Agent": "AzureAI900HybridRAG/1.0 (student@azure.edu)"})
                    try:
                        with urllib.request.urlopen(sum_req, timeout=4) as s_resp:
                            s_data = json.loads(s_resp.read().decode("utf-8"))
                            extract = s_data.get("extract")
                            page_url = s_data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}")
                            if extract and len(extract) > 40:
                                snippets.append(f"{title}: {extract}")
                                sources.append({"title": f"Wikipedia: {title}", "url": page_url})
                                if len(snippets) >= 2:
                                    break
                    except Exception:
                        pass
        except Exception as e:
            print(f"Wikipedia search error: {e}")

    if not snippets:
        return (
            "🌐 **Web Grounding:** Could not retrieve live web results for that query. Please try rephrasing.",
            []
        )

    # 3. Azure AI Language Question Answering
    best_short = ""
    best_ans = ""
    try:
        resp = qa_client.get_answers_from_text(question=query, text_documents=snippets[:3])
        if resp.answers:
            a = resp.answers[0]
            if a.confidence and a.confidence > 0.10 and a.answer:
                best_ans = a.answer
                short = getattr(a, "short_answer", None)
                cand_short = getattr(short, "text", "").strip() if short else ""
                # Avoid dangling prepositional fragments
                if cand_short and not re.match(r'^(?:between|and|or|in|of|at|to|from|by|with|for|as|on)\b', cand_short, re.I):
                    best_short = cand_short
    except Exception as e:
        print(f"Azure QA web error: {e}")

    out = "🌐 **Web-Grounded Answer** *(via Live Internet Search)*:\n\n"
    if best_short:
        out += f"🎯 **Direct Answer:** **{best_short}**\n\n"

    if best_ans:
        clean_ans = re.sub(r'^[A-Za-z0-9\s\-]+:\s+', '', best_ans)
        out += f"{clean_ans}\n\n"
    elif snippets:
        clean_snippet = re.sub(r'^[A-Za-z0-9\s\-]+:\s+', '', snippets[0])
        out += f"{clean_snippet}\n\n"

    out += "📌 **Web Sources:**\n"
    web_sources = []
    seen_urls = set()
    for s in sources:
        if s["url"] not in seen_urls:
            seen_urls.add(s["url"])
            out += f"- [{s['title']}]({s['url']})\n"
            web_sources.append({"source": f"🌐 {s['title']}", "score": 0.95})
        if len(web_sources) >= 3:
            break

    return out, web_sources


class AskRequest(BaseModel):
    question:   str
    top_k:      Optional[int] = 5
    web_search: Optional[bool] = False


@app.post("/api/ask")
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 0. Check real-time clock queries immediately
    temporal = check_temporal_query(req.question)
    if temporal:
        answer, sources = temporal
        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": 0,
        }

    # 1. If user explicitly toggled Web Search
    if req.web_search:
        answer, sources = get_web_grounding(req.question)
        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(sources),
        }

    # 2. Search local documents in Azure AI Search
    chunks = retrieve_chunks(req.question, top_k=req.top_k)

    # 3. Handle greetings
    if not chunks:
        q_clean = req.question.strip().lower().strip("!?,.")
        if q_clean in {"hello", "hi", "hey", "help", "who are you", "good morning", "good evening"}:
            return {
                "answer": "👋 **Hello!** I am your **Azure AI Knowledge Assistant**.\n\nAsk me anything about your uploaded documents or Azure AI fundamentals, or enable **🌐 Web Search** to search the live internet!",
                "sources": [{"source": "Azure Assistant", "score": 1.0}],
                "chunks_used": 0,
            }

    # 4. Generate answer from local documents
    answer, sources = generate_answer(req.question, chunks)

    return {
        "answer":      answer,
        "sources":     sources,
        "chunks_used": len(chunks),
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".txt", ".pdf", ".md"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file: {ext}. Allowed: {', '.join(allowed)}",
        )

    content = await file.read()
    text = extract_text_from_pdf(content) if ext == ".pdf" else content.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from the file.")

    doc_id = index_document(text, file.filename)
    return {
        "doc_id":      doc_id,
        "name":        file.filename,
        "chunk_count": documents[doc_id]["chunk_count"],
        "message":     f"✅ Indexed {documents[doc_id]['chunk_count']} chunks from '{file.filename}' into Azure AI Search.",
    }


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Delete all chunks from Azure AI Search that belong to this document
    results = search_client.search(
        search_text="*",
        filter=f"doc_id eq '{doc_id}'",
        select=["id"],
        top=1000,
    )
    ids_to_delete = [{"id": r["id"]} for r in results]
    if ids_to_delete:
        search_client.delete_documents(documents=ids_to_delete)

    name = documents.pop(doc_id)["name"]
    save_documents()
    return {"message": f"✅ Removed '{name}' from Azure AI Search index."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
