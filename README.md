<div align="center">

# CDSS — Clinical Decision Support System

### An AI-powered patient research tool built with LangGraph + Groq

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq%20API-F55036?logo=groq&logoColor=white)](https://groq.com)
[![LangGraph](https://img.shields.io/badge/Agents-LangGraph-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Not a replacement for your doctor — a research amplifier.**  
> Upload your medical reports, get a structured plain-English briefing to bring to your next appointment.

</div>

---

## What does it do?

You paste or upload your medical documents. CDSS reads them and produces a **patient research report** covering three questions doctors rarely have time to answer for you:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   1. STANDARD OF CARE     What do clinical guidelines say about     │
│      ───────────────      my condition and stage?                   │
│                                                                     │
│   2. CLINICAL TRIALS      Are there recruiting studies worldwide    │
│      ───────────────      that match my profile and biomarkers?     │
│                                                                     │
│   3. OFF-LABEL OPTIONS    Are there drugs approved for OTHER        │
│      ─────────────────    diseases that share my biology?           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Every section ends with "bring this to your doctor" framing. CDSS never prescribes or diagnoses.

---

## Pipeline Architecture

CDSS runs a **5-agent LangGraph pipeline** on every query. Each agent is a specialised function that reads the shared state, does its job, and writes results back.

```
                        PATIENT INPUT
                   (text / PDF / image file)
                            │
                            ▼
            ┌───────────────────────────────┐
            │         AGENT 1               │
            │      Intake & Parse           │
            │                               │
            │  Pass 1 ─ detect doc type     │  ← soap_note / lab_report /
            │  Pass 2 ─ type-matched        │    genomic_panel / radiology /
            │           extraction          │    discharge_summary / etc.
            │  Pass 3 ─ clinical            │
            │           intelligence        │  ← missed diagnoses, urgent flags,
            │                               │    risk factors, actionable insights
            │  Rule-based quality score     │
            └───────────────┬───────────────┘
                            │  PatientProfileState
                            │  (condition, stage, biomarkers,
                            │   labs, vitals, history, ...)
                            ▼
            ┌───────────────────────────────┐
            │         AGENT 2               │
            │   Standard Care Retrieval     │
            │                               │
            │  ┌─────────┐  ┌───────────┐  │
            │  │ PubMed  │  │  Qdrant   │  │
            │  │ E-utils │  │ (PDF RAG) │  │
            │  └────┬────┘  └─────┬─────┘  │
            │       └──────┬──────┘         │
            │           deduplicate         │
            │           synthesise          │
            └───────────────┬───────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │         AGENT 3               │
            │   Clinical Trials Finder      │
            │                               │
            │  ClinicalTrials.gov v2 API    │  ← No LLM — fully deterministic
            │  filter: RECRUITING           │
            │  query: condition + biomarkers│
            │  NCT IDs validated live       │
            └───────────────┬───────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │         AGENT 4               │
            │  Cross-Indication Discovery   │
            │                               │
            │  5-lens biomedical reasoning: │
            │  • Loss-of-function pathway   │
            │  • Gain-of-function kinase    │
            │  • Chromatin remodelling      │
            │  • Tissue-agnostic approvals  │
            │  • Basket trial evidence      │
            └───────────────┬───────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │         AGENT 5               │
            │     Report Synthesiser        │
            │                               │
            │  Assembles structured         │  ← Pure code, no LLM
            │  Markdown report:             │
            │  • Urgent flags (top)         │
            │  • Patient profile            │
            │  • Clinical insights          │
            │  • Standard care              │
            │  • Trials table               │
            │  • Off-label hypotheses       │
            │  • Questions to ask doctor    │
            └───────────────┬───────────────┘
                            │
                            ▼
                    PATIENT REPORT
              (4-tab Streamlit display
               + downloadable .md file)
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Agent framework** | [LangGraph](https://langchain-ai.github.io/langgraph/) | Stateful multi-agent graph with conditional retry loops |
| **LLM inference** | [Groq API](https://groq.com) (free tier) | DeepSeek-R1 70B / Llama 3.3 70B — OpenAI-compatible |
| **Embeddings** | [BioBERT](https://huggingface.co/pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb) | 768-dim biomedical sentence vectors |
| **Vector store** | [Qdrant](https://qdrant.tech) (in-memory) | RAG over user-uploaded clinical guideline PDFs |
| **Guideline search** | [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) | Free NCBI API — auto-fetches practice guidelines, no setup |
| **Clinical trials** | [ClinicalTrials.gov v2](https://clinicaltrials.gov/data-api/api) | Live REST API for recruiting studies worldwide |
| **State schema** | [Pydantic v2](https://docs.pydantic.dev/) | Strict typed state shared across all 5 agents |
| **UI** | [Streamlit](https://streamlit.io) | Web interface with live status updates |
| **PDF parsing** | [pypdf](https://github.com/py-pdf/pypdf) | Text extraction from uploaded PDF reports |
| **Image OCR** | Groq Vision + pytesseract | Handwritten notes and scanned document support |

---

## Supported Input Types

CDSS auto-detects what kind of document you uploaded and applies a specialised extraction prompt:

```
┌──────────────────────┬──────────────────────────────────────────────────────┐
│ Document Type        │ What gets extracted                                  │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ free_text            │ Condition, stage, biomarkers, medications             │
│ soap_note            │ S/O/A/P sections → vitals, chief complaint, plan      │
│ lab_report           │ Every test with value / unit / flag / reference range │
│ pathology_report     │ Grade, tumor size, margins, IHC, TNM staging          │
│ genomic_panel        │ Mutations, fusions, TMB, MSI, PD-L1                   │
│ radiology_report     │ Modality, date, structured findings per study          │
│ discharge_summary    │ Admission/discharge diagnosis, procedures              │
│ referral_letter      │ Referring clinician, reason, key history               │
│ oncology_note        │ Treatment response, chemo plan, performance status     │
└──────────────────────┴──────────────────────────────────────────────────────┘
```

Accepted file formats: **PDF · PNG · JPG · JPEG · TXT · CSV · RTF**  
Multiple files can be uploaded and merged in a single run.

---

## Data Quality Score

After extraction, CDSS scores how much clinically useful information was found — no LLM involved, fully rule-based:

```
  Dimension              Weight   Scoring rule
  ─────────────────────  ──────   ────────────────────────────────────────────
  Core oncology           30%     condition (+15%) · stage (+10%) · biomarkers (+5%)
  Treatment history       15%     medications (+8%) · prior therapies (+7%)
  Demographics            10%     age (+5%) · sex (+5%)
  Lab data                15%     0 tests=0% · 1-3=47% · 4-10=80% · >10=100%
  Biomarker enrichment    10%     TMB (+4%) · MSI (+4%) · PD-L1 (+4%) — cap 10%
  Clinical context        10%     comorbidities (+4%) · symptoms (+3%) · imaging (+3%)
  History                 10%     family hx (+4%) · social hx (+3%) · allergies (+3%)

  Result shown as:  HIGH ≥ 70%  ·  MEDIUM ≥ 40%  ·  LOW < 40%
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com) (takes 60 seconds to get)

### Install

```bash
git clone https://github.com/YOUR_USERNAME/cdss.git
cd cdss
pip install -r requirements.txt
```

### Run

```bash
streamlit run cdss_streamlit.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

1. Paste your Groq API key in the sidebar
2. Upload your medical reports **and/or** type a description in the text area
3. Click **Run CDSS Analysis**
4. Read your report across 4 tabs — download as `.md` when done

### Optional: Upload clinical guidelines

In the sidebar, upload NCCN or ESMO PDF guidelines. They get chunked, embedded with BioBERT, and stored in Qdrant. Agent 2 will use them in addition to PubMed for every subsequent run in that session.

---

## Results — 4 Tabs

```
┌─────────────┬──────────────────┬──────────────────┬──────────────┐
│   Report    │  Patient Profile │   Data Quality   │   Raw JSON   │
├─────────────┼──────────────────┼──────────────────┼──────────────┤
│ Full        │ Demographics ·   │ Progress bars    │ Raw Agent 1  │
│ Markdown    │ Vitals ·         │ per dimension    │ extraction   │
│ report      │ Biomarkers ·     │                  │              │
│             │ Lab results ·    │ Actionable       │ All          │
│ Download    │ Imaging          │ insights         │ validation   │
│ as .md      │ findings         │                  │ flags        │
└─────────────┴──────────────────┴──────────────────┴──────────────┘
```

---

## Safety Design

- Every output section ends with *"Ask your doctor which of these options apply to you"*
- Evidence levels 1–4 shown on every off-label hypothesis (in-vitro → Phase III)
- No prescriptive language ("you should take X") — only "this exists"
- NCT IDs are validated with a live HTTP check before appearing in the report
- System Notes section only shows genuine errors, never internal routing messages
- **This tool does not store any patient data.** Everything runs locally in your session.

---

## Roadmap

- [ ] Web-hosted version (no local install)
- [ ] DrugBank / OpenTargets integration for richer pathway data
- [ ] Longitudinal tracking — compare reports across appointments
- [ ] HIPAA-compliant infrastructure for real clinical deployment
- [ ] Evidence-level calibration against published clinical outcomes
- [ ] Broader rare disease coverage

---

## Disclaimer

> CDSS is for **research and education only**. It is not a medical device, does not provide medical advice, and has not been reviewed or approved by any regulatory authority. Always consult a qualified healthcare professional before making any medical decision. The authors accept no liability for clinical decisions made based on this tool's output.

---

<div align="center">

Built with Python · LangGraph · Groq · Streamlit · BioBERT · Qdrant · PubMed · ClinicalTrials.gov

</div>
