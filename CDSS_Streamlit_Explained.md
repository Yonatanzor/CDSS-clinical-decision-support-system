# CDSS Streamlit App — How It Works

A step-by-step walkthrough of `cdss_streamlit.py`, from startup to final report.

---

## Table of Contents

1. [What the app does](#1-what-the-app-does)
2. [Startup — what happens when you run it](#2-startup)
3. [Data model — PatientProfileState](#3-data-model)
4. [Resource caching — the heavy objects](#4-resource-caching)
5. [The sidebar — API key and guideline PDFs](#5-the-sidebar)
6. [Input area — files and free text](#6-input-area)
7. [The pipeline — 5 agents in sequence](#7-the-pipeline)
   - [Agent 1 — 3-pass intake](#agent-1--3-pass-intake)
   - [Agent 2 — Standard care retrieval](#agent-2--standard-care-retrieval)
   - [Agent 3 — Clinical trials search](#agent-3--clinical-trials-search)
   - [Agent 4 — Cross-indication discovery](#agent-4--cross-indication-discovery)
   - [Agent 5 — Report synthesis](#agent-5--report-synthesis)
8. [Results display — 4 tabs](#8-results-display)
9. [Data quality scoring](#9-data-quality-scoring)
10. [How image and PDF files are handled](#10-how-image-and-pdf-files-are-handled)
11. [Session state — how data persists across reruns](#11-session-state)
12. [Full execution flow diagram](#12-full-execution-flow-diagram)

---

## 1. What the app does

CDSS is a **patient research tool**, not a diagnostic system. Given a patient's medical documents or a text description of their condition, it produces a structured report covering:

- **Standard of care** — what treatment guidelines say (sourced from PubMed and optionally from uploaded clinical PDFs)
- **Active clinical trials** — recruiting studies worldwide matching the patient's condition and biomarkers (sourced from ClinicalTrials.gov)
- **Off-label drug hypotheses** — drugs approved for *other* conditions that share biological pathways with the patient's genomic alterations
- **Clinical intelligence** — missed diagnoses to consider, modifiable risk factors, actionable next steps, urgent flags

Every section ends with "bring this to your doctor" framing. The app never prescribes or diagnoses.

---

## 2. Startup

When you run `streamlit run cdss_streamlit.py`, Streamlit executes the file top-to-bottom, then waits for user interaction. On every user action (button click, text input, file upload), Streamlit **re-runs the entire file** from scratch. This is the core Streamlit execution model — it is not a bug.

The first lines that execute are:

```python
st.set_page_config(...)   # Must be the very first Streamlit call
```

Then all class definitions, constants, and function definitions are loaded into memory. Nothing expensive runs yet — those are deferred until explicitly called.

At the very bottom of the file:

```python
if __name__ == "__main__":
    for key in ("llm_ready", "embedder_loaded", "last_result", "api_key_input"):
        if key not in st.session_state:
            st.session_state[key] = ...
    main()
```

This initializes session state defaults on the **first run only** (the `if key not in` check prevents overwriting values that already exist from a previous rerun), then calls `main()`.

---

## 3. Data Model

The entire pipeline is built around a single Pydantic model called `PatientProfileState`. Think of it as a **shared clipboard** that gets passed from agent to agent, with each agent reading what it needs and writing its results into it.

```
PatientProfileState
├── raw_input              ← the original text the user provided
├── input_type             ← detected document type (e.g. "soap_note", "genomic_panel")
│
├── Core extraction (Agents 2-5 read these)
│   ├── condition          ← e.g. "Non-small cell lung cancer"
│   ├── stage              ← e.g. "Stage III"
│   ├── biomarkers         ← list of Biomarker objects (gene, variant_type, details)
│   ├── current_medications
│   └── prior_therapies
│
├── Rich clinical fields (new in v5)
│   ├── patient_age, patient_sex, bmi
│   ├── vitals, chief_complaint, symptoms
│   ├── lab_results        ← list of {test, value, unit, flag, ref}
│   ├── imaging_findings   ← list of {modality, date, finding}
│   ├── comorbidities, family_history, social_history, allergies
│   └── pathology_grade, tumor_size_cm, lymph_node_status, surgical_margins,
│       immunohistochemistry, tumor_mutational_burden, microsatellite_status, pd_l1_expression
│
├── Clinical intelligence (Pass 3 of Agent 1)
│   ├── missed_diagnosis_flags
│   ├── modifiable_risk_factors
│   ├── non_modifiable_risk_factors
│   ├── actionable_insights
│   └── urgent_flags
│
├── Agent outputs
│   ├── standard_therapies      ← Agent 2
│   ├── clinical_trials         ← Agent 3
│   └── off_label_hypotheses    ← Agent 4
│
└── Quality & bookkeeping
    ├── data_quality_score
    ├── data_quality_breakdown
    ├── validation_flags        ← error messages and routing notes
    ├── final_report            ← Agent 5 output (markdown string)
    └── retry_count / max_retries
```

Three supporting Pydantic models are defined separately:
- `Biomarker` — one genomic alteration (gene + variant type + details)
- `ClinicalTrial` — one ClinicalTrials.gov study
- `OffLabelHypothesis` — one drug candidate with evidence level and citation

---

## 4. Resource Caching

Three expensive objects are created once and reused across all reruns using `@st.cache_resource`. They survive page refreshes for the lifetime of the server process.

### Biomedical embedding model (~400 MB, loaded once)

```python
@st.cache_resource
def _load_embedder():
    return SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
```

BioBERT is a BERT model fine-tuned on biomedical text. It converts sentences into 768-dimensional vectors. Used only when the user uploads guideline PDFs — it turns each text chunk into a number vector so Qdrant can do similarity search.

### In-memory vector database

```python
@st.cache_resource
def _create_qdrant():
    client = QdrantClient(":memory:")
    client.create_collection("guidelines", ...)
    return client
```

Qdrant stores the vectorized chunks of any clinical PDFs the user uploads. When Agent 2 searches for standard-of-care guidelines, it queries this store first. The `":memory:"` mode means data is lost when the server stops — it is not saved to disk.

### Groq LLM client (keyed on API key)

```python
@st.cache_resource
def _create_llm(api_key: str):
    llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    available = {m.id for m in llm.models.list().data}
    model = next((m for m in PREFERENCE_ORDER if m in available), ...)
    return llm, model, sorted(available)
```

Uses the OpenAI-compatible SDK (Groq speaks the same protocol as OpenAI). Checks which models are currently live and picks the best one from a preference list: DeepSeek-R1 70B → Llama 3.3 70B → smaller fallbacks. Because `api_key` is a parameter, Streamlit caches a separate instance per key — changing the key auto-refreshes the client.

All LLM calls go through a single `chat()` helper:

```python
def chat(prompt, max_tokens=1024):
    llm   = st.session_state["llm"]
    MODEL = st.session_state["MODEL"]
    return llm.chat.completions.create(...).choices[0].message.content.strip()
```

---

## 5. The Sidebar

The sidebar runs on every page load via `render_sidebar()`.

### API key input

The user pastes their Groq API key (`gsk_...`). When it changes, `llm_ready` is set to `False`, which triggers a reconnect. A green "Connected: model-name" message confirms success.

### Guideline PDF upload (optional)

The user can upload NCCN, ESMO, or any clinical PDF. When they click "Ingest", the app:

1. Loads BioBERT (first time only — may take 30–60 seconds)
2. Extracts text from each PDF page with `pypdf`
3. Splits text into overlapping 500-word chunks (step of 400 words, so consecutive chunks overlap by 100 words to preserve context across boundaries)
4. Encodes each chunk into a 768-dimensional vector
5. Upserts all vectors into Qdrant with the filename as metadata

These guideline chunks will be searched by Agent 2 during every subsequent pipeline run in this session.

---

## 6. Input Area

The main page has two side-by-side input zones.

### Left — file upload

Accepts: PDF, PNG, JPG, JPEG, TXT, CSV, RTF (multiple files allowed).

Each file is processed by `extract_text_from_upload()`:
- **PDF** → `pypdf` extracts text page by page
- **PNG / JPG / JPEG** → `_ocr_image_bytes()` (see section 10)
- **TXT / CSV / RTF** → decoded as UTF-8 (falls back to latin-1, then cp1252)

### Right — free text

A plain text area where the user can type or paste a patient description, clinical note, or any medical narrative.

### Combining inputs

All extracted texts and the free text are joined into one string:

```python
combined_input = "\n\n---\n\n".join([
    "[Source: filename.pdf]\nextracted text...",
    "free text from the user...",
])
```

The `[Source: filename]` header helps the LLM understand where each section came from when multiple files are uploaded.

---

## 7. The Pipeline

When the user clicks **Run CDSS Analysis**, a `PatientProfileState` is created with `raw_input = combined_input` and passed through `run_pipeline()`, which calls all five agents in sequence:

```
PatientProfileState --> Agent 1 --> Agent 2 --> Agent 3 --> Agent 4 --> Agent 5 --> final_report
```

Each agent receives the full state object, writes its results into it, and returns it. The next agent sees everything the previous ones wrote.

---

### Agent 1 — 3-pass intake

**`agent_intake()` — lines ~509–598**

This is the most complex agent. It makes **3 LLM calls** in sequence.

#### Pass 1 — Document type detection (10 tokens)

```
Prompt: "Classify this text into one of: soap_note | discharge_summary |
         lab_report | pathology_report | radiology_report | referral_letter |
         genomic_panel | oncology_note | free_text"
```

Only the first 3,000 characters of the input are sent to minimize cost. The model returns a single label. If the label is not in the known set, it defaults to `free_text`.

**Why this matters:** different document types contain clinical information in completely different structures. A lab report uses test/value/flag rows; a SOAP note uses S/O/A/P sections; a genomic panel report uses alteration tables. A single generic prompt would miss structure-specific details.

#### Pass 2 — Type-specific extraction (up to 1500 tokens)

The detected label selects one of 9 specialized prompts from `_EXTRACTION_PROMPTS`. Each prompt tells the LLM what to look for in that specific document type. For example:

- `genomic_panel` prompt: "For fusions, list the full fusion name AND each component gene separately. Extract TMB, microsatellite_status, pd_l1_expression."
- `soap_note` prompt: "Map S section to chief_complaint/symptoms, O section to vitals/lab_results, A section to condition/stage, P section to current_medications."
- `pathology_report` prompt: "Extract tumor_size_cm, lymph_node_status, surgical_margins, immunohistochemistry."

All 9 prompts share the same output JSON schema (`_BASE_JSON`), so the parsing code is identical regardless of type.

After parsing, **fusion expansion** runs on biomarkers: if a gene is listed as a fusion (e.g. `EML4-ALK Fusion`), the individual components (`EML4`, `ALK`) are also added as separate `Fusion component` entries so downstream agents can query them individually.

#### Pass 3 — Clinical intelligence (up to 1200 tokens)

A compact summary of the extracted profile is sent to a senior-clinician role prompt:

```
"Perform a CLINICAL INTELLIGENCE ANALYSIS — surface patterns a busy clinician might miss."
```

The model returns 5 JSON arrays:

| Array | Description | Example |
|---|---|---|
| `missed_diagnosis_flags` | Conditions that fit the data but are not listed | "Consider PE because bilateral infiltrates + SpO2 91%" |
| `modifiable_risk_factors` | Things the patient can change (citing specific extracted values) | "BMI 32.1 — obesity raises recurrence risk" |
| `non_modifiable_risk_factors` | Age, germline mutations, family history | "BRCA1 germline — hereditary breast/ovarian cancer syndrome" |
| `actionable_insights` | Evidence-based next steps from the profile | "TMB-High — discuss pembrolizumab eligibility" |
| `urgent_flags` | Findings needing immediate clinical attention | "WBC 0.2 K/uL — critical neutropenia" |

Any `urgent_flags` are also appended to `validation_flags` with an `"URGENT:"` prefix, which causes them to appear at the very top of the final report.

#### After Pass 3 — Data quality score

`compute_data_quality_score()` runs with **no LLM call**. See Section 9 for details.

---

### Agent 2 — Standard care retrieval

**`agent_standard_care()` — lines ~697–735**

Searches for clinical practice guidelines about the patient's condition from two sources.

**Source 1 — PubMed (automatic, no setup required)**

`search_pubmed_guidelines()` uses the free NCBI E-utilities API with two cascading queries:
1. `"[condition]"[tiab] AND "practice guideline"[pt]` — strict: only official guidelines
2. Broader fallback using `"systematic review"` or `"meta-analysis"` if fewer than 3 results are found

It fetches up to 8 article IDs, retrieves full abstracts in XML format, and truncates each to 600 words. The tool adds `&tool=CDSS&email=...` to all requests as required by NCBI etiquette.

**Source 2 — Qdrant (only if guideline PDFs were uploaded)**

Encodes the query `"[condition] [stage] treatment guidelines"` into a vector and finds the top-6 most similar chunks from ingested PDFs.

**Deduplication then synthesis**

Both lists are merged, with Qdrant results taking priority (they contain full guideline text; PubMed provides abstracts). The merged list is capped at 10 items. The `CARE_PROMPT` then asks the LLM to write a plain-English treatment summary from these excerpts, noting evidence quality differences.

---

### Agent 3 — Clinical trials search

**`agent_clinical_trials()` — lines ~741–790**

Calls the ClinicalTrials.gov v2 REST API directly. **No LLM is used.**

Key query parameters:
- `query.cond` = patient's condition
- `filter.overallStatus` = RECRUITING
- `query.term` = patient's biomarker gene names joined with OR (e.g. `EGFR OR TP53`)
- `pageSize` = 10

For each study, it extracts the NCT ID, title, phase, up to 5 locations, and eligibility criteria (truncated at 500 characters). Results are stored as `ClinicalTrial` objects.

---

### Agent 4 — Cross-indication discovery

**`agent_cross_indication()` — lines ~824–857**

Finds drugs approved for *other* diseases that share a biological mechanism with the patient's genomic alterations.

In the full-power version (notebook), this agent can use PrimeKG — a Harvard biomedical knowledge graph with 29,000 nodes and 8 million edges — to trace molecular pathways from gene to drug. In the Streamlit app, PrimeKG is not loaded (it requires a separate download), so the agent uses **pure LLM biomedical reasoning** instead.

The `HYPOTHESIS_PROMPT_LLM` gives the model a 5-lens reasoning framework to analyze every alteration:

| Lens | Logic | Example |
|---|---|---|
| Loss-of-function | What pathway is unrestrained? What is the synthetic lethal partner? | SMARCB1 deletion → EZH2 unrestrained → Tazemetostat |
| Gain-of-function | What kinase is constitutively active? What inhibitor targets it? | EGFR amplification → Erlotinib |
| Chromatin remodeling | SWI/SNF complex loss suggests EZH2/HDAC/BET inhibitors | ARID1A loss → EZH2 inhibitor |
| Tissue-agnostic | FDA approvals that apply to any tumor type with a specific marker | TMB-High → Pembrolizumab |
| Basket trial | Pre-clinical data from cell lines with this exact alteration | KRAS G12C in any solid tumor → Sotorasib |

The model returns up to 8 drug candidates, each with evidence level 1–4 and a PubMed search URL.

---

### Agent 5 — Report synthesis

**`agent_synthesizer()` — lines ~894–1031**

No LLM call — this agent is entirely programmatic. It assembles the `final_report` markdown string from all the state fields populated by the previous four agents.

**Sections assembled in order:**

1. **Urgent findings** (top, only if urgent flags exist) — red callout block
2. **Your Profile** — document type, condition, stage, demographics, biomarkers, TMB/MSI/PD-L1, medications, comorbidities + data quality badge
3. **Clinical Insights** — actionable insights, modifiable/non-modifiable risk factors, missed diagnosis considerations
4. **Standard Treatment Options** — synthesized text from Agent 2
5. **Active Clinical Trials** — each NCT ID is validated with a live HTTP check before appearing; markdown table with links
6. **Off-Label Therapy Hypotheses** — markdown table with drug, approved indication, shared mechanism, evidence level
7. **Questions to Ask Your Doctor** — auto-generated from the trials and high-evidence hypotheses
8. **System Notes** — only genuine errors are shown here (parse failures, API errors, excluded trials); internal routing messages are filtered out
9. **Disclaimer** — always last

---

## 8. Results Display

After the pipeline finishes, the result is stored in session state and the page reruns. Four tabs are shown:

| Tab | Contents |
|---|---|
| **Report** | Full `final_report` string rendered as markdown. Download button exports as `.md`. |
| **Patient Profile** | Two-column structured view: demographics / vitals / symptoms on the left; biomarkers / lab results table / imaging on the right. |
| **Data Quality** | Progress bars per dimension + actionable insights + missed diagnosis flags for clinical discussion. |
| **Raw JSON** | The raw extraction dict from Agent 1 Pass 2 + all validation flags (color-coded: red = URGENT, yellow = error, blue = informational). |

---

## 9. Data Quality Scoring

The score is **entirely rule-based** — no LLM involved. It answers: "How much clinically useful information did we extract from this document?"

| Dimension | Max weight | Scoring rule |
|---|---|---|
| Core oncology | 30% | condition (+15%), stage (+10%), any biomarkers (+5%) |
| Treatment history | 15% | current medications (+8%), prior therapies (+7%) |
| Demographics | 10% | age (+5%), sex (+5%) |
| Lab data | 15% | 0 tests = 0%, 1–3 = 47%, 4–10 = 80%, >10 = 100% |
| Biomarker enrichment | 10% | TMB present (+4%), MSI present (+4%), PD-L1 present (+4%), capped at 10% |
| Clinical context | 10% | comorbidities (+4%), symptoms (+3%), imaging (+3%) |
| History | 10% | family history (+4%), social history (+3%), allergies (+3%) |

A typical oncology free-text description scores 40–65%. A full NGS genomic panel report scores 60–80%. A discharge summary with labs scores 70–90%.

Displayed in the report header as **HIGH** (≥70%) / **MEDIUM** (≥40%) / **LOW** (<40%).

---

## 10. How Image and PDF Files Are Handled

### PDFs

`pypdf.PdfReader` extracts text page by page from the PDF binary. This works well for digitally-created PDFs. Scanned PDFs (where content is embedded as images) return empty strings from `extract_text()` — for those, save as a PNG/JPG and use the image path instead.

### Images (PNG, JPG, JPEG)

`_ocr_image_bytes()` tries two strategies in order:

**Strategy 1 — Groq vision model (primary)**

The image is base64-encoded and sent to `meta-llama/llama-4-scout-17b-16e-instruct` with the instruction: *"Transcribe ALL text exactly as it appears, preserving structure, labels, values, and units."* This handles handwritten notes, scanned documents, and complex lab report layouts. No extra Python package needed — it reuses the existing Groq client.

**Strategy 2 — pytesseract fallback**

If the Groq vision call fails (model unavailable, rate limit, network error), the app tries `pytesseract.image_to_string()` — local OCR using the open-source Tesseract engine. This requires:
- `pip install pytesseract pillow`
- [Tesseract binary](https://github.com/tesseract-ocr/tesseract) installed on your system

If neither strategy works, a clear error message is returned in the text instead of silently failing.

---

## 11. Session State

Because Streamlit re-runs the entire script on every user interaction, data that needs to persist is stored in `st.session_state`:

| Key | What it holds | When set |
|---|---|---|
| `api_key_input` | The raw API key string typed by the user | Sidebar text input |
| `llm_ready` | Boolean: is the LLM client initialized? | Sidebar connection logic |
| `llm` | The OpenAI client object pointing to Groq | `_create_llm()` result |
| `MODEL` | The selected model ID string (e.g. `deepseek-r1-distill-llama-70b`) | `_create_llm()` result |
| `available_models` | Sorted list of all Groq model IDs | `_create_llm()` result |
| `embedder_loaded` | Boolean: has BioBERT been loaded? | Sidebar "Ingest" button |
| `ingested_guidelines` | Set of already-ingested PDF filenames | Sidebar "Ingest" button |
| `last_result` | The completed `PatientProfileState` object | `run_pipeline()` |

`@st.cache_resource` objects (`_load_embedder`, `_create_qdrant`, `_create_llm`) are stored in Streamlit's internal cache — they persist across reruns and page refreshes for the lifetime of the server process, and are shared across all browser tabs connected to the same server.

---

## 12. Full Execution Flow Diagram

```
User opens http://localhost:8501
         |
         v
Streamlit runs cdss_streamlit.py top-to-bottom
  - loads all classes, constants, function definitions
  - calls main()
         |
         v
render_sidebar()
  - user enters Groq API key
  - _create_llm(api_key) called once, result cached
  - user optionally uploads guideline PDFs
      - _load_embedder() called once, cached (~400 MB BioBERT)
      - ingest_guideline_pdf() chunks + embeds each PDF into Qdrant
         |
         v
main() renders input section
  - user uploads patient report files  ---|
  - user types free text               ---|
                                          v
                            extract_text_from_upload()
                              .pdf  -> pypdf page text
                              .png/.jpg -> Groq vision OCR -> pytesseract fallback
                              .txt/.csv -> UTF-8 decode
                                          |
                                          v
                            combined_input = all texts merged with source headers
                                          |
                                          v
                         user clicks "Run CDSS Analysis"
                                          |
                                          v
                         PatientProfileState(raw_input=combined_input)
                                          |
                                          v
                                  run_pipeline()
                                          |
              +-----------+-----------+-----------+-----------+
              v           v           v           v           v
          Agent 1     Agent 2     Agent 3     Agent 4     Agent 5
         3-pass       Standard    Clinical    Cross-      Report
         intake       care        trials      indication  synthesis
            |         PubMed +    ClinTrials  LLM         (no LLM)
         Pass 1:      Qdrant      .gov API    reasoning   validate
         type                     (no LLM)                NCT IDs
         detect          |            |            |       assemble
            |            v            v            v       markdown
         Pass 2:  state.standard_ state.      state.off_      |
         extract  therapies       clinical_   label_          v
            |                     trials      hypotheses  state.final_
         Pass 3:                                          report
         clinical
         intelligence
            |
         quality
         score
              |
              v
   st.session_state["last_result"] = state
   st.rerun()
              |
              v
   Results rendered in 4 tabs:
   [Report] [Patient Profile] [Data Quality] [Raw JSON]
   + Download button (.md file)
```

---

*Generated for CDSS v5 — `cdss_streamlit.py`*
