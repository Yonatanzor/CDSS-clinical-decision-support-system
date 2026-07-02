# -*- coding: utf-8 -*-
"""
CDSS Streamlit App
Clinical Decision Support System -- patient research tool.
"""
import os
import io
import json
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CDSS -- Clinical Decision Support",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Pydantic models (same as notebook)
# ---------------------------------------------------------------------------
from pydantic import BaseModel, Field


class Biomarker(BaseModel):
    gene: str
    variant_type: str
    details: str


class ClinicalTrial(BaseModel):
    nct_id: str
    title: str
    phase: str
    status: str
    locations: List[str]
    eligibility_summary: str
    url: str


class OffLabelHypothesis(BaseModel):
    drug_name: str
    approved_indication: str
    shared_mechanism: str
    evidence_level: int
    evidence_label: str
    citation: str


class PatientProfileState(BaseModel):
    raw_input: str
    input_is_pdf: bool = False
    condition: str = ""
    stage: str = ""
    biomarkers: List[Biomarker] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    prior_therapies: List[str] = Field(default_factory=list)
    standard_therapies: List[Dict[str, Any]] = Field(default_factory=list)
    clinical_trials: List[ClinicalTrial] = Field(default_factory=list)
    off_label_hypotheses: List[OffLabelHypothesis] = Field(default_factory=list)
    validation_flags: List[str] = Field(default_factory=list)
    final_report: str = ""
    retry_count: int = 0
    max_retries: int = 2
    input_type: str = ""
    patient_age: Optional[int] = None
    patient_sex: str = ""
    patient_weight_kg: Optional[float] = None
    patient_height_cm: Optional[float] = None
    bmi: Optional[float] = None
    vitals: Dict[str, str] = Field(default_factory=dict)
    chief_complaint: str = ""
    symptoms: List[str] = Field(default_factory=list)
    symptom_duration: str = ""
    symptom_severity: str = ""
    lab_results: List[Dict[str, str]] = Field(default_factory=list)
    imaging_findings: List[Dict[str, str]] = Field(default_factory=list)
    comorbidities: List[str] = Field(default_factory=list)
    family_history: List[str] = Field(default_factory=list)
    social_history: Dict[str, str] = Field(default_factory=dict)
    allergies: List[str] = Field(default_factory=list)
    pathology_grade: str = ""
    tumor_size_cm: Optional[float] = None
    lymph_node_status: str = ""
    surgical_margins: str = ""
    immunohistochemistry: List[Dict[str, str]] = Field(default_factory=list)
    tumor_mutational_burden: str = ""
    microsatellite_status: str = ""
    pd_l1_expression: str = ""
    missed_diagnosis_flags: List[str] = Field(default_factory=list)
    modifiable_risk_factors: List[str] = Field(default_factory=list)
    non_modifiable_risk_factors: List[str] = Field(default_factory=list)
    actionable_insights: List[str] = Field(default_factory=list)
    urgent_flags: List[str] = Field(default_factory=list)
    data_quality_score: float = 0.0
    data_quality_breakdown: Dict[str, float] = Field(default_factory=dict)
    intake_raw_json: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_MODEL   = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
VECTOR_DIM    = 768
COLLECTION    = "guidelines"
TRIALS_BASE   = "https://clinicaltrials.gov/api/v2/studies"
NCBI_ESEARCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_PARAMS   = "&tool=CDSS&email=anonymous%40cdss"

PREFERENCE_ORDER = [
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
    "deepseek-r1-distill-llama-8b",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
]

EVIDENCE_LABELS = {
    1: "Pre-clinical (in-vitro only)",
    2: "Pre-clinical (animal models)",
    3: "Early clinical (Phase I/II data)",
    4: "Clinical (Phase III data for adjacent indication)",
}

VALID_INPUT_TYPES = {
    "soap_note", "discharge_summary", "lab_report", "pathology_report",
    "radiology_report", "referral_letter", "genomic_panel", "oncology_note", "free_text",
}

# ---------------------------------------------------------------------------
# Cached heavy resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading biomedical embedding model (~400 MB, once)...")
def _load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def _create_qdrant():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    return client


@st.cache_resource(show_spinner="Connecting to Groq API...")
def _create_llm(api_key: str):
    """Returns (llm, MODEL). Keyed on api_key so it refreshes when the key changes."""
    from openai import OpenAI
    llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    available = {m.id for m in llm.models.list().data}
    model = next((m for m in PREFERENCE_ORDER if m in available), None)
    if model is None:
        model = sorted(available)[0]
    return llm, model, sorted(available)


# ---------------------------------------------------------------------------
# LLM helpers (use session state refs)
# ---------------------------------------------------------------------------
def chat(prompt: str, max_tokens: int = 1024, status_cb=None) -> str:
    llm   = st.session_state["llm"]
    MODEL = st.session_state["MODEL"]
    resp  = llm.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------
def qdrant_search(query: str, top_k: int = 6) -> list:
    if not st.session_state.get("embedder_loaded"):
        return []
    try:
        embedder = _load_embedder()
        qdrant   = _create_qdrant()
        from qdrant_client.models import QueryRequest
        vec = embedder.encode(query).tolist()
        hits = qdrant.query_points(
            collection_name=COLLECTION,
            query=vec,
            limit=top_k,
        ).points
        return [h.payload for h in hits if h.payload]
    except Exception:
        return []


def ingest_guideline_pdf(pdf_bytes: bytes, filename: str) -> int:
    """Chunk a guideline PDF into Qdrant. Returns number of chunks ingested."""
    try:
        import pypdf
        embedder = _load_embedder()
        qdrant   = _create_qdrant()
        from qdrant_client.models import PointStruct

        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text   = "\n".join(p.extract_text() or "" for p in reader.pages)

        # Chunk by ~500 words
        words  = text.split()
        chunks = [" ".join(words[i:i+500]) for i in range(0, len(words), 400)]
        chunks = [c for c in chunks if len(c.strip()) > 80]

        points = []
        for i, chunk in enumerate(chunks):
            vec = embedder.encode(chunk).tolist()
            points.append(PointStruct(
                id=abs(hash(filename + str(i))) % (2**63),
                vector=vec,
                payload={"text": chunk, "source": filename, "title": filename},
            ))

        if points:
            qdrant.upsert(collection_name=COLLECTION, points=points)
        return len(points)
    except Exception as e:
        st.warning(f"Could not ingest {filename}: {e}")
        return 0


# ---------------------------------------------------------------------------
# PDF / text extraction (for patient reports)
# ---------------------------------------------------------------------------
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def _ocr_image_bytes(raw: bytes, filename: str) -> str:
    """
    Extract text from an image via Groq vision (llama-3.2-11b-vision-preview).
    Falls back to pytesseract if the vision call fails or the model is unavailable.
    """
    import base64

    ext = os.path.splitext(filename)[1].lower()
    mime = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".bmp": "image/bmp",
        ".tiff": "image/tiff", ".tif": "image/tiff",
    }.get(ext, "image/png")

    # --- Try Groq vision first (no extra install needed) ---
    llm   = st.session_state.get("llm")
    model = st.session_state.get("MODEL", "")

    # Pick a vision-capable model: prefer scout, fall back to the session model
    VISION_PREFERENCE = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "llama-3.2-90b-vision-preview",
        "llama-3.2-11b-vision-preview",
    ]
    available = set(st.session_state.get("available_models", []))
    vision_model = next((m for m in VISION_PREFERENCE if m in available), None)

    vision_error = ""
    if llm and vision_model:
        try:
            b64  = base64.b64encode(raw).decode()
            resp = llm.chat.completions.create(
                model=vision_model,
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This is a medical document image. "
                                "Transcribe ALL text exactly as it appears, preserving structure, "
                                "labels, values, and units. Do not summarize or interpret."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }],
            )
            text = resp.choices[0].message.content.strip()
            if text:
                return text
        except Exception as e:
            vision_error = str(e)
    elif not vision_model:
        vision_error = "No vision-capable model found in your Groq account."

    # --- Fallback: pytesseract (local OCR) ---
    try:
        import pytesseract
        from PIL import Image
        img  = Image.open(io.BytesIO(raw))
        result = pytesseract.image_to_string(img).strip()
        if result:
            return result
        raise ValueError("pytesseract returned empty text")
    except ImportError:
        pass
    except Exception as e:
        vision_error += f" | pytesseract: {e}"

    # Both failed — surface a clear error so the user knows
    msg = (
        f"IMAGE OCR FAILED for '{filename}'. "
        f"Reason: {vision_error or 'unknown error'}. "
        "Please paste the text content manually in the free text box instead."
    )
    st.error(msg)
    return ""


def extract_text_from_upload(uploaded_file) -> str:
    """Extract text from an uploaded file (PDF, image, or plain text)."""
    name = uploaded_file.name.lower()
    raw  = uploaded_file.read()
    ext  = os.path.splitext(name)[1]

    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw))
            pages  = [p.extract_text() or "" for p in reader.pages]
            return "\n\n".join(p for p in pages if p.strip())
        except Exception as e:
            return f"[PDF extraction failed: {e}]"

    if ext in _IMAGE_EXTS:
        return _ocr_image_bytes(raw, uploaded_file.name)

    # Plain text variants
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Agent 1 -- 3-pass intake pipeline
# ---------------------------------------------------------------------------
_BASE_JSON = """{
  "condition": "",
  "stage": "",
  "biomarkers": [{"gene": "GENE", "variant_type": "Mutation|Fusion|Amplification|Deletion|Overexpression", "details": "description"}],
  "current_medications": [],
  "prior_therapies": [],
  "patient_age": null,
  "patient_sex": "",
  "patient_weight_kg": null,
  "patient_height_cm": null,
  "vitals": {},
  "chief_complaint": "",
  "symptoms": [],
  "symptom_duration": "",
  "symptom_severity": "",
  "lab_results": [{"test": "", "value": "", "unit": "", "flag": "", "ref": ""}],
  "imaging_findings": [{"modality": "", "date": "", "finding": ""}],
  "comorbidities": [],
  "family_history": [],
  "social_history": {},
  "allergies": [],
  "pathology_grade": "",
  "tumor_size_cm": null,
  "lymph_node_status": "",
  "surgical_margins": "",
  "immunohistochemistry": [{"marker": "", "result": "", "percentage": ""}],
  "tumor_mutational_burden": "",
  "microsatellite_status": "",
  "pd_l1_expression": ""
}"""

INPUT_TYPE_PROMPT = (
    "You are a clinical document classifier.\n\n"
    "Classify the following medical text into EXACTLY ONE of these labels:\n"
    "  soap_note           -- structured S/O/A/P note from a physician visit\n"
    "  discharge_summary   -- hospital discharge document\n"
    "  lab_report          -- laboratory test results (CBC, CMP, tumor markers, etc.)\n"
    "  pathology_report    -- biopsy / surgical pathology / cytology report\n"
    "  radiology_report    -- imaging report (CT, MRI, PET, X-ray, ultrasound)\n"
    "  referral_letter     -- letter from one clinician referring to another\n"
    "  genomic_panel       -- next-generation sequencing / genomic panel / molecular report\n"
    "  oncology_note       -- oncologist clinic note (chemo plan, response assessment)\n"
    "  free_text           -- patient self-description, unstructured narrative, or none of the above\n\n"
    "Rules:\n"
    "- Return ONLY the label, no explanation, no punctuation.\n"
    "- If the text shows characteristics of multiple types, pick the PRIMARY purpose.\n"
    "- If fewer than 30 words are present, return free_text.\n\n"
    "Text:\n"
    "{raw_input}"
)

_EXTRACTION_PROMPTS = {
    "free_text": (
        "You are a clinical information extractor. Extract ALL available clinical information "
        "from the patient text below. Only extract what is explicitly stated -- never infer or hallucinate. "
        "Use null for absent numeric fields, empty strings for absent text, empty arrays for absent lists.\n\n"
        "Patient text:\n{raw_input}\n\nReturn ONLY valid JSON:\n" + _BASE_JSON
    ),
    "soap_note": (
        "You are extracting structured data from a SOAP clinical note.\n"
        "Map sections: chief_complaint/symptoms -> Subjective; vitals/lab_results/imaging_findings -> Objective; "
        "condition/stage -> Assessment; current_medications -> Plan.\n\n"
        "SOAP Note:\n{raw_input}\n\nReturn ONLY valid JSON:\n" + _BASE_JSON
    ),
    "lab_report": (
        "You are extracting structured data from a clinical laboratory report.\n"
        "Extract EVERY test result. Flag field: H (high), L (low), C (critical), '' (normal).\n"
        "If TMB, MSI, or PD-L1 appear, populate those fields too.\n\n"
        "Lab Report:\n{raw_input}\n\nReturn ONLY valid JSON:\n" + _BASE_JSON
    ),
    "pathology_report": (
        "You are extracting structured data from a surgical pathology or biopsy report.\n"
        "Focus: condition (final diagnosis), stage (TNM), pathology_grade, tumor_size_cm, "
        "lymph_node_status, surgical_margins, immunohistochemistry, biomarkers.\n\n"
        "Pathology Report:\n{raw_input}\n\nReturn ONLY valid JSON:\n" + _BASE_JSON
    ),
    "genomic_panel": (
        "You are extracting structured data from an NGS / genomic panel report.\n"
        "Focus: ALL alterations (mutations, fusions, amplifications, deletions, CNVs). "
        "For fusions, list the full name AND each component gene separately. "
        "Also extract TMB, microsatellite_status, pd_l1_expression.\n\n"
        "Genomic Report:\n{raw_input}\n\nReturn ONLY valid JSON:\n" + _BASE_JSON
    ),
    "discharge_summary": (
        "You are extracting structured data from a hospital discharge summary.\n"
        "Focus: admission/discharge diagnosis, procedures, lab highlights, discharge medications.\n\n"
        "Discharge Summary:\n{raw_input}\n\nReturn ONLY valid JSON:\n" + _BASE_JSON
    ),
    "radiology_report": (
        "You are extracting structured data from a radiology / imaging report.\n"
        "Create one imaging_findings entry per study. Extract tumor measurements -> tumor_size_cm.\n\n"
        "Radiology Report:\n{raw_input}\n\nReturn ONLY valid JSON:\n" + _BASE_JSON
    ),
    "referral_letter": (
        "You are extracting structured data from a clinical referral letter.\n"
        "Extract: patient history, reason for referral (-> chief_complaint), condition, medications.\n\n"
        "Referral Letter:\n{raw_input}\n\nReturn ONLY valid JSON:\n" + _BASE_JSON
    ),
    "oncology_note": (
        "You are extracting structured data from an oncology clinic note.\n"
        "Focus: ECOG status, current treatment cycle, response assessment (CR/PR/SD/PD), toxicities. "
        "Put treatment regimen in current_medications. Put response in stage if no formal stage.\n\n"
        "Oncology Note:\n{raw_input}\n\nReturn ONLY valid JSON:\n" + _BASE_JSON
    ),
}

CLINICAL_INTELLIGENCE_PROMPT = (
    "You are a senior clinical decision support AI with expertise in oncology, "
    "internal medicine, and evidence-based risk stratification.\n\n"
    "A structured clinical profile has been extracted from a patient document. "
    "Perform a CLINICAL INTELLIGENCE ANALYSIS -- surface patterns a busy clinician might miss. "
    "This is NOT a diagnosis -- it is a research tool to support clinical conversations.\n\n"
    "Patient Profile:\n{profile_json}\n\n"
    "Return ONLY valid JSON with these five keys:\n"
    "missed_diagnosis_flags -- conditions fitting the data not already named "
    "(format: 'Consider X because [specific data point]')\n"
    "modifiable_risk_factors -- citing specific profile values\n"
    "non_modifiable_risk_factors -- with clinical significance\n"
    "actionable_insights -- evidence-based next steps from profile data\n"
    "urgent_flags -- critical findings requiring immediate attention (empty [] if none)"
)


def _expand_fusions(biomarkers: list) -> list:
    expanded = list(biomarkers)
    for bm in biomarkers:
        if bm.variant_type == "Fusion" and "-" in bm.gene:
            for part in bm.gene.split("-"):
                expanded.append(Biomarker(
                    gene=part, variant_type="Fusion component",
                    details=f"Component of {bm.gene} fusion",
                ))
    return expanded


def compute_data_quality_score(state: PatientProfileState):
    bd = {}
    core = (0.15 if state.condition else 0) + (0.10 if state.stage else 0) + (0.05 if state.biomarkers else 0)
    bd["core_oncology"] = round(core / 0.30, 2)
    tx = (0.08 if state.current_medications else 0) + (0.07 if state.prior_therapies else 0)
    bd["treatment_history"] = round(tx / 0.15, 2)
    demo = (0.05 if state.patient_age is not None else 0) + (0.05 if state.patient_sex else 0)
    bd["demographics"] = round(demo / 0.10, 2)
    n = len(state.lab_results)
    lab = 0.0 if n == 0 else (0.07 if n <= 3 else (0.12 if n <= 10 else 0.15))
    bd["lab_data"] = round(lab / 0.15, 2)
    bio = min(0.10, (0.04 if state.tumor_mutational_burden else 0) +
                    (0.04 if state.microsatellite_status else 0) +
                    (0.04 if state.pd_l1_expression else 0))
    bd["biomarker_enrichment"] = round(bio / 0.10, 2)
    ctx = (0.04 if state.comorbidities else 0) + (0.03 if state.symptoms else 0) + (0.03 if state.imaging_findings else 0)
    bd["clinical_context"] = round(ctx / 0.10, 2)
    hist = (0.04 if state.family_history else 0) + (0.03 if state.social_history else 0) + (0.03 if state.allergies else 0)
    bd["history"] = round(hist / 0.10, 2)
    total = core + tx + demo + lab + bio + ctx + hist
    return round(total, 3), bd


def agent_intake(state: PatientProfileState, status_cb=None) -> PatientProfileState:
    raw = state.raw_input

    def log(msg):
        if status_cb:
            status_cb(msg)

    # PASS 1: type detection
    log("Pass 1/3: Detecting document type...")
    detected = chat(INPUT_TYPE_PROMPT.replace("{raw_input}", raw[:3000]), max_tokens=10).strip().lower().rstrip(".")
    state.input_type = detected if detected in VALID_INPUT_TYPES else "free_text"
    log(f"Pass 1/3: Detected -- {state.input_type}")

    # PASS 2: extraction
    log("Pass 2/3: Extracting clinical data...")
    prompt = _EXTRACTION_PROMPTS[state.input_type]
    raw_json = strip_json_fences(chat(prompt.replace("{raw_input}", raw), max_tokens=1500))
    try:
        data = json.loads(raw_json)
        state.intake_raw_json = data
        state.condition           = data.get("condition", "")
        state.stage               = data.get("stage", "")
        state.current_medications = data.get("current_medications", []) or []
        state.prior_therapies     = data.get("prior_therapies", []) or []
        raw_bm = [
            Biomarker(gene=b.get("gene", ""), variant_type=b.get("variant_type", ""), details=b.get("details", ""))
            for b in (data.get("biomarkers") or []) if b.get("gene")
        ]
        state.biomarkers          = _expand_fusions(raw_bm)
        state.patient_age         = data.get("patient_age")
        state.patient_sex         = data.get("patient_sex", "")
        state.patient_weight_kg   = data.get("patient_weight_kg")
        state.patient_height_cm   = data.get("patient_height_cm")
        if state.patient_weight_kg and state.patient_height_cm:
            h = state.patient_height_cm / 100
            state.bmi = round(state.patient_weight_kg / (h * h), 1)
        state.vitals              = data.get("vitals") or {}
        state.chief_complaint     = data.get("chief_complaint", "")
        state.symptoms            = data.get("symptoms") or []
        state.symptom_duration    = data.get("symptom_duration", "")
        state.symptom_severity    = data.get("symptom_severity", "")
        state.lab_results         = [r for r in (data.get("lab_results") or []) if r.get("test")]
        state.imaging_findings    = [r for r in (data.get("imaging_findings") or []) if r.get("finding")]
        state.comorbidities       = data.get("comorbidities") or []
        state.family_history      = data.get("family_history") or []
        state.social_history      = data.get("social_history") or {}
        state.allergies           = data.get("allergies") or []
        state.pathology_grade     = data.get("pathology_grade", "")
        state.tumor_size_cm       = data.get("tumor_size_cm")
        state.lymph_node_status   = data.get("lymph_node_status", "")
        state.surgical_margins    = data.get("surgical_margins", "")
        state.immunohistochemistry     = data.get("immunohistochemistry") or []
        state.tumor_mutational_burden  = data.get("tumor_mutational_burden", "")
        state.microsatellite_status    = data.get("microsatellite_status", "")
        state.pd_l1_expression         = data.get("pd_l1_expression", "")
    except (json.JSONDecodeError, KeyError) as e:
        state.validation_flags.append(f"Intake parse error: {e}")

    # PASS 3: clinical intelligence
    log("Pass 3/3: Running clinical intelligence analysis...")
    profile_summary = {
        "condition": state.condition, "stage": state.stage,
        "biomarkers": [{"gene": b.gene, "variant_type": b.variant_type} for b in state.biomarkers[:10]],
        "current_medications": state.current_medications, "prior_therapies": state.prior_therapies,
        "patient_age": state.patient_age, "patient_sex": state.patient_sex, "bmi": state.bmi,
        "vitals": state.vitals, "chief_complaint": state.chief_complaint, "symptoms": state.symptoms,
        "lab_results": state.lab_results[:15], "imaging_findings": state.imaging_findings,
        "comorbidities": state.comorbidities, "family_history": state.family_history,
        "social_history": state.social_history,
        "tumor_mutational_burden": state.tumor_mutational_burden,
        "microsatellite_status": state.microsatellite_status,
        "pd_l1_expression": state.pd_l1_expression,
    }
    intel_raw = strip_json_fences(
        chat(CLINICAL_INTELLIGENCE_PROMPT.replace("{profile_json}", json.dumps(profile_summary, indent=2)), max_tokens=1200)
    )
    try:
        intel = json.loads(intel_raw)
        state.missed_diagnosis_flags      = intel.get("missed_diagnosis_flags", [])
        state.modifiable_risk_factors     = intel.get("modifiable_risk_factors", [])
        state.non_modifiable_risk_factors = intel.get("non_modifiable_risk_factors", [])
        state.actionable_insights         = intel.get("actionable_insights", [])
        state.urgent_flags                = intel.get("urgent_flags", [])
        for flag in state.urgent_flags:
            state.validation_flags.append(f"URGENT: {flag}")
    except (json.JSONDecodeError, KeyError) as e:
        state.validation_flags.append(f"Clinical intelligence parse error: {e}")

    state.data_quality_score, state.data_quality_breakdown = compute_data_quality_score(state)
    return state


# ---------------------------------------------------------------------------
# Agent 2 -- Standard care (PubMed + Qdrant)
# ---------------------------------------------------------------------------
CARE_PROMPT = (
    "You are helping a patient understand standard treatment options for their condition.\n"
    "The sources below are a mix of clinical practice guideline abstracts and published research summaries "
    "retrieved from PubMed and/or uploaded guideline PDFs. Synthesise them into a plain-English summary "
    "of current standard-of-care approaches.\n\n"
    "Rules:\n"
    "- Only use information present in the sources below. Do not add external knowledge.\n"
    "- Do not mention specific doses or brand names unless they appear in the sources.\n"
    "- Distinguish between first-line and subsequent options if the sources make this distinction.\n"
    "- End with: 'Ask your doctor which of these options apply to your specific situation.'\n\n"
    "Patient condition: {condition} -- Stage: {stage}\n"
    "Prior therapies already tried: {prior_therapies}\n\n"
    "Sources ({source_count} total -- {pubmed_count} from PubMed, {qdrant_count} from uploaded PDFs):\n"
    "{excerpts}\n\n"
    "Write a plain-English summary (3-5 sentences per treatment category). Use markdown headers."
)


def _ncbi_get(url: str, retries: int = 1) -> bytes:
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries:
                time.sleep(2)
                continue
            raise


def search_pubmed_guidelines(condition: str, stage: str) -> list:
    if not condition:
        return []
    stage_clause = f' "{stage}"[tiab]' if stage and stage.lower() not in ("", "unspecified") else ""
    base  = f'"{condition}"[tiab]{stage_clause}'
    queries = [
        f'{base} AND "practice guideline"[pt]',
        f'{base} AND ("systematic review"[pt] OR "meta-analysis"[pt] OR "guideline"[tiab])',
    ]
    pmids = []
    for query in queries:
        try:
            enc  = urllib.parse.quote(query)
            url  = f"{NCBI_ESEARCH}?db=pubmed&term={enc}&retmax=8&sort=relevance&retmode=json{NCBI_PARAMS}"
            data = json.loads(_ncbi_get(url))
            pmids = data.get("esearchresult", {}).get("idlist", [])
            if len(pmids) >= 3:
                break
        except Exception:
            return []
    if not pmids:
        return []
    try:
        ids_str   = ",".join(pmids)
        url       = f"{NCBI_EFETCH}?db=pubmed&id={ids_str}&rettype=abstract&retmode=xml{NCBI_PARAMS}"
        xml_bytes = _ncbi_get(url)
        root      = ET.fromstring(xml_bytes)
    except Exception:
        return []
    results = []
    for article in root.findall(".//PubmedArticle"):
        try:
            pmid  = article.findtext(".//PMID", "")
            title = article.findtext(".//ArticleTitle", "")
            parts = article.findall(".//AbstractText")
            abstract = " ".join(
                (f"{p.get('Label', '')}: " if p.get("Label") else "") + (p.text or "")
                for p in parts
            ).strip()
            if not abstract:
                continue
            words = abstract.split()
            if len(words) > 600:
                abstract = " ".join(words[:600]) + " [truncated]"
            results.append({"pmid": pmid, "title": title, "text": abstract, "source": f"PubMed PMID:{pmid}"})
        except Exception:
            continue
    return results


def _deduplicate(pubmed: list, qdrant: list, cap: int = 10) -> list:
    seen, merged = set(), []
    for hit in qdrant + pubmed:
        key = hit.get("text", "")[:120].strip()
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
        if len(merged) >= cap:
            break
    return merged


def agent_standard_care(state: PatientProfileState, status_cb=None) -> PatientProfileState:
    if not state.condition:
        state.validation_flags.append("Standard care skipped: no condition extracted.")
        return state
    if status_cb:
        status_cb(f"Searching PubMed for guidelines: {state.condition}...")

    try:
        pubmed_hits = search_pubmed_guidelines(state.condition, state.stage or "")
    except Exception as e:
        pubmed_hits = []
        state.validation_flags.append(f"PubMed search failed: {e}")

    qdrant_hits = qdrant_search(f"{state.condition} {state.stage} treatment guidelines")
    all_hits    = _deduplicate(pubmed_hits, qdrant_hits)

    if not all_hits:
        state.standard_therapies = [{"summary": (
            "No guideline data found. PubMed returned no matching practice guidelines. "
            "Try a broader condition name, or upload NCCN/ESMO PDFs in the sidebar."
        )}]
        return state

    excerpt_blocks = [
        f"[Source: {h['source']}]" + (f"\n{h.get('title','')}" if h.get("title") else "") + f"\n{h['text']}"
        for h in all_hits
    ]
    summary = chat(CARE_PROMPT.format(
        condition=state.condition, stage=state.stage or "unspecified",
        prior_therapies=", ".join(state.prior_therapies) or "none mentioned",
        source_count=len(all_hits), pubmed_count=len(pubmed_hits), qdrant_count=len(qdrant_hits),
        excerpts="\n\n---\n\n".join(excerpt_blocks),
    ), max_tokens=1500)

    state.standard_therapies = [{"summary": summary, "sources": [
        *[f"PubMed:{h['pmid']}" for h in pubmed_hits],
        *[h["source"] for h in qdrant_hits],
    ]}]
    return state


# ---------------------------------------------------------------------------
# Agent 3 -- Clinical trials
# ---------------------------------------------------------------------------
def agent_clinical_trials(state: PatientProfileState, status_cb=None) -> PatientProfileState:
    if not state.condition:
        state.validation_flags.append("Clinical trials skipped: no condition extracted.")
        return state
    if status_cb:
        status_cb("Searching ClinicalTrials.gov for recruiting trials...")

    biomarker_terms = [b.gene for b in state.biomarkers if b.gene]
    params = {
        "query.cond": state.condition,
        "filter.overallStatus": "RECRUITING",
        "pageSize": 10,
        "format": "json",
        "fields": "NCTId,BriefTitle,Phase,OverallStatus,LocationCity,LocationCountry,EligibilityCriteria",
    }
    if biomarker_terms:
        params["query.term"] = " OR ".join(biomarker_terms[:5])

    url = TRIALS_BASE + "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v})
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        state.validation_flags.append(f"Trials API error: {e}")
        return state

    for study in data.get("studies", []):
        proto   = study.get("protocolSection", {})
        id_mod  = proto.get("identificationModule", {})
        st_mod  = proto.get("statusModule", {})
        de_mod  = proto.get("designModule", {})
        lc_mod  = proto.get("contactsLocationsModule", {})
        el_mod  = proto.get("eligibilityModule", {})
        nct_id  = id_mod.get("nctId", "")
        phases  = de_mod.get("phases", [])
        locs    = list({
            f"{l.get('city','')}, {l.get('country','')}".strip(", ")
            for l in lc_mod.get("locations", []) if l.get("city") or l.get("country")
        })[:5]
        elig    = el_mod.get("eligibilityCriteria", "")
        state.clinical_trials.append(ClinicalTrial(
            nct_id=nct_id,
            title=id_mod.get("briefTitle", ""),
            phase=", ".join(phases) if phases else "N/A",
            status=st_mod.get("overallStatus", ""),
            locations=locs,
            eligibility_summary=(elig[:500] + "...") if len(elig) > 500 else elig,
            url=f"https://clinicaltrials.gov/study/{nct_id}",
        ))
    return state


# ---------------------------------------------------------------------------
# Agent 4 -- Cross-indication (LLM fallback, optional PrimeKG)
# ---------------------------------------------------------------------------
HYPOTHESIS_PROMPT_LLM = (
    "You are an expert oncologist and molecular biologist specializing in rare and refractory cancers.\n\n"
    "A patient has {condition} (stage: {stage}).\n"
    "Genomic profile: {biomarkers}\n"
    "Prior treatments that failed: {prior_therapies}\n\n"
    "Identify up to 8 off-label drug candidates approved for OTHER conditions that have a strong "
    "mechanistic rationale for this patient's specific genomic alterations.\n\n"
    "Reasoning framework:\n"
    "1. LOSS-OF-FUNCTION/DELETION: what pathway is unrestrained? what is the synthetic lethal partner?\n"
    "   (e.g. SMARCB1 loss -> unopposed EZH2 -> Tazemetostat; PTEN loss -> PI3K/mTOR -> Everolimus)\n"
    "2. GAIN-OF-FUNCTION/FUSION/AMPLIFICATION: what kinase is constitutively active?\n"
    "3. CHROMATIN REMODELING complex loss: consider EZH2, HDAC, BET bromodomain inhibitors\n"
    "4. TISSUE-AGNOSTIC approvals: TMB-High -> Pembrolizumab; NTRK fusion -> Larotrectinib\n"
    "5. BASKET TRIAL evidence even if pre-clinical\n\n"
    "Return ONLY a JSON array:\n"
    "[\n"
    "  {{\n"
    "    \"drug_name\": \"generic name\",\n"
    "    \"approved_indication\": \"current approved use\",\n"
    "    \"shared_mechanism\": \"specific 2-3 sentence mechanistic rationale\",\n"
    "    \"evidence_level\": 1|2|3|4,\n"
    "    \"citation\": \"https://pubmed.ncbi.nlm.nih.gov/?term=DRUG+ALTERATION+cancer\"\n"
    "  }}\n"
    "]\n"
    "Evidence: 1=in-vitro, 2=animal model, 3=Phase I/II, 4=Phase III adjacent indication."
)


def agent_cross_indication(state: PatientProfileState, status_cb=None) -> PatientProfileState:
    if not state.biomarkers:
        state.validation_flags.append("Cross-indication skipped: no biomarkers extracted.")
        return state
    if status_cb:
        status_cb("Running cross-indication discovery (LLM reasoning)...")

    biomarkers_text = ", ".join(
        f"{b.gene} {b.variant_type} ({b.details})" for b in state.biomarkers
        if b.variant_type != "Fusion component"
    )
    state.validation_flags.append("PrimeKG not loaded -- using LLM biomedical reasoning for cross-indication discovery.")

    text = strip_json_fences(chat(HYPOTHESIS_PROMPT_LLM.format(
        condition=state.condition, stage=state.stage or "unspecified",
        biomarkers=biomarkers_text,
        prior_therapies=", ".join(state.prior_therapies) or "none",
    ), max_tokens=3000))

    try:
        for h in json.loads(text):
            level = int(h.get("evidence_level", 1))
            state.off_label_hypotheses.append(OffLabelHypothesis(
                drug_name=h.get("drug_name", ""),
                approved_indication=h.get("approved_indication", ""),
                shared_mechanism=h.get("shared_mechanism", ""),
                evidence_level=level,
                evidence_label=EVIDENCE_LABELS.get(level, "Unknown"),
                citation=h.get("citation", ""),
            ))
    except (json.JSONDecodeError, KeyError) as e:
        state.validation_flags.append(f"Cross-indication parse error: {e}")

    return state


# ---------------------------------------------------------------------------
# Agent 5 -- Synthesizer
# ---------------------------------------------------------------------------
DISCLAIMER = (
    "\n---\n"
    "**Important Disclaimer**\n\n"
    "This report was generated by an AI research tool and is for **informational and "
    "educational purposes only**. It does **not** constitute medical advice, diagnosis, "
    "or a treatment recommendation.\n\n"
    "- Clinical trial eligibility must be confirmed by a qualified physician\n"
    "- Off-label therapy hypotheses require evaluation by your specialist\n"
    "- Never start, stop, or change a treatment based on this report alone\n\n"
    "Always consult a licensed medical professional before making any treatment decisions.\n"
)


def validate_nct_id(nct_id: str) -> bool:
    try:
        url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}?format=json&fields=NCTId"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _quality_badge(score: float) -> str:
    if score >= 0.70:
        return f"HIGH ({score:.0%})"
    elif score >= 0.40:
        return f"MEDIUM ({score:.0%})"
    else:
        return f"LOW ({score:.0%})"


def agent_synthesizer(state: PatientProfileState, status_cb=None) -> PatientProfileState:
    if status_cb:
        status_cb("Composing final report...")

    lines = [
        "# Patient Research Report",
        "*Generated by CDSS -- for informational purposes only. Not medical advice.*\n",
    ]

    if state.urgent_flags:
        lines.append("## Urgent Findings\n")
        lines.append("> **These items may require immediate clinical attention.**\n")
        for flag in state.urgent_flags:
            lines.append(f"- **[!] {flag}**")
        lines.append("\n---\n")

    badge = _quality_badge(state.data_quality_score)
    lines.append(f"## Your Profile  |  Data quality: {badge}\n")
    lines.append(f"- **Document type:** {state.input_type.replace('_', ' ').title() if state.input_type else 'Unknown'}")
    lines.append(f"- **Condition:** {state.condition or 'Not specified'}")
    lines.append(f"- **Stage / Severity:** {state.stage or 'Not specified'}")

    demo_parts = list(filter(None, [
        f"{state.patient_age}yo" if state.patient_age else "",
        state.patient_sex,
        f"BMI {state.bmi}" if state.bmi else "",
    ]))
    if demo_parts:
        lines.append(f"- **Demographics:** {', '.join(demo_parts)}")

    core_bm = [b for b in state.biomarkers if b.variant_type != "Fusion component"]
    if core_bm:
        lines.append(f"- **Key Biomarkers:** {', '.join(f'{b.gene} {b.variant_type}' for b in core_bm)}")
    if state.tumor_mutational_burden:
        lines.append(f"- **TMB:** {state.tumor_mutational_burden}")
    if state.microsatellite_status:
        lines.append(f"- **MSI Status:** {state.microsatellite_status}")
    if state.pd_l1_expression:
        lines.append(f"- **PD-L1:** {state.pd_l1_expression}")
    if state.current_medications:
        lines.append(f"- **Current Medications:** {', '.join(state.current_medications)}")
    if state.prior_therapies:
        lines.append(f"- **Prior Therapies:** {', '.join(state.prior_therapies)}")
    if state.comorbidities:
        lines.append(f"- **Comorbidities:** {', '.join(state.comorbidities)}")
    lines.append("")

    if state.actionable_insights or state.modifiable_risk_factors or state.missed_diagnosis_flags:
        lines.append("## Clinical Insights\n")
        if state.actionable_insights:
            lines.append("**Actionable next steps based on your profile:**\n")
            for i in state.actionable_insights:
                lines.append(f"- {i}")
            lines.append("")
        if state.modifiable_risk_factors:
            lines.append("**Modifiable risk factors:**\n")
            for r in state.modifiable_risk_factors:
                lines.append(f"- {r}")
            lines.append("")
        if state.non_modifiable_risk_factors:
            lines.append("**Non-modifiable risk factors:**\n")
            for r in state.non_modifiable_risk_factors:
                lines.append(f"- {r}")
            lines.append("")
        if state.missed_diagnosis_flags:
            lines.append("**Considerations for clinical discussion:**\n")
            lines.append("> *These are patterns for clinical discussion, not diagnoses.*\n")
            for f in state.missed_diagnosis_flags:
                lines.append(f"- {f}")
            lines.append("")

    lines.append("## Standard Treatment Options\n")
    lines.append(state.standard_therapies[0]["summary"] if state.standard_therapies
                 else "*No guideline data available. Upload NCCN/ESMO PDFs in the sidebar.*")
    lines.append("")

    lines.append("## Active Clinical Trials (Worldwide)\n")
    if status_cb:
        status_cb("Validating NCT IDs...")
    valid_trials = [t for t in state.clinical_trials if validate_nct_id(t.nct_id)]
    for t in state.clinical_trials:
        if t not in valid_trials:
            state.validation_flags.append(f"Trial {t.nct_id} could not be verified -- excluded.")

    if valid_trials:
        lines += ["| Trial | Phase | Locations | NCT ID |", "|-------|-------|-----------|--------|"]
        for t in valid_trials:
            locs  = "; ".join(t.locations[:3]) or "See link"
            title = (t.title[:55] + "...") if len(t.title) > 55 else t.title
            lines.append(f"| [{title}]({t.url}) | {t.phase} | {locs} | {t.nct_id} |")
        lines += ["", '> **Next step:** Ask your doctor: *"Do I qualify for [trial name] given my profile?"*']
    else:
        lines.append("*No recruiting trials found matching your profile at this time.*")
    lines.append("")

    lines.append("## Off-Label Therapy Hypotheses\n")
    lines.append("Treatments approved for **other conditions** that share biological pathways with yours. "
                 "Require physician evaluation before any consideration.\n")
    lines.append("**Evidence:** 1 = in-vitro only | 2 = animal models | 3 = Phase I/II | 4 = Phase III adjacent\n")
    if state.off_label_hypotheses:
        lines += ["| Drug | Approved For | Shared Mechanism | Evidence | Source |",
                  "|------|-------------|-----------------|----------|--------|"]
        for h in state.off_label_hypotheses:
            note  = " [!]" if h.evidence_level < 2 else ""
            lines.append(
                f"| **{h.drug_name}** | {h.approved_indication} | {h.shared_mechanism} | "
                f"{h.evidence_level} -- {h.evidence_label}{note} | [PubMed]({h.citation}) |"
            )
        lines += ["", "> **Next step:** For drugs with evidence >= 3, ask your doctor."]
    else:
        lines.append("*No cross-indication hypotheses found for your biomarker profile.*")
    lines.append("")

    lines.append("## Questions to Ask Your Doctor\n")
    for t in valid_trials[:3]:
        lines.append(f'- Am I eligible for the trial "{t.title[:50]}..." ({t.nct_id})?')
    for h in [h for h in state.off_label_hypotheses if h.evidence_level >= 3]:
        genes = ", ".join(b.gene for b in state.biomarkers[:2] if b.variant_type != "Fusion component")
        lines.append(f"- Has {h.drug_name} (approved for {h.approved_indication}) been considered for my {genes} profile?")
    if not valid_trials and not any(h.evidence_level >= 3 for h in state.off_label_hypotheses):
        lines += [
            "- What clinical trials are currently recruiting for my condition and biomarkers?",
            "- Are there off-label treatments being studied for my specific mutation?",
        ]
    lines.append("")

    _ERROR_KW = ("error", "failed", "excluded", "could not")
    error_flags = [f for f in state.validation_flags
                   if any(k in f.lower() for k in _ERROR_KW) and not f.startswith("URGENT:")]
    if error_flags:
        lines.append("## System Notes\n")
        for flag in error_flags:
            lines.append(f"- [!] {flag}")
        lines.append("")

    lines.append(DISCLAIMER)
    state.final_report = "\n".join(lines)
    return state


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------
def run_pipeline(state: PatientProfileState, status_container) -> PatientProfileState:
    steps = [
        ("Agent 1/5: Parsing patient profile (3-pass)...", agent_intake),
        ("Agent 2/5: Retrieving standard care guidelines...",  agent_standard_care),
        ("Agent 3/5: Searching clinical trials...",            agent_clinical_trials),
        ("Agent 4/5: Discovering cross-indication hypotheses...", agent_cross_indication),
        ("Agent 5/5: Composing final report...",               agent_synthesizer),
    ]
    with status_container.status("Running CDSS pipeline...", expanded=True) as st_status:
        for label, fn in steps:
            st.write(label)
            def _cb(msg, _label=label):
                st.write(f"  {msg}")
            state = fn(state, status_cb=_cb)

        # Retry loop (same as notebook)
        retries = 0
        while any("error" in f.lower() for f in state.validation_flags) and retries < state.max_retries:
            retries += 1
            st.write(f"Retrying pipeline (attempt {retries}/{state.max_retries})...")
            state.retry_count = retries
            state = agent_intake(state)

        st_status.update(label="Pipeline complete!", state="complete")
    return state


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def _render_quality_bars(breakdown: dict, total: float):
    import streamlit as st

    dimension_weights = {
        "core_oncology":      0.30,
        "treatment_history":  0.15,
        "demographics":       0.10,
        "lab_data":           0.15,
        "biomarker_enrichment": 0.10,
        "clinical_context":   0.10,
        "history":            0.10,
    }
    labels = {
        "core_oncology":        "Core Oncology (condition, stage, biomarkers)",
        "treatment_history":    "Treatment History (medications, prior therapies)",
        "demographics":         "Demographics (age, sex)",
        "lab_data":             "Lab Data (test results)",
        "biomarker_enrichment": "Biomarker Enrichment (TMB, MSI, PD-L1)",
        "clinical_context":     "Clinical Context (comorbidities, symptoms, imaging)",
        "history":              "History (family, social, allergies)",
    }
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown("**Dimension-level completeness**")
    with cols[1]:
        color = "green" if total >= 0.70 else ("orange" if total >= 0.40 else "red")
        st.markdown(f"**Overall: :{color}[{total:.0%}]**")

    for key, weight in dimension_weights.items():
        ratio = breakdown.get(key, 0.0)
        contrib = ratio * weight
        st.progress(ratio, text=f"{labels[key]} — {ratio:.0%} ({contrib:.2f}/{weight:.2f})")


def _render_profile_card(state: PatientProfileState):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Core Profile**")
        st.markdown(f"- **Document type:** {state.input_type.replace('_', ' ').title() or 'Unknown'}")
        st.markdown(f"- **Condition:** {state.condition or 'Not extracted'}")
        st.markdown(f"- **Stage:** {state.stage or 'Not extracted'}")
        if state.patient_age or state.patient_sex:
            demo = ", ".join(filter(None, [
                f"{state.patient_age}yo" if state.patient_age else "",
                state.patient_sex,
                f"BMI {state.bmi}" if state.bmi else "",
            ]))
            st.markdown(f"- **Demographics:** {demo}")
        if state.vitals:
            st.markdown(f"- **Vitals:** {', '.join(f'{k}: {v}' for k, v in state.vitals.items())}")
        if state.chief_complaint:
            st.markdown(f"- **Chief complaint:** {state.chief_complaint}")
        if state.symptoms:
            st.markdown(f"- **Symptoms:** {', '.join(state.symptoms)}")
        if state.comorbidities:
            st.markdown(f"- **Comorbidities:** {', '.join(state.comorbidities)}")
        if state.allergies:
            st.markdown(f"- **Allergies:** {', '.join(state.allergies)}")

    with c2:
        st.markdown("**Biomarkers & Molecular**")
        core_bm = [b for b in state.biomarkers if b.variant_type != "Fusion component"]
        if core_bm:
            for b in core_bm:
                st.markdown(f"- **{b.gene}** {b.variant_type}: {b.details}")
        else:
            st.markdown("*None extracted*")
        if state.tumor_mutational_burden:
            st.markdown(f"- **TMB:** {state.tumor_mutational_burden}")
        if state.microsatellite_status:
            st.markdown(f"- **MSI:** {state.microsatellite_status}")
        if state.pd_l1_expression:
            st.markdown(f"- **PD-L1:** {state.pd_l1_expression}")

        if state.lab_results:
            st.markdown(f"**Lab Results ({len(state.lab_results)} tests)**")
            rows = []
            for r in state.lab_results[:12]:
                flag = r.get("flag", "")
                val  = r.get("value", "") + " " + r.get("unit", "")
                rows.append({"Test": r.get("test", ""), "Value": val.strip(),
                             "Flag": flag, "Ref": r.get("ref", "")})
            st.dataframe(rows, use_container_width=True)

        if state.imaging_findings:
            st.markdown(f"**Imaging ({len(state.imaging_findings)} studies)**")
            for img in state.imaging_findings:
                st.markdown(f"- **{img.get('modality', '')}** ({img.get('date', '')}): {img.get('finding', '')}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.title("CDSS Settings")
        st.markdown("*Clinical Decision Support System*")
        st.divider()

        # API key
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state.get("api_key_input", ""),
            placeholder="gsk_...",
            help="Get a free key at console.groq.com",
        )

        if api_key and api_key != st.session_state.get("api_key_input"):
            st.session_state["api_key_input"] = api_key
            # Clear cached client so it re-initializes with new key
            st.session_state["llm_ready"] = False

        if api_key and not st.session_state.get("llm_ready"):
            with st.spinner("Connecting to Groq..."):
                try:
                    llm, model, available = _create_llm(api_key)
                    st.session_state["llm"]      = llm
                    st.session_state["MODEL"]     = model
                    st.session_state["llm_ready"] = True
                    st.session_state["available_models"] = available
                except Exception as e:
                    st.error(f"API key error: {e}")
                    st.session_state["llm_ready"] = False

        if st.session_state.get("llm_ready"):
            model = st.session_state.get("MODEL", "")
            st.success(f"Connected: **{model}**")
            with st.expander("Available Groq models", expanded=False):
                for m in st.session_state.get("available_models", []):
                    st.code(m, language=None)
        else:
            st.info("Enter your Groq API key to begin.")

        st.divider()

        # Guideline PDF upload
        st.markdown("**Upload Guideline PDFs** (optional)")
        st.caption("NCCN, ESMO, or any clinical PDF — boosts standard care quality.")
        guideline_files = st.file_uploader(
            "Drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            key="guideline_upload",
            label_visibility="collapsed",
        )

        if guideline_files:
            already = st.session_state.get("ingested_guidelines", set())
            new_files = [f for f in guideline_files if f.name not in already]
            if new_files:
                if st.button(f"Ingest {len(new_files)} PDF(s) into knowledge base"):
                    # Trigger embedder load
                    with st.spinner("Loading embedding model..."):
                        _load_embedder()
                        st.session_state["embedder_loaded"] = True
                    total_chunks = 0
                    for f in new_files:
                        chunks = ingest_guideline_pdf(f.read(), f.name)
                        total_chunks += chunks
                        already.add(f.name)
                    st.session_state["ingested_guidelines"] = already
                    st.success(f"Ingested {total_chunks} chunks from {len(new_files)} PDF(s).")
            else:
                st.caption(f"{len(already)} PDF(s) already ingested.")

        st.divider()
        st.caption("v5 | Groq + LangGraph + PubMed + ClinicalTrials.gov")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    render_sidebar()

    # --- Header ---
    st.markdown(
        "<h1 style='margin-bottom:0'>🧬 CDSS</h1>"
        "<p style='color:gray;font-size:1.1rem;margin-top:4px'>"
        "Clinical Decision Support System &mdash; AI-powered patient research tool</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    if not st.session_state.get("llm_ready"):
        st.info("Enter your **Groq API key** in the sidebar to get started. "
                "Get a free key at [console.groq.com](https://console.groq.com).")
        st.stop()

    # --- Input Section ---
    st.markdown("## Step 1 — Provide Patient Information")
    st.caption(
        "Upload one or more medical reports **and/or** paste text below. "
        "Multiple files are merged automatically. Supported: PDF, PNG, JPG, TXT, CSV, RTF. "
        "Images are transcribed via Groq vision (no extra install needed)."
    )

    col_upload, col_text = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("**Upload Medical Reports**")
        uploaded_reports = st.file_uploader(
            "Drop files here",
            type=["pdf", "png", "jpg", "jpeg", "txt", "csv", "rtf"],
            accept_multiple_files=True,
            key="patient_reports",
            label_visibility="collapsed",
        )
        if uploaded_reports:
            st.caption(f"{len(uploaded_reports)} file(s) selected:")
            for f in uploaded_reports:
                st.markdown(f"- `{f.name}` ({f.size // 1024 + 1} KB)")

    with col_text:
        st.markdown("**Free-text Description**")
        free_text = st.text_area(
            "Paste clinical notes, patient description, or any medical text here.",
            height=200,
            placeholder=(
                "Example: I am a 54-year-old patient with Stage III NSCLC.\n"
                "My genomic report shows an EGFR exon 19 deletion mutation.\n"
                "I am currently on osimertinib (Tagrisso) as first-line treatment.\n"
                "No prior chemotherapy."
            ),
            label_visibility="collapsed",
        )

    # Combine: extracted file text + free text
    combined_parts = []
    if uploaded_reports:
        for f in uploaded_reports:
            extracted = extract_text_from_upload(f)
            if extracted.strip():
                combined_parts.append(f"[Source: {f.name}]\n{extracted}")

    if free_text.strip():
        combined_parts.append(free_text.strip())

    combined_input = "\n\n---\n\n".join(combined_parts)

    if not combined_input.strip():
        st.warning("Please upload at least one file or enter text before running the analysis.")

    st.divider()

    # --- Run button ---
    run_col, _ = st.columns([2, 3])
    with run_col:
        run_clicked = st.button(
            "Run CDSS Analysis",
            type="primary",
            disabled=not combined_input.strip(),
            use_container_width=True,
        )

    if run_clicked and combined_input.strip():
        st.session_state["last_result"] = None
        initial_state = PatientProfileState(raw_input=combined_input)
        status_area   = st.container()

        with st.spinner("Analyzing..."):
            result = run_pipeline(initial_state, status_area)

        st.session_state["last_result"] = result
        st.rerun()

    # --- Results ---
    result: PatientProfileState = st.session_state.get("last_result")
    if result is None:
        return

    st.divider()
    st.markdown("## Results")

    tabs = st.tabs(["Report", "Patient Profile", "Data Quality", "Raw JSON"])

    with tabs[0]:
        if result.urgent_flags:
            st.error("**Urgent findings detected** — see report below for details.")

        st.markdown(result.final_report)

        st.download_button(
            label="Download Report (.md)",
            data=result.final_report.encode("utf-8"),
            file_name="cdss_report.md",
            mime="text/markdown",
        )

    with tabs[1]:
        _render_profile_card(result)

    with tabs[2]:
        _render_quality_bars(result.data_quality_breakdown, result.data_quality_score)

        if result.actionable_insights:
            st.divider()
            st.markdown("**Actionable Insights from Clinical Intelligence Pass**")
            for insight in result.actionable_insights:
                st.markdown(f"- {insight}")

        if result.missed_diagnosis_flags:
            st.divider()
            st.caption("*Patterns for clinical discussion only — not diagnoses*")
            st.markdown("**Considerations for Clinical Discussion**")
            for flag in result.missed_diagnosis_flags:
                st.markdown(f"- {flag}")

    with tabs[3]:
        st.caption("Raw extracted JSON from Agent 1 Pass 2 (for debugging)")
        st.json(result.intake_raw_json)
        st.divider()
        st.caption("All validation flags")
        for flag in result.validation_flags:
            if flag.startswith("URGENT:"):
                st.error(flag)
            elif "error" in flag.lower() or "failed" in flag.lower():
                st.warning(flag)
            else:
                st.info(flag)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Session state defaults
    for key in ("llm_ready", "embedder_loaded", "last_result", "api_key_input"):
        if key not in st.session_state:
            st.session_state[key] = None if key == "last_result" else False

    main()
