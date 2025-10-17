# main.py
import os
import io
import time
import json
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from groq import Groq
from crewai import Agent, Task, Crew, LLM as CrewLLM

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

load_dotenv()

REQUIRED_DISCLOSURES = {
    "insurance": [
        "policy terms and conditions",
        "exclusions and waiting periods",
        "premium, charges, and taxes",
        "free-look period",
        "claim process and documentation",
        "consent to call recording and use of data",
    ],
    "loan": [
        "interest rate and APR equivalent",
        "processing and other fees",
        "EMI amount and tenure",
        "prepayment/foreclosure charges",
        "KYC and consent",
        "impact of defaults on credit score",
    ],
    "credit_card": [
        "annual and joining fees",
        "interest/finance charges",
        "late payment and overlimit fees",
        "billing cycle and due dates",
        "KYC and consent",
        "grievance redressal details",
    ],
    "investment": [
        "risk of capital loss",
        "fees/loads/expense ratio",
        "lock-in or exit load",
        "suitability vs risk tolerance",
        "KYC and consent",
        "past performance is not indicative",
    ],
}

DEFAULT_MODEL = os.getenv("LLM_MODEL", "groq/llama-3.1-70b-versatile")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

def get_llm(model: Optional[str] = None, temperature: float = DEFAULT_TEMPERATURE):
    m = model or DEFAULT_MODEL
    if not m.startswith("groq/"):
        m = f"groq/{m}"
    return CrewLLM(model=m, temperature=temperature)

def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        cleaned = text.strip().strip("`").replace("\n``````", "")
        try:
            return json.loads(cleaned)
        except Exception:
            return {"raw": text}

def transcribe_audio_groq(audio_path: str, language: str = "en") -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            file=f,
            model=os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo"),
            response_format="verbose_json",
            language=language,
            temperature=0.0,
        )
    try:
        text = getattr(resp, "text", None)
        if isinstance(resp, dict):
            text = resp.get("text", text)
        return (text or "").strip()
    except Exception:
        try:
            return json.dumps(resp, ensure_ascii=False)[:10000]
        except Exception:
            return ""

EXTRACT_SCHEMA = """
You are an expert compliance analyst for BFSI calls.
Extract a compact JSON with:
- customer_profile: { "income_bracket": "...", "age": "...", "risk_tolerance": "...", "location": "...", "existing_products": ["..."] }
- product_interest: { "type": "loan|credit_card|insurance|investment", "variant": "...", "tenure_or_term": "...", "amount_or_sum_assured": "..." }
- disclosures_mentioned: ["..."]
- red_flags: ["..."]
- sentiment_summary: "..."
- competitor_mentions: ["..."]
Only output minified JSON.
"""

COMPLIANCE_SCHEMA = """
Given product_type and transcript facts, output JSON:
{
 "missing_disclosures": ["..."],
 "suitability_issues": ["..."],
 "risk_flags": ["..."],
 "coaching_points": ["..."],
 "summary": "1-2 lines"
}
Be strict and grounded in transcript snippets; do not invent.
Output minified JSON only.
"""

EMAIL_PROMPT = """
Draft a concise, compliant corrective email to the customer acknowledging our conversation.
Include: correct disclosures missing earlier, neutral tone, clear next steps, and link placeholders.
Use Indian English style and keep it under 180 words.
Variables:
customer_name: {customer_name}
product_type: {product_type}
issues: {issues}
"""

SCRIPT_PROMPT = """
Draft a compliant next-call script for the sales agent to address gaps.
Include: greeting, recap, mandatory disclosures (short), suitability confirmation questions, and graceful close.
Keep it under 180 words.
Variables:
agent_name: {agent_name}
product_type: {product_type}
issues: {issues}
"""

def build_agents(model: Optional[str] = None):
    llm = get_llm(model=model)
    extractor = Agent(
        role="Entity & Fact Extractor",
        goal="Extract structured entities, disclosures, risks, competitor mentions, and sentiment.",
        backstory="Specialist in BFSI conversation analysis for compliance and suitability.",
        verbose=False, allow_delegation=False, llm=llm,
    )
    checker = Agent(
        role="Compliance & Suitability Checker",
        goal="Assess mandatory disclosures and suitability concerns for the specified product.",
        backstory="Experienced BFSI compliance reviewer grounded in regulatory best practices.",
        verbose=False, allow_delegation=False, llm=llm,
    )
    writer = Agent(
        role="Communication Drafter",
        goal="Create corrective customer email and next‑call script adhering to compliance norms.",
        backstory="Seasoned writer for regulated financial communications.",
        verbose=False, allow_delegation=False, llm=llm,
    )
    return extractor, checker, writer

def rule_based_missing_disclosures(product_type: str, transcript: str, mentions: List[str]) -> List[str]:
    transcript_lc = transcript.lower()
    required = REQUIRED_DISCLOSURES.get(product_type, [])
    found = set([m.lower() for m in (mentions or [])])
    return [req for req in required if (req.lower() not in transcript_lc) and (req.lower() not in found)]

def risk_score_from_issues(missing_cnt: int, suitability_cnt: int, risk_cnt: int, sentiment_hint: str) -> float:
    base = 0.25 * missing_cnt + 0.25 * suitability_cnt + 0.4 * risk_cnt
    if "negative" in (sentiment_hint or "").lower():
        base += 0.1
    return max(0.0, min(1.0, base))

def analyze_call(
    audio_path: str,
    product_type: str,
    language: str = "en",
    llm_model: Optional[str] = None,
    risk_threshold: float = 0.6,
    ctx: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    ctx = ctx or {}
    extractor, checker, writer = build_agents(model=llm_model)

    transcript = transcribe_audio_groq(audio_path=audio_path, language=language)

    extract_task = Task(
        description=f"{EXTRACT_SCHEMA}\nTranscript:\n{transcript}",
        expected_output="Valid minified JSON per schema.",
        agent=extractor,
    )
    extract_result = Crew(agents=[extractor], tasks=[extract_task], verbose=False).kickoff()
    extract_json = safe_json_loads(str(extract_result))

    compliance_task = Task(
        description=f"{COMPLIANCE_SCHEMA}\nproduct_type={product_type}\nentities={json.dumps(extract_json, ensure_ascii=False)}",
        expected_output="Valid minified JSON per schema.",
        agent=checker,
    )
    compliance_result = Crew(agents=[checker], tasks=[compliance_task], verbose=False).kickoff()
    compliance_json = safe_json_loads(str(compliance_result))

    rb_missing = rule_based_missing_disclosures(
        product_type=product_type,
        transcript=transcript,
        mentions=extract_json.get("disclosures_mentioned", []),
    )
    missing_disclosures = sorted(set(rb_missing + compliance_json.get("missing_disclosures", [])))
    suitability_issues = list(dict.fromkeys(compliance_json.get("suitability_issues", [])))
    risk_flags = list(dict.fromkeys((extract_json.get("red_flags") or []) + (compliance_json.get("risk_flags") or [])))
    sentiment = extract_json.get("sentiment_summary", "")

    score = risk_score_from_issues(len(missing_disclosures), len(suitability_issues), len(risk_flags), sentiment)
    requires_review = score >= risk_threshold

    issues_text = json.dumps(
        {"missing_disclosures": missing_disclosures, "suitability_issues": suitability_issues, "risk_flags": risk_flags},
        ensure_ascii=False,
    )

    email_task = Task(
        description=EMAIL_PROMPT.format(
            customer_name=ctx.get("customer_name", "Customer"),
            product_type=product_type,
            issues=issues_text,
        ),
        expected_output="A concise corrective email in < 180 words.",
        agent=writer,
    )
    email_out = Crew(agents=[writer], tasks=[email_task], verbose=False).kickoff()
    email_text = str(email_out)

    script_task = Task(
        description=SCRIPT_PROMPT.format(
            agent_name=ctx.get("agent_name", "Agent"),
            product_type=product_type,
            issues=issues_text,
        ),
        expected_output="A concise call script in < 180 words.",
        agent=writer,
    )
    script_out = Crew(agents=[writer], tasks=[script_task], verbose=False).kickoff()
    script_text = str(script_out)

    report = {
        "transcription": transcript[:3000],
        "entities": extract_json.get("customer_profile", {}),
        "product_interest": extract_json.get("product_interest", {}),
        "disclosures_mentioned": extract_json.get("disclosures_mentioned", []),
        "competitor_mentions": extract_json.get("competitor_mentions", []),
        "sentiment_summary": sentiment,
        "missing_disclosures": missing_disclosures,
        "suitability_issues": suitability_issues,
        "risk_flags": risk_flags,
        "coaching_points": compliance_json.get("coaching_points", []),
        "summary": compliance_json.get("summary", ""),
        "risk_score": score,
        "requires_human_review": requires_review,
    }

    audit = {
        "timestamp": int(time.time()),
        "duration_sec": round(time.time() - t0, 2),
        "product_type": product_type,
        "language": language,
        "risk_threshold": risk_threshold,
        "model": llm_model or DEFAULT_MODEL,
        "steps": ["transcription", "extraction", "compliance_llm", "rule_overlay", "scoring", "drafts"],
    }

    return {
        "report": report,
        "email_draft": email_text.strip(),
        "call_script": script_text.strip(),
        "audit": audit,
    }

# -------- PDF Builder --------
def build_pdf_bytes(report: Dict[str, Any], email_text: str, script_text: str, audit: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    title = styles["Title"]
    h = styles["Heading2"]
    p = styles["BodyText"]

    story = []
    story.append(Paragraph("BFSI Sales‑Compliance Analysis", title))
    story.append(Spacer(1, 10))

    # Summary table
    data = [
        ["Risk score", f"{report.get('risk_score', 0):.2f}", "Needs review", str(report.get("requires_human_review", False))],
        ["Product type", report.get("product_interest", {}).get("type", "n/a"), "Language", audit.get("language", "n/a")],
        ["Model", audit.get("model", "n/a"), "Duration (s)", str(audit.get("duration_sec", "n/a"))],
    ]
    tbl = Table(data, colWidths=[90, 210, 90, 120])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.5, colors.lightgrey),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.beige]),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Summary", h))
    story.append(Paragraph(report.get("summary", "n/a"), p))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Missing disclosures", h))
    md = report.get("missing_disclosures", [])
    story.append(Paragraph(", ".join(md) if md else "None", p))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Suitability issues", h))
    si = report.get("suitability_issues", [])
    story.append(Paragraph(", ".join(si) if si else "None", p))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Risk flags", h))
    rf = report.get("risk_flags", [])
    story.append(Paragraph(", ".join(rf) if rf else "None", p))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Transcription (preview)", h))
    story.append(Paragraph(report.get("transcription", "")[:2000] or "n/a", p))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Draft: Corrective Email", h))
    story.append(Paragraph(email_text.replace("\n", "<br/>") or "n/a", p))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Draft: Next‑Call Script", h))
    story.append(Paragraph(script_text.replace("\n", "<br/>") or "n/a", p))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Audit", h))
    story.append(Paragraph(json.dumps(audit, ensure_ascii=False, indent=2).replace("\n", "<br/>"), p))

    doc.build(story)
    buf.seek(0)
    return buf.read()
