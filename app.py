# app.py — Plain-text UI (no JSON renderers)
import os
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv
from main import analyze_call, build_pdf_bytes

load_dotenv()
st.set_page_config(page_title="BFSI Sales‑Compliance Analyzer", page_icon="✅", layout="wide")

# -------- Helpers for plain text --------
def as_lines_list(items, label):
    items = items or []
    if not items:
        return f"{label}: None"
    return f"{label}:\n" + "\n".join([f"  - {x}" for x in items])

def as_kv_text(d, title):
    if not d:
        return f"{title}: None"
    lines = [f"{title}:"]
    for k, v in d.items():
        lines.append(f"  - {k}: {v}")
    return "\n".join(lines)

# -------- Header --------
st.markdown(
    """
    <h2 style="margin-bottom:0">BFSI Sales‑Compliance Analyzer</h2>
    <p style="color:#666; margin-top:4px">
      Upload a sales call, analyze disclosures & suitability, get risk and ready‑to‑send follow‑ups.
    </p>
    """,
    unsafe_allow_html=True,
)

# -------- Sidebar --------
with st.sidebar:
    st.header("Settings")
    product = st.selectbox("Product line", ["loan", "credit_card", "insurance", "investment"])
    language = st.selectbox("Language", ["en", "hi"])
    risk_threshold = st.slider(
        "Risk threshold (review if ≥)",
        min_value=0.0, max_value=1.0,
        value=float(os.getenv("RISK_THRESHOLD", "0.6")),
        step=0.05
    )
    model = st.text_input("AI model", value=os.getenv("LLM_MODEL", "groq/llama-3.1-70b-versatile"))
    st.caption("Use a provider‑prefixed model if your setup requires it.")

# -------- Upload --------
st.subheader("Upload audio")
uploaded = st.file_uploader("Choose a file", type=["mp3", "wav", "m4a", "ogg"])
if uploaded:
    st.audio(uploaded, format="audio/mpeg")

# -------- Optional context --------
colA, colB, colC = st.columns([1,1,1])
with colA:
    customer_name = st.text_input("Customer name (optional)", value="")
with colB:
    customer_email = st.text_input("Customer email (optional)", value="")
with colC:
    agent_name = st.text_input("Agent name (optional)", value="")

# -------- Run --------
run = st.button("Analyze", type="primary", use_container_width=True, disabled=uploaded is None)

if run:
    if not uploaded:
        st.warning("Please upload an audio file.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Analyzing…"):
        out = analyze_call(
            audio_path=tmp_path,
            product_type=product,
            language=language,
            llm_model=model,
            risk_threshold=risk_threshold,
            ctx={"customer_name": customer_name, "customer_email": customer_email, "agent_name": agent_name},
        )

    report = out.get("report", {})
    email_text = out.get("email_draft", "")
    script_text = out.get("call_script", "")
    audit = out.get("audit", {})

    # -------- Status row --------
    top1, top2, top3, top4 = st.columns([1,1,1,1])
    with top1:
        st.metric("Risk score", f"{report.get('risk_score', 0):.2f}")
    with top2:
        need_review = report.get("requires_human_review", False)
        badge_color = "#e11d48" if need_review else "#16a34a"
        badge_text = "Needs review" if need_review else "OK"
        st.markdown(
            f"<div style='background:{badge_color};color:#fff;padding:6px 10px;border-radius:6px;width:fit-content'>{badge_text}</div>",
            unsafe_allow_html=True
        )
    with top3:
        st.metric("Product", report.get("product_interest", {}).get("type", product))
    with top4:
        st.metric("Language", audit.get("language", language))

    # -------- Tabs with plain text --------
    tabs = st.tabs(["Report", "Email", "Script", "Audit"])

    with tabs[0]:
        # Plain text report
        summary = report.get("summary", "—")
        sentiment = report.get("sentiment_summary", "—")
        entities_txt = as_kv_text(report.get("entities", {}), "Customer profile")
        product_txt = as_kv_text(report.get("product_interest", {}), "Product interest")
        missing_txt = as_lines_list(report.get("missing_disclosures", []), "Missing disclosures")
        suit_txt = as_lines_list(report.get("suitability_issues", []), "Suitability issues")
        risk_txt = as_lines_list(report.get("risk_flags", []), "Risk flags")
        comp_txt = as_lines_list(report.get("competitor_mentions", []), "Competitor mentions")

        report_text = f"""
Summary: {summary}
Sentiment: {sentiment}

{entities_txt}

{product_txt}

{missing_txt}

{suit_txt}

{risk_txt}

{comp_txt}
""".strip()
        st.text(report_text)

        st.markdown("#### Transcription (preview)")
        st.text(report.get("transcription", "")[:3000] or "—")

    with tabs[1]:
        st.markdown("#### Draft email")
        st.text(email_text or "—")

    with tabs[2]:
        st.markdown("#### Next‑call script")
        st.text(script_text or "—")

    with tabs[3]:
        # Plain text audit
        audit_lines = []
        for k in ["timestamp", "duration_sec", "product_type", "language", "risk_threshold", "model"]:
            audit_lines.append(f"{k}: {audit.get(k, '—')}")
        steps = audit.get("steps", [])
        audit_text = "\n".join(audit_lines) + "\n" + as_lines_list(steps, "steps")
        st.text(audit_text)

    # -------- Download PDF --------
    pdf_bytes = build_pdf_bytes(report=report, email_text=email_text, script_text=script_text, audit=audit)
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name="bfsi_call_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
