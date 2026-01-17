import streamlit as st
import json
from services.prompt_categorization import PromptEnrichmentService
from utils.file_handler import load_jsonl


# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(
    page_title="Prompt Enrichment Service",
    layout="wide",
)

st.title("🧠 Prompt Enrichment – Stage 1")
st.caption("Rule-based features + Intent categorization using Portkey")

# ===============================
# Sidebar Controls
# ===============================
st.sidebar.header("⚙️ Controls")

processing_mode = st.sidebar.radio(
    "Processing Mode",
    options=[
        "Process raw JSONL (Run Stage 1)",
        "Load processed Stage 1 JSONL"
    ]
)

file_path = st.sidebar.text_input(
    "JSONL file path",
    value="data/sample.jsonl"
)

limit = st.sidebar.slider(
    "Number of prompts to process",
    min_value=10,
    max_value=1000,
    value=100
)

run_btn = st.sidebar.button("▶ Run")

# ===============================
# Main Flow
# ===============================
if run_btn:

    # =====================================================
    # MODE 1: PROCESS RAW JSONL (RUN STAGE 1)
    # =====================================================
    if processing_mode == "Process raw JSONL (Run Stage 1)":

        st.subheader("📥 Stage 0: Data Ingestion")
        records = load_jsonl(file_path, limit=limit)

        st.success(f"Loaded {len(records)} prompts")
        st.json(records[0], expanded=False)

        # ---------------------------------
        st.subheader("🧮 Stage 1A: Rule-Based Feature Extraction")
        service = PromptEnrichmentService()

        feature_progress = st.progress(0)
        enriched_results = []
        output_records = []

        for i, record in enumerate(records):
            prompt = record["prompt"]
            enriched = service.enrich(prompt)

            enriched_results.append(enriched)
            output_records.append({
                "features": enriched.__dict__,
                "record": record
            })

            feature_progress.progress((i + 1) / len(records))

        st.success("Feature extraction + classification completed")

        output_path = "data/enriched_prompts_stage1.jsonl"
        with open(output_path, "w") as outfile:
            for r in output_records:
                outfile.write(json.dumps(r) + "\n")

        st.info(f"💾 Saved processed file to `{output_path}`")

    # =====================================================
    # MODE 2: LOAD ALREADY PROCESSED STAGE-1 JSONL
    # =====================================================
    else:
        st.subheader("📂 Loading Processed Stage-1 Data")

        processed_records = load_jsonl(file_path, limit=limit)

        enriched_results = [
            type("EnrichedPrompt", (), rec["features"])
            for rec in processed_records
        ]

        st.success(f"Loaded {len(enriched_results)} enriched prompts")
