import streamlit as st
import json
import pandas as pd
from collections import defaultdict

from utils.file_handler import load_jsonl, persist_stage2_result
from services.evaluation_service import EvaluationOrchestrator
from data.dataclass import ReplayResponse
from config.constants.constant import MODEL_CONFIGS, PRICING_TABLE
from utils.portkey import get_portkey_client

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="Stage 2 – Evaluation", layout="wide")

st.title("📊 Stage 2 – Multi-Model Evaluation")
st.caption("Replay • Rank • Persist")

# ===============================
# Sidebar
# ===============================
st.sidebar.header("⚙️ Controls")

input_path = st.sidebar.text_input(
    "Stage-1 Enriched JSONL",
    value="data/enriched_prompts_stage1.jsonl"
)

output_path = st.sidebar.text_input(
    "Stage-2 Output JSONL",
    value="data/stage2_evaluation.jsonl"
)

limit = st.sidebar.slider(
    "Prompts to evaluate",
    1, 100, 5
)

selected_models = st.sidebar.multiselect(
    "Models",
    options=[m["slug"] for m in MODEL_CONFIGS],
    default=[m["slug"] for m in MODEL_CONFIGS[:3]]
)

run_btn = st.sidebar.button("▶ Run Stage-2")

# ===============================
# Execution
# ===============================
if run_btn:

    records = load_jsonl(input_path, limit)
    orchestrator = EvaluationOrchestrator(
        model_configs=[
            m for m in MODEL_CONFIGS if m["slug"] in selected_models
        ],
        pricing_table=PRICING_TABLE
    )

    st.subheader("⚙️ Running Evaluations")
    progress = st.progress(0)

    for idx, record in enumerate(records):

        enriched = record["features"]
        prompt = enriched["text"]
        conversation_id = record["record"]["conversation_id"]

        reference = next(
            t["content"]
            for t in record["record"]["turns"]
            if t["role"] == "assistant"
        )

        results, judge_reasoning = orchestrator.evaluate_prompt(
            prompt, reference
        )

        # ===============================
        # Persist JSONL
        # ===============================
        persist_stage2_result(
            output_path,
            {
                "conversation_id": conversation_id,
                "prompt": prompt,
                "reference_answer": reference,
                "judge_reasoning": judge_reasoning,
                "features": record["features"]["features"],
                "results": [
                    {
                        "model_slug": r.model_slug,
                        "provider": r.provider,
                        "cost_usd": r.cost_usd,
                        "latency_ms": r.latency_ms,
                        "refusal": r.refusal,
                        "similarity_score": r.similarity_score,
                        "judge_score": r.judge_score,
                        "response_text": r.response_text
                    }
                    for r in results
                ]
            }
        )

        progress.progress((idx + 1) / len(records))

    st.success("Stage-2 evaluation completed")

    # ===============================
    # Visualization
    # ===============================
    st.subheader("🏆 Model Rankings (Latest Prompt)")

    df = pd.DataFrame(
        sorted(
            results,
            key=lambda r: r.judge_score,
            reverse=True
        )
    )

    st.dataframe(
        df[[
            "model_slug",
            "judge_score",
            "similarity_score",
            "cost_usd",
            "latency_ms",
            "refusal"
        ]]
    )

    # ===============================
    # Cost vs Quality Plot
    # ===============================
    st.subheader("📈 Cost vs Quality")

    chart_df = pd.DataFrame([
        {
            "model": r.model_slug,
            "quality": r.judge_score,
            "cost": r.cost_usd
        }
        for r in results
    ])

    st.scatter_chart(
        chart_df,
        x="cost",
        y="quality",
        size=100
    )

    # ===============================
    # Report Card
    # ===============================
    best_cost = min(results, key=lambda r: r.cost_usd)
    best_quality = max(results, key=lambda r: r.judge_score)

    st.subheader("✅ Stage-2 Report Card")

    st.markdown(f"""
    **Prompts Evaluated:** {limit}  
    **Models Compared:** {len(selected_models)}  

    🟢 **Lowest Cost:** `{best_cost.model_slug}` (${best_cost.cost_usd})  
    ⭐ **Best Quality:** `{best_quality.model_slug}` (Score {best_quality.judge_score})  

    **Artifacts Saved:** `{output_path}`  
    """)

    st.success("Stage-2 artifacts persisted successfully")