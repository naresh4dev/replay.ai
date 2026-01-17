import streamlit as st
import json
from services.decision_service import Stage3Service
from services.routing_policy_service import RoutingPolicyService

st.set_page_config(page_title="Stage 4 – Routing Policy", layout="wide")
st.title("🚦 Stage 4 – Routing Policy Generator")

# -------------------------------------
stage2_path = st.sidebar.text_input(
    "Stage-2 Evaluation JSONL",
    value="data/stage2_evaluation.jsonl",
    key="stage4_stage2_path"
)

stage1_path = st.sidebar.text_input(
    "Stage-1 Enrichment JSONL",
    value="data/enriched_prompts_stage1.jsonl",
    key="stage4_stage1_path"
)

st.sidebar.header("⚙️ Policy Controls")

quality_weight = st.sidebar.slider(
    "Quality Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05,
    key="stage4_quality_weight"
)

cost_weight = st.sidebar.slider(
    "Cost Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
    key="stage4_cost_weight"
)

latency_threshold = st.sidebar.slider(
    "Max Latency (ms)",
    min_value=0,
    max_value=10000,
    value=5000,
    step=100,
    key="stage4_latency_threshold"
)

max_cost = st.sidebar.slider(
    "Max Cost per Request ($)",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01,
    key="stage4_max_cost"
)

min_quality = st.sidebar.slider(
    "Minimum Quality Score",
    min_value=0,
    max_value=10,
    value=7,
    step=1,
    key="stage4_min_quality"
)

run_btn = st.sidebar.button("▶ Generate Policy", key="stage4_run_btn")

# -------------------------------------
if run_btn or st.session_state.get('stage4_loaded', False):
    if run_btn:
        stage3 = Stage3Service(stage2_path, stage1_path)
        router = RoutingPolicyService(stage3)
        st.session_state.stage3 = stage3
        st.session_state.router = router
        st.session_state.stage4_loaded = True
    
    stage3 = st.session_state.stage3
    router = st.session_state.router

    categories = sorted(stage3.df.category.unique())
    selected = st.multiselect(
        "Categories",
        categories,
        default=list(categories),
        key="stage4_category_select"
    )

    policies = []

    st.header("📌 Routing Recommendations")

    for cat in selected:
        rec = router.recommend(
            category=cat,
            quality_weight=quality_weight,
            max_cost=max_cost if max_cost > 0 else None,
            min_quality=min_quality if min_quality > 0 else None,
            cost_weight=cost_weight
        )
        
        if not rec:
            st.warning(f"No valid routing for `{cat}`")
            continue

        policies.append(rec)

        with st.expander(f"🏷️ {cat}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Quality", round(rec["evidence"]["avg_quality"], 2))
            col2.metric("Avg Cost ($)", round(rec["evidence"]["avg_cost"], 4))
            col3.metric("Score", round(rec["evidence"]["score"], 3))
            st.json(rec)

    # -------------------------------------
    st.header("📦 Export Portkey Routing Rules")

    if st.button("⬇ Export Config", key="stage4_export_btn"):
        config = router.export_portkey_config(policies)

        st.download_button(
            label="Download portkey-routing.json",
            data=json.dumps(config, indent=2),
            file_name="portkey-routing.json",
            mime="application/json",
            key="stage4_download_btn"
        )

    st.success("Routing policies generated successfully")
else:
    st.info("👈 Click 'Generate Policy' in the sidebar to begin.")
