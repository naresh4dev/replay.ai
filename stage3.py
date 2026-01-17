import streamlit as st
import pandas as pd

from services.decision_service import Stage3Service

# =====================================
# Page Config
# =====================================
st.set_page_config(
    page_title="Stage 3 – Trade-off Analysis",
    layout="wide"
)

st.title("📐 Stage 3 – Aggregation & Trade-off Analysis")
st.caption("Pareto Frontier • Confidence • Dominance")

# =====================================
# Sidebar
# =====================================
st.sidebar.header("⚙️ Controls")

stage2_path = st.sidebar.text_input(
    "Stage-2 Evaluation JSONL",
    value="data/stage2_evaluation.jsonl"
)

stage1_path = st.sidebar.text_input(
    "Stage-1 Enrichment JSONL",
    value="data/enriched_prompts_stage1.jsonl"
)

run_btn = st.sidebar.button("▶ Run Stage-3 Analysis")

# =====================================
# Cached Data Loading
# =====================================
@st.cache_data
def load_service(stage2_path: str, stage1_path: str):
    """Cache the service initialization and data loading."""
    service = Stage3Service(stage2_path, stage1_path)
    outputs = service.run()
    return service, outputs

# =====================================
# Main
# =====================================
if run_btn:
    # Initialize session state
    if 'analysis_loaded' not in st.session_state:
        st.session_state.analysis_loaded = False

    # Load data (cached)
    service, outputs = load_service(stage2_path, stage1_path)
    st.session_state.analysis_loaded = True
    st.session_state.service = service
    st.session_state.outputs = outputs

if st.session_state.get('analysis_loaded', False):
    service = st.session_state.service
    outputs = st.session_state.outputs
    
    per_model = outputs["per_model"]
    pareto = outputs["pareto"]

    # ---------------------------------
    st.subheader("📊 Per-Model Aggregates")

    st.dataframe(
        per_model[[
            "model",
            "avg_quality",
            "avg_cost",
            "refusal_rate",
            "sample_size",
            "confidence_interval"
        ]],
        use_container_width=True
    )

    # ---------------------------------
    st.subheader("🏆 Pareto Frontier (Cost vs Quality)")

    chart_df = per_model.copy()
    chart_df["is_pareto"] = chart_df["model"].isin(
        [p["model"] for p in pareto]
    )

    st.scatter_chart(
        chart_df,
        x="avg_cost",
        y="avg_quality",
        size="sample_size",
        color="is_pareto"
    )

    # ---------------------------------
    st.subheader("📈 Confidence Intervals (Quality)")

    ci_df = per_model.dropna(subset=["confidence_interval"])

    for _, row in ci_df.iterrows():
        low, high = row["confidence_interval"]
        st.markdown(
            f"**{row['model']}** → "
            f"{round(row['avg_quality'],2)} "
            f"([{low}, {high}])"
        )

    # ---------------------------------
    st.subheader("✅ Stage-3 Report Card")

    best_cost = per_model.loc[
        per_model["avg_cost"].idxmin()
    ]

    best_quality = per_model.loc[
        per_model["avg_quality"].idxmax()
    ]

    st.markdown(f"""
    **Models Evaluated:** {len(per_model)}  
    **Total Samples:** {per_model.sample_size.sum()}  

    🟢 **Lowest Cost Model:** `{best_cost.model}`  
    ⭐ **Highest Quality Model:** `{best_quality.model}`  

    **Pareto-Optimal Models:**  
    {", ".join(p["model"] for p in pareto)}
    """)

    st.subheader("🏷️ Category-specific Pareto Frontier")

    categories = sorted(service.df.category.unique())
    selected_cat = st.selectbox("Select category", categories, key="category_selector")

    cat_df, cat_pareto = service.category_pareto(selected_cat)

    # Display model names with metrics
    st.dataframe(
        cat_df[["model", "avg_quality", "avg_cost", "n"]],
        use_container_width=True
    )

    st.scatter_chart(
        cat_df,
        x="avg_cost",
        y="avg_quality",
        size="n"
    )

    st.markdown(
        "**Pareto-optimal models:** " +
        ", ".join(p["model"] for p in cat_pareto)
    )    

    st.subheader("🎚️ Cost vs Quality Sensitivity")

    alpha = st.slider(
        "Quality importance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="alpha_slider"
    )

    @st.cache_data
    def get_sensitivity_rank(_service, alpha):
        """Cache sensitivity ranking computation."""
        return _service.sensitivity_rank(alpha)

    ranked = get_sensitivity_rank(service, alpha)

    st.dataframe(
        ranked[["model", "avg_quality", "avg_cost", "score"]],
        use_container_width=True
    )

    best = ranked.iloc[0]
    st.success(
        f"🏆 Best model at α={alpha}: "
        f"{best.model}"
    )

    

    st.success("Stage-3 analysis completed successfully")
else:
    st.info("👈 Click 'Run Stage-3 Analysis' in the sidebar to begin.")
