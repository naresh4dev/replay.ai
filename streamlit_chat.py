import streamlit as st
from services.chat_orchestrator import ChatOrchestrator
from services.prompt_categorization import PromptEnrichmentService
from services.routing_policy_service import RoutingPolicyService
from services.decision_service import Stage3Service
from utils.portkey import get_portkey_client

st.set_page_config(page_title="Smart AI Chat", layout="wide")
st.title("💬 Decision-Aware AI Chat")

# -------------------------------
# Sidebar – User Profile
# -------------------------------
st.sidebar.header("🎛️ Your Preferences")

quality_weight = st.sidebar.slider("Quality Priority", 0.0, 1.0, 0.6)
cost_weight = st.sidebar.slider("Cost Priority", 0.0, 1.0, 0.4)
max_cost = st.sidebar.number_input("Max Cost ($)", value=0.0, step=0.01)

user_prefs = {
    "quality_weight": quality_weight,
    "cost_weight": cost_weight,
    "max_cost": max_cost if max_cost > 0 else None
}

# -------------------------------
# Initialize Services
# -------------------------------
stage3 = Stage3Service("data/stage2_evaluation.jsonl")
router = RoutingPolicyService(stage3)
enrichment = PromptEnrichmentService()

client = get_portkey_client()

orchestrator = ChatOrchestrator(
    enrichment, router, client
)

# -------------------------------
# Chat Interface
# -------------------------------
prompt = st.chat_input("Ask anything...")

if prompt:
    with st.spinner("Thinking with the best model..."):
        result = orchestrator.handle_prompt(prompt, user_prefs)
    
    st.chat_message("assistant").markdown(result["answer"])

    with st.expander("🔍 Why this model?"):
        st.markdown(result["explanation"])
        st.json(result["metadata"])
