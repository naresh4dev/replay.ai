import streamlit as st
from services.chat_orchestrator import ChatOrchestrator
from services.prompt_categorization import PromptEnrichmentService
from services.routing_policy_service import RoutingPolicyService
from services.decision_service import Stage3Service
from utils.portkey import get_portkey_client

st.set_page_config(page_title="Smart AI Chat", layout="wide")
st.title("💬 Decision-Aware AI Chat")

# -------------------------------
# Initialize Session State
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_recommendations" not in st.session_state:
    st.session_state.chat_recommendations = []

# -------------------------------
# Sidebar – User Profile & Controls
# -------------------------------
st.sidebar.header("🎛️ Your Preferences")

quality_weight = st.sidebar.slider("Quality Priority", 0.0, 1.0, 0.6, key="chat_quality_weight")
cost_weight = st.sidebar.slider("Cost Priority", 0.0, 1.0, 0.4, key="chat_cost_weight")
max_cost = st.sidebar.number_input("Max Cost ($)", value=0.20, step=0.01, key="chat_max_cost")

user_prefs = {
    "quality_weight": quality_weight,
    "cost_weight": cost_weight,
    "max_cost": max_cost if max_cost > 0 else None
}

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History", key="clear_chat"):
    st.session_state.chat_history = []
    st.session_state.chat_recommendations = []
    st.rerun()

# Show chat statistics
st.sidebar.divider()
st.sidebar.subheader("📊 Chat Statistics")
st.sidebar.metric("Total Messages", len(st.session_state.chat_history))
st.sidebar.metric("Conversations", len([m for m in st.session_state.chat_history if m["role"] == "user"]))

# -------------------------------
# Initialize Services
# -------------------------------
try:
    stage3 = Stage3Service("data/stage2_evaluation.jsonl", "data/enriched_prompts_stage1.jsonl")
    router = RoutingPolicyService(stage3)
    enrichment = PromptEnrichmentService()
    client = get_portkey_client()

    orchestrator = ChatOrchestrator(
        enrichment, router, client
    )
except Exception as e:
    st.error(f"❌ Error initializing services: {e}")
    st.stop()

# -------------------------------
# Display Chat History
# -------------------------------
for i, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Show model recommendation for user messages
        if msg["role"] == "user" and i < len(st.session_state.chat_recommendations):
            rec = st.session_state.chat_recommendations[i // 2]  # Every other message
            with st.expander("🔍 Model Selection Details"):
                st.markdown(f"**Selected Model:** `{rec['model_slug']}`")
                st.json(rec["recommendation"])

# -------------------------------
# Chat Input
# -------------------------------
prompt = st.chat_input("Ask anything...", key="chat_input_main")

if prompt:
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get model recommendation
    with st.spinner("🤔 Selecting best model..."):
        try:
            recommendation, explanation = orchestrator.get_recommended_model(prompt, user_prefs)
            
            # Default model slug if recommendation is None
            if recommendation is None:
                model_slug = "@openai/gpt-4o"
                recommendation = {
                    "primary_model": "gpt-4o",
                    "primary_provider": "openai",
                    "fallback_model": "gpt-4o-mini",
                    "fallback_provider": "openai",
                    "category": "unknown",
                    "evidence": {}
                }
            else:
                model_slug = f"@{recommendation['primary_provider']}/{recommendation['primary_model']}"
            
            # Store recommendation
            st.session_state.chat_recommendations.append({
                "model_slug": model_slug,
                "recommendation": recommendation,
                "explanation": explanation
            })
        except Exception as e:
            st.error(f"Error in model selection: {e}")
            model_slug = "@openai/gpt-4o"
            recommendation = {}
            explanation = "Using default model due to error"
        # Show model selection details
    with st.expander("🔍 Model Selection Details"):
        st.markdown(f"**Selected Model:** `{model_slug}`")
        st.markdown(f"**Reasoning:** {explanation}")
        st.json(recommendation)
        # Show cost estimate if available
        if "evidence" in recommendation and "avg_cost" in recommendation["evidence"]:
            st.metric("Estimated Cost", f"${recommendation['evidence']['avg_cost']:.6f}")
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("✨ Generating response..."):
            try:
                result = orchestrator.handle_prompt(prompt, model_slug)
                answer = result["answer"]
                
                # Add assistant message to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })
                st.markdown(answer)
                
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg
                })


