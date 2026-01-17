import math
class ChatOrchestrator:

    def __init__(self, enrichment_service, router, portkey_client):
        self.enrichment = enrichment_service
        self.router = router
        self.client = portkey_client

    def get_recommended_model(self, prompt, user_prefs):
        enriched = self.enrichment.enrich(prompt)

        recommendation = self.router.recommend(
            category=enriched.category,
            quality_weight=user_prefs["quality_weight"],
            cost_weight=user_prefs["cost_weight"],
            max_cost=user_prefs.get("max_cost")
        )

        if not recommendation:
            explanation = (
                "No suitable model found based on your preferences. "
                "Using default model(@openai/gpt-4o)."
            )
            return None, explanation

        explanation = (
            f"Switched to **{recommendation['primary_model']}** "
            f"to reduce cost while maintaining quality. "
            f"Estimated avg cost: ${(recommendation['evidence']['avg_cost']):.4f}, "
            f"quality score: {recommendation['evidence']['avg_quality']}."
        )
        return recommendation, explanation

    def handle_prompt(self, prompt, model_slug):
       
        # -----------------------------
        # Model Invocation
        # -----------------------------
        response = self.client.chat.completions.create(
            model=model_slug,
            messages=[{"role": "user", "content": prompt}]
        )

        # -----------------------------
        # Explanation
        # -----------------------------

        return {
            "answer": response.choices[0].message.content,
        }
