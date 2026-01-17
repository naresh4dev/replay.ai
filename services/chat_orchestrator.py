class ChatOrchestrator:

    def __init__(self, enrichment_service, router, portkey_client):
        self.enrichment = enrichment_service
        self.router = router
        self.client = portkey_client

    def handle_prompt(self, prompt, user_prefs):
        # -----------------------------
        # Stage 1: Enrichment
        # -----------------------------
        enriched = self.enrichment.enrich(prompt)

        # -----------------------------
        # Stage 4: Routing
        # -----------------------------
        recommendation = self.router.recommend(
            category=enriched.category,
            quality_weight=user_prefs["quality_weight"],
            cost_weight=user_prefs["cost_weight"],
            max_cost=user_prefs.get("max_cost")
        )

        if not recommendation:
            raise ValueError("No suitable model found")

        # -----------------------------
        # Model Invocation
        # -----------------------------
        response = self.client.chat.completions.create(
            model=f"@{recommendation['primary_provider']}/{recommendation['primary_model']}",
            messages=[{"role": "user", "content": prompt}]
        )

        # -----------------------------
        # Explanation
        # -----------------------------
        explanation = (
            f"Switched to **{recommendation['primary_model']}** "
            f"to reduce cost while maintaining quality. "
            f"Estimated avg cost: ${recommendation['evidence']['avg_cost']}, "
            f"quality score: {recommendation['evidence']['avg_quality']}."
        )

        return {
            "answer": response.choices[0].message.content,
            "explanation": explanation,
            "metadata": {
                "category": enriched.category,
                "complexity": enriched.complexity,
                "tokens": enriched.features["token_count"],
                "model_used": recommendation["primary_model"]
            }
        }
