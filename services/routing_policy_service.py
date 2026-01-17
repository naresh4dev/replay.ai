import json

class RoutingPolicyService:

    def __init__(self, stage3_service):
        self.stage3 = stage3_service

    # ----------------------------------
    def recommend(
        self,
        category: str,
        quality_weight: float = 0.7,
        cost_weight: float = 0.3,
        max_cost: float = None,
        min_quality: int = None
    ):
        """
        Recommend routing policy for a given category.
        """
        subset = self.stage3.df[self.stage3.df.category == category]
        
        if subset.empty:
            return None
        
        agg = subset.groupby(["model", "provider"]).agg({
            "quality": "mean",
            "cost": "mean",
            "latency": "mean"
        }).reset_index()
        
        agg.columns = ["model", "provider", "avg_quality", "avg_cost", "avg_latency"]
        
        # Apply filters
        if max_cost:
            agg = agg[agg.avg_cost <= max_cost]
        
        if min_quality:
            agg = agg[agg.avg_quality >= min_quality]
        
        if agg.empty:
            return None
        
        # Normalize and score
        q_norm = (agg.avg_quality - agg.avg_quality.min()) / (
            agg.avg_quality.max() - agg.avg_quality.min() + 0.0001
        )
        c_norm = (agg.avg_cost - agg.avg_cost.min()) / (
            agg.avg_cost.max() - agg.avg_cost.min() + 0.0001
        )
        
        agg["score"] = quality_weight * q_norm - cost_weight * c_norm
        agg = agg.sort_values("score", ascending=False)
        
        # Get primary and fallback as dictionary records
        primary_row = agg.iloc[0].to_dict()
        fallback_row = agg.iloc[1].to_dict() if len(agg) > 1 else primary_row
        
        return {
            "category": category,
            "primary_model": primary_row["model"],
            "primary_provider": primary_row["provider"],
            "fallback_model": fallback_row["model"],
            "fallback_provider": fallback_row["provider"],
            "evidence": {
                "avg_quality": primary_row["avg_quality"],
                "avg_cost": primary_row["avg_cost"],
                "avg_latency": primary_row["avg_latency"],
                "score": primary_row["score"]
            }
        }

    def export_portkey_config(self, policies: list) -> dict:
        """
        Export policies as Portkey routing configuration.
        """
        strategies = []
        
        for policy in policies:
            strategies.append({
                "strategy": {
                    "mode": "fallback"
                },
                "targets": [
                    {
                        "provider": policy["primary_provider"],
                        "model": policy["primary_model"],
                        "weight": 1
                    },
                    {
                        "provider": policy["fallback_provider"],
                        "model": policy["fallback_model"],
                        "weight": 0
                    }
                ],
                "metadata": {
                    "category": policy["category"],
                    "avg_quality": policy["evidence"]["avg_quality"],
                    "avg_cost": policy["evidence"]["avg_cost"]
                }
            })
        
        return {
            "strategies": strategies,
            "retry": {
                "attempts": 2
            },
            "cache": {
                "mode": "simple",
                "max_age": 3600
            }
        }
