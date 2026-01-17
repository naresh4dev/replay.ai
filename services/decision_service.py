import json
import math
import pandas as pd
import scipy.stats as st

# =====================================
# Loader
# =====================================
class Stage3DataLoader:

    @staticmethod
    def load_stage2_jsonl(path: str) -> pd.DataFrame:
        rows = []

        with open(path, "r") as f:
            for line in f:
                rec = json.loads(line)

                for r in rec["results"]:
                    rows.append({
                        "conversation_id": rec["conversation_id"],
                        "prompt": rec["prompt"],
                        "model": r["model_slug"],
                        "provider": r["provider"],
                        "cost": r["cost_usd"],
                        "latency": r["latency_ms"],
                        "refusal": r["refusal"],
                        "judge_score": r["judge_score"]
                    })

        return pd.DataFrame(rows)


# =====================================
# Aggregations
# =====================================
class Aggregator:

    @staticmethod
    def per_model(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.groupby("model")
            .agg(
                avg_quality=("judge_score", "mean"),
                std_quality=("judge_score", "std"),
                avg_cost=("cost", "mean"),
                avg_latency=("latency", "mean"),
                refusal_rate=("refusal", "mean"),
                sample_size=("judge_score", "count"),
            )
            .reset_index()
        )

    

# =====================================
# Confidence Intervals
# =====================================
class ConfidenceAnalyzer:

    @staticmethod
    def confidence_interval(mean, std, n, confidence=0.95):
        if n < 2 or pd.isna(std):
            return None

        z = st.norm.ppf((1 + confidence) / 2)
        margin = z * (std / math.sqrt(n))
        return round(mean - margin, 3), round(mean + margin, 3)

    @staticmethod
    def apply(df: pd.DataFrame) -> pd.DataFrame:
        df["confidence_interval"] = df.apply(
            lambda r: ConfidenceAnalyzer.confidence_interval(
                r.avg_quality, r.std_quality, r.sample_size
            ),
            axis=1
        )
        return df


# =====================================
# Pareto Frontier
# =====================================
class ParetoAnalyzer:

    @staticmethod
    def compute(df: pd.DataFrame):
        points = df.to_dict("records")
        frontier = []

        for p in points:
            dominated = False
            for q in points:
                if (
                    q["avg_cost"] <= p["avg_cost"]
                    and q["avg_quality"] >= p["avg_quality"]
                    and (
                        q["avg_cost"] < p["avg_cost"]
                        or q["avg_quality"] > p["avg_quality"]
                    )
                ):
                    dominated = True
                    break

            if not dominated:
                # Include model name in pareto result
                frontier.append({
                    "model": p.get("model"),
                    "avg_cost": p["avg_cost"],
                    "avg_quality": p["avg_quality"]
                })

        return frontier


# =====================================
# Stage-3 Orchestrator
# =====================================
class Stage3Service:

    def __init__(self, stage2_path: str, stage1_path: str = "data/enriched_prompts_stage1.jsonl"):
        """
        Initialize Stage3Service with stage2 evaluation data and stage1 enrichment data.
        
        Args:
            stage2_path: Path to stage2 evaluation JSONL file
            stage1_path: Path to stage1 enrichment JSONL file (contains category info)
        """
        self.stage2_path = stage2_path
        self.stage1_path = stage1_path
        self.df = self._load_and_merge_data()

    def _load_and_merge_data(self) -> pd.DataFrame:
        """Load stage2 data and merge with stage1 category information."""
        # Load stage2 evaluation results
        stage2_records = []
        with open(self.stage2_path, "r") as f:
            for line in f:
                stage2_records.append(json.loads(line))
        
        # Load stage1 enrichment data
        stage1_records = []
        with open(self.stage1_path, "r") as f:
            for line in f:
                stage1_records.append(json.loads(line))
        
        # Create stage1 lookup dict by conversation_id
        stage1_lookup = {}
        for record in stage1_records:
            conv_id = record["record"]["conversation_id"]
            stage1_lookup[conv_id] = {
                "category": record["features"].get("category", "unknown"),
                "intent": record["features"].get("intent", "unknown"),
                "complexity": record["features"].get("complexity", "unknown")
            }
        
        # Flatten stage2 data and merge with stage1 info
        rows = []
        for record in stage2_records:
            conv_id = record["conversation_id"]
            
            # Get category info from stage1
            stage1_info = stage1_lookup.get(conv_id, {
                "category": "unknown",
                "intent": "unknown", 
                "complexity": "unknown"
            })
            
            for result in record["results"]:
                rows.append({
                    "conversation_id": conv_id,
                    "model": result["model_slug"],
                    "provider": result["provider"],
                    "quality": result["judge_score"],
                    "cost": result["cost_usd"],
                    "latency": result["latency_ms"],
                    "refusal": result["refusal"],
                    "similarity": result.get("similarity_score", 0),
                    "category": stage1_info["category"],
                    "intent": stage1_info["intent"],
                    "complexity": stage1_info["complexity"],
                    "response_text": result.get("response_text", ""),
                    "judge_score": result.get("judge_score", 0)
                })
        
        return pd.DataFrame(rows)

    def per_category(self, category: str) -> pd.DataFrame:
        subset = self.df[self.df.category == category]
        return (
            subset.groupby(["model"])
            .agg(
                avg_quality=("judge_score", "mean"),
                avg_cost=("cost", "mean"),
                sample_size=("judge_score", "count"),
            )
            .reset_index()
        )
    
    def category_pareto(self, category: str):
        subset = self.df[self.df.category == category]
        grouped = subset.groupby("model").agg(
            avg_quality=("quality", "mean"),
            avg_cost=("cost", "mean"),
            n=("quality", "count")
        ).reset_index()

        return grouped, ParetoAnalyzer.compute(grouped)
    
    def sensitivity_rank(self, alpha: float = 0.5):
        """
        alpha = weight on quality
        (1-alpha) = weight on cost
        """
        agg = Aggregator.per_model(self.df)

        q_norm = (agg.avg_quality - agg.avg_quality.min()) / (
            agg.avg_quality.max() - agg.avg_quality.min()
        )

        c_norm = (agg.avg_cost - agg.avg_cost.min()) / (
            agg.avg_cost.max() - agg.avg_cost.min()
        )

        agg["score"] = alpha * q_norm - (1 - alpha) * c_norm
        return agg.sort_values("score", ascending=False)
    
    def time_drift(self, freq: str = "W"):
        df = self.df.copy()
        
        # Add timestamp if not present
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.date_range(
                start="2024-01-01",
                periods=len(df),
                freq="H"
            )
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["bucket"] = df.timestamp.dt.to_period(freq).astype(str)

        return (
            df.groupby(["bucket", "model"])
            .agg(
                avg_quality=("quality", "mean"),
                avg_cost=("cost", "mean")
            )
            .reset_index()
        )

    def run(self):
        per_model = Aggregator.per_model(self.df)
        per_model = ConfidenceAnalyzer.apply(per_model)
        pareto = ParetoAnalyzer.compute(per_model)

        return {
            "raw": self.df,
            "per_model": per_model,
            "pareto": pareto

        }
