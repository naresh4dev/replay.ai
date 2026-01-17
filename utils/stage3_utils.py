import scipy.stats as st
import math

def per_model_aggregate(df):
    return (
        df.groupby("model")
        .agg(
            avg_quality=("judge_score", "mean"),
            avg_cost=("cost", "mean"),
            avg_latency=("latency", "mean"),
            refusal_rate=("refusal", "mean"),
            sample_size=("judge_score", "count"),
            std_quality=("judge_score", "std")
        )
        .reset_index()
    )

def per_category_aggregate(df):
    return (
        df.groupby(["category", "model"])
        .agg(
            avg_quality=("judge_score", "mean"),
            avg_cost=("cost", "mean"),
            refusal_rate=("refusal", "mean"),
            sample_size=("judge_score", "count")
        )
        .reset_index()
    )

def pareto_frontier(df):
    """
    Expects per-model aggregate dataframe
    """
    points = df[["model", "avg_cost", "avg_quality"]].to_dict("records")
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
            frontier.append(p)

    return frontier


def confidence_interval(mean, std, n, confidence=0.95):
    if n < 2:
        return None

    z = st.norm.ppf((1 + confidence) / 2)
    margin = z * (std / math.sqrt(n))

    return (mean - margin, mean + margin)

def add_confidence_intervals(df):
    df["ci"] = df.apply(
        lambda r: confidence_interval(
            r.avg_quality, r.std_quality, r.sample_size
        ),
        axis=1
    )
    return df
