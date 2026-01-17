CATEGORIES = [
    "code",
    "question_answering",
    "creative_writing",
    "data_analysis",
    "reasoning",
    "summarization",
    "translation",
    "other"
]

MODEL_CONFIGS = [
    {"slug": "gpt-4o", "provider": "openai"},
    {"slug": "gpt-4o-mini", "provider": "openai"},
    {"slug": "claude-3-7-sonnet-20250219", "provider": "anthropic"},
    {"slug": "claude-sonnet-4-20250514", "provider": "anthropic"},
    {"slug": "gemini-2.0-flash-exp", "provider": "vertex"},
]

PRICING_TABLE = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-3-7-sonnet-20250219": {"input": 0.003, "output": 0.015},
    "claude-sonnet-4-20250514": {"input": 0.008, "output": 0.03},
    "grok-3-mini-fast-latest": {"input": 0.0005, "output": 0.002},
    "gemini-2.0-flash-exp": {"input": 0.0001, "output": 0.0004},
}

REFUSAL_PATTERNS = [
        "i cannot",
        "i can't",
        "i am unable",
        "i’m unable",
        "cannot help with",
        "not allowed",
        "policy",
        "as an ai language model"
    ]