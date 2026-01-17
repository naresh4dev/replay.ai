from dataclasses import dataclass
from typing import Dict, Optional
from abc import ABC, abstractmethod

@dataclass
class EnrichedPrompt:
    text: str
    features: Dict
    category: str
    intent: str
    complexity: str
    confidence: float
    embedding: Optional[list] = None
    cluster_id: Optional[str] = None

@dataclass
class ReplayResponse:
    model_slug: str
    provider: str
    response_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int

@dataclass
class EvaluationResult:
    model_slug: str
    provider: str
    response_text: str
    cost_usd: float
    refusal: bool
    similarity_score: float
    latency_ms: int
    judge_score: int
    judge_reasoning: str
    raw_data: Optional[Dict] = None