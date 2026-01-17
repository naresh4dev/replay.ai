from config.constants.constant import PRICING_TABLE, REFUSAL_PATTERNS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CostCalculator:

    def calculate(self, model_slug, input_tokens, output_tokens):
        pricing = PRICING_TABLE[model_slug]
        return round(
            input_tokens * pricing["input"] +
            output_tokens * pricing["output"],
            6
        )
class RefusalDetector:
    def __init__(self):
        self.REFUSAL_PATTERNS = REFUSAL_PATTERNS
    def detect(self, response_text: str) -> bool:
        text = response_text.lower()
        return any(p in text for p in self.REFUSAL_PATTERNS)

class SimilarityEvaluator:

    def score(self, reference: str, candidate: str) -> float:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([reference, candidate])
        score = cosine_similarity(vectors[0], vectors[1])[0][0]
        return round(float(score), 3)