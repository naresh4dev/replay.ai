from config.interfaces.interface import RuleBasedFeatureExtractor
from config.llm.classifier import PortkeyFewShotClassifier
from config.llm.embedding import EmbeddingGenerator
from data.dataclass import EnrichedPrompt

from utils.portkey import get_portkey_client

class PromptEnrichmentService:

    def __init__(
        self,
        embedder: EmbeddingGenerator = None
    ):
        self.client = get_portkey_client(provider="@openai")
        self.feature_extractor = RuleBasedFeatureExtractor()
        self.classifier = PortkeyFewShotClassifier(self.client)
        self.embedder = embedder

    def enrich(self, prompt_text: str) -> EnrichedPrompt:
        features = self.feature_extractor.extract(prompt_text)

        classification = self.classifier.classify(
            prompt_text, features
        )

        embedding = None
        if self.embedder:
            embedding = self.embedder.embed(prompt_text)

        return EnrichedPrompt(
            text=prompt_text,
            features=features,
            category=classification["category"],
            intent=classification["intent"],
            complexity=classification["complexity"],
            confidence=classification["confidence"],
            embedding=embedding
        )
