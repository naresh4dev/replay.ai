from abc import ABC, abstractmethod
from typing import Dict
import re
import tiktoken


class PromptClassifier(ABC):
    @abstractmethod
    def classify(self, prompt: str, features: Dict) -> Dict:
        pass

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, prompt: str) -> Dict:
        pass

class RuleBasedFeatureExtractor:

    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def extract(self, prompt: str) -> Dict:
        tokens = self.encoder.encode(prompt)
        token_count = len(tokens)

        code_blocks = len(re.findall(r"```", prompt))
        inline_code = len(re.findall(r"`.+?`", prompt))
        code_ratio = min((code_blocks + inline_code) / max(token_count, 1), 1.0)

        language="en"

        return {
            "token_count": token_count,
            "code_ratio": round(code_ratio, 3),
            "language": language
        }