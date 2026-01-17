import json
from typing import Dict
from config.interfaces.interface import PromptClassifier

class PortkeyFewShotClassifier(PromptClassifier):

    def __init__(self, portkey_client, model="gpt-4o-mini"):
        self.client = portkey_client
        self.model = f"@openai/{model}"

    def classify(self, prompt: str, features: Dict) -> Dict:
        system_prompt = """
You are a prompt classifier.

Return JSON only.

Schema:
{
  "category": "code | question_answering | creative_writing | data_analysis | reasoning | summarization | translation | other",
  "intent": "short_snake_case_label",
  "complexity": "low | medium | high",
  "confidence": 0.0-1.0
}
"""

        user_prompt = f"""
Prompt:
{prompt}

Extracted Features:
{json.dumps(features)}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )

        return json.loads(response.choices[0].message.content)
