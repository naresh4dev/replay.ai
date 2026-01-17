from data.dataclass import ReplayResponse
import time

class ReplayExecutor:

    def __init__(self, client):
        self.client = client
        
    def run_single(self, prompt: str, model_slug: str, provider: str) -> ReplayResponse:
        """Execute replay for a single model."""
        start = time.time()

        # Build request parameters
        params = {
            "model": f"@{provider}/{model_slug}",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }
        
        # Anthropic requires max_tokens
        if provider == "anthropic":
            params["max_tokens"] = 4096

        completion = self.client.with_options(
        ).chat.completions.create(**params)

        latency = int((time.time() - start) * 1000)

        msg = completion.choices[0].message.content
        usage = completion.usage

        return ReplayResponse(
            model_slug=model_slug,
            provider=provider,
            response_text=msg,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=latency
        )
    
    def run(self, prompt: str, model_configs: list) -> dict:
        """
        Returns:
        {
          model_slug: ReplayResponse
        }
        """
        responses = {}

        for model in model_configs:
            response = self.run_single(prompt, model["slug"], model["provider"])
            responses[model["slug"]] = response

        return responses
