import json


class LLMJudge:

    def __init__(self, client, model="gpt-4o-mini"):
        self.client = client
        self.model = model

    def evaluate_group(self, prompt, reference, candidates: dict):
        formatted_answers = "\n\n".join(
            f"{k}) {v}" for k, v in candidates.items()
        )

        system_prompt = """
You are an impartial evaluator.

Compare multiple answers to the same prompt.
Score each answer from 1–10.
Higher is better.
Have to justify your scores.

THINGS TO NOTE:
- Consider if any model answer uses less tokens to achieve the same quality.
- Consider if any model answer refuses to answer the prompt.
- Consider relevance, correctness, creativity, and depth. depending on the prompt.
- Consider latency as one of the factors for time-sensitive prompts.


Return JSON only.
{
  "scores": 
    {
    "model_slug": score_integer,
    } ,
  "reasoning": "short explanation"
}
"""

        user_prompt = f"""
Prompt:
{prompt}

Reference Answer:
{reference}

Candidate Answers:
{formatted_answers}
"""

        try:
            response = self.client.with_options(
                provider="@openai"
            ).chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=2000
            )

            content = response.choices[0].message.content
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback if response isn't valid JSON
            return {
                "scores": {slug: 5 for slug in candidates.keys()},
                "reasoning": "Failed to parse judge response"
            }
        except Exception as e:
            print(f"Judge evaluation error: {e}")
            return {
                "scores": {slug: 5 for slug in candidates.keys()},
                "reasoning": f"Judge error: {str(e)}"
            }
