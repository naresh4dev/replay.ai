import random
import json
from datasets import load_dataset

SUPPORTED_MODEL_SLUGS = ["o3-deep-research", "o3-2025-04-16", "o3", "o1-pro-2025-03-19", "o1-pro", "o1-preview-2024-09-12", "o1-preview", "o1-mini-2024-09-12", "o1-mini", "o1-2024-12-17", "o1", "multimodalembedding@001", "mistral.pixtral-large-2502-v1:0", "mistral.mixtral-8x7b-instruct-v0:1", "mistral.mistral-small-2402-v1:0", "mistral.mistral-large-2407-v1:0", "mistral.mistral-large-2402-v1:0", "mistral.mistral-7b-instruct-v0:2", "meta.llama3-8b-instruct-v1:0", "meta.llama3-70b-instruct-v1:0", "meta.llama3-3-70b-instruct-v1:0", "meta.llama3-2-90b-instruct-v1:0", "meta.llama3-2-11b-instruct-v1:0", "meta.llama3-1-8b-instruct-v1:0", "meta.llama3-1-70b-instruct-v1:0", "meta.llama3-1-405b-instruct-v1:0", "meta.llama2-70b-chat-v1", "meta.llama2-13b-chat-v1", "meta.llama-3.2-90b-vision-instruct-maas", "meta.llama-3.1-8b-instruct-maas", "meta.llama-3.1-70b-instruct-maas", "meta.llama-3.1-405b-instruct-maas", "llama4@llama-4-scout-17b-16e-instruct", "llama3_1@llama-3.1-8b-instruct", "llama3-8b-8192", "llama3-70b-8192", "llama3-3@llama-3.3-70b-instruct", "llama3-2@llama-3.2-3b-instruct", "llama3-2@llama-3.2-1b-instruct", "imagen-4.0-ultra-generate-preview-06-06", "imagen-4.0-ultra-generate-preview-05-20", "imagen-4.0-ultra-generate-001", "imagen-4.0-generate-preview-06-06", "imagen-4.0-generate-preview-05-20", "imagen-4.0-generate-001", "imagen-4.0-fast-generate-preview-06-06", "imagen-4.0-fast-generate-001", "imagen-3.0-generate-002", "imagen-3.0-generate-001", "imagen-3.0-fast-generate-001", "imagegeneration@006", "imagegeneration@005", "imagegeneration@002", "grok-vision-beta", "grok-code-fast-1-0825", "grok-code-fast-1", "grok-code-fast", "grok-beta", "grok-4-latest", "grok-4-fast-reasoning-latest", "grok-4-fast-reasoning", "grok-4-fast-non-reasoning-latest", "grok-4-fast-non-reasoning", "grok-4-fast", "grok-4-1-fast-reasoning-latest", "grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning-latest", "grok-4-1-fast-non-reasoning", "grok-4-1-fast", "grok-4-0709", "grok-4", "grok-3-mini-latest", "grok-3-mini-fast-latest", "grok-3-mini-fast-beta", "grok-3-mini-fast", "grok-3-mini-beta", "grok-3-mini", "grok-3-latest", "grok-3-fast-latest", "grok-3-fast-beta", "grok-3-fast", "grok-3-beta", "grok-3", "grok-2-vision-latest", "grok-2-vision-1212", "grok-2-vision", "grok-2-latest", "grok-2-1212", "grok-2", "gpt-realtime-mini-2025-10-06", "gpt-realtime-mini", "gpt-realtime-2025-08-28", "gpt-realtime", "gpt-image-1-mini", "gpt-image-1", "gpt-audio-mini-2025-10-06", "gpt-audio-mini", "gpt-audio-2025-08-28", "gpt-audio", "gpt-5.2-pro-2025-12-11"]

def infer_provider(slug: str) -> str:
    if slug.startswith(("o1", "o3", "gpt")):
        return "openai"
    if slug.startswith(("meta.", "llama")):
        return "meta"
    if slug.startswith("mistral"):
        return "mistral"
    if slug.startswith("grok"):
        return "xai"
    if slug.startswith(("imagen", "imagegeneration", "multimodalembedding")):
        return "google"
    return "unknown"

dataset = load_dataset(
    "lmsys/lmsys-chat-1m",
    split="train",
    streaming=True
)

OUTPUT_FILE = "lmsys_randomized_models.jsonl"
with open(OUTPUT_FILE, "w") as f:
    for row in dataset:
        chosen_model = random.choice(SUPPORTED_MODEL_SLUGS)
        provider = infer_provider(chosen_model)

        output_row = {
            "conversation_id": row.get("conversation_id"),
            "prompt": row.get("conversation")[0]["content"],
            "original_model": row.get("model"),
            "replay_model_slug": chosen_model,
            "replay_provider": provider,
            "turns": row.get("conversation"),
            "language": row.get("language"),
            "source": "lmsys-chat-1m"
        }

        f.write(json.dumps(output_row) + "\n")

print(f"✅ Randomized replay dataset written to {OUTPUT_FILE}")