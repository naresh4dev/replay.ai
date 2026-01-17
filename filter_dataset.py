# Random filter length of 1000 rows from lmsys-chat-1m dataset and also select only English language prompts
import random
import json

# Define model slugs and their providers
MODEL_CONFIGS = [
    {"slug": "gpt-4o", "provider": "openai"},
    {"slug": "gpt-4o-mini", "provider": "openai"},
    {"slug": "claude-3-7-sonnet-20250219", "provider": "anthropic"},
    {"slug": "claude-sonnet-4-20250514", "provider": "anthropic"},
    {"slug": "gemini-2.0-flash-exp", "provider": "vertex"},
]

def filter_dataset(input_file: str, output_file: str, sample_size: int = 1000, language: str = "English"):
    filtered_rows = []
    
    with open(input_file, "r") as infile:
        for line in infile:
            row = json.loads(line)
            if row.get("language") == language:
                filtered_rows.append(row)
    
    # Randomly sample the specified number of rows
    sampled_rows = random.sample(filtered_rows, min(sample_size, len(filtered_rows)))
    
    # Add randomized model slug and provider to each row
    for row in sampled_rows:
        model_config = random.choice(MODEL_CONFIGS)
        row["replay_model_slug"] = model_config["slug"]
        row["replay_provider"] = model_config["provider"]
    
    # Write sampled rows to output file
    with open(output_file, "w") as outfile:
        for row in sampled_rows:
            outfile.write(json.dumps(row) + "\n")
    
    print(f"Filtered {len(filtered_rows)} English rows, sampled {len(sampled_rows)} rows to {output_file}")

if __name__ == "__main__":
    input_path = "lmsys_randomized_models.jsonl"
    output_path = "lmsys_filtered_english_1k.jsonl"
    filter_dataset(input_path, output_path)