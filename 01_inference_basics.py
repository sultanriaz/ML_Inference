import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = r"C:\black_paper\FYP\MedMind_Unified_Project\medmind_refactored\models\base_model"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def test_inference(prompt_text, max_new_tokens=20):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    print(f"\n--- Testing Prompt (Length: {input_length} tokens) ---")
    
    # 1. Measure Prefill (Time to First Token / TTFT)
    start_prefill = time.perf_counter()
    with torch.no_grad():
        # Generate exactly 1 token to isolate the prefill phase
        model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_time = time.perf_counter() - start_prefill
    
    # 2. Measure Total Time (Prefill + Decode)
    start_total = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_total
    
    # 3. Calculate Decode-only metrics
    decode_only_time = total_time - prefill_time
    tokens_generated = max_new_tokens - 1
    
    print(f"Prefill Time (Compute-bound): {prefill_time:.4f} seconds")
    print(f"Decode Time for {tokens_generated} tokens (Memory-bound): {decode_only_time:.4f} seconds")
    print(f"Decode Speed: {tokens_generated / decode_only_time:.2f} tokens/sec")

# Run tests as required by your schedule
short_prompt = "What is hypertension?"
long_prompt = "What is hypertension? " * 100  # Artificially inflate the prompt length

test_inference(short_prompt)
test_inference(long_prompt)