# LLM Inference Optimization Study Log

## 1. Inference Basics: Prefill vs. Decode
* **File:** `01_inference_basics.py`
* **Environment:** NVIDIA T1200 (4GB VRAM) | 16GB RAM | Qwen2.5-1.5B (fp16)

**Findings:**
1. The **prefill phase** is compute-bound and processes the entire input prompt at once; processing time spikes drastically as prompt length increases because the computational matrix math scales quadratically ($O(N^2)$). 
2. The **decode phase** is memory-bound and generates output one token at a time; its speed is entirely bottlenecked by how fast the model weights and the accumulating KV cache can be read from memory. 
3. Because the 1.5B model weights exceeded the 4GB VRAM and spilled over to slower system RAM, the hardware vividly demonstrated memory bottlenecks.

**Metrics:**
* **Short Prompt (4 tokens):** Prefill: 0.91s | Decode Speed: 4.23 tokens/sec
* **Long Prompt (401 tokens):** Prefill: 29.07s | Decode Speed: 2.74 tokens/sec