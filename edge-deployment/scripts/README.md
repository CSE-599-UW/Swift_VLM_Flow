# Build TensorRT Engine

## Run build engine bash
Inside container:
```bash
chmod +x /workspace/scripts/build_trt_engines_baseline.sh
/workspace/scripts/build_trt_engines_baseline.sh
```

## Each Stage Explain
1. Converting HF checkpoint
   - Quantization (PTQ) is here.
   > note: Qwen2-VL is BF16 model.
2. Building LLM decoder engine
   - Set `max_batch_size=1`. 
     - Not considering multi-request situation.
   - `max_multimodal_len=1024`
     - Assume max 1024 visual tokens per image.
   - `--max_input_len=1440`
     - Assume max visual + text token = 1440.
   - `--max_seq_len=2048` = max_input_len + max_output_tokens
  
3. Building vision encoder engine
   - Check and adjust data type in `multimodal_builder.py`
    > [!CAUTION] OOM Warning
    > 30GB+ system RAM required
    > - Recommend 32GB swap for Stage 3 ONNX export:
    > ```bash
    >   sudo swapoff -a
    >   sudo dd if=/dev/zero of=/swapfile bs=1G count=32
    >   sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
    > ```

  
4. Smoke Test
   ```bash
   [Q] ['Describe what you see in this image in one sentence.']
   [A]: ['A skateboarder in a green hat is performing a trick over a picnic table, with a crowd watching.']
   Generated 19 tokens
   ```
   Successed

---
## Result
**Memory Usage** 
```bash
Vision engine loaded:   1282 MiB (GPU)
LLM engine loaded:      3414 MiB (GPU)
KV cache allocated:     6.84 GiB
Execution context:       131 MiB
─────────────────────────────────────
Total GPU used:        ~11.7 GiB / 15 GiB
Available remaining:    ~3.7 GiB
```

**Final Engine Size**
```bash
LLM decoder:     3.4 GB  (rank0.engine)
Vision encoder:  1.3 GB  (model.engine)
Total:           4.7 GB
```

---
Next step: run TRT benchmark
```bash
python3 /workspace/benchmark/run_benchmark_trt.py --num_samples 50 --warmup 3
```