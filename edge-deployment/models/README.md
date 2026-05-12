# Model Weights

Model weights are not tracked in git (too large).

## Download Options

**Option A — Google Drive** (recommended):
[Qwen2-VL-2B-Instruct weights](https://drive.google.com/file/d/1F2dkfXLpiJB5lpe7LzvDe3IoqOGmW07Y/view?usp=sharing)

Extract to: `edge-deployment/models/Qwen2-VL-2B-Instruct/`

**Option B — HuggingFace**:

```bash
pip install huggingface_hub
hf download Qwen/Qwen2-VL-2B-Instruct \
  --local-dir ./models/Qwen2-VL-2B-Instruct
```

## Expected Folder Structure After Download

```
models/Qwen2-VL-2B-Instruct/
├── config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── tokenizer.json
└── ...
```
