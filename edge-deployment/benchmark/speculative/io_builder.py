"""io_builder.py — build TensorRT-Edge-LLM `input.json` payloads for VLM inference.

Format per docs/source/user_guide/format/input-format.md:
  {batch_size, temperature, top_p, top_k, max_generate_length, requests: [...]}
  request = {messages: [{role, content}]}
  content = str (text-only) OR [{type:"image", image:path}, {type:"text", text:...}]
Images are referenced by absolute/relative file path.
"""
import json


def build_request(prompt: str, image_path: str | None = None) -> dict:
    """One chat request. With an image -> content is a [image, text] list;
    text-only -> content is a plain string."""
    if image_path:
        content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt
    return {"messages": [{"role": "user", "content": content}]}


def build_input_json(samples: list[dict], max_new_tokens: int,
                     temperature: float = 0.0, top_k: int = 1,
                     top_p: float = 1.0, batch_size: int = 1) -> dict:
    """Assemble a full input.json payload from benchmark samples.

    Each sample: {"prompt": str, "image_path": str|None, "question_id": ...}.
    Defaults are greedy (temperature 0.0, top_k 1) so SD-on vs SD-off output is
    directly comparable for the losslessness check.
    """
    return {
        "batch_size": batch_size,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_generate_length": max_new_tokens,
        "apply_chat_template": True,
        "requests": [build_request(s["prompt"], s.get("image_path")) for s in samples],
    }


def write_input_json(payload: dict, path: str) -> str:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
