import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import io_builder


def test_build_request_with_image():
    req = io_builder.build_request("Describe the image.", image_path="/abs/img.jpg")
    assert req["messages"][0]["role"] == "user"
    content = req["messages"][0]["content"]
    assert content[0] == {"type": "image", "image": "/abs/img.jpg"}
    assert content[1] == {"type": "text", "text": "Describe the image."}


def test_build_request_text_only():
    req = io_builder.build_request("Hello?", image_path=None)
    # text-only -> content is a plain string per the input format
    assert req["messages"][0]["content"] == "Hello?"


def test_build_input_json_structure():
    samples = [
        {"question_id": 1, "prompt": "What is this?", "image_path": "/a.jpg"},
        {"question_id": 2, "prompt": "And this?", "image_path": "/b.jpg"},
    ]
    payload = io_builder.build_input_json(samples, max_new_tokens=128,
                                          temperature=0.0, top_k=1, top_p=1.0)
    assert payload["batch_size"] == 1
    assert payload["max_generate_length"] == 128
    assert payload["temperature"] == 0.0
    assert payload["top_k"] == 1
    assert len(payload["requests"]) == 2
    assert payload["requests"][0]["messages"][0]["content"][0]["image"] == "/a.jpg"
    # round-trips through json
    import json
    s = json.dumps(payload)
    assert "What is this?" in s and "/b.jpg" in s


def test_write_input_json(tmp_path):
    samples = [{"question_id": 1, "prompt": "Hi", "image_path": "/a.jpg"}]
    p = tmp_path / "in.json"
    io_builder.write_input_json(io_builder.build_input_json(samples, max_new_tokens=16), str(p))
    import json
    loaded = json.load(open(p))
    assert loaded["requests"][0]["messages"][0]["content"][1]["text"] == "Hi"
