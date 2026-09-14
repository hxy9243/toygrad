"""Tests for the YuE2 OpenAI-compatible serving server."""
import base64
import io
import json
import zipfile
import pytest
from fastapi.testclient import TestClient

# Use mock mode for fast and deterministic unit testing
import os
os.environ["MOCK_YUE2"] = "1"

try:
    from app.server import app, SAMPLE_ABC
    from app.handler import handler
except ImportError:
    from server import app, SAMPLE_ABC
    from handler import handler

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["mock_mode"] is True


def test_list_models():
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "m-a-p/YuE2-3B" in ids or "YuE2-3B" in ids


def test_chat_completions_abc_only():
    payload = {
        "model": "YuE2-3B",
        "messages": [
            {"role": "system", "content": "Style: Jazz-funk, piano"},
            {"role": "user", "content": "[Verse]\nTonight is alive\n[Chorus]\nMusic will thrive"}
        ],
        "return_format": "abc"
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/json")
    data = res.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    content = data["choices"][0]["message"]["content"]
    assert "X:1" in content or "T:" in content
    assert "audio" not in data["choices"][0]["message"]
    assert data["yue2_metadata"]["stage"] == "plan"


def test_chat_completions_abc_plain_text():
    payload = {
        "model": "YuE2-3B",
        "style": "Pop",
        "lyrics": "Hello world",
        "return_format": "abc"
    }
    res = client.post("/v1/chat/completions", json=payload, headers={"Accept": "text/plain"})
    assert res.status_code == 200
    assert "text/plain" in res.headers["content-type"]
    text = res.text
    assert "X:1" in text or "T:" in text


def test_chat_completions_arraybuffer_option():
    payload = {
        "model": "YuE2-3B",
        "style": "Cyber Metal, guitar, heavy drums",
        "lyrics": "[Verse]\nNeon pulse\n[Chorus]\nRise and fight",
        "return_format": "arraybuffer",
        "seed": 9999
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/octet-stream"
    assert "attachment; filename=" in res.headers.get("content-disposition", "")
    assert res.headers.get("x-yue2-status") == "complete"
    assert res.headers.get("x-yue2-seed") == "9999"

    # Verify that the binary response is a valid ArrayBuffer / ZIP with all generated files
    buf = io.BytesIO(res.content)
    with zipfile.ZipFile(buf, "r") as zf:
        file_names = zf.namelist()
        assert "score.abc" in file_names
        assert "audio.flac" in file_names
        assert "plan.json" in file_names
        assert "result.json" in file_names
        assert "semantic.npy" in file_names
        assert "latent.npy" in file_names

        # Verify abc content inside zip
        abc_content = zf.read("score.abc").decode("utf-8")
        assert "X:1" in abc_content

        # Verify plan.json inside zip
        plan_data = json.loads(zf.read("plan.json").decode("utf-8"))
        assert plan_data["style"] == "Cyber Metal, guitar, heavy drums"


def test_chat_completions_accept_header_arraybuffer():
    payload = {
        "model": "YuE2-3B",
        "messages": [
            {"role": "user", "content": "Style: Rock\nLyrics: Rock on"}
        ]
    }
    res = client.post("/v1/chat/completions", json=payload, headers={"Accept": "application/octet-stream"})
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/octet-stream"
    buf = io.BytesIO(res.content)
    with zipfile.ZipFile(buf, "r") as zf:
        assert "score.abc" in zf.namelist()
        assert "audio.flac" in zf.namelist()


def test_chat_completions_json_with_audio():
    payload = {
        "model": "YuE2-3B",
        "style": "Jazz",
        "lyrics": "Autumn leaves falling",
        "return_format": "json"
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "chat.completion"
    choice = data["choices"][0]
    assert "audio" in choice["message"]
    assert choice["message"]["audio"]["format"] == "flac"
    assert len(choice["message"]["audio"]["data"]) > 0


def test_runpod_handler_plan():
    job = {
        "id": "test-job-plan",
        "input": {
            "style": "Cinematic Orchestral",
            "lyrics": "[Verse]\nEchoes in the dark",
            "stage": "plan",
            "return_format": "abc",
        }
    }
    res = handler(job)
    assert res["status"] == "success"
    assert res["stage"] == "plan"
    assert "abc" in res
    assert "X:1" in res["abc"] or "T:" in res["abc"]
    assert "plan" in res


def test_runpod_handler_audio_json():
    job = {
        "id": "test-job-audio",
        "input": {
            "style": "Synthwave",
            "lyrics": "[Chorus]\nNeon nights",
            "stage": "audio",
            "return_format": "json",
            "duration": 5.0,
        }
    }
    res = handler(job)
    assert res["status"] == "success"
    assert res["stage"] == "audio"
    assert "audio_base64" in res
    assert res["audio_format"] == "flac"
    assert res["sample_rate"] == 48000


def test_runpod_handler_arraybuffer_zip():
    job = {
        "id": "test-job-zip",
        "input": {
            "style": "Acoustic Folk",
            "lyrics": "[Verse]\nGentle breeze",
            "return_format": "arraybuffer",
            "duration": 2.0,
        }
    }
    res = handler(job)
    assert res["status"] == "success"
    assert "zip_base64" in res
    raw_zip = base64.b64decode(res["zip_base64"])
    with zipfile.ZipFile(io.BytesIO(raw_zip), "r") as zf:
        assert "score.abc" in zf.namelist()
        assert "audio.flac" in zf.namelist()
        assert "plan.json" in zf.namelist()
        assert "result.json" in zf.namelist()


def test_runpod_handler_missing_input():
    res = handler({})
    assert res["status"] == "error"
    assert "Missing 'input'" in res["message"]

