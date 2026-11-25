"""Tests for runtime API/model settings, model discovery, connection testing, and task deletion."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings_store import RuntimeSettingsStore
from core.config import Settings


def _wait_completed(client: TestClient, run_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/tasks/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in {"completed", "failed"}:
            return data
        time.sleep(0.1)
    raise AssertionError(f"Task {run_id} timed out")


def test_runtime_settings_masking_persistence_and_health(tmp_path: Path) -> None:
    base_settings = Settings(
        llm_api_key="sk-secret1234567890",
        llm_base_url="https://api.deepseek.com",
        llm_model="deepseek-flash",
        embedding_provider="openai",
        embedding_api_key="sk-embsecret123456",
        embedding_base_url="https://api.siliconflow.cn/v1",
        embedding_model="BAAI/bge-m3",
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "outputs"),
    )
    app = create_app(settings=base_settings, persist_rag=False)
    client = TestClient(app)

    res = client.get("/api/settings")
    assert res.status_code == 200
    body = res.json()
    assert body["llm_api_key_configured"] is True
    assert "secret1234567890" not in body["llm_api_key_masked"]
    assert body["llm_api_key_masked"].startswith("sk-sec")
    assert body["llm_model"] == "deepseek-flash"
    assert "deepseek-flash" in body["enabled_llm_models"]

    # Update models without sending api_key -> existing secret must remain intact
    upd = client.put(
        "/api/settings",
        json={
            "llm_model": "deepseek-v4-pro",
            "available_llm_models": ["deepseek-flash", "deepseek-v4-pro", "deepseek-chat"],
            "enabled_llm_models": ["deepseek-flash", "deepseek-v4-pro"],
            "llm_api_key": "",
        },
    )
    assert upd.status_code == 200
    upd_body = upd.json()
    assert upd_body["llm_model"] == "deepseek-v4-pro"
    assert upd_body["enabled_llm_models"] == ["deepseek-flash", "deepseek-v4-pro"]
    assert upd_body["llm_api_key_configured"] is True
    assert upd_body["llm_api_key_masked"].startswith("sk-sec")

    # Reflects live in /api/health
    health = client.get("/api/health").json()
    assert health["llm_model"] == "deepseek-v4-pro"
    assert health["enabled_llm_models"] == ["deepseek-flash", "deepseek-v4-pro"]

    # Survives RuntimeSettingsStore reload
    reloaded = RuntimeSettingsStore(base_settings)
    assert reloaded.get_effective_settings().llm_model == "deepseek-v4-pro"
    assert reloaded.get_effective_settings().llm_api_key == "sk-secret1234567890"


def test_fetch_models_with_v1_fallback_and_connection_tests(tmp_path: Path) -> None:
    base_settings = Settings(
        llm_api_key="sk-initial-key-12345",
        llm_base_url="https://api.example.com",
        llm_model="deepseek-flash",
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "outputs"),
    )
    app = create_app(settings=base_settings, persist_rag=False)
    client = TestClient(app)

    def fake_get(self, url: str, headers=None, **kwargs):
        req = httpx.Request("GET", url)
        if url == "https://api.example.com/models":
            return httpx.Response(404, request=req)
        if url == "https://api.example.com/v1/models":
            return httpx.Response(
                200,
                request=req,
                json={
                    "data": [
                        {"id": "deepseek-v4-pro"},
                        {"id": "deepseek-flash"},
                        {"id": "deepseek-flash"},
                    ]
                },
            )
        return httpx.Response(404, request=req)

    with patch.object(httpx.Client, "get", fake_get):
        models_resp = client.post(
            "/api/settings/models",
            json={"target": "llm", "base_url": "https://api.example.com", "api_key": ""},
        )
    assert models_resp.status_code == 200
    assert models_resp.json() == {
        "models": ["deepseek-flash", "deepseek-v4-pro"],
        "count": 2,
    }

    def fake_post(self, url: str, headers=None, json=None, **kwargs):
        req = httpx.Request("POST", url)
        if url.endswith("/chat/completions"):
            if headers and headers.get("Authorization") == "Bearer bad-key":
                return httpx.Response(401, request=req, text="Invalid API key")
            return httpx.Response(
                200,
                request=req,
                json={"choices": [{"message": {"content": "OK"}}]},
            )
        if url.endswith("/embeddings"):
            return httpx.Response(
                200,
                request=req,
                json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]}]},
            )
        return httpx.Response(404, request=req)

    with patch.object(httpx.Client, "post", fake_post):
        ok_llm = client.post(
            "/api/settings/test",
            json={
                "target": "llm",
                "base_url": "https://api.example.com/v1",
                "api_key": "",
                "model": "deepseek-v4-pro",
            },
        )
        bad_llm = client.post(
            "/api/settings/test",
            json={
                "target": "llm",
                "base_url": "https://api.example.com/v1",
                "api_key": "bad-key",
                "model": "deepseek-v4-pro",
            },
        )
        ok_emb = client.post(
            "/api/settings/test",
            json={
                "target": "embedding",
                "base_url": "https://api.example.com/v1",
                "api_key": "",
                "model": "BAAI/bge-m3",
            },
        )

    assert ok_llm.status_code == 200
    assert ok_llm.json()["ok"] is True
    assert "deepseek-v4-pro" in ok_llm.json()["message"]

    assert bad_llm.status_code == 200
    assert bad_llm.json()["ok"] is False
    assert "401" in bad_llm.json()["message"]

    assert ok_emb.status_code == 200
    assert ok_emb.json()["ok"] is True
    assert "维度 4" in ok_emb.json()["message"]


def test_per_run_model_selection_and_delete_task(tmp_path: Path) -> None:
    settings = Settings(
        llm_api_key="",
        embedding_provider="local_hash",
        max_papers_per_query=4,
        top_k_papers=4,
        max_iterations=0,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "outputs"),
    )
    app = create_app(settings=settings, persist_rag=False)
    client = TestClient(app)

    create_resp = client.post(
        "/api/tasks",
        json={
            "topic": "Agentic RAG model selection test",
            "top_k_papers": 4,
            "max_papers_per_query": 4,
            "max_iterations": 0,
            "download_pdfs": False,
            "require_plan_approval": False,
            "mock": True,
            "llm_model": "deepseek-v4-pro",
        },
    )
    assert create_resp.status_code == 202
    run_id = create_resp.json()["run_id"]

    completed = _wait_completed(client, run_id)
    assert completed["status"] == "completed"
    out_dir = Path(completed["output_dir"])
    assert out_dir.exists()

    del_resp = client.delete(f"/api/tasks/{run_id}")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"deleted": True, "run_id": run_id}
    assert not out_dir.exists()
    assert client.get(f"/api/tasks/{run_id}").status_code == 404
    assert client.delete(f"/api/tasks/{run_id}").status_code == 404
