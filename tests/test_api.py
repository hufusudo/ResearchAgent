"""Tests for FastAPI REST + SSE endpoints and UI decoupling enforcement."""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from api.main import create_app
from core.config import Settings


def _wait_for_status(client: TestClient, run_id: str, targets: set[str], timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/tasks/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in targets:
            return data
        time.sleep(0.1)
    raise AssertionError(f"Task {run_id} did not reach {targets} within {timeout}s")


def test_ui_app_is_strictly_decoupled_from_core_and_api() -> None:
    ui_source = Path("ui/app.py").read_text(encoding="utf-8")
    tree = ast.parse(ui_source)
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("core", "api")):
                    forbidden.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(("core", "api")):
                forbidden.append(node.module)
    assert forbidden == []


def test_api_rest_hitl_sse_report_and_chunks(tmp_path: Path) -> None:
    settings = Settings(
        llm_api_key="",
        embedding_provider="local_hash",
        max_papers_per_query=4,
        top_k_papers=4,
        max_iterations=1,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "outputs"),
    )
    app = create_app(settings=settings, persist_rag=False)
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    create_resp = client.post(
        "/api/tasks",
        json={
            "topic": "Agentic RAG literature survey",
            "top_k_papers": 4,
            "max_papers_per_query": 4,
            "max_iterations": 1,
            "download_pdfs": False,
            "require_plan_approval": True,
            "mock": True,
        },
    )
    assert create_resp.status_code == 202
    run_id = create_resp.json()["run_id"]

    paused = _wait_for_status(client, run_id, {"awaiting_approval", "failed"})
    assert paused["status"] == "awaiting_approval"
    assert paused["plan"] is not None

    edited_plan = dict(paused["plan"])
    edited_plan["refined_scope"] = "API Approved Scope"

    approve_resp = client.post(f"/api/tasks/{run_id}/approve", json={"plan": edited_plan})
    assert approve_resp.status_code == 202

    completed = _wait_for_status(client, run_id, {"completed", "failed"})
    assert completed["status"] == "completed"
    assert completed["plan"]["refined_scope"] == "API Approved Scope"
    assert completed["verification_report"] is not None
    assert len(completed["evaluation_history"]) >= 1

    tasks_resp = client.get("/api/tasks")
    assert tasks_resp.status_code == 200
    assert any(t["run_id"] == run_id for t in tasks_resp.json())

    report_resp = client.get(f"/api/tasks/{run_id}/report")
    assert report_resp.status_code == 200
    assert "## References" in report_resp.text
    assert "## Citation Verification Summary" in report_resp.text

    chunks_resp = client.get(f"/api/tasks/{run_id}/chunks", params={"paper_id": "P01"})
    assert chunks_resp.status_code == 200
    chunks_data = chunks_resp.json()
    assert chunks_data["total"] >= 1
    assert all(c["paper_id"] == "P01" for c in chunks_data["chunks"])

    events_resp = client.get(f"/api/tasks/{run_id}/events")
    assert events_resp.status_code == 200
    assert "event: trace" in events_resp.text
    assert "event: status" in events_resp.text


def _build_pdf_bytes(title: str, body: str) -> bytes:
    doc = pymupdf.open()
    doc.set_metadata({"title": title, "author": "Upload Tester"})
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 545, 780), body, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def test_api_upload_download_bibtex_and_spa_hosting(tmp_path: Path) -> None:
    fake_dist = tmp_path / "dist"
    fake_dist.mkdir()
    (fake_dist / "index.html").write_text(
        "<!doctype html><html><title>ReacherAgent</title><body><div id='root'></div></body></html>",
        encoding="utf-8",
    )

    settings = Settings(
        llm_api_key="",
        embedding_provider="local_hash",
        max_papers_per_query=4,
        top_k_papers=8,
        max_iterations=0,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "outputs"),
    )
    app = create_app(settings=settings, persist_rag=False, static_dir=fake_dist)
    client = TestClient(app)

    spa_resp = client.get("/")
    assert spa_resp.status_code == 200
    assert "ReacherAgent" in spa_resp.text

    bad_upload = client.post(
        "/api/uploads",
        files=[("files", ("notes.txt", b"plain text", "text/plain"))],
    )
    assert bad_upload.status_code == 400

    pdf_bytes = _build_pdf_bytes(
        "Uploaded Local Paper on Agentic RAG",
        "1. Introduction\nUploaded PDF content about hybrid RAG and citation grounding.",
    )
    ok_upload = client.post(
        "/api/uploads",
        files=[("files", ("uploaded_rag.pdf", pdf_bytes, "application/pdf"))],
    )
    assert ok_upload.status_code == 201
    upload_data = ok_upload.json()
    batch_id = upload_data["upload_batch_id"]
    assert upload_data["files"][0]["filename"] == "uploaded_rag.pdf"

    create_resp = client.post(
        "/api/tasks",
        json={
            "topic": "Agentic RAG grounding",
            "top_k_papers": 8,
            "max_papers_per_query": 4,
            "max_iterations": 0,
            "download_pdfs": False,
            "require_plan_approval": False,
            "mock": True,
            "upload_batch_id": batch_id,
            "sources": ["arxiv", "openalex"],
        },
    )
    assert create_resp.status_code == 202
    run_id = create_resp.json()["run_id"]

    completed = _wait_for_status(client, run_id, {"completed", "failed"})
    assert completed["status"] == "completed"
    sources = {p["source"] for p in completed["selected_papers"]}
    assert "local_pdf" in sources

    dl_report = client.get(f"/api/tasks/{run_id}/report", params={"download": "true"})
    assert dl_report.status_code == 200
    assert "attachment;" in dl_report.headers.get("content-disposition", "")
    assert f"reacher_survey_{run_id}.md" in dl_report.headers.get("content-disposition", "")

    dl_bib = client.get(f"/api/tasks/{run_id}/bibtex", params={"download": "true"})
    assert dl_bib.status_code == 200
    assert "attachment;" in dl_bib.headers.get("content-disposition", "")
    assert "@article{P01," in dl_bib.text
