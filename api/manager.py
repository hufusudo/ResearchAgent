"""Task execution manager with SQLite persistence and asyncio SSE broadcasting."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from api.schemas import (
    CreateTaskRequest,
    TaskDetailResponse,
    TaskSummary,
    UploadedFileInfo,
    UploadResponse,
)
from api.settings_store import RuntimeSettingsStore
from core.config import Settings, get_settings
from core.graph.workflow import resume_research_session, start_research_session
from core.schemas import ResearchPlan, TextChunk, TraceEvent


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskManager:
    """Manages background LangGraph tasks, SQLite metadata, CLI history discovery, and SSE."""

    def __init__(
        self,
        settings: Settings | None = None,
        persist_rag: bool = True,
        settings_store: RuntimeSettingsStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.persist_rag = persist_rag
        self.settings_store = settings_store or RuntimeSettingsStore(self.settings)
        self.settings.data_path.mkdir(parents=True, exist_ok=True)
        self.settings.output_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.settings.data_path / "tasks.db"

        self._lock = threading.Lock()
        self._live_events: dict[str, list[dict[str, Any]]] = {}
        self._subscribers: dict[
            str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any] | None]]]
        ] = {}
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                        run_id TEXT PRIMARY KEY,
                        topic TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        output_dir TEXT NOT NULL,
                        options_json TEXT NOT NULL,
                        error_message TEXT
                    )
                    """
                )
                conn.commit()

    def _read_state_json(self, run_id: str) -> dict[str, Any] | None:
        state_file = self.settings.output_path / run_id / "state.json"
        if not state_file.exists():
            return None
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _read_report_md(self, run_id: str) -> str:
        report_file = self.settings.output_path / run_id / "report.md"
        if not report_file.exists():
            return ""
        try:
            return report_file.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _push_event(self, run_id: str, event_name: str, data: dict[str, Any]) -> None:
        item = {"event": event_name, "data": data}
        with self._lock:
            self._live_events.setdefault(run_id, []).append(item)
            subs = list(self._subscribers.get(run_id, []))
        for loop, queue in subs:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                pass

    def _close_subscribers(self, run_id: str) -> None:
        with self._lock:
            subs = list(self._subscribers.get(run_id, []))
        for loop, queue in subs:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except RuntimeError:
                pass

    def _update_task_row(
        self,
        run_id: str,
        status: str,
        output_dir: str | None = None,
        error_message: str | None = None,
    ) -> None:
        now = _now_iso()
        with self._lock:
            with self._connect() as conn:
                if output_dir is not None:
                    conn.execute(
                        """
                        UPDATE tasks
                        SET status = ?, updated_at = ?, output_dir = ?, error_message = ?
                        WHERE run_id = ?
                        """,
                        (status, now, output_dir, error_message, run_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE tasks
                        SET status = ?, updated_at = ?, error_message = ?
                        WHERE run_id = ?
                        """,
                        (status, now, error_message, run_id),
                    )
                conn.commit()

    def _get_task_row(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM tasks WHERE run_id = ?", (run_id,)
                ).fetchone()
        return dict(row) if row is not None else None

    def create_task(self, req: CreateTaskRequest) -> TaskSummary:
        """Register a new survey task and start execution in a background thread."""
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        now = _now_iso()
        output_dir = str(self.settings.output_path / run_id)
        options = req.model_dump(mode="json")

        with self._lock:
            self._live_events[run_id] = []
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO tasks (run_id, topic, status, created_at, updated_at, output_dir, options_json, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        req.topic,
                        "running",
                        now,
                        now,
                        output_dir,
                        json.dumps(options, ensure_ascii=False),
                        None,
                    ),
                )
                conn.commit()

        worker = threading.Thread(
            target=self._run_start_worker,
            args=(run_id, req),
            daemon=True,
        )
        worker.start()

        return TaskSummary(
            run_id=run_id,
            topic=req.topic,
            status="running",
            created_at=now,
            updated_at=now,
        )

    def _resolve_upload_dir(self, upload_batch_id: str | None) -> Path | None:
        if not upload_batch_id or not re.fullmatch(r"[A-Za-z0-9_-]+", upload_batch_id):
            return None
        candidate = self.settings.data_path / "uploads" / upload_batch_id
        return candidate if candidate.exists() and candidate.is_dir() else None

    def save_uploaded_pdfs(self, files: list[tuple[str, bytes]]) -> UploadResponse:
        """Validate and persist uploaded local PDF files into a temporary batch directory."""
        if not files:
            raise ValueError("Please upload at least one PDF file.")

        batch_id = f"up_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        batch_dir = self.settings.data_path / "uploads" / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        saved_files: list[UploadedFileInfo] = []
        for idx, (raw_name, content) in enumerate(files, start=1):
            base_name = Path(raw_name).name.strip() or f"document_{idx}.pdf"
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name)
            if not safe_name.lower().endswith(".pdf"):
                raise ValueError(f"Unsupported file '{base_name}': only .pdf files are allowed.")
            if not content or not content[:1024].lstrip().startswith(b"%PDF"):
                raise ValueError(f"Invalid PDF content in '{base_name}': missing %PDF header.")

            dest = batch_dir / safe_name
            if dest.exists():
                dest = batch_dir / f"{idx}_{safe_name}"
            dest.write_bytes(content)
            saved_files.append(UploadedFileInfo(filename=dest.name, size_bytes=len(content)))

        return UploadResponse(upload_batch_id=batch_id, files=saved_files)

    def _run_start_worker(self, run_id: str, req: CreateTaskRequest) -> None:
        def on_event(event: TraceEvent) -> None:
            self._push_event(run_id, "trace", event.model_dump(mode="json"))

        try:
            local_pdf_dir = self._resolve_upload_dir(req.upload_batch_id)
            effective_settings = self.settings_store.get_effective_settings()
            _, status, out_dir = start_research_session(
                topic=req.topic,
                settings=effective_settings,
                mock=req.mock,
                top_k_papers=req.top_k_papers,
                max_papers_per_query=req.max_papers_per_query,
                max_iterations=req.max_iterations,
                download_pdfs=req.download_pdfs,
                local_pdf_dir=local_pdf_dir,
                persist_rag=self.persist_rag,
                require_plan_approval=req.require_plan_approval,
                run_id=run_id,
                sources=req.sources,
                llm_model=req.llm_model,
                on_event=on_event,
            )
            self._update_task_row(run_id, status=status, output_dir=str(out_dir))
            self._push_event(run_id, "status", {"run_id": run_id, "status": status})
        except Exception as exc:
            self._update_task_row(run_id, status="failed", error_message=str(exc))
            self._push_event(
                run_id,
                "status",
                {"run_id": run_id, "status": "failed", "error": str(exc)},
            )
        finally:
            self._close_subscribers(run_id)

    def approve_task(self, run_id: str, approved_plan: ResearchPlan | None = None) -> TaskSummary:
        """Resume a paused HITL task with optional user plan modifications."""
        row = self._get_task_row(run_id)
        if row is None:
            raise KeyError(f"Task '{run_id}' not found.")
        if row["status"] != "awaiting_approval":
            raise ValueError(
                f"Task '{run_id}' is in status '{row['status']}', expected 'awaiting_approval'."
            )

        options = json.loads(row["options_json"])
        req = CreateTaskRequest.model_validate(options)
        self._update_task_row(run_id, status="running")
        self._push_event(run_id, "status", {"run_id": run_id, "status": "running"})

        worker = threading.Thread(
            target=self._run_resume_worker,
            args=(run_id, req, approved_plan),
            daemon=True,
        )
        worker.start()

        updated_row = self._get_task_row(run_id) or row
        return TaskSummary(
            run_id=run_id,
            topic=updated_row["topic"],
            status="running",
            created_at=updated_row["created_at"],
            updated_at=updated_row["updated_at"],
        )

    def _run_resume_worker(
        self,
        run_id: str,
        req: CreateTaskRequest,
        approved_plan: ResearchPlan | None,
    ) -> None:
        def on_event(event: TraceEvent) -> None:
            self._push_event(run_id, "trace", event.model_dump(mode="json"))

        try:
            local_pdf_dir = self._resolve_upload_dir(req.upload_batch_id)
            effective_settings = self.settings_store.get_effective_settings()
            _, out_dir = resume_research_session(
                run_id=run_id,
                approved_plan=approved_plan,
                settings=effective_settings,
                mock=req.mock,
                top_k_papers=req.top_k_papers,
                max_papers_per_query=req.max_papers_per_query,
                max_iterations=req.max_iterations,
                download_pdfs=req.download_pdfs,
                local_pdf_dir=local_pdf_dir,
                persist_rag=self.persist_rag,
                sources=req.sources,
                llm_model=req.llm_model,
                on_event=on_event,
            )
            self._update_task_row(run_id, status="completed", output_dir=str(out_dir))
            self._push_event(run_id, "status", {"run_id": run_id, "status": "completed"})
        except Exception as exc:
            self._update_task_row(run_id, status="failed", error_message=str(exc))
            self._push_event(
                run_id,
                "status",
                {"run_id": run_id, "status": "failed", "error": str(exc)},
            )
        finally:
            self._close_subscribers(run_id)

    def list_tasks(self) -> list[TaskSummary]:
        """List tasks from SQLite plus any pre-existing CLI runs found in outputs/."""
        with self._lock:
            with self._connect() as conn:
                rows = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM tasks ORDER BY created_at DESC"
                    ).fetchall()
                ]

        summaries: dict[str, TaskSummary] = {}
        for row in rows:
            run_id = row["run_id"]
            state = self._read_state_json(run_id) or {}
            summaries[run_id] = TaskSummary(
                run_id=run_id,
                topic=row["topic"],
                status=row["status"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                paper_count=len(state.get("selected_papers") or []),
                chunk_count=len(state.get("chunks") or []),
                iteration=int(state.get("iteration") or 0),
                error_message=row["error_message"],
            )

        # Include historical CLI runs stored only on disk.
        if self.settings.output_path.exists():
            for child in sorted(self.settings.output_path.iterdir(), reverse=True):
                if not child.is_dir():
                    continue
                run_id = child.name
                if run_id in summaries:
                    continue
                state = self._read_state_json(run_id)
                if not state:
                    continue
                mtime = datetime.fromtimestamp(child.stat().st_mtime, tz=timezone.utc).isoformat()
                summaries[run_id] = TaskSummary(
                    run_id=run_id,
                    topic=state.get("topic", run_id),
                    status="completed",
                    created_at=mtime,
                    updated_at=mtime,
                    paper_count=len(state.get("selected_papers") or []),
                    chunk_count=len(state.get("chunks") or []),
                    iteration=int(state.get("iteration") or 0),
                )

        return sorted(summaries.values(), key=lambda s: s.created_at, reverse=True)

    def get_task(self, run_id: str) -> TaskDetailResponse | None:
        """Return full task details from SQLite and persisted state.json."""
        row = self._get_task_row(run_id)
        state = self._read_state_json(run_id)

        if row is None and state is None:
            return None

        if row is not None:
            topic = row["topic"]
            status = row["status"]
            created_at = row["created_at"]
            updated_at = row["updated_at"]
            output_dir = row["output_dir"]
            error_message = row["error_message"]
        else:
            out_path = self.settings.output_path / run_id
            mtime = datetime.fromtimestamp(out_path.stat().st_mtime, tz=timezone.utc).isoformat()
            topic = (state or {}).get("topic", run_id)
            status = "completed"
            created_at = mtime
            updated_at = mtime
            output_dir = str(out_path)
            error_message = None

        state = state or {}
        draft = state.get("draft_markdown") or self._read_report_md(run_id)

        return TaskDetailResponse.model_validate(
            {
                "run_id": run_id,
                "topic": topic,
                "status": status,
                "created_at": created_at,
                "updated_at": updated_at,
                "output_dir": output_dir,
                "error_message": error_message,
                "plan": state.get("plan"),
                "selected_papers": state.get("selected_papers") or [],
                "evidence_by_subquestion": state.get("evidence_by_subquestion") or {},
                "evaluation_history": state.get("evaluation_history") or [],
                "rewrite_history": state.get("rewrite_history") or [],
                "latest_evaluation": state.get("latest_evaluation"),
                "verification_report": state.get("verification_report"),
                "trace": state.get("trace") or [],
                "errors": state.get("errors") or [],
                "draft_markdown": draft,
            }
        )

    def delete_task(self, run_id: str) -> bool:
        """Delete a task record from SQLite and remove its persisted output/PDF folders."""
        if not run_id or not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
            raise KeyError(f"Task '{run_id}' not found.")

        row = self._get_task_row(run_id)
        output_dir = self.settings.output_path / run_id
        if row is None and not output_dir.exists():
            raise KeyError(f"Task '{run_id}' not found.")
        if row is not None and row["status"] == "running":
            raise ValueError(f"Task '{run_id}' is currently running and cannot be deleted.")

        with self._lock:
            self._live_events.pop(run_id, None)
            with self._connect() as conn:
                conn.execute("DELETE FROM tasks WHERE run_id = ?", (run_id,))
                conn.commit()

        out_root = self.settings.output_path.resolve()
        if output_dir.exists() and output_dir.resolve().is_relative_to(out_root):
            shutil.rmtree(output_dir, ignore_errors=True)

        pdf_dir = self.settings.data_path / "pdfs" / run_id
        data_root = self.settings.data_path.resolve()
        if pdf_dir.exists() and pdf_dir.resolve().is_relative_to(data_root):
            shutil.rmtree(pdf_dir, ignore_errors=True)

        return True

    def get_report(self, run_id: str) -> str | None:
        """Return generated Markdown survey text for a run."""
        md = self._read_report_md(run_id)
        if md:
            return md
        state = self._read_state_json(run_id)
        if state is not None:
            return state.get("draft_markdown", "")
        return None

    def get_bibtex(self, run_id: str) -> str | None:
        """Generate BibTeX bibliography entries for selected papers in a run."""
        state = self._read_state_json(run_id)
        row = self._get_task_row(run_id)
        if state is None and row is None:
            return None

        papers = (state or {}).get("selected_papers") or []
        entries: list[str] = []
        for idx, paper in enumerate(papers, start=1):
            pid = paper.get("paper_id") or f"P{idx:02d}"
            title = str(paper.get("title") or "").replace("{", "\\{").replace("}", "\\}")
            authors = " and ".join(paper.get("authors") or ["Unknown"])
            year = paper.get("year") or "n.d."
            venue = str(paper.get("venue") or paper.get("source") or "arXiv")
            url = str(paper.get("url") or "")
            entries.append(
                "\n".join(
                    [
                        f"@article{{{pid},",
                        f"  title = {{{title}}},",
                        f"  author = {{{authors}}},",
                        f"  year = {{{year}}},",
                        f"  journal = {{{venue}}},",
                        f"  url = {{{url}}}",
                        "}",
                    ]
                )
            )
        return "\n\n".join(entries) + ("\n" if entries else "")

    def get_chunks(self, run_id: str, paper_id: str | None = None) -> list[TextChunk] | None:
        """Return indexed source chunks for a run, optionally filtered by paper_id."""
        state = self._read_state_json(run_id)
        row = self._get_task_row(run_id)
        if state is None and row is None:
            return None
        raw_chunks = (state or {}).get("chunks") or []
        chunks = [TextChunk.model_validate(c) for c in raw_chunks]
        if paper_id:
            chunks = [c for c in chunks if c.paper_id == paper_id]
        return chunks

    async def subscribe(self, run_id: str) -> AsyncIterator[str]:
        """Yield SSE-formatted trace and status events for a task."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        with self._lock:
            past_events = list(self._live_events.get(run_id, []))
            row = None
            with self._connect() as conn:
                db_row = conn.execute(
                    "SELECT status, error_message FROM tasks WHERE run_id = ?", (run_id,)
                ).fetchone()
                if db_row is not None:
                    row = dict(db_row)

            # Fallback to persisted trace if server restarted after run completion.
            if not past_events:
                state = self._read_state_json(run_id)
                if state:
                    for ev in state.get("trace") or []:
                        past_events.append({"event": "trace", "data": ev})

            status = row["status"] if row is not None else ("completed" if past_events else "unknown")
            is_active = status == "running"
            if is_active:
                self._subscribers.setdefault(run_id, []).append((loop, queue))

        try:
            for item in past_events:
                yield f"event: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"

            if not is_active:
                payload: dict[str, Any] = {"run_id": run_id, "status": status}
                if row and row.get("error_message"):
                    payload["error"] = row["error_message"]
                yield f"event: status\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                return

            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"event: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
        finally:
            if is_active:
                with self._lock:
                    subs = self._subscribers.get(run_id, [])
                    self._subscribers[run_id] = [pair for pair in subs if pair[1] is not queue]
