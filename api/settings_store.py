"""Runtime settings persistence, secret masking, model discovery, and connectivity checks."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Literal

import httpx

from api.schemas import (
    RuntimeSettingsResponse,
    TestConnectionResponse,
    UpdateSettingsRequest,
)
from core.config import Settings


def mask_secret(secret: str) -> str:
    """Return a safely masked representation of an API key."""
    cleaned = secret.strip()
    if not cleaned:
        return ""
    if len(cleaned) <= 8:
        return "***"
    return f"{cleaned[:6]}...{cleaned[-4:]}"


def _dedupe_str_list(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in items:
        item = raw.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _candidate_endpoints(base_url: str, suffix: str) -> list[str]:
    clean = base_url.strip().rstrip("/")
    if not clean:
        return []
    urls = [f"{clean}{suffix}"]
    if not clean.endswith("/v1"):
        urls.append(f"{clean}/v1{suffix}")
    return urls


def fetch_provider_models(
    base_url: str,
    api_key: str,
    timeout: float = 15.0,
) -> list[str]:
    """Fetch available model IDs from an OpenAI-compatible /models endpoint."""
    endpoints = _candidate_endpoints(base_url, "/models")
    if not endpoints:
        raise ValueError("请先填写 Base URL。")

    headers: dict[str, str] = {}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    last_error = "无法连接到模型列表接口"
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for idx, url in enumerate(endpoints):
            try:
                resp = client.get(url, headers=headers)
            except httpx.HTTPError as exc:
                last_error = f"网络请求失败：{exc}"
                continue

            if resp.status_code == 404 and idx + 1 < len(endpoints):
                continue
            if resp.status_code >= 400:
                detail = resp.text.strip()[:240]
                raise RuntimeError(f"获取模型失败 (HTTP {resp.status_code})：{detail}")

            payload = resp.json()
            models: list[str] = []
            if isinstance(payload, dict):
                if isinstance(payload.get("data"), list):
                    for item in payload["data"]:
                        if isinstance(item, dict) and item.get("id"):
                            models.append(str(item["id"]).strip())
                elif isinstance(payload.get("models"), list):
                    for item in payload["models"]:
                        if isinstance(item, dict):
                            mid = item.get("id") or item.get("name") or item.get("model")
                            if mid:
                                models.append(str(mid).strip())
            elif isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict) and item.get("id"):
                        models.append(str(item["id"]).strip())

            unique = sorted(set(m for m in models if m))
            if not unique:
                raise RuntimeError("接口返回成功，但未在响应中找到模型 ID 列表。")
            return unique

    raise RuntimeError(last_error)


def test_provider_connection(
    target: Literal["llm", "embedding"],
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 20.0,
) -> TestConnectionResponse:
    """Test live connectivity and response latency for an LLM or Embedding endpoint."""
    clean_model = model.strip()
    if not base_url.strip():
        return TestConnectionResponse(ok=False, message="请先填写 Base URL。")
    if not clean_model:
        return TestConnectionResponse(ok=False, message="请先选择或填写模型名称。")

    suffix = "/chat/completions" if target == "llm" else "/embeddings"
    endpoints = _candidate_endpoints(base_url, suffix)
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    body: dict[str, Any] = (
        {
            "model": clean_model,
            "messages": [{"role": "user", "content": "Reply with OK"}],
            "max_tokens": 5,
            "temperature": 0,
        }
        if target == "llm"
        else {
            "model": clean_model,
            "input": ["ping"],
        }
    )

    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for idx, url in enumerate(endpoints):
            try:
                resp = client.post(url, headers=headers, json=body)
            except httpx.HTTPError as exc:
                latency = int((time.perf_counter() - t0) * 1000)
                return TestConnectionResponse(
                    ok=False,
                    message=f"网络请求失败：{exc}",
                    latency_ms=latency,
                    model_used=clean_model,
                )

            if resp.status_code == 404 and idx + 1 < len(endpoints):
                continue

            latency = int((time.perf_counter() - t0) * 1000)
            if resp.status_code >= 400:
                return TestConnectionResponse(
                    ok=False,
                    message=f"接口返回错误 (HTTP {resp.status_code})",
                    latency_ms=latency,
                    model_used=clean_model,
                    detail=resp.text.strip()[:240],
                )

            try:
                payload = resp.json()
            except Exception:
                return TestConnectionResponse(
                    ok=False,
                    message="接口返回内容不是合法 JSON",
                    latency_ms=latency,
                    model_used=clean_model,
                )

            if target == "llm":
                choices = payload.get("choices") if isinstance(payload, dict) else None
                if not choices:
                    return TestConnectionResponse(
                        ok=False,
                        message="响应缺少 choices 字段",
                        latency_ms=latency,
                        model_used=clean_model,
                    )
                return TestConnectionResponse(
                    ok=True,
                    message=f"大模型连接测试成功（{clean_model}）",
                    latency_ms=latency,
                    model_used=clean_model,
                )

            data = payload.get("data") if isinstance(payload, dict) else None
            if not data or not isinstance(data, list) or not data[0].get("embedding"):
                return TestConnectionResponse(
                    ok=False,
                    message="响应缺少 embedding 向量数据",
                    latency_ms=latency,
                    model_used=clean_model,
                )
            dim = len(data[0]["embedding"])
            return TestConnectionResponse(
                ok=True,
                message=f"Embedding 连接测试成功（维度 {dim}）",
                latency_ms=latency,
                model_used=clean_model,
            )

    latency = int((time.perf_counter() - t0) * 1000)
    return TestConnectionResponse(
        ok=False,
        message="未找到可用的 API 端点 (404)",
        latency_ms=latency,
        model_used=clean_model,
    )


class RuntimeSettingsStore:
    """Thread-safe JSON persistence for runtime LLM, Embedding, and academic source settings."""

    def __init__(self, base_settings: Settings) -> None:
        self._base = base_settings
        self._base.data_path.mkdir(parents=True, exist_ok=True)
        self._file: Path = self._base.data_path / "runtime_settings.json"
        self._lock = threading.Lock()
        self._overrides: dict[str, Any] = self._load_file()

    def _load_file(self) -> dict[str, Any]:
        if not self._file.exists():
            return {}
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_file(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(
            json.dumps(self._overrides, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_effective_settings(self) -> Settings:
        """Merge persisted overrides on top of .env base settings."""
        with self._lock:
            ov = dict(self._overrides)

        update_fields: dict[str, Any] = {}
        for key in (
            "llm_base_url",
            "llm_api_key",
            "llm_model",
            "embedding_provider",
            "embedding_base_url",
            "embedding_api_key",
            "embedding_model",
            "openalex_email",
            "semantic_scholar_api_key",
        ):
            if key in ov and ov[key] is not None:
                update_fields[key] = ov[key]

        if "academic_sources" in ov and isinstance(ov["academic_sources"], list):
            cleaned_sources = _dedupe_str_list([str(s).lower() for s in ov["academic_sources"]])
            if cleaned_sources:
                update_fields["academic_sources"] = ",".join(cleaned_sources)

        return self._base.model_copy(update=update_fields) if update_fields else self._base

    def get_public_settings(self) -> RuntimeSettingsResponse:
        """Return public non-secret settings with masked keys and enabled model lists."""
        cfg = self.get_effective_settings()
        with self._lock:
            avail_llm = list(self._overrides.get("available_llm_models") or [])
            enabled_llm = list(self._overrides.get("enabled_llm_models") or [])
            avail_emb = list(self._overrides.get("available_embedding_models") or [])

        enabled_llm = _dedupe_str_list(enabled_llm)
        if cfg.llm_model.strip() and cfg.llm_model.strip() not in enabled_llm:
            enabled_llm = [cfg.llm_model.strip(), *enabled_llm]

        avail_llm = _dedupe_str_list([*avail_llm, *enabled_llm])
        avail_emb = _dedupe_str_list([*avail_emb, cfg.embedding_model])

        return RuntimeSettingsResponse(
            llm_base_url=cfg.llm_base_url,
            llm_model=cfg.llm_model,
            llm_api_key_configured=bool(cfg.llm_api_key.strip()),
            llm_api_key_masked=mask_secret(cfg.llm_api_key),
            available_llm_models=avail_llm,
            enabled_llm_models=enabled_llm,
            embedding_provider=cfg.embedding_provider,
            embedding_base_url=cfg.effective_embedding_base_url,
            embedding_model=cfg.embedding_model,
            embedding_api_key_configured=bool(cfg.effective_embedding_api_key),
            embedding_api_key_masked=mask_secret(cfg.effective_embedding_api_key),
            available_embedding_models=avail_emb,
            academic_sources=cfg.academic_sources_list,
            openalex_email=cfg.openalex_email,
            semantic_scholar_key_configured=bool(cfg.semantic_scholar_api_key.strip()),
        )

    def update(self, payload: UpdateSettingsRequest) -> RuntimeSettingsResponse:
        """Persist updated settings while preserving existing secrets if omitted/empty."""
        with self._lock:
            if payload.llm_base_url is not None and payload.llm_base_url.strip():
                self._overrides["llm_base_url"] = payload.llm_base_url.strip().rstrip("/")
            if payload.llm_api_key is not None and payload.llm_api_key.strip():
                self._overrides["llm_api_key"] = payload.llm_api_key.strip()
            if payload.llm_model is not None and payload.llm_model.strip():
                self._overrides["llm_model"] = payload.llm_model.strip()

            if payload.available_llm_models is not None:
                self._overrides["available_llm_models"] = _dedupe_str_list(
                    payload.available_llm_models
                )
            if payload.enabled_llm_models is not None:
                cleaned_enabled = _dedupe_str_list(payload.enabled_llm_models)
                active_model = (
                    self._overrides.get("llm_model") or self._base.llm_model
                ).strip()
                if active_model and active_model not in cleaned_enabled:
                    cleaned_enabled = [active_model, *cleaned_enabled]
                self._overrides["enabled_llm_models"] = cleaned_enabled

            if payload.embedding_provider is not None:
                self._overrides["embedding_provider"] = payload.embedding_provider
            if payload.embedding_base_url is not None:
                self._overrides["embedding_base_url"] = payload.embedding_base_url.strip().rstrip("/")
            if payload.embedding_api_key is not None and payload.embedding_api_key.strip():
                self._overrides["embedding_api_key"] = payload.embedding_api_key.strip()
            if payload.embedding_model is not None and payload.embedding_model.strip():
                self._overrides["embedding_model"] = payload.embedding_model.strip()
            if payload.available_embedding_models is not None:
                self._overrides["available_embedding_models"] = _dedupe_str_list(
                    payload.available_embedding_models
                )

            if payload.academic_sources is not None:
                cleaned_sources = _dedupe_str_list([s.lower() for s in payload.academic_sources])
                if cleaned_sources:
                    self._overrides["academic_sources"] = cleaned_sources
            if payload.openalex_email is not None:
                self._overrides["openalex_email"] = payload.openalex_email.strip()
            if (
                payload.semantic_scholar_api_key is not None
                and payload.semantic_scholar_api_key.strip()
            ):
                self._overrides["semantic_scholar_api_key"] = payload.semantic_scholar_api_key.strip()

            self._save_file()

        return self.get_public_settings()

    def resolve_credentials(
        self,
        target: Literal["llm", "embedding"],
        base_url: str = "",
        api_key: str = "",
    ) -> tuple[str, str]:
        """Resolve effective base_url and api_key using draft values or saved fallback."""
        cfg = self.get_effective_settings()
        if target == "llm":
            eff_url = base_url.strip() or cfg.llm_base_url
            eff_key = api_key.strip() or cfg.llm_api_key
            return eff_url, eff_key

        eff_url = base_url.strip() or cfg.effective_embedding_base_url
        eff_key = api_key.strip() or cfg.effective_embedding_api_key
        return eff_url, eff_key
