import type {
  CreateTaskPayload,
  FetchModelsResponse,
  HealthResponse,
  ResearchPlan,
  RuntimeSettingsResponse,
  TaskDetailResponse,
  TaskSummary,
  TestConnectionResponse,
  TextChunk,
  TraceEvent,
  UpdateSettingsRequest,
  UploadResponse,
} from "./types";

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init);
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const errBody = (await resp.json()) as { detail?: string };
      if (errBody.detail) detail = errBody.detail;
    } catch {
      // ignore non-json error body
    }
    throw new Error(detail);
  }
  return (await resp.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/api/health");
}

export function listTasks(): Promise<TaskSummary[]> {
  return requestJson<TaskSummary[]>("/api/tasks");
}

export function getTask(runId: string): Promise<TaskDetailResponse> {
  return requestJson<TaskDetailResponse>(`/api/tasks/${encodeURIComponent(runId)}`);
}

export function createTask(payload: CreateTaskPayload): Promise<TaskSummary> {
  return requestJson<TaskSummary>("/api/tasks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function approveTask(runId: string, plan: ResearchPlan): Promise<TaskSummary> {
  return requestJson<TaskSummary>(`/api/tasks/${encodeURIComponent(runId)}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan }),
  });
}

export async function uploadPdfs(files: File[]): Promise<UploadResponse> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  return requestJson<UploadResponse>("/api/uploads", {
    method: "POST",
    body: form,
  });
}

export async function getChunks(runId: string, paperId?: string): Promise<TextChunk[]> {
  const query = paperId ? `?paper_id=${encodeURIComponent(paperId)}` : "";
  const data = await requestJson<{ chunks: TextChunk[] }>(
    `/api/tasks/${encodeURIComponent(runId)}/chunks${query}`
  );
  return data.chunks;
}

export function getReportDownloadUrl(runId: string): string {
  return `/api/tasks/${encodeURIComponent(runId)}/report?download=true`;
}

export function getBibtexDownloadUrl(runId: string): string {
  return `/api/tasks/${encodeURIComponent(runId)}/bibtex?download=true`;
}

export function deleteTask(runId: string): Promise<{ deleted: boolean; run_id: string }> {
  return requestJson<{ deleted: boolean; run_id: string }>(
    `/api/tasks/${encodeURIComponent(runId)}`,
    { method: "DELETE" }
  );
}

export function getSettings(): Promise<RuntimeSettingsResponse> {
  return requestJson<RuntimeSettingsResponse>("/api/settings");
}

export function updateSettings(
  payload: UpdateSettingsRequest
): Promise<RuntimeSettingsResponse> {
  return requestJson<RuntimeSettingsResponse>("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function fetchProviderModels(payload: {
  target: "llm" | "embedding";
  base_url: string;
  api_key: string;
}): Promise<FetchModelsResponse> {
  return requestJson<FetchModelsResponse>("/api/settings/models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function testProviderConnection(payload: {
  target: "llm" | "embedding";
  base_url: string;
  api_key: string;
  model: string;
}): Promise<TestConnectionResponse> {
  return requestJson<TestConnectionResponse>("/api/settings/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function streamTaskEvents(
  runId: string,
  callbacks: {
    onTrace: (event: TraceEvent) => void;
    onStatus: (status: string, error?: string) => void;
  },
  signal?: AbortSignal
): Promise<void> {
  const resp = await fetch(`/api/tasks/${encodeURIComponent(runId)}/events`, {
    signal,
    headers: { Accept: "text/event-stream" },
  });
  if (!resp.ok || !resp.body) {
    throw new Error(`无法连接实时事件流 (${resp.status})`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const processBlock = (block: string) => {
    const lines = block.split(/\r?\n/);
    let eventName = "message";
    const dataLines: string[] = [];
    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }
    if (dataLines.length === 0) return;
    try {
      const parsed = JSON.parse(dataLines.join("\n"));
      if (eventName === "trace") {
        callbacks.onTrace(parsed as TraceEvent);
      } else if (eventName === "status") {
        callbacks.onStatus(String(parsed.status || "unknown"), parsed.error);
      }
    } catch {
      // ignore malformed SSE chunk
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split(/\r?\n\r?\n/);
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      if (part.trim()) processBlock(part);
    }
  }
  if (buffer.trim()) {
    processBlock(buffer);
  }
}
