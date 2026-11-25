import { useEffect, useState } from "react";
import {
  Check,
  CheckCircle2,
  Eye,
  EyeOff,
  Loader2,
  Plus,
  RefreshCw,
  Search,
  Wifi,
  X,
  XCircle,
} from "lucide-react";
import {
  fetchProviderModels,
  testProviderConnection,
  updateSettings,
} from "../api";
import type {
  RuntimeSettingsResponse,
  TestConnectionResponse,
  UpdateSettingsRequest,
} from "../types";

interface SettingsModalProps {
  open: boolean;
  settings: RuntimeSettingsResponse | null;
  onClose: () => void;
  onSaved: (updated: RuntimeSettingsResponse) => void;
}

type SettingsTab = "llm" | "embedding" | "sources";

const SOURCE_OPTIONS = [
  { id: "arxiv", label: "arXiv", desc: "免费开源预印本，全文 PDF 下载最稳定" },
  { id: "openalex", label: "OpenAlex", desc: "含引用量与期刊会议信息，免密钥" },
  { id: "semantic_scholar", label: "Semantic Scholar", desc: "高覆盖学术图谱（建议配 API Key）" },
];

export function SettingsModal({ open, settings, onClose, onSaved }: SettingsModalProps) {
  const [tab, setTab] = useState<SettingsTab>("llm");

  // LLM state
  const [llmBaseUrl, setLlmBaseUrl] = useState("");
  const [llmApiKey, setLlmApiKey] = useState("");
  const [showLlmKey, setShowLlmKey] = useState(false);
  const [llmModel, setLlmModel] = useState("");
  const [availableLlmModels, setAvailableLlmModels] = useState<string[]>([]);
  const [enabledLlmModels, setEnabledLlmModels] = useState<string[]>([]);
  const [manualLlmModel, setManualLlmModel] = useState("");
  const [modelSearch, setModelSearch] = useState("");
  const [llmFetching, setLlmFetching] = useState(false);
  const [llmTesting, setLlmTesting] = useState(false);
  const [llmTestResult, setLlmTestResult] = useState<TestConnectionResponse | null>(null);
  const [llmFetchMsg, setLlmFetchMsg] = useState<string | null>(null);

  // Embedding state
  const [embProvider, setEmbProvider] = useState<"openai" | "local_hash">("local_hash");
  const [embBaseUrl, setEmbBaseUrl] = useState("");
  const [embApiKey, setEmbApiKey] = useState("");
  const [showEmbKey, setShowEmbKey] = useState(false);
  const [embModel, setEmbModel] = useState("");
  const [availableEmbModels, setAvailableEmbModels] = useState<string[]>([]);
  const [embFetching, setEmbFetching] = useState(false);
  const [embTesting, setEmbTesting] = useState(false);
  const [embTestResult, setEmbTestResult] = useState<TestConnectionResponse | null>(null);
  const [embFetchMsg, setEmbFetchMsg] = useState<string | null>(null);

  // Academic sources state
  const [academicSources, setAcademicSources] = useState<string[]>(["arxiv", "openalex"]);
  const [openalexEmail, setOpenalexEmail] = useState("");
  const [s2ApiKey, setS2ApiKey] = useState("");

  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !settings) return;
    setLlmBaseUrl(settings.llm_base_url);
    setLlmApiKey("");
    setLlmModel(settings.llm_model);
    setAvailableLlmModels(settings.available_llm_models);
    setEnabledLlmModels(settings.enabled_llm_models);
    setLlmTestResult(null);
    setLlmFetchMsg(null);
    setModelSearch("");

    setEmbProvider(settings.embedding_provider);
    setEmbBaseUrl(settings.embedding_base_url);
    setEmbApiKey("");
    setEmbModel(settings.embedding_model);
    setAvailableEmbModels(settings.available_embedding_models);
    setEmbTestResult(null);
    setEmbFetchMsg(null);

    setAcademicSources(settings.academic_sources);
    setOpenalexEmail(settings.openalex_email);
    setS2ApiKey("");
    setErrorMsg(null);
  }, [open, settings]);

  if (!open) return null;

  const handleFetchLlmModels = async () => {
    setErrorMsg(null);
    setLlmFetchMsg(null);
    setLlmFetching(true);
    try {
      const res = await fetchProviderModels({
        target: "llm",
        base_url: llmBaseUrl,
        api_key: llmApiKey,
      });
      const merged = Array.from(new Set([...res.models, ...enabledLlmModels, llmModel].filter(Boolean)));
      setAvailableLlmModels(merged);
      setLlmFetchMsg(`已获取到 ${res.count} 个模型，请勾选你要在调研中使用的模型。`);
      if (res.models.length > 0 && !llmModel) {
        setLlmModel(res.models[0]);
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "获取模型列表失败");
    } finally {
      setLlmFetching(false);
    }
  };

  const handleTestLlm = async () => {
    setErrorMsg(null);
    setLlmTestResult(null);
    setLlmTesting(true);
    try {
      const res = await testProviderConnection({
        target: "llm",
        base_url: llmBaseUrl,
        api_key: llmApiKey,
        model: llmModel,
      });
      setLlmTestResult(res);
    } catch (err) {
      setLlmTestResult({
        ok: false,
        message: err instanceof Error ? err.message : "测试连接失败",
        latency_ms: 0,
        model_used: llmModel,
        detail: "",
      });
    } finally {
      setLlmTesting(false);
    }
  };

  const toggleEnabledLlm = (modelId: string) => {
    setEnabledLlmModels((prev) => {
      if (prev.includes(modelId)) {
        if (prev.length <= 1) return prev;
        const next = prev.filter((m) => m !== modelId);
        if (llmModel === modelId) {
          setLlmModel(next[0]);
        }
        return next;
      }
      return [...prev, modelId];
    });
  };

  const addManualLlmModel = () => {
    const id = manualLlmModel.trim();
    if (!id) return;
    if (!availableLlmModels.includes(id)) {
      setAvailableLlmModels((prev) => [id, ...prev]);
    }
    if (!enabledLlmModels.includes(id)) {
      setEnabledLlmModels((prev) => [...prev, id]);
    }
    if (!llmModel) {
      setLlmModel(id);
    }
    setManualLlmModel("");
  };

  const handleFetchEmbModels = async () => {
    setErrorMsg(null);
    setEmbFetchMsg(null);
    setEmbFetching(true);
    try {
      const res = await fetchProviderModels({
        target: "embedding",
        base_url: embBaseUrl,
        api_key: embApiKey,
      });
      const merged = Array.from(new Set([...res.models, embModel].filter(Boolean)));
      setAvailableEmbModels(merged);
      setEmbFetchMsg(`已获取到 ${res.count} 个模型，可从中选择 Embedding 模型。`);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "获取 Embedding 模型列表失败");
    } finally {
      setEmbFetching(false);
    }
  };

  const handleTestEmb = async () => {
    setErrorMsg(null);
    setEmbTestResult(null);
    setEmbTesting(true);
    try {
      const res = await testProviderConnection({
        target: "embedding",
        base_url: embBaseUrl,
        api_key: embApiKey,
        model: embModel,
      });
      setEmbTestResult(res);
    } catch (err) {
      setEmbTestResult({
        ok: false,
        message: err instanceof Error ? err.message : "Embedding 连接测试失败",
        latency_ms: 0,
        model_used: embModel,
        detail: "",
      });
    } finally {
      setEmbTesting(false);
    }
  };

  const toggleSource = (id: string) => {
    setAcademicSources((prev) => {
      if (prev.includes(id)) {
        return prev.length > 1 ? prev.filter((s) => s !== id) : prev;
      }
      return [...prev, id];
    });
  };

  const handleSave = async () => {
    setErrorMsg(null);
    setSaving(true);
    try {
      const payload: UpdateSettingsRequest = {
        llm_base_url: llmBaseUrl.trim(),
        llm_model: llmModel.trim(),
        available_llm_models: availableLlmModels,
        enabled_llm_models: enabledLlmModels,
        embedding_provider: embProvider,
        embedding_base_url: embBaseUrl.trim(),
        embedding_model: embModel.trim(),
        available_embedding_models: availableEmbModels,
        academic_sources: academicSources,
        openalex_email: openalexEmail.trim(),
      };
      if (llmApiKey.trim()) payload.llm_api_key = llmApiKey.trim();
      if (embApiKey.trim()) payload.embedding_api_key = embApiKey.trim();
      if (s2ApiKey.trim()) payload.semantic_scholar_api_key = s2ApiKey.trim();

      const updated = await updateSettings(payload);
      onSaved(updated);
      onClose();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "保存设置失败");
    } finally {
      setSaving(false);
    }
  };

  const filteredLlmModels = availableLlmModels.filter((m) =>
    !modelSearch.trim() ? true : m.toLowerCase().includes(modelSearch.trim().toLowerCase())
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-[1px] p-4">
      <div className="w-full max-w-2xl max-h-[88vh] bg-[#FAF9F6] border border-[#E7E5DF] rounded-2xl shadow-xl flex flex-col overflow-hidden">
        {/* Modal Header */}
        <div className="px-6 py-4 bg-white border-b border-[#E7E5DF] flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-[#18181B]">模型与接口设置</h2>
            <p className="text-xs text-[#71717A] mt-0.5">
              支持任意 OpenAI 兼容接口（DeepSeek、硅基流动、Qwen、中转站等），修改后立即生效
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-[#71717A] hover:text-[#18181B] hover:bg-[#F3F2EE] cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Clean Tab Bar */}
        <div className="px-6 bg-white border-b border-[#E7E5DF] flex gap-6">
          {(
            [
              ["llm", "大模型 API (LLM)"],
              ["embedding", "向量模型 API (Embedding)"],
              ["sources", "学术数据源"],
            ] as Array<[SettingsTab, string]>
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              className={`py-2.5 text-xs font-medium border-b-2 transition-colors cursor-pointer ${
                tab === id
                  ? "border-[#3B5BDB] text-[#18181B]"
                  : "border-transparent text-[#71717A] hover:text-[#18181B]"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-5">
          {errorMsg && (
            <div className="px-4 py-2.5 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-700">
              {errorMsg}
            </div>
          )}

          {/* TAB 1: LLM */}
          {tab === "llm" && (
            <div className="space-y-4">
              <div className="rounded-xl bg-white border border-[#E7E5DF] p-4 space-y-3">
                <label className="block text-xs font-medium text-[#3F3F46]">
                  Base URL
                  <input
                    type="text"
                    value={llmBaseUrl}
                    onChange={(e) => setLlmBaseUrl(e.target.value)}
                    placeholder="https://api.deepseek.com 或 https://api.openai.com/v1"
                    className="mt-1.5 w-full rounded-lg border border-[#E2DFD7] px-3 py-2 text-sm text-[#18181B] focus:outline-none focus:border-[#3B5BDB]"
                  />
                </label>

                <label className="block text-xs font-medium text-[#3F3F46]">
                  API Key
                  <div className="mt-1.5 relative">
                    <input
                      type={showLlmKey ? "text" : "password"}
                      value={llmApiKey}
                      onChange={(e) => setLlmApiKey(e.target.value)}
                      placeholder={
                        settings?.llm_api_key_configured
                          ? `已配置 (${settings.llm_api_key_masked})，留空保持当前密钥`
                          : "输入 sk-..."
                      }
                      className="w-full rounded-lg border border-[#E2DFD7] pl-3 pr-10 py-2 text-sm text-[#18181B] focus:outline-none focus:border-[#3B5BDB]"
                    />
                    <button
                      type="button"
                      onClick={() => setShowLlmKey((prev) => !prev)}
                      className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[#71717A] hover:text-[#18181B] cursor-pointer"
                    >
                      {showLlmKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </label>

                <div className="flex flex-wrap items-center gap-2 pt-1">
                  <button
                    type="button"
                    disabled={llmFetching}
                    onClick={() => void handleFetchLlmModels()}
                    className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-[#EEF2FF] text-[#3B5BDB] hover:bg-[#DBE4FF] text-xs font-medium transition-colors cursor-pointer disabled:opacity-50"
                  >
                    {llmFetching ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <RefreshCw className="w-3.5 h-3.5" />
                    )}
                    获取模型
                  </button>

                  <button
                    type="button"
                    disabled={llmTesting}
                    onClick={() => void handleTestLlm()}
                    className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-white border border-[#E2DFD7] hover:border-[#3B5BDB]/50 text-[#18181B] text-xs font-medium transition-colors cursor-pointer disabled:opacity-50"
                  >
                    {llmTesting ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Wifi className="w-3.5 h-3.5" />
                    )}
                    测试连接
                  </button>
                </div>

                {llmFetchMsg && (
                  <div className="text-xs text-[#3B5BDB] bg-[#EEF2FF]/60 rounded-lg px-3 py-2">
                    {llmFetchMsg}
                  </div>
                )}

                {llmTestResult && (
                  <div
                    className={`flex items-start gap-2 rounded-lg px-3 py-2 text-xs ${
                      llmTestResult.ok
                        ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
                        : "bg-rose-50 text-rose-700 border border-rose-200"
                    }`}
                  >
                    {llmTestResult.ok ? (
                      <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
                    ) : (
                      <XCircle className="w-4 h-4 shrink-0 mt-0.5" />
                    )}
                    <div>
                      <div className="font-medium">
                        {llmTestResult.message}
                        {llmTestResult.latency_ms > 0 ? ` · ${llmTestResult.latency_ms} ms` : ""}
                      </div>
                      {llmTestResult.detail && (
                        <div className="mt-0.5 opacity-80 break-all">{llmTestResult.detail}</div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Enabled Models Selection */}
              <div className="rounded-xl bg-white border border-[#E7E5DF] p-4 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <div className="text-xs font-semibold text-[#18181B]">
                      选择要在调研界面中使用的模型（已选 {enabledLlmModels.length} 个）
                    </div>
                    <div className="text-[11px] text-[#71717A] mt-0.5">
                      勾选后的模型会出现在输入框右下角的快速切换菜单中
                    </div>
                  </div>
                  <div className="text-xs text-[#52525B]">
                    默认模型：<code className="font-mono text-[#3B5BDB]">{llmModel || "未设置"}</code>
                  </div>
                </div>

                <div className="flex flex-col sm:flex-row gap-2">
                  <div className="relative flex-1">
                    <Search className="w-3.5 h-3.5 text-[#71717A] absolute left-3 top-1/2 -translate-y-1/2" />
                    <input
                      type="text"
                      value={modelSearch}
                      onChange={(e) => setModelSearch(e.target.value)}
                      placeholder="搜索筛选已获取的模型..."
                      className="w-full pl-8 pr-3 py-1.5 rounded-lg border border-[#E2DFD7] text-xs focus:outline-none focus:border-[#3B5BDB]"
                    />
                  </div>
                  <div className="flex gap-1.5">
                    <input
                      type="text"
                      value={manualLlmModel}
                      onChange={(e) => setManualLlmModel(e.target.value)}
                      placeholder="手动输入模型 ID"
                      className="w-40 px-2.5 py-1.5 rounded-lg border border-[#E2DFD7] text-xs font-mono focus:outline-none focus:border-[#3B5BDB]"
                    />
                    <button
                      type="button"
                      onClick={addManualLlmModel}
                      className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-[#F3F2EE] hover:bg-[#E7E5DF] text-xs font-medium text-[#18181B] cursor-pointer"
                    >
                      <Plus className="w-3.5 h-3.5" />
                      添加
                    </button>
                  </div>
                </div>

                <div className="max-h-52 overflow-y-auto divide-y divide-[#F3F2EE] border border-[#ECEAE4] rounded-lg">
                  {filteredLlmModels.length === 0 ? (
                    <div className="px-3 py-4 text-xs text-[#71717A] text-center">
                      暂无匹配模型，请点击上方“获取模型”或手动添加模型 ID。
                    </div>
                  ) : (
                    filteredLlmModels.map((mid) => {
                      const isEnabled = enabledLlmModels.includes(mid);
                      const isDefault = llmModel === mid;
                      return (
                        <div
                          key={mid}
                          className="px-3 py-2 flex items-center justify-between gap-3 text-xs hover:bg-[#FAF9F6]"
                        >
                          <label className="flex items-center gap-2.5 min-w-0 flex-1 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={isEnabled}
                              onChange={() => toggleEnabledLlm(mid)}
                              className="accent-[#3B5BDB]"
                            />
                            <span className="font-mono text-[#18181B] truncate">{mid}</span>
                          </label>
                          <button
                            type="button"
                            onClick={() => {
                              setLlmModel(mid);
                              if (!enabledLlmModels.includes(mid)) {
                                setEnabledLlmModels((prev) => [...prev, mid]);
                              }
                            }}
                            className={`px-2 py-0.5 rounded text-[11px] shrink-0 cursor-pointer ${
                              isDefault
                                ? "bg-[#3B5BDB] text-white font-medium"
                                : "bg-[#F3F2EE] text-[#52525B] hover:text-[#18181B]"
                            }`}
                          >
                            {isDefault ? (
                              <span className="inline-flex items-center gap-1">
                                <Check className="w-3 h-3" />
                                当前默认
                              </span>
                            ) : (
                              "设为默认"
                            )}
                          </button>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: Embedding */}
          {tab === "embedding" && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                {(
                  [
                    ["openai", "OpenAI 兼容向量接口", "语义检索精度高（推荐如 BAAI/bge-m3）"],
                    ["local_hash", "本地哈希向量 (local_hash)", "零网络零费用，离线快速测试"],
                  ] as const
                ).map(([mode, title, desc]) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setEmbProvider(mode)}
                    className={`text-left p-3.5 rounded-xl border transition-colors cursor-pointer ${
                      embProvider === mode
                        ? "bg-[#EEF2FF]/70 border-[#3B5BDB] text-[#18181B]"
                        : "bg-white border-[#E7E5DF] text-[#52525B]"
                    }`}
                  >
                    <div className="text-xs font-semibold">{title}</div>
                    <div className="text-[11px] opacity-80 mt-1">{desc}</div>
                  </button>
                ))}
              </div>

              {embProvider === "openai" && (
                <div className="rounded-xl bg-white border border-[#E7E5DF] p-4 space-y-3">
                  <label className="block text-xs font-medium text-[#3F3F46]">
                    Embedding Base URL
                    <input
                      type="text"
                      value={embBaseUrl}
                      onChange={(e) => setEmbBaseUrl(e.target.value)}
                      placeholder="https://api.siliconflow.cn/v1（留空则复用大模型 Base URL）"
                      className="mt-1.5 w-full rounded-lg border border-[#E2DFD7] px-3 py-2 text-sm text-[#18181B] focus:outline-none focus:border-[#3B5BDB]"
                    />
                  </label>

                  <label className="block text-xs font-medium text-[#3F3F46]">
                    Embedding API Key
                    <div className="mt-1.5 relative">
                      <input
                        type={showEmbKey ? "text" : "password"}
                        value={embApiKey}
                        onChange={(e) => setEmbApiKey(e.target.value)}
                        placeholder={
                          settings?.embedding_api_key_configured
                            ? `已配置 (${settings.embedding_api_key_masked})，留空保持不变`
                            : "留空则复用大模型 API Key"
                        }
                        className="w-full rounded-lg border border-[#E2DFD7] pl-3 pr-10 py-2 text-sm text-[#18181B] focus:outline-none focus:border-[#3B5BDB]"
                      />
                      <button
                        type="button"
                        onClick={() => setShowEmbKey((prev) => !prev)}
                        className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[#71717A] hover:text-[#18181B] cursor-pointer"
                      >
                        {showEmbKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                  </label>

                  <label className="block text-xs font-medium text-[#3F3F46]">
                    Embedding 模型名称
                    <input
                      type="text"
                      value={embModel}
                      onChange={(e) => setEmbModel(e.target.value)}
                      placeholder="BAAI/bge-m3 或 text-embedding-3-small"
                      className="mt-1.5 w-full rounded-lg border border-[#E2DFD7] px-3 py-2 text-sm font-mono text-[#18181B] focus:outline-none focus:border-[#3B5BDB]"
                    />
                  </label>

                  <div className="flex flex-wrap items-center gap-2 pt-1">
                    <button
                      type="button"
                      disabled={embFetching}
                      onClick={() => void handleFetchEmbModels()}
                      className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-[#EEF2FF] text-[#3B5BDB] hover:bg-[#DBE4FF] text-xs font-medium cursor-pointer disabled:opacity-50"
                    >
                      {embFetching ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <RefreshCw className="w-3.5 h-3.5" />
                      )}
                      获取模型
                    </button>
                    <button
                      type="button"
                      disabled={embTesting}
                      onClick={() => void handleTestEmb()}
                      className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-white border border-[#E2DFD7] hover:border-[#3B5BDB]/50 text-[#18181B] text-xs font-medium cursor-pointer disabled:opacity-50"
                    >
                      {embTesting ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <Wifi className="w-3.5 h-3.5" />
                      )}
                      测试连接
                    </button>
                  </div>

                  {embFetchMsg && (
                    <div className="text-xs text-[#3B5BDB] bg-[#EEF2FF]/60 rounded-lg px-3 py-2">
                      {embFetchMsg}
                    </div>
                  )}

                  {availableEmbModels.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 pt-1">
                      {availableEmbModels.slice(0, 24).map((m) => (
                        <button
                          key={m}
                          type="button"
                          onClick={() => setEmbModel(m)}
                          className={`px-2.5 py-1 rounded-lg text-[11px] font-mono border cursor-pointer ${
                            embModel === m
                              ? "bg-[#3B5BDB] text-white border-[#3B5BDB]"
                              : "bg-[#FAF9F6] text-[#3F3F46] border-[#E2DFD7] hover:border-[#3B5BDB]/50"
                          }`}
                        >
                          {m}
                        </button>
                      ))}
                    </div>
                  )}

                  {embTestResult && (
                    <div
                      className={`flex items-start gap-2 rounded-lg px-3 py-2 text-xs ${
                        embTestResult.ok
                          ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
                          : "bg-rose-50 text-rose-700 border border-rose-200"
                      }`}
                    >
                      {embTestResult.ok ? (
                        <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
                      ) : (
                        <XCircle className="w-4 h-4 shrink-0 mt-0.5" />
                      )}
                      <div>
                        <div className="font-medium">
                          {embTestResult.message}
                          {embTestResult.latency_ms > 0 ? ` · ${embTestResult.latency_ms} ms` : ""}
                        </div>
                        {embTestResult.detail && (
                          <div className="mt-0.5 opacity-80 break-all">{embTestResult.detail}</div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* TAB 3: Academic Sources */}
          {tab === "sources" && (
            <div className="space-y-4">
              <div className="rounded-xl bg-white border border-[#E7E5DF] p-4 space-y-3">
                <div className="text-xs font-semibold text-[#18181B]">默认启用的学术检索源</div>
                <div className="space-y-2">
                  {SOURCE_OPTIONS.map((opt) => (
                    <label
                      key={opt.id}
                      className="flex items-start gap-2.5 p-2.5 rounded-lg border border-[#ECEAE4] hover:bg-[#FAF9F6] cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={academicSources.includes(opt.id)}
                        onChange={() => toggleSource(opt.id)}
                        className="accent-[#3B5BDB] mt-0.5"
                      />
                      <div>
                        <div className="text-xs font-medium text-[#18181B]">{opt.label}</div>
                        <div className="text-[11px] text-[#71717A]">{opt.desc}</div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              <div className="rounded-xl bg-white border border-[#E7E5DF] p-4 space-y-3">
                <label className="block text-xs font-medium text-[#3F3F46]">
                  OpenAlex 礼貌池邮箱（可选，免 Key 提速）
                  <input
                    type="email"
                    value={openalexEmail}
                    onChange={(e) => setOpenalexEmail(e.target.value)}
                    placeholder="your_email@university.edu"
                    className="mt-1.5 w-full rounded-lg border border-[#E2DFD7] px-3 py-2 text-sm text-[#18181B] focus:outline-none focus:border-[#3B5BDB]"
                  />
                </label>

                <label className="block text-xs font-medium text-[#3F3F46]">
                  Semantic Scholar API Key（可选）
                  <input
                    type="password"
                    value={s2ApiKey}
                    onChange={(e) => setS2ApiKey(e.target.value)}
                    placeholder={
                      settings?.semantic_scholar_key_configured
                        ? "已配置，留空保持当前密钥"
                        : "如需高频调用 Semantic Scholar 可选填"
                    }
                    className="mt-1.5 w-full rounded-lg border border-[#E2DFD7] px-3 py-2 text-sm text-[#18181B] focus:outline-none focus:border-[#3B5BDB]"
                  />
                </label>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3.5 bg-white border-t border-[#E7E5DF] flex items-center justify-end gap-2.5">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-white border border-[#E2DFD7] text-xs font-medium text-[#52525B] hover:text-[#18181B] cursor-pointer"
          >
            取消
          </button>
          <button
            type="button"
            disabled={saving}
            onClick={() => void handleSave()}
            className="px-4 py-2 rounded-xl bg-[#3B5BDB] hover:bg-[#364FC7] disabled:opacity-50 text-white text-xs font-medium cursor-pointer shadow-2xs"
          >
            {saving ? "保存中..." : "保存配置"}
          </button>
        </div>
      </div>
    </div>
  );
}
