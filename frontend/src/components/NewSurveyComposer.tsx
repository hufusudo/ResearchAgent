import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  Check,
  ChevronDown,
  FileText,
  Paperclip,
  Settings,
  SlidersHorizontal,
  Sparkles,
  X,
} from "lucide-react";
import { uploadPdfs } from "../api";
import type { CreateTaskPayload, UploadedFileInfo } from "../types";

interface NewSurveyComposerProps {
  defaultSources: string[];
  enabledModels: string[];
  defaultModel: string;
  submitting: boolean;
  onSubmit: (payload: CreateTaskPayload) => Promise<void>;
  onOpenSettings: () => void;
}

const EXAMPLE_TOPICS = [
  "基于图检索增强生成（GraphRAG）的多跳推理与幻觉抑制研究",
  "KV Cache Compression and Eviction Methods for Long-Context LLM Inference",
  "Test-Time Compute Scaling and Process Reward Models for Code Reasoning",
  "多智能体协同下的自动化科研文献综述与事实核验方法",
];

const SOURCE_OPTIONS: Array<{ id: string; label: string; hint: string }> = [
  { id: "arxiv", label: "arXiv", hint: "预印本全文稳定" },
  { id: "openalex", label: "OpenAlex", hint: "含引用量与期刊会议" },
  { id: "semantic_scholar", label: "Semantic Scholar", hint: "高覆盖学术图谱" },
];

export function NewSurveyComposer({
  defaultSources,
  enabledModels,
  defaultModel,
  submitting,
  onSubmit,
  onOpenSettings,
}: NewSurveyComposerProps) {
  const [topic, setTopic] = useState("");
  const [topK, setTopK] = useState(5);
  const [maxPerQuery, setMaxPerQuery] = useState(5);
  const [maxIterations, setMaxIterations] = useState(2);
  const [downloadPdfs, setDownloadPdfs] = useState(true);
  const [hitl, setHitl] = useState(false);
  const [mockMode, setMockMode] = useState(false);
  const [sources, setSources] = useState<string[]>(
    defaultSources.length > 0 ? defaultSources : ["arxiv", "openalex"]
  );
  const [selectedModel, setSelectedModel] = useState(
    defaultModel || enabledModels[0] || "gpt-4o-mini"
  );
  const [showSettings, setShowSettings] = useState(false);
  const [showModelMenu, setShowModelMenu] = useState(false);

  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFileInfo[]>([]);
  const [uploadBatchId, setUploadBatchId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const modelMenuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const modelList = enabledModels.length > 0 ? enabledModels : [defaultModel].filter(Boolean);
    if (modelList.length > 0 && !modelList.includes(selectedModel)) {
      setSelectedModel(defaultModel || modelList[0]);
    }
  }, [enabledModels, defaultModel, selectedModel]);

  useEffect(() => {
    if (!showModelMenu) return;
    const handleClickOutside = (event: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(event.target as Node)) {
        setShowModelMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showModelMenu]);

  const modelChoices = Array.from(
    new Set([defaultModel, ...enabledModels, selectedModel].filter(Boolean))
  );

  const handleFilesSelected = async (incoming: FileList | File[]) => {
    const pdfs = Array.from(incoming).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (pdfs.length === 0) {
      setErrorMsg("请仅上传 .pdf 格式的英文或中文文献文件。");
      return;
    }
    const merged = [...pendingFiles];
    for (const file of pdfs) {
      if (!merged.some((existing) => existing.name === file.name && existing.size === file.size)) {
        merged.push(file);
      }
    }

    setErrorMsg(null);
    setUploading(true);
    try {
      const resp = await uploadPdfs(merged);
      setPendingFiles(merged);
      setUploadedFiles(resp.files);
      setUploadBatchId(resp.upload_batch_id);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "PDF 上传失败，请重试。");
    } finally {
      setUploading(false);
    }
  };

  const removeUploadedFile = async (index: number) => {
    const nextFiles = pendingFiles.filter((_, idx) => idx !== index);
    if (nextFiles.length === 0) {
      setPendingFiles([]);
      setUploadedFiles([]);
      setUploadBatchId(null);
      return;
    }
    setUploading(true);
    try {
      const resp = await uploadPdfs(nextFiles);
      setPendingFiles(nextFiles);
      setUploadedFiles(resp.files);
      setUploadBatchId(resp.upload_batch_id);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "更新上传文件列表失败。");
    } finally {
      setUploading(false);
    }
  };

  const toggleSource = (id: string) => {
    setSources((prev) => {
      if (prev.includes(id)) {
        return prev.length > 1 ? prev.filter((s) => s !== id) : prev;
      }
      return [...prev, id];
    });
  };

  const handleLaunch = async () => {
    if (topic.trim().length < 2) {
      setErrorMsg("请输入有效的研究主题（至少 2 个字符）。");
      return;
    }
    setErrorMsg(null);
    await onSubmit({
      topic: topic.trim(),
      top_k_papers: topK,
      max_papers_per_query: maxPerQuery,
      max_iterations: maxIterations,
      download_pdfs: downloadPdfs,
      require_plan_approval: hitl,
      mock: mockMode,
      upload_batch_id: uploadBatchId,
      sources,
      llm_model: selectedModel || null,
    });
  };

  return (
    <div className="min-h-full flex flex-col items-center justify-center px-6 py-12">
      <div className="w-full max-w-3xl">
        {/* Editorial Welcome Heading */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-semibold text-[#18181B] tracking-tight">
            今天想调研什么研究方向？
          </h1>
          <p className="mt-2.5 text-sm text-[#52525B]">
            输入研究主题或拖入本地 PDF，智能体将自主完成规划拆解、多源检索、混合 RAG 溯源与反思迭代。
          </p>
        </div>

        {/* Spacious Composer Card */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            if (e.dataTransfer.files.length > 0) {
              void handleFilesSelected(e.dataTransfer.files);
            }
          }}
          className={`relative rounded-2xl bg-white border transition-all shadow-xs ${
            dragOver
              ? "border-[#3B5BDB] ring-2 ring-[#3B5BDB]/15"
              : "border-[#E5E2DC] focus-within:border-[#3B5BDB]/60"
          }`}
        >
          <textarea
            rows={3}
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            onKeyDown={(e) => {
              if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !submitting) {
                e.preventDefault();
                void handleLaunch();
              }
            }}
            placeholder="输入研究主题（支持中文或英文，Ctrl + Enter 快速启动）..."
            className="w-full resize-none bg-transparent px-5 pt-4 pb-2 text-[15px] text-[#18181B] placeholder:text-[#A1A1AA] focus:outline-none leading-relaxed"
          />

          {/* Attached Local PDF Pills */}
          {uploadedFiles.length > 0 && (
            <div className="px-5 pb-2 flex flex-wrap gap-2">
              {uploadedFiles.map((file, idx) => (
                <span
                  key={`${file.filename}-${idx}`}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#F5F4F0] border border-[#E5E2DC] text-xs text-[#27272A]"
                >
                  <FileText className="w-3.5 h-3.5 text-[#3B5BDB]" />
                  <span className="max-w-[200px] truncate">{file.filename}</span>
                  <span className="text-[#71717A]">
                    ({Math.max(1, Math.round(file.size_bytes / 1024))} KB)
                  </span>
                  <button
                    type="button"
                    onClick={() => void removeUploadedFile(idx)}
                    className="text-[#71717A] hover:text-rose-600 cursor-pointer"
                    title="移除该文件"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </span>
              ))}
            </div>
          )}

          {/* Clean Bottom Toolbar */}
          <div className="px-4 py-3 border-t border-[#F0EEE9] flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                multiple
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0) {
                    void handleFilesSelected(e.target.files);
                    e.target.value = "";
                  }
                }}
              />
              <button
                type="button"
                disabled={uploading}
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-[#52525B] hover:text-[#18181B] hover:bg-[#F5F4F0] transition-colors cursor-pointer"
              >
                <Paperclip className="w-3.5 h-3.5" />
                {uploading ? "正在上传..." : "上传本地 PDF"}
              </button>

              <button
                type="button"
                onClick={() => setShowSettings((prev) => !prev)}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
                  showSettings
                    ? "bg-[#EEF2FF] text-[#3B5BDB]"
                    : "text-[#52525B] hover:text-[#18181B] hover:bg-[#F5F4F0]"
                }`}
              >
                <SlidersHorizontal className="w-3.5 h-3.5" />
                <span>
                  {topK} 篇 · {maxIterations} 轮反思 · {sources.length} 源
                </span>
              </button>
            </div>

            {/* Bottom-Right Quick Model Switcher + Launch CTA */}
            <div className="flex items-center gap-2">
              <div ref={modelMenuRef} className="relative">
                <button
                  type="button"
                  onClick={() => setShowModelMenu((prev) => !prev)}
                  className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-[#FAF9F6] border border-[#E2DFD7] hover:border-[#3B5BDB]/50 text-xs font-medium text-[#27272A] transition-colors cursor-pointer"
                  title="切换本次调研使用的大模型"
                >
                  <Sparkles className="w-3.5 h-3.5 text-[#3B5BDB]" />
                  <span className="max-w-[160px] truncate font-mono">{selectedModel}</span>
                  <ChevronDown className="w-3.5 h-3.5 text-[#71717A]" />
                </button>

                {showModelMenu && (
                  <div className="absolute right-0 bottom-full mb-2 w-60 rounded-xl bg-white border border-[#E5E2DC] shadow-lg py-1.5 z-30">
                    <div className="px-3 py-1 text-[11px] font-medium text-[#71717A]">
                      选择调研模型
                    </div>
                    <div className="max-h-52 overflow-y-auto">
                      {modelChoices.map((mid) => {
                        const isSelected = mid === selectedModel;
                        return (
                          <button
                            key={mid}
                            type="button"
                            onClick={() => {
                              setSelectedModel(mid);
                              setShowModelMenu(false);
                            }}
                            className={`w-full px-3 py-2 text-left text-xs flex items-center justify-between gap-2 hover:bg-[#FAF9F6] cursor-pointer ${
                              isSelected ? "text-[#3B5BDB] font-medium bg-[#EEF2FF]/40" : "text-[#27272A]"
                            }`}
                          >
                            <span className="font-mono truncate">{mid}</span>
                            {isSelected && <Check className="w-3.5 h-3.5 shrink-0" />}
                          </button>
                        );
                      })}
                    </div>
                    <div className="mt-1 pt-1 border-t border-[#F0EEE9]">
                      <button
                        type="button"
                        onClick={() => {
                          setShowModelMenu(false);
                          onOpenSettings();
                        }}
                        className="w-full px-3 py-2 text-left text-xs text-[#52525B] hover:text-[#3B5BDB] hover:bg-[#FAF9F6] flex items-center gap-2 cursor-pointer"
                      >
                        <Settings className="w-3.5 h-3.5" />
                        管理 API 与可用模型...
                      </button>
                    </div>
                  </div>
                )}
              </div>

              <button
                type="button"
                disabled={submitting || uploading}
                onClick={() => void handleLaunch()}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-[#3B5BDB] hover:bg-[#364FC7] disabled:opacity-50 text-white text-sm font-medium transition-colors cursor-pointer shadow-2xs"
              >
                <span>{submitting ? "正在启动..." : "开始调研"}</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Expandable Clean Configuration Drawer */}
          {showSettings && (
            <div className="px-5 py-4 border-t border-[#ECEAE4] bg-[#FCFBF9] rounded-b-2xl space-y-4">
              <div>
                <div className="text-xs font-medium text-[#18181B] mb-2">学术检索数据源</div>
                <div className="flex flex-wrap gap-2">
                  {SOURCE_OPTIONS.map((opt) => {
                    const active = sources.includes(opt.id);
                    return (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => toggleSource(opt.id)}
                        className={`px-3 py-1.5 rounded-lg text-xs border transition-colors cursor-pointer ${
                          active
                            ? "bg-[#EEF2FF] border-[#3B5BDB]/40 text-[#3B5BDB] font-medium"
                            : "bg-white border-[#E5E2DC] text-[#52525B] hover:border-[#D4D0C7]"
                        }`}
                      >
                        {opt.label}
                        <span className="ml-1.5 text-[11px] opacity-75">({opt.hint})</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <label className="block text-xs text-[#52525B]">
                  <div className="flex justify-between mb-1">
                    <span>初始入选上限 (Top-K)</span>
                    <span className="font-medium text-[#18181B]">{topK} 篇</span>
                  </div>
                  <input
                    type="range"
                    min={2}
                    max={15}
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="w-full accent-[#3B5BDB]"
                  />
                </label>

                <label className="block text-xs text-[#52525B]">
                  <div className="flex justify-between mb-1">
                    <span>单检索词召回上限</span>
                    <span className="font-medium text-[#18181B]">{maxPerQuery} 篇</span>
                  </div>
                  <input
                    type="range"
                    min={2}
                    max={15}
                    value={maxPerQuery}
                    onChange={(e) => setMaxPerQuery(Number(e.target.value))}
                    className="w-full accent-[#3B5BDB]"
                  />
                </label>

                <label className="block text-xs text-[#52525B]">
                  <div className="flex justify-between mb-1">
                    <span>最大反思补充轮次</span>
                    <span className="font-medium text-[#18181B]">{maxIterations} 轮</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={3}
                    value={maxIterations}
                    onChange={(e) => setMaxIterations(Number(e.target.value))}
                    className="w-full accent-[#3B5BDB]"
                  />
                </label>
              </div>

              <div className="flex flex-wrap gap-5 pt-1 text-xs text-[#3F3F46]">
                <label className="inline-flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={downloadPdfs}
                    onChange={(e) => setDownloadPdfs(e.target.checked)}
                    className="accent-[#3B5BDB] rounded"
                  />
                  下载并解析开源全文 PDF（取消则仅用摘要）
                </label>
                <label className="inline-flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={hitl}
                    onChange={(e) => setHitl(e.target.checked)}
                    className="accent-[#3B5BDB] rounded"
                  />
                  开启 HITL（检索前暂停确认大纲）
                </label>
                <label className="inline-flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={mockMode}
                    onChange={(e) => setMockMode(e.target.checked)}
                    className="accent-[#3B5BDB] rounded"
                  />
                  离线模拟模式（不消耗 Token）
                </label>
              </div>
            </div>
          )}
        </div>

        {errorMsg && (
          <div className="mt-3 px-4 py-2.5 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-700">
            {errorMsg}
          </div>
        )}

        {/* Quiet Example Topic Suggestions */}
        <div className="mt-6">
          <div className="text-xs text-[#71717A] mb-2.5 text-center">示例研究主题（点击快速填充）</div>
          <div className="flex flex-wrap justify-center gap-2">
            {EXAMPLE_TOPICS.map((sample) => (
              <button
                key={sample}
                type="button"
                onClick={() => setTopic(sample)}
                className="px-3 py-1.5 rounded-full bg-white border border-[#E7E5DF] text-xs text-[#52525B] hover:border-[#3B5BDB]/40 hover:text-[#18181B] transition-colors cursor-pointer"
              >
                {sample}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
