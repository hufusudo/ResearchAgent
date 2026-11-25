import { useEffect, useState } from "react";
import { ExternalLink, FileText, Search, X } from "lucide-react";
import { getChunks } from "../api";
import { INGEST_LABELS, SOURCE_LABELS } from "../labels";
import type { Paper, TextChunk } from "../types";

interface ProvenanceDrawerProps {
  open: boolean;
  runId: string;
  papers: Paper[];
  selectedPaperId: string | null;
  highlightChunkId: string | null;
  onSelectPaper: (paperId: string) => void;
  onClose: () => void;
}

export function ProvenanceDrawer({
  open,
  runId,
  papers,
  selectedPaperId,
  highlightChunkId,
  onSelectPaper,
  onClose,
}: ProvenanceDrawerProps) {
  const [chunks, setChunks] = useState<TextChunk[]>([]);
  const [loading, setLoading] = useState(false);
  const [filterText, setFilterText] = useState("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const activePaper =
    papers.find((p) => p.paper_id === selectedPaperId) ?? papers[0] ?? null;

  useEffect(() => {
    if (!open || !activePaper) return;
    let cancelled = false;
    setLoading(true);
    setErrorMsg(null);
    getChunks(runId, activePaper.paper_id)
      .then((res) => {
        if (!cancelled) setChunks(res);
      })
      .catch((err) => {
        if (!cancelled) setErrorMsg(err instanceof Error ? err.message : "加载分块失败");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, runId, activePaper?.paper_id]);

  if (!open) return null;

  const filteredChunks = chunks.filter((c) => {
    if (!filterText.trim()) return true;
    const q = filterText.trim().toLowerCase();
    return (
      c.chunk_id.toLowerCase().includes(q) ||
      c.section.toLowerCase().includes(q) ||
      c.text.toLowerCase().includes(q)
    );
  });

  return (
    <aside className="w-full lg:w-[420px] shrink-0 border-l border-[#E7E5DF] bg-[#FCFCFA] flex flex-col h-full">
      {/* Drawer Header */}
      <div className="px-5 py-4 border-b border-[#E7E5DF] flex items-center justify-between gap-2 bg-white">
        <div className="flex items-center gap-2 min-w-0">
          <FileText className="w-4 h-4 text-[#3B5BDB] shrink-0" />
          <span className="text-sm font-semibold text-[#18181B] truncate">
            原文引用溯源检查
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="p-1.5 rounded-lg text-[#71717A] hover:text-[#18181B] hover:bg-[#F3F2EE] cursor-pointer"
          title="关闭溯源侧栏"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Paper Selector Tabs */}
      {papers.length > 0 && (
        <div className="px-5 py-3 border-b border-[#ECEAE4] bg-[#F8F7F3]">
          <div className="flex flex-wrap gap-1.5">
            {papers.map((p) => {
              const active = activePaper?.paper_id === p.paper_id;
              return (
                <button
                  key={p.paper_id}
                  type="button"
                  onClick={() => onSelectPaper(p.paper_id)}
                  className={`px-2.5 py-1 rounded-lg font-mono text-xs font-medium transition-colors cursor-pointer ${
                    active
                      ? "bg-[#3B5BDB] text-white"
                      : "bg-white border border-[#E2DFD7] text-[#52525B] hover:border-[#3B5BDB]/50"
                  }`}
                >
                  {p.paper_id}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Selected Paper Metadata */}
      {activePaper ? (
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          <div className="rounded-xl bg-white border border-[#E7E5DF] p-4 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <span className="px-2 py-0.5 rounded bg-[#EEF2FF] text-[#3B5BDB] font-mono text-xs font-semibold">
                {activePaper.paper_id}
              </span>
              <span className="text-xs text-[#71717A]">
                {SOURCE_LABELS[activePaper.source] ?? activePaper.source} ·{" "}
                {INGEST_LABELS[activePaper.ingest_status] ?? activePaper.ingest_status}
              </span>
            </div>
            <h3 className="text-sm font-semibold text-[#18181B] leading-snug">
              {activePaper.title}
            </h3>
            <div className="text-xs text-[#52525B]">
              {activePaper.authors.slice(0, 4).join(", ")}
              {activePaper.authors.length > 4 ? " et al." : ""} · {activePaper.year || "n.d."}
              {activePaper.venue ? ` · ${activePaper.venue}` : ""}
              {activePaper.citation_count !== null ? ` · 引用 ${activePaper.citation_count}` : ""}
            </div>
            <a
              href={activePaper.url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-xs text-[#3B5BDB] hover:underline pt-1"
            >
              查看原始文献链接
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>

          {/* Search within paper chunks */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-[#71717A] absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              placeholder="在当前文献分块中检索关键词或 ID..."
              className="w-full pl-8 pr-3 py-2 rounded-xl bg-white border border-[#E5E2DC] text-xs text-[#18181B] placeholder:text-[#A1A1AA] focus:outline-none focus:border-[#3B5BDB]"
            />
          </div>

          {loading ? (
            <div className="text-xs text-[#71717A] py-6 text-center">正在加载原文分块...</div>
          ) : errorMsg ? (
            <div className="text-xs text-rose-600 py-4">{errorMsg}</div>
          ) : (
            <div className="space-y-3">
              <div className="text-[11px] text-[#71717A]">
                共 {filteredChunks.length} 个分块
              </div>
              {filteredChunks.slice(0, 30).map((chunk) => {
                const isHighlighted = highlightChunkId === chunk.chunk_id;
                return (
                  <div
                    key={chunk.chunk_id}
                    className={`rounded-xl border p-3.5 text-xs leading-relaxed transition-colors ${
                      isHighlighted
                        ? "bg-[#EEF2FF]/60 border-[#3B5BDB]"
                        : "bg-white border-[#E7E5DF]"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <span className="font-mono font-semibold text-[#3B5BDB]">
                        {chunk.chunk_id}
                      </span>
                      <span className="text-[11px] text-[#71717A] truncate">
                        {chunk.section} · 第 {chunk.page} 页
                      </span>
                    </div>
                    <p className="text-[#27272A] whitespace-pre-wrap break-words">{chunk.text}</p>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      ) : (
        <div className="p-6 text-xs text-[#71717A]">暂无可溯源的文献。</div>
      )}
    </aside>
  );
}
