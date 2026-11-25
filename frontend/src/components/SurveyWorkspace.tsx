import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  BookMarked,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  Download,
  ExternalLink,
  PanelRightOpen,
} from "lucide-react";
import { getBibtexDownloadUrl, getReportDownloadUrl } from "../api";
import {
  INGEST_LABELS,
  NODE_LABELS,
  SOURCE_LABELS,
  STOP_REASON_LABELS,
  VERDICT_LABELS,
} from "../labels";
import type { TaskDetailResponse } from "../types";
import { ProvenanceDrawer } from "./ProvenanceDrawer";

interface SurveyWorkspaceProps {
  detail: TaskDetailResponse;
}

type ActiveTab = "survey" | "matrix" | "reflection" | "literature";

function linkifyCitations(markdown: string): string {
  // Split around fenced or inline code so we never alter code snippets.
  const segments = markdown.split(/(```[\s\S]*?```|`[^`\n]+`)/g);
  return segments
    .map((segment, index) => {
      if (index % 2 === 1) return segment;
      return segment.replace(/\[((?:P\d{2,})(?:\s*,\s*P\d{2,})*)\](?!\()/g, (_full, inner: string) => {
        const ids = inner
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean);
        return ids.map((pid) => `[[${pid}]](#cite-${pid})`).join(" ");
      });
    })
    .join("");
}

export function SurveyWorkspace({ detail }: SurveyWorkspaceProps) {
  const [activeTab, setActiveTab] = useState<ActiveTab>("survey");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerPaperId, setDrawerPaperId] = useState<string | null>(null);
  const [highlightChunkId, setHighlightChunkId] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [selectedSqFilter, setSelectedSqFilter] = useState<string>("ALL");
  const [traceExpanded, setTraceExpanded] = useState(false);

  const papers = detail.selected_papers || [];
  const ver = detail.verification_report;
  const evalHist = detail.evaluation_history || [];
  const rewriteHist = detail.rewrite_history || [];
  const subQuestions = detail.plan?.sub_questions || [];

  const linkedMarkdown = useMemo(
    () => linkifyCitations(detail.draft_markdown || ""),
    [detail.draft_markdown]
  );

  const openProvenanceForPaper = (paperId: string, chunkId: string | null = null) => {
    setDrawerPaperId(paperId);
    setHighlightChunkId(chunkId);
    setDrawerOpen(true);
  };

  const handleCopyMarkdown = async () => {
    if (!detail.draft_markdown) return;
    try {
      await navigator.clipboard.writeText(detail.draft_markdown);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // ignore clipboard permissions error
    }
  };

  const tabs: Array<{ id: ActiveTab; label: string }> = [
    { id: "survey", label: "综述初稿" },
    { id: "matrix", label: "证据矩阵" },
    { id: "reflection", label: "反思与核验" },
    { id: "literature", label: "文献与轨迹" },
  ];

  const filteredSubQuestions =
    selectedSqFilter === "ALL"
      ? subQuestions
      : subQuestions.filter((sq) => sq.id === selectedSqFilter);

  return (
    <div className="flex h-full overflow-hidden">
      {/* Main Content Area */}
      <div className="flex-1 min-w-0 flex flex-col h-full overflow-hidden">
        {/* Clean Sticky Workspace Header */}
        <header className="px-6 lg:px-10 pt-6 pb-0 border-b border-[#E7E5DF] bg-[#FAF9F6]/90 backdrop-blur-xs shrink-0">
          <div className="max-w-5xl mx-auto">
            <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
              <div className="min-w-0">
                <h1 className="text-xl lg:text-2xl font-semibold text-[#18181B] tracking-tight leading-snug">
                  {detail.topic}
                </h1>
                <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-[#71717A]">
                  <span>{papers.length} 篇文献</span>
                  <span>·</span>
                  <span>{rewriteHist.length} 轮反思补充</span>
                  {ver && (
                    <>
                      <span>·</span>
                      <span className="text-emerald-700 font-medium">
                        引用真实率 {(ver.reference_validity_rate * 100).toFixed(0)}%
                      </span>
                      <span>·</span>
                      <span className="text-[#3B5BDB] font-medium">
                        论断支撑率 {(ver.claim_support_rate * 100).toFixed(0)}%
                      </span>
                    </>
                  )}
                </div>
              </div>

              {/* Compact Action Toolbar */}
              <div className="flex flex-wrap items-center gap-2 shrink-0">
                <a
                  href={getReportDownloadUrl(detail.run_id)}
                  download={`reacher_survey_${detail.run_id}.md`}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#3B5BDB] hover:bg-[#364FC7] text-white text-xs font-medium transition-colors shadow-2xs"
                >
                  <Download className="w-3.5 h-3.5" />
                  下载综述 (.md)
                </a>
                <a
                  href={getBibtexDownloadUrl(detail.run_id)}
                  download={`reacher_refs_${detail.run_id}.bib`}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-[#E2DFD7] hover:border-[#3B5BDB]/40 text-xs font-medium text-[#27272A] transition-colors"
                >
                  <BookMarked className="w-3.5 h-3.5" />
                  BibTeX
                </a>
                <button
                  type="button"
                  onClick={() => void handleCopyMarkdown()}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-[#E2DFD7] hover:border-[#3B5BDB]/40 text-xs font-medium text-[#27272A] transition-colors cursor-pointer"
                >
                  {copied ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-600" />
                      已复制
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      复制
                    </>
                  )}
                </button>
                <button
                  type="button"
                  onClick={() => setDrawerOpen((prev) => !prev)}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors cursor-pointer ${
                    drawerOpen
                      ? "bg-[#EEF2FF] border-[#3B5BDB]/40 text-[#3B5BDB]"
                      : "bg-white border-[#E2DFD7] text-[#27272A] hover:border-[#3B5BDB]/40"
                  }`}
                >
                  <PanelRightOpen className="w-3.5 h-3.5" />
                  原文溯源
                </button>
              </div>
            </div>

            {/* Minimal Underline Tabs */}
            <nav className="mt-5 flex items-center gap-6">
              {tabs.map((tab) => {
                const active = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setActiveTab(tab.id)}
                    className={`pb-2.5 text-sm font-medium border-b-2 transition-colors cursor-pointer ${
                      active
                        ? "border-[#3B5BDB] text-[#18181B]"
                        : "border-transparent text-[#71717A] hover:text-[#18181B]"
                    }`}
                  >
                    {tab.label}
                  </button>
                );
              })}
            </nav>
          </div>
        </header>

        {/* Scrollable Tab Body */}
        <div className="flex-1 overflow-y-auto px-6 lg:px-10 py-8">
          <div className="max-w-5xl mx-auto">
            {detail.error_message && (
              <div className="mb-6 px-4 py-3 rounded-xl bg-rose-50 border border-rose-200 text-sm text-rose-700">
                任务执行异常：{detail.error_message}
              </div>
            )}

            {/* TAB 1: Survey Editorial Reader */}
            {activeTab === "survey" && (
              <div className="max-w-[780px] mx-auto">
                <div className="mb-4 flex items-center justify-between text-xs text-[#71717A]">
                  <span>点击正文中的蓝色引用标签（如 [P01]）可直接在右侧查看原文分块证据。</span>
                </div>
                <article className="rounded-2xl bg-white border border-[#E7E5DF] px-6 sm:px-10 py-8 shadow-2xs prose-academic">
                  {detail.draft_markdown ? (
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        a: ({ href, children, ...props }) => {
                          if (href && href.startsWith("#cite-")) {
                            const pid = href.replace("#cite-", "");
                            return (
                              <button
                                type="button"
                                onClick={() => openProvenanceForPaper(pid)}
                                className="inline-flex items-center px-1.5 py-0.5 mx-0.5 text-[11px] font-mono font-medium rounded bg-[#EEF2FF] text-[#3B5BDB] hover:bg-[#DBE4FF] transition-colors cursor-pointer align-baseline"
                                title={`点击查看 ${pid} 的原文分块`}
                              >
                                {children}
                              </button>
                            );
                          }
                          return (
                            <a
                              href={href}
                              target="_blank"
                              rel="noreferrer"
                              className="text-[#3B5BDB] hover:underline"
                              {...props}
                            >
                              {children}
                            </a>
                          );
                        },
                      }}
                    >
                      {linkedMarkdown}
                    </ReactMarkdown>
                  ) : (
                    <p className="text-sm text-[#71717A]">暂未生成综述正文。</p>
                  )}
                </article>
              </div>
            )}

            {/* TAB 2: Structured Evidence Matrix */}
            {activeTab === "matrix" && (
              <div className="space-y-6">
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setSelectedSqFilter("ALL")}
                    className={`px-3 py-1.5 rounded-full text-xs font-medium cursor-pointer transition-colors ${
                      selectedSqFilter === "ALL"
                        ? "bg-[#3B5BDB] text-white"
                        : "bg-white border border-[#E2DFD7] text-[#52525B]"
                    }`}
                  >
                    全部子问题 ({subQuestions.length})
                  </button>
                  {subQuestions.map((sq) => (
                    <button
                      key={sq.id}
                      type="button"
                      onClick={() => setSelectedSqFilter(sq.id)}
                      className={`px-3 py-1.5 rounded-full text-xs font-medium cursor-pointer transition-colors ${
                        selectedSqFilter === sq.id
                          ? "bg-[#3B5BDB] text-white"
                          : "bg-white border border-[#E2DFD7] text-[#52525B]"
                      }`}
                    >
                      {sq.id} · {sq.title}
                    </button>
                  ))}
                </div>

                {filteredSubQuestions.map((sq) => {
                  const items = detail.evidence_by_subquestion[sq.id] || [];
                  return (
                    <section
                      key={sq.id}
                      className="rounded-2xl bg-white border border-[#E7E5DF] overflow-hidden shadow-2xs"
                    >
                      <div className="px-5 py-4 border-b border-[#ECEAE4] bg-[#FCFBF9]">
                        <div className="flex items-center gap-2">
                          <span className="px-2 py-0.5 rounded bg-[#EEF2FF] text-[#3B5BDB] font-mono text-xs font-semibold">
                            {sq.id}
                          </span>
                          <h3 className="text-sm font-semibold text-[#18181B]">{sq.title}</h3>
                        </div>
                        <p className="mt-1 text-xs text-[#71717A]">{sq.description}</p>
                      </div>

                      {items.length === 0 ? (
                        <div className="px-5 py-6 text-xs text-[#71717A]">
                          该子问题暂未抽取出结构化证据。
                        </div>
                      ) : (
                        <div className="overflow-x-auto">
                          <table className="w-full text-left border-collapse text-xs leading-relaxed">
                            <thead className="bg-[#F6F5F0] text-[#52525B]">
                              <tr>
                                <th className="px-4 py-3 font-medium">文献</th>
                                <th className="px-4 py-3 font-medium">核心方法</th>
                                <th className="px-4 py-3 font-medium">数据集 / 场景</th>
                                <th className="px-4 py-3 font-medium">关键发现</th>
                                <th className="px-4 py-3 font-medium">局限性</th>
                                <th className="px-4 py-3 font-medium">原文分块</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-[#EFECE6]">
                              {items.map((ev, i) => (
                                <tr key={`${ev.paper_id}-${i}`} className="hover:bg-[#FAF9F6]">
                                  <td className="px-4 py-3 align-top">
                                    <button
                                      type="button"
                                      onClick={() => openProvenanceForPaper(ev.paper_id)}
                                      className="px-2 py-0.5 rounded bg-[#EEF2FF] text-[#3B5BDB] font-mono font-medium hover:bg-[#DBE4FF] cursor-pointer"
                                    >
                                      {ev.paper_id}
                                    </button>
                                  </td>
                                  <td className="px-4 py-3 align-top text-[#27272A] min-w-[180px]">
                                    {ev.method}
                                  </td>
                                  <td className="px-4 py-3 align-top text-[#52525B] min-w-[150px]">
                                    {ev.dataset_or_setting}
                                  </td>
                                  <td className="px-4 py-3 align-top text-[#27272A] min-w-[200px]">
                                    {ev.key_findings}
                                  </td>
                                  <td className="px-4 py-3 align-top text-[#52525B] min-w-[170px]">
                                    {ev.limitations}
                                  </td>
                                  <td className="px-4 py-3 align-top">
                                    <div className="flex flex-wrap gap-1">
                                      {ev.source_chunk_ids.map((cid) => (
                                        <button
                                          key={cid}
                                          type="button"
                                          onClick={() =>
                                            openProvenanceForPaper(ev.paper_id, cid)
                                          }
                                          className="px-1.5 py-0.5 rounded bg-[#F3F2EE] hover:bg-[#EEF2FF] text-[#3F3F46] hover:text-[#3B5BDB] font-mono text-[11px] cursor-pointer"
                                        >
                                          {cid}
                                        </button>
                                      ))}
                                    </div>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </section>
                  );
                })}
              </div>
            )}

            {/* TAB 3: Reflection & Citation Audit */}
            {activeTab === "reflection" && (
              <div className="space-y-8">
                {/* Evaluator & Query Rewriter Rounds */}
                <section className="space-y-4">
                  <h2 className="text-base font-semibold text-[#18181B]">
                    动态评估与反思轨迹
                  </h2>
                  {evalHist.length === 0 ? (
                    <div className="rounded-2xl bg-white border border-[#E7E5DF] p-6 text-xs text-[#71717A]">
                      暂无评估迭代记录（例如早期阶段一历史任务）。
                    </div>
                  ) : (
                    evalHist.map((ev) => {
                      const rw = rewriteHist.find((r) => r.iteration === ev.iteration + 1);
                      return (
                        <div
                          key={ev.iteration}
                          className="rounded-2xl bg-white border border-[#E7E5DF] p-5 space-y-4 shadow-2xs"
                        >
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <div className="flex items-center gap-2.5">
                              <span className="px-2.5 py-0.5 rounded-full bg-[#F3F2EE] text-xs font-semibold text-[#18181B]">
                                第 {ev.iteration} 轮评估
                              </span>
                              <span className="text-xs text-[#52525B]">
                                {STOP_REASON_LABELS[ev.stop_reason] ?? ev.stop_reason}
                              </span>
                            </div>
                            <span
                              className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                ev.needs_more_retrieval
                                  ? "bg-amber-50 text-amber-700 border border-amber-200"
                                  : "bg-emerald-50 text-emerald-700 border border-emerald-200"
                              }`}
                            >
                              {ev.needs_more_retrieval ? "触发补充检索" : "评估通过"}
                            </span>
                          </div>

                          {ev.reasoning && (
                            <p className="text-xs text-[#52525B] leading-relaxed">{ev.reasoning}</p>
                          )}

                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {ev.subquestion_evaluations.map((sqEv) => (
                              <div
                                key={sqEv.sub_question_id}
                                className="rounded-xl bg-[#FAF9F6] border border-[#ECEAE4] p-3.5 space-y-2"
                              >
                                <div className="flex items-center justify-between text-xs">
                                  <span className="font-mono font-semibold text-[#18181B]">
                                    {sqEv.sub_question_id}
                                  </span>
                                  <span className="text-[#52525B]">
                                    {sqEv.unique_paper_count} 篇 · {sqEv.evidence_count} 条证据 ·{" "}
                                    <strong>{sqEv.coverage_score.toFixed(1)}/10</strong>
                                  </span>
                                </div>
                                <div className="h-1.5 rounded-full bg-[#E7E5DF] overflow-hidden">
                                  <div
                                    className={`h-full rounded-full ${
                                      sqEv.sufficient ? "bg-emerald-500" : "bg-amber-500"
                                    }`}
                                    style={{
                                      width: `${Math.min(100, Math.max(8, sqEv.coverage_score * 10))}%`,
                                    }}
                                  />
                                </div>
                                {sqEv.gaps.length > 0 && (
                                  <p className="text-[11px] text-amber-800">
                                    缺口：{sqEv.gaps.join("；")}
                                  </p>
                                )}
                              </div>
                            ))}
                          </div>

                          {rw && rw.rewritten_queries.length > 0 && (
                            <div className="pt-2 border-t border-[#ECEAE4] space-y-2">
                              <div className="text-xs font-medium text-[#3B5BDB]">
                                检索词改写（第 {rw.iteration} 轮补充）
                              </div>
                              <div className="space-y-1.5">
                                {rw.rewritten_queries.map((qItem, qi) => (
                                  <div
                                    key={qi}
                                    className="flex flex-wrap items-center gap-2 text-xs bg-[#EEF2FF]/50 rounded-lg px-3 py-2"
                                  >
                                    <span className="font-mono font-semibold text-[#3B5BDB]">
                                      {qItem.sub_question_id}
                                    </span>
                                    <code className="text-[#18181B] font-mono">{qItem.query}</code>
                                    {qItem.rationale && (
                                      <span className="text-[#71717A]">— {qItem.rationale}</span>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })
                  )}
                </section>

                {/* Citation Verification Audit */}
                <section className="space-y-4">
                  <h2 className="text-base font-semibold text-[#18181B]">
                    引用核验审计报告
                  </h2>
                  {!ver ? (
                    <div className="rounded-2xl bg-white border border-[#E7E5DF] p-6 text-xs text-[#71717A]">
                      暂无引用核验报告。
                    </div>
                  ) : (
                    <>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="rounded-xl bg-white border border-[#E7E5DF] p-4">
                          <div className="text-xs text-[#71717A]">参考文献真实率</div>
                          <div className="mt-1 text-2xl font-semibold text-emerald-700">
                            {(ver.reference_validity_rate * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="rounded-xl bg-white border border-[#E7E5DF] p-4">
                          <div className="text-xs text-[#71717A]">论断原文支撑率</div>
                          <div className="mt-1 text-2xl font-semibold text-[#3B5BDB]">
                            {(ver.claim_support_rate * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="rounded-xl bg-white border border-[#E7E5DF] p-4">
                          <div className="text-xs text-[#71717A]">已核验论断</div>
                          <div className="mt-1 text-2xl font-semibold text-[#18181B]">
                            {ver.checked_claims} / {ver.total_cited_claims}
                          </div>
                        </div>
                        <div className="rounded-xl bg-white border border-[#E7E5DF] p-4">
                          <div className="text-xs text-[#71717A]">非法引用编号</div>
                          <div className="mt-1 text-2xl font-semibold text-[#18181B]">
                            {ver.invalid_citations.length}
                          </div>
                        </div>
                      </div>

                      {ver.details.length > 0 && (
                        <div className="rounded-2xl bg-white border border-[#E7E5DF] overflow-hidden shadow-2xs">
                          <table className="w-full text-left border-collapse text-xs leading-relaxed">
                            <thead className="bg-[#F6F5F0] text-[#52525B]">
                              <tr>
                                <th className="px-4 py-3 font-medium">带引用论断</th>
                                <th className="px-4 py-3 font-medium">引用文献</th>
                                <th className="px-4 py-3 font-medium">核验结论</th>
                                <th className="px-4 py-3 font-medium">核验说明</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-[#EFECE6]">
                              {ver.details.map((item, idx) => (
                                <tr key={idx} className="hover:bg-[#FAF9F6]">
                                  <td className="px-4 py-3 text-[#27272A] min-w-[260px]">
                                    {item.claim}
                                  </td>
                                  <td className="px-4 py-3 align-top">
                                    <div className="flex flex-wrap gap-1">
                                      {item.cited_paper_ids.map((pid) => (
                                        <button
                                          key={pid}
                                          type="button"
                                          onClick={() => openProvenanceForPaper(pid)}
                                          className="px-1.5 py-0.5 rounded bg-[#EEF2FF] text-[#3B5BDB] font-mono text-[11px] cursor-pointer"
                                        >
                                          {pid}
                                        </button>
                                      ))}
                                    </div>
                                  </td>
                                  <td className="px-4 py-3 align-top whitespace-nowrap">
                                    <span
                                      className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${
                                        item.verdict === "supported"
                                          ? "bg-emerald-50 text-emerald-700"
                                          : item.verdict === "partially_supported"
                                          ? "bg-amber-50 text-amber-700"
                                          : "bg-rose-50 text-rose-700"
                                      }`}
                                    >
                                      {VERDICT_LABELS[item.verdict] ?? item.verdict}
                                    </span>
                                  </td>
                                  <td className="px-4 py-3 align-top text-[#52525B] min-w-[180px]">
                                    {item.reason}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </>
                  )}
                </section>
              </div>
            )}

            {/* TAB 4: Literature & Execution Trace */}
            {activeTab === "literature" && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 gap-3">
                  {papers.map((paper) => (
                    <div
                      key={paper.paper_id}
                      className="rounded-2xl bg-white border border-[#E7E5DF] p-5 flex flex-col sm:flex-row sm:items-start justify-between gap-4 shadow-2xs"
                    >
                      <div className="min-w-0 space-y-1.5">
                        <div className="flex flex-wrap items-center gap-2 text-xs">
                          <span className="px-2 py-0.5 rounded bg-[#EEF2FF] text-[#3B5BDB] font-mono font-semibold">
                            {paper.paper_id}
                          </span>
                          <span className="px-2 py-0.5 rounded bg-[#F3F2EE] text-[#52525B]">
                            {SOURCE_LABELS[paper.source] ?? paper.source}
                          </span>
                          <span className="text-[#71717A]">{paper.year || "n.d."}</span>
                          {paper.venue && (
                            <span className="text-[#52525B]">· {paper.venue}</span>
                          )}
                          {paper.citation_count !== null && (
                            <span className="text-[#52525B]">
                              · 被引 {paper.citation_count}
                            </span>
                          )}
                          <span className="text-[#71717A]">
                            · {INGEST_LABELS[paper.ingest_status] ?? paper.ingest_status}
                          </span>
                        </div>
                        <h3 className="text-sm font-semibold text-[#18181B] leading-snug">
                          {paper.title}
                        </h3>
                        <p className="text-xs text-[#71717A]">
                          {paper.authors.slice(0, 5).join(", ")}
                          {paper.authors.length > 5 ? " et al." : ""}
                        </p>
                      </div>

                      <div className="flex items-center gap-2 shrink-0">
                        <button
                          type="button"
                          onClick={() => openProvenanceForPaper(paper.paper_id)}
                          className="px-3 py-1.5 rounded-lg bg-[#FAF9F6] border border-[#E2DFD7] hover:border-[#3B5BDB]/50 text-xs font-medium text-[#27272A] cursor-pointer"
                        >
                          查看分块
                        </button>
                        <a
                          href={paper.url}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-white border border-[#E2DFD7] hover:border-[#3B5BDB]/50 text-xs text-[#3B5BDB]"
                        >
                          原文
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Collapsible Execution Trace */}
                <div className="rounded-2xl bg-white border border-[#E7E5DF] overflow-hidden">
                  <button
                    type="button"
                    onClick={() => setTraceExpanded((prev) => !prev)}
                    className="w-full px-5 py-3.5 flex items-center justify-between text-xs font-medium text-[#18181B] hover:bg-[#FAF9F6] cursor-pointer"
                  >
                    <span>节点执行轨迹明细（共 {detail.trace.length} 步）</span>
                    {traceExpanded ? (
                      <ChevronDown className="w-4 h-4 text-[#71717A]" />
                    ) : (
                      <ChevronRight className="w-4 h-4 text-[#71717A]" />
                    )}
                  </button>
                  {traceExpanded && (
                    <div className="px-5 pb-4 border-t border-[#ECEAE4] divide-y divide-[#F0EEE9]">
                      {detail.trace.map((ev, idx) => (
                        <div key={idx} className="py-2.5 flex items-start justify-between gap-4 text-xs">
                          <div>
                            <span className="font-semibold text-[#18181B]">
                              {NODE_LABELS[ev.node] ?? ev.node}
                            </span>
                            <span className="ml-2 text-[#52525B]">{ev.message}</span>
                          </div>
                          <span className="text-[11px] text-[#A1A1AA] shrink-0 font-mono">
                            {ev.timestamp.slice(11, 19)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {detail.errors.length > 0 && (
                  <div className="rounded-2xl bg-amber-50/70 border border-amber-200 p-4 space-y-1">
                    <div className="text-xs font-medium text-amber-900">
                      非致命降级与提示（{detail.errors.length}）
                    </div>
                    {detail.errors.map((err, idx) => (
                      <div key={idx} className="text-xs text-amber-800">
                        - {err}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Provenance Side Drawer */}
      <ProvenanceDrawer
        open={drawerOpen}
        runId={detail.run_id}
        papers={papers}
        selectedPaperId={drawerPaperId}
        highlightChunkId={highlightChunkId}
        onSelectPaper={(pid) => {
          setDrawerPaperId(pid);
          setHighlightChunkId(null);
        }}
        onClose={() => setDrawerOpen(false)}
      />
    </div>
  );
}
