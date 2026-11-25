export const STATUS_LABELS: Record<string, string> = {
  running: "运行中",
  awaiting_approval: "待确认大纲",
  completed: "已完成",
  failed: "失败",
  unknown: "未知",
};

export const NODE_LABELS: Record<string, string> = {
  planner: "任务规划",
  retriever_screener: "多源检索与筛选",
  ingestion: "全文解析与建库",
  extractor: "结构化证据抽取",
  evaluator: "结果评估与反思",
  query_rewriter: "检索词改写",
  synthesizer: "综述合成",
  verifier: "引用核验",
  reviser: "核验驱动改写",
};

export const STOP_REASON_LABELS: Record<string, string> = {
  evidence_sufficient: "证据充分，进入综述合成",
  insufficient_coverage: "部分子问题覆盖不足，触发补充检索",
  max_iterations_reached: "达到最大反思轮次上限",
  marginal_gain_circuit_breaker: "边际收益熔断（新增文献不足或重合度过高）",
};

export const SOURCE_LABELS: Record<string, string> = {
  arxiv: "arXiv",
  openalex: "OpenAlex",
  semantic_scholar: "Semantic Scholar",
  local_pdf: "本地 PDF",
};

export const INGEST_LABELS: Record<string, string> = {
  full_text: "全文解析",
  abstract_only: "仅摘要",
  pending: "待处理",
  failed: "失败",
};

export const VERDICT_LABELS: Record<string, string> = {
  supported: "完全支撑",
  partially_supported: "部分支撑",
  unsupported: "未支撑",
};
