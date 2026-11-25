import { CheckCircle2, Loader2, RefreshCw } from "lucide-react";
import { NODE_LABELS } from "../labels";
import type { TraceEvent } from "../types";

interface LiveTraceTimelineProps {
  runId: string;
  topic: string;
  events: TraceEvent[];
}

export function LiveTraceTimeline({ runId, topic, events }: LiveTraceTimelineProps) {
  return (
    <div className="min-h-full flex flex-col items-center justify-center px-6 py-12">
      <div className="w-full max-w-2xl">
        <div className="mb-6">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-[#EEF2FF] text-[#3B5BDB] text-xs font-medium">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            正在执行调研 · {runId}
          </div>
          <h2 className="mt-3 text-xl font-semibold text-[#18181B] leading-snug">{topic}</h2>
        </div>

        <div className="rounded-2xl bg-white border border-[#E7E5DF] p-6 shadow-2xs">
          {events.length === 0 ? (
            <div className="flex items-center gap-3 text-sm text-[#52525B] py-4">
              <Loader2 className="w-4 h-4 animate-spin text-[#3B5BDB]" />
              <span>正在初始化 LangGraph 状态机并拆解研究主题...</span>
            </div>
          ) : (
            <ol className="space-y-4">
              {events.map((ev, idx) => {
                const isReflection = ev.node === "evaluator" || ev.node === "query_rewriter";
                const isLast = idx === events.length - 1;
                return (
                  <li key={`${ev.timestamp}-${idx}`} className="flex items-start gap-3">
                    <div className="mt-0.5 shrink-0">
                      {isReflection ? (
                        <RefreshCw className="w-4 h-4 text-[#3B5BDB]" />
                      ) : (
                        <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-[#18181B]">
                          {NODE_LABELS[ev.node] ?? ev.node}
                        </span>
                        {isLast && (
                          <span className="text-[11px] text-[#3B5BDB] font-medium">最新进度</span>
                        )}
                      </div>
                      <p className="text-sm text-[#52525B] mt-0.5 leading-relaxed">{ev.message}</p>
                    </div>
                  </li>
                );
              })}
            </ol>
          )}
        </div>
      </div>
    </div>
  );
}
