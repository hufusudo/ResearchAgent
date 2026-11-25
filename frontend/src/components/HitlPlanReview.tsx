import { useState } from "react";
import { Check, Plus, Trash2 } from "lucide-react";
import type { ResearchPlan, SubQuestion } from "../types";

interface HitlPlanReviewProps {
  plan: ResearchPlan;
  submitting: boolean;
  onApprove: (updatedPlan: ResearchPlan) => Promise<void>;
}

export function HitlPlanReview({ plan, submitting, onApprove }: HitlPlanReviewProps) {
  const [scope, setScope] = useState(plan.refined_scope);
  const [subQuestions, setSubQuestions] = useState<SubQuestion[]>(plan.sub_questions);

  const updateSq = (index: number, patch: Partial<SubQuestion>) => {
    setSubQuestions((prev) =>
      prev.map((item, idx) => (idx === index ? { ...item, ...patch } : item))
    );
  };

  const addSubQuestion = () => {
    const nextIdx = subQuestions.length + 1;
    setSubQuestions((prev) => [
      ...prev,
      {
        id: `SQ${nextIdx}`,
        title: "New Research Sub-Question",
        description: "",
        search_queries: [],
      },
    ]);
  };

  const removeSubQuestion = (index: number) => {
    if (subQuestions.length <= 1) return;
    setSubQuestions((prev) => prev.filter((_, idx) => idx !== index));
  };

  const handleConfirm = async () => {
    const cleaned = subQuestions.map((sq, idx) => ({
      id: sq.id.trim() || `SQ${idx + 1}`,
      title: sq.title.trim(),
      description: sq.description.trim(),
      search_queries: sq.search_queries.map((q) => q.trim()).filter(Boolean),
    }));
    await onApprove({
      topic: plan.topic,
      refined_scope: scope.trim(),
      sub_questions: cleaned,
    });
  };

  return (
    <div className="max-w-3xl mx-auto px-6 py-10 space-y-6">
      <div>
        <span className="inline-flex items-center px-2.5 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-800 text-xs font-medium">
          人机协同暂停 · 等待确认大纲
        </span>
        <h2 className="mt-3 text-2xl font-semibold text-[#18181B]">{plan.topic}</h2>
        <p className="mt-1.5 text-sm text-[#52525B]">
          你可以审阅或修改研究范围、子问题和英文检索词，确认后系统将按此计划开始文献检索与分析。
        </p>
      </div>

      <div className="rounded-2xl bg-white border border-[#E7E5DF] p-5 space-y-2">
        <label className="block text-xs font-medium text-[#52525B]">调研范围界定</label>
        <textarea
          rows={2}
          value={scope}
          onChange={(e) => setScope(e.target.value)}
          className="w-full rounded-xl border border-[#E5E2DC] px-3.5 py-2.5 text-sm text-[#18181B] focus:outline-none focus:border-[#3B5BDB]"
        />
      </div>

      <div className="space-y-4">
        {subQuestions.map((sq, idx) => (
          <div
            key={`${sq.id}-${idx}`}
            className="rounded-2xl bg-white border border-[#E7E5DF] p-5 space-y-3"
          >
            <div className="flex items-center justify-between gap-3">
              <span className="px-2 py-0.5 rounded-md bg-[#EEF2FF] text-[#3B5BDB] font-mono text-xs font-semibold">
                {sq.id}
              </span>
              {subQuestions.length > 1 && (
                <button
                  type="button"
                  onClick={() => removeSubQuestion(idx)}
                  className="text-xs text-[#71717A] hover:text-rose-600 inline-flex items-center gap-1 cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  移除
                </button>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="space-y-2.5">
                <label className="block text-xs text-[#52525B]">
                  子问题标题
                  <input
                    type="text"
                    value={sq.title}
                    onChange={(e) => updateSq(idx, { title: e.target.value })}
                    className="mt-1 w-full rounded-lg border border-[#E5E2DC] px-3 py-2 text-sm text-[#18181B] focus:outline-none focus:border-[#3B5BDB]"
                  />
                </label>
                <label className="block text-xs text-[#52525B]">
                  研究范围说明
                  <textarea
                    rows={2}
                    value={sq.description}
                    onChange={(e) => updateSq(idx, { description: e.target.value })}
                    className="mt-1 w-full rounded-lg border border-[#E5E2DC] px-3 py-2 text-sm text-[#18181B] focus:outline-none focus:border-[#3B5BDB]"
                  />
                </label>
              </div>

              <label className="block text-xs text-[#52525B]">
                英文检索词（每行一条）
                <textarea
                  rows={5}
                  value={sq.search_queries.join("\n")}
                  onChange={(e) =>
                    updateSq(idx, { search_queries: e.target.value.split(/\r?\n/) })
                  }
                  className="mt-1 w-full rounded-lg border border-[#E5E2DC] px-3 py-2 text-xs font-mono text-[#18181B] focus:outline-none focus:border-[#3B5BDB]"
                />
              </label>
            </div>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-between pt-2">
        <button
          type="button"
          onClick={addSubQuestion}
          className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-white border border-[#E2DFD7] text-xs font-medium text-[#52525B] hover:text-[#18181B] cursor-pointer"
        >
          <Plus className="w-3.5 h-3.5" />
          新增子问题
        </button>

        <button
          type="button"
          disabled={submitting}
          onClick={() => void handleConfirm()}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[#3B5BDB] hover:bg-[#364FC7] disabled:opacity-50 text-white text-sm font-medium cursor-pointer shadow-2xs"
        >
          <Check className="w-4 h-4" />
          {submitting ? "正在恢复执行..." : "确认计划并开始检索"}
        </button>
      </div>
    </div>
  );
}
