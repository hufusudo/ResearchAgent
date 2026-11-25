import { useState } from "react";
import { BookOpen, Plus, Settings, Sparkles, Trash2 } from "lucide-react";
import { STATUS_LABELS } from "../labels";
import type { HealthResponse, TaskSummary } from "../types";

interface SidebarProps {
  tasks: TaskSummary[];
  activeRunId: string | null;
  health: HealthResponse | null;
  healthError: string | null;
  onNewSurvey: () => void;
  onSelectTask: (runId: string) => void;
  onDeleteTask: (runId: string) => Promise<void>;
  onOpenSettings: () => void;
}

function statusDotClass(status: string): string {
  switch (status) {
    case "completed":
      return "bg-emerald-500";
    case "running":
      return "bg-indigo-500 animate-pulse";
    case "awaiting_approval":
      return "bg-amber-500";
    case "failed":
      return "bg-rose-500";
    default:
      return "bg-zinc-400";
  }
}

export function Sidebar({
  tasks,
  activeRunId,
  health,
  healthError,
  onNewSurvey,
  onSelectTask,
  onDeleteTask,
  onOpenSettings,
}: SidebarProps) {
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const handleConfirmDelete = async (runId: string) => {
    setDeletingId(runId);
    try {
      await onDeleteTask(runId);
      setConfirmingId(null);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <aside className="w-[268px] shrink-0 bg-[#F3F2EE] border-r border-[#E7E5DF] flex flex-col h-full select-none">
      {/* Brand Header */}
      <div className="px-4 pt-5 pb-4 flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-[#3B5BDB] text-white flex items-center justify-center shadow-xs">
          <BookOpen className="w-4 h-4" />
        </div>
        <div>
          <div className="text-sm font-semibold text-[#18181B] tracking-tight leading-none">
            ReacherAgent
          </div>
          <div className="text-[11px] text-[#71717A] mt-1">自主文献调研智能体</div>
        </div>
      </div>

      {/* New Survey CTA */}
      <div className="px-3 pb-3">
        <button
          type="button"
          onClick={onNewSurvey}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl bg-white border border-[#E2DFD7] text-sm font-medium text-[#18181B] hover:border-[#3B5BDB]/50 hover:text-[#3B5BDB] shadow-2xs transition-colors cursor-pointer"
        >
          <Plus className="w-4 h-4" />
          新建调研任务
        </button>
      </div>

      {/* Task History */}
      <div className="px-4 pt-1 pb-2 text-[11px] font-medium text-[#71717A] uppercase tracking-wider">
        历史调研记录 ({tasks.length})
      </div>
      <div className="flex-1 overflow-y-auto px-2 space-y-1 pb-4">
        {tasks.length === 0 ? (
          <div className="px-3 py-6 text-xs text-[#71717A] text-center">
            暂无调研记录，点击上方发起新任务。
          </div>
        ) : (
          tasks.map((task) => {
            const isActive = task.run_id === activeRunId;
            const isConfirming = confirmingId === task.run_id;
            const isDeleting = deletingId === task.run_id;
            const canDelete = task.status !== "running";

            return (
              <div
                key={task.run_id}
                className={`relative rounded-xl transition-colors group ${
                  isActive
                    ? "bg-white shadow-2xs border border-[#E2DFD7]"
                    : "hover:bg-[#EAE8E2]/70 border border-transparent"
                }`}
              >
                <div className="flex items-start gap-2 px-3 py-2.5">
                  <button
                    type="button"
                    onClick={() => {
                      setConfirmingId(null);
                      onSelectTask(task.run_id);
                    }}
                    className="flex-1 min-w-0 text-left flex items-start gap-2 cursor-pointer"
                  >
                    <span
                      className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${statusDotClass(task.status)}`}
                      title={STATUS_LABELS[task.status] ?? task.status}
                    />
                    <div className="min-w-0 flex-1">
                      <div
                        className={`text-[13px] leading-snug line-clamp-2 ${
                          isActive ? "font-medium text-[#18181B]" : "text-[#3F3F46]"
                        }`}
                      >
                        {task.topic}
                      </div>
                      <div className="mt-1 flex items-center gap-1.5 text-[11px] text-[#71717A]">
                        <span>{STATUS_LABELS[task.status] ?? task.status}</span>
                        {task.paper_count > 0 && (
                          <>
                            <span>·</span>
                            <span>{task.paper_count} 篇</span>
                          </>
                        )}
                        {task.iteration > 0 && (
                          <>
                            <span>·</span>
                            <span>{task.iteration} 轮反思</span>
                          </>
                        )}
                      </div>
                    </div>
                  </button>

                  {canDelete && !isConfirming && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setConfirmingId(task.run_id);
                      }}
                      className="opacity-0 group-hover:opacity-100 focus:opacity-100 p-1 rounded-md text-[#71717A] hover:text-rose-600 hover:bg-rose-50 transition-all cursor-pointer shrink-0"
                      title="删除该条记录"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>

                {isConfirming && (
                  <div className="px-3 pb-2.5 pt-0.5 flex items-center justify-between gap-2 text-[11px]">
                    <span className="text-rose-700 font-medium">确认彻底删除？</span>
                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        disabled={isDeleting}
                        onClick={() => setConfirmingId(null)}
                        className="px-2 py-0.5 rounded bg-[#F3F2EE] text-[#52525B] hover:text-[#18181B] cursor-pointer"
                      >
                        取消
                      </button>
                      <button
                        type="button"
                        disabled={isDeleting}
                        onClick={() => void handleConfirmDelete(task.run_id)}
                        className="px-2 py-0.5 rounded bg-rose-600 text-white hover:bg-rose-700 disabled:opacity-50 cursor-pointer"
                      >
                        {isDeleting ? "删除中" : "删除"}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Compact Runtime Status & Settings Footer */}
      <div className="p-3 border-t border-[#E7E5DF] bg-[#EFECE6]/70">
        <div className="flex items-center justify-between gap-2">
          {health ? (
            <div className="flex items-center gap-2 text-xs text-[#52525B] min-w-0 flex-1">
              <span
                className={`w-2 h-2 rounded-full shrink-0 ${
                  health.llm_api_key_configured ? "bg-emerald-500" : "bg-amber-500"
                }`}
              />
              <div className="min-w-0 flex-1">
                <div className="font-medium text-[#18181B] truncate flex items-center gap-1">
                  <Sparkles className="w-3 h-3 text-[#3B5BDB] shrink-0" />
                  {health.llm_model}
                </div>
                <div className="text-[11px] text-[#71717A] truncate mt-0.5">
                  {health.academic_sources.join(" + ")}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-xs text-rose-600 truncate flex-1">
              {healthError || "后端未连接"}
            </div>
          )}

          <button
            type="button"
            onClick={onOpenSettings}
            className="p-2 rounded-lg bg-white border border-[#E2DFD7] text-[#52525B] hover:text-[#3B5BDB] hover:border-[#3B5BDB]/50 transition-colors cursor-pointer shrink-0 shadow-2xs"
            title="模型与 API 设置"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>
    </aside>
  );
}
