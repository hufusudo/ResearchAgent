import { useCallback, useEffect, useRef, useState } from "react";
import { Menu, X } from "lucide-react";
import {
  approveTask,
  createTask,
  deleteTask,
  getHealth,
  getSettings,
  getTask,
  listTasks,
  streamTaskEvents,
} from "./api";
import { HitlPlanReview } from "./components/HitlPlanReview";
import { LiveTraceTimeline } from "./components/LiveTraceTimeline";
import { NewSurveyComposer } from "./components/NewSurveyComposer";
import { SettingsModal } from "./components/SettingsModal";
import { Sidebar } from "./components/Sidebar";
import { SurveyWorkspace } from "./components/SurveyWorkspace";
import type {
  CreateTaskPayload,
  HealthResponse,
  ResearchPlan,
  RuntimeSettingsResponse,
  TaskDetailResponse,
  TaskSummary,
  TraceEvent,
} from "./types";

export function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [runtimeSettings, setRuntimeSettings] = useState<RuntimeSettingsResponse | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [activeDetail, setActiveDetail] = useState<TaskDetailResponse | null>(null);
  const [liveEvents, setLiveEvents] = useState<TraceEvent[]>([]);
  const [streamingRunId, setStreamingRunId] = useState<string | null>(null);
  const [streamingTopic, setStreamingTopic] = useState<string>("");
  const [submitting, setSubmitting] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const abortRef = useRef<AbortController | null>(null);

  const refreshHistory = useCallback(async () => {
    try {
      const rows = await listTasks();
      setTasks(rows);
    } catch {
      // ignore background refresh failures
    }
  }, []);

  const refreshConfig = useCallback(async () => {
    try {
      const [h, s] = await Promise.all([getHealth(), getSettings()]);
      setHealth(h);
      setRuntimeSettings(s);
      setHealthError(null);
    } catch (err) {
      setHealthError(err instanceof Error ? err.message : "后端连接失败");
    }
  }, []);

  useEffect(() => {
    void refreshConfig();
    void refreshHistory();
  }, [refreshConfig, refreshHistory]);

  const startEventStream = useCallback(
    (runId: string, topic: string) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setStreamingRunId(runId);
      setStreamingTopic(topic);
      setLiveEvents([]);

      void streamTaskEvents(
        runId,
        {
          onTrace: (event) => {
            setLiveEvents((prev) => [...prev, event]);
          },
          onStatus: () => {
            // handled once stream ends
          },
        },
        controller.signal
      )
        .catch(() => {
          // ignore abort or transient disconnect; still fetch final detail
        })
        .finally(async () => {
          if (controller.signal.aborted) return;
          try {
            const detail = await getTask(runId);
            setActiveDetail(detail);
          } finally {
            setStreamingRunId(null);
            await refreshHistory();
          }
        });
    },
    [refreshHistory]
  );

  const handleSelectTask = async (runId: string) => {
    abortRef.current?.abort();
    setMobileMenuOpen(false);
    setActiveRunId(runId);
    setStreamingRunId(null);
    const detail = await getTask(runId);
    setActiveDetail(detail);
    if (detail.status === "running") {
      startEventStream(runId, detail.topic);
    }
  };

  const handleDeleteTask = async (runId: string) => {
    await deleteTask(runId);
    if (activeRunId === runId) {
      abortRef.current?.abort();
      setActiveRunId(null);
      setActiveDetail(null);
      setStreamingRunId(null);
      setLiveEvents([]);
    }
    await refreshHistory();
  };

  const handleNewSurvey = () => {
    abortRef.current?.abort();
    setMobileMenuOpen(false);
    setActiveRunId(null);
    setActiveDetail(null);
    setStreamingRunId(null);
    setLiveEvents([]);
  };

  const handleCreateTask = async (payload: CreateTaskPayload) => {
    setSubmitting(true);
    try {
      const created = await createTask(payload);
      setActiveRunId(created.run_id);
      setActiveDetail(null);
      await refreshHistory();
      startEventStream(created.run_id, created.topic);
    } finally {
      setSubmitting(false);
    }
  };

  const handleApprovePlan = async (updatedPlan: ResearchPlan) => {
    if (!activeDetail) return;
    setSubmitting(true);
    try {
      await approveTask(activeDetail.run_id, updatedPlan);
      setActiveDetail(null);
      await refreshHistory();
      startEventStream(activeDetail.run_id, activeDetail.topic);
    } finally {
      setSubmitting(false);
    }
  };

  const enabledModels =
    runtimeSettings?.enabled_llm_models ||
    health?.enabled_llm_models ||
    (health?.llm_model ? [health.llm_model] : []);
  const defaultModel = runtimeSettings?.llm_model || health?.llm_model || "gpt-4o-mini";

  return (
    <div className="h-screen w-screen overflow-hidden flex bg-[#FAF9F6] text-[#18181B]">
      {/* Desktop Left Sidebar */}
      <div className="hidden md:flex h-full">
        <Sidebar
          tasks={tasks}
          activeRunId={activeRunId}
          health={health}
          healthError={healthError}
          onNewSurvey={handleNewSurvey}
          onSelectTask={(runId) => void handleSelectTask(runId)}
          onDeleteTask={handleDeleteTask}
          onOpenSettings={() => setSettingsOpen(true)}
        />
      </div>

      {/* Mobile Sidebar Overlay */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-40 flex md:hidden">
          <div className="h-full shadow-xl">
            <Sidebar
              tasks={tasks}
              activeRunId={activeRunId}
              health={health}
              healthError={healthError}
              onNewSurvey={handleNewSurvey}
              onSelectTask={(runId) => void handleSelectTask(runId)}
              onDeleteTask={handleDeleteTask}
              onOpenSettings={() => {
                setMobileMenuOpen(false);
                setSettingsOpen(true);
              }}
            />
          </div>
          <div className="flex-1 bg-black/25" onClick={() => setMobileMenuOpen(false)} />
        </div>
      )}

      {/* Main Workspace Container */}
      <main className="flex-1 min-w-0 h-full flex flex-col overflow-hidden">
        {/* Mobile Top Bar */}
        <div className="md:hidden px-4 py-3 border-b border-[#E7E5DF] bg-[#F3F2EE] flex items-center justify-between">
          <button
            type="button"
            onClick={() => setMobileMenuOpen((prev) => !prev)}
            className="p-1.5 rounded-lg text-[#18181B]"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
          <span className="text-sm font-semibold">ReacherAgent</span>
          <span className="w-8" />
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto">
          {streamingRunId ? (
            <LiveTraceTimeline
              runId={streamingRunId}
              topic={streamingTopic}
              events={liveEvents}
            />
          ) : activeDetail?.status === "awaiting_approval" && activeDetail.plan ? (
            <HitlPlanReview
              plan={activeDetail.plan}
              submitting={submitting}
              onApprove={handleApprovePlan}
            />
          ) : activeDetail ? (
            <SurveyWorkspace detail={activeDetail} />
          ) : (
            <NewSurveyComposer
              defaultSources={
                runtimeSettings?.academic_sources || health?.academic_sources || ["arxiv", "openalex"]
              }
              enabledModels={enabledModels}
              defaultModel={defaultModel}
              submitting={submitting}
              onSubmit={handleCreateTask}
              onOpenSettings={() => setSettingsOpen(true)}
            />
          )}
        </div>
      </main>

      <SettingsModal
        open={settingsOpen}
        settings={runtimeSettings}
        onClose={() => setSettingsOpen(false)}
        onSaved={(updated) => {
          setRuntimeSettings(updated);
          void refreshConfig();
        }}
      />
    </div>
  );
}

export default App;
