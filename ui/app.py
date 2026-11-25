"""前后端完全解耦的 Streamlit 中文测试台，仅通过 FastAPI REST + SSE 通信。"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_API_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

_STATUS_LABELS = {
    "running": "运行中",
    "awaiting_approval": "等待人工确认大纲",
    "completed": "已完成",
    "failed": "执行失败",
    "unknown": "未知状态",
}

_NODE_LABELS = {
    "planner": "任务规划 (planner)",
    "retriever_screener": "文献检索与筛选 (retriever_screener)",
    "ingestion": "全文解析与混合建库 (ingestion)",
    "extractor": "结构化证据抽取 (extractor)",
    "evaluator": "结果评估与反思 (evaluator)",
    "query_rewriter": "检索词改写 (query_rewriter)",
    "synthesizer": "综述合成 (synthesizer)",
    "verifier": "引用核验 (verifier)",
}

_STOP_REASON_LABELS = {
    "evidence_sufficient": "证据充分，进入综述合成",
    "insufficient_coverage": "部分子问题覆盖不足，触发补充检索",
    "max_iterations_reached": "达到最大反思轮次上限，停止迭代",
    "marginal_gain_circuit_breaker": "边际收益熔断（无新增文献或重合度过高）",
}

_INGEST_LABELS = {
    "full_text": "全文解析",
    "abstract_only": "仅摘要",
    "pending": "待处理",
    "failed": "解析失败",
}

_VERDICT_LABELS = {
    "supported": "完全支撑",
    "partially_supported": "部分支撑",
    "unsupported": "未支撑",
}


def _api_get(api_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(f"{api_url}{path}", params=params)
        resp.raise_for_status()
        return resp.json()


def _api_post(api_url: str, path: str, payload: dict[str, Any]) -> Any:
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(f"{api_url}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()


def _stream_run_events(api_url: str, run_id: str, trace_box: Any) -> str:
    """消费 FastAPI SSE 事件流，实时更新轨迹并等待暂停或结束。"""
    rendered_lines: list[str] = []
    final_status = "unknown"
    current_event = "message"
    data_lines: list[str] = []

    def flush_sse() -> str | None:
        nonlocal current_event, data_lines
        if not data_lines:
            current_event = "message"
            return None
        raw_data = "\n".join(data_lines)
        data_lines = []
        ev_type = current_event
        current_event = "message"
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            return None

        if ev_type == "trace":
            node = payload.get("node", "agent")
            node_label = _NODE_LABELS.get(node, node)
            msg = payload.get("message", "")
            rendered_lines.append(f"- **[{node_label}]** {msg}")
            trace_box.markdown("\n".join(rendered_lines))
        elif ev_type == "status":
            return str(payload.get("status", "unknown"))
        return None

    with httpx.Client(timeout=None) as client:
        with client.stream("GET", f"{api_url}/api/tasks/{run_id}/events") as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line == "":
                    maybe_status = flush_sse()
                    if maybe_status is not None:
                        final_status = maybe_status
                        if final_status in {"awaiting_approval", "completed", "failed"}:
                            break
                    continue
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].strip())

    maybe_status = flush_sse()
    if maybe_status is not None:
        final_status = maybe_status
    return final_status


def main() -> None:
    st.set_page_config(page_title="ReacherAgent 文献调研工作台", layout="wide")
    st.title("ReacherAgent - 自主文献调研智能体测试台")
    st.caption(
        "基于 LangGraph 状态图规划、混合 RAG（向量+BM25）检索、动态反思与引用核验；"
        "前端仅通过 FastAPI REST + SSE 接口通信。"
    )

    if "active_run_id" not in st.session_state:
        st.session_state["active_run_id"] = None
    if "pending_stream_run_id" not in st.session_state:
        st.session_state["pending_stream_run_id"] = None

    with st.sidebar:
        st.header("后端服务连接")
        api_url = st.text_input("FastAPI 服务地址", value=DEFAULT_API_URL).rstrip("/")

        try:
            health_data = _api_get(api_url, "/api/health")
            key_badge = "已配置" if health_data.get("llm_api_key_configured") else "未配置"
            emb_label = (
                "OpenAI 向量接口"
                if health_data.get("embedding_provider") == "openai"
                else "本地哈希向量 (local_hash)"
            )
            st.success(
                f"后端在线 | 模型: {health_data.get('llm_model')} | "
                f"API 密钥: {key_badge} | 向量方案: {emb_label}"
            )
        except Exception as exc:
            st.error(f"无法连接 FastAPI 后端（{exc}）。请先运行 .\\start_api.ps1。")
            st.stop()

        st.divider()
        st.header("调研参数设置")
        top_k_papers = st.slider("初始入选论文上限 (Top-K)", min_value=2, max_value=15, value=5)
        max_papers_per_query = st.slider(
            "单条检索词召回上限", min_value=2, max_value=15, value=5
        )
        max_iterations = st.slider("最大反思补充轮次", min_value=0, max_value=3, value=2)
        download_pdfs = st.checkbox("下载并解析全文 PDF（取消则仅用摘要）", value=True)
        require_plan_approval = st.checkbox(
            "开启人机协同确认（HITL：检索前暂停审核大纲）", value=False
        )
        mock_mode = st.checkbox("离线模拟模式（不消耗 Token）", value=False)

        st.divider()
        st.header("历史调研任务")
        tasks_list: list[dict[str, Any]] = []
        try:
            tasks_list = _api_get(api_url, "/api/tasks")
        except Exception as exc:
            st.warning(f"获取历史任务列表失败：{exc}")

        if tasks_list:
            options = {
                (
                    f"{t['run_id']} | "
                    f"[{_STATUS_LABELS.get(t['status'], t['status'])}] "
                    f"{t['topic'][:28]}"
                ): t["run_id"]
                for t in tasks_list
            }
            labels = ["（请选择要查看的历史任务）"] + list(options.keys())
            chosen_label = st.selectbox("选择历史任务", options=labels)
            if chosen_label != "（请选择要查看的历史任务）":
                chosen_id = options[chosen_label]
                if st.button("加载选中任务", use_container_width=True):
                    st.session_state["active_run_id"] = chosen_id
                    st.session_state["pending_stream_run_id"] = None
                    st.rerun()
        else:
            st.info("暂无历史调研记录。")

    # 新建文献调研任务卡片
    with st.container(border=True):
        st.subheader("发起新的文献调研")
        topic = st.text_input(
            "输入研究主题（支持中文或英文，系统会自动拆解为英文学术检索计划）",
            placeholder="例如：基于图检索与多智能体协同的复杂问答方法研究 / Agentic RAG for academic literature review",
        )
        col_btn, col_note = st.columns([1, 3])
        with col_btn:
            launch_clicked = st.button("开始文献调研", type="primary", use_container_width=True)
        with col_note:
            st.caption(
                "提示：勾选左侧“开启人机协同确认 (HITL)”后，智能体会在拆解完子问题与检索词后暂停，"
                "供你人工修改后再开始文献检索。"
            )

        if launch_clicked:
            if len(topic.strip()) < 2:
                st.warning("请输入有效的研究主题（至少 2 个字符）。")
            else:
                created = _api_post(
                    api_url,
                    "/api/tasks",
                    {
                        "topic": topic.strip(),
                        "top_k_papers": top_k_papers,
                        "max_papers_per_query": max_papers_per_query,
                        "max_iterations": max_iterations,
                        "download_pdfs": download_pdfs,
                        "require_plan_approval": require_plan_approval,
                        "mock": mock_mode,
                    },
                )
                st.session_state["active_run_id"] = created["run_id"]
                st.session_state["pending_stream_run_id"] = created["run_id"]
                st.rerun()

    # 实时消费 SSE 状态流
    stream_id = st.session_state.get("pending_stream_run_id")
    if stream_id:
        with st.status(f"正在执行任务 {stream_id} ...", expanded=True) as status_widget:
            trace_box = st.empty()
            final_status = _stream_run_events(api_url, stream_id, trace_box)
            if final_status == "completed":
                status_widget.update(label=f"任务 {stream_id} 执行完成。", state="complete")
            elif final_status == "awaiting_approval":
                status_widget.update(
                    label=f"任务 {stream_id} 已暂停，等待人工确认调研大纲。",
                    state="complete",
                )
            else:
                status_widget.update(
                    label=f"任务 {stream_id} 已结束（{_STATUS_LABELS.get(final_status, final_status)}）。",
                    state="error",
                )
        st.session_state["pending_stream_run_id"] = None

    active_run_id = st.session_state.get("active_run_id")
    if not active_run_id:
        return

    try:
        detail = _api_get(api_url, f"/api/tasks/{active_run_id}")
    except Exception as exc:
        st.error(f"无法获取任务 {active_run_id} 的详情：{exc}")
        return

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("任务编号 (Run ID)", detail["run_id"])
    m2.metric("当前状态", _STATUS_LABELS.get(detail["status"], detail["status"]))
    m3.metric("入选论文数", len(detail.get("selected_papers") or []))
    m4.metric("反思补充轮次", len(detail.get("rewrite_history") or []))

    if detail.get("error_message"):
        st.error(f"任务执行出错：{detail['error_message']}")

    # HITL 人工审核与修改大纲表单
    if detail["status"] == "awaiting_approval" and detail.get("plan"):
        plan = detail["plan"]
        st.subheader("人机协同（HITL）：审核与微调调研计划")
        st.info("你可以直接修改研究范围、子问题表述或补充英文检索词，确认后智能体将从断点继续执行。")

        with st.form("hitl_plan_form"):
            edited_scope = st.text_area("调研范围界定 (Refined Scope)", value=plan.get("refined_scope", ""))
            edited_sqs: list[dict[str, Any]] = []
            for idx, sq in enumerate(plan.get("sub_questions") or []):
                st.markdown(f"**子问题 {sq['id']}**")
                c1, c2 = st.columns(2)
                with c1:
                    sq_title = st.text_input(
                        f"子问题标题 ({sq['id']})", value=sq.get("title", ""), key=f"title_{idx}"
                    )
                    sq_desc = st.text_area(
                        f"子问题描述 ({sq['id']})",
                        value=sq.get("description", ""),
                        key=f"desc_{idx}",
                    )
                with c2:
                    sq_queries_raw = st.text_area(
                        f"英文检索词（每行一条，{sq['id']}）",
                        value="\n".join(sq.get("search_queries") or []),
                        key=f"queries_{idx}",
                    )
                edited_sqs.append(
                    {
                        "id": sq["id"],
                        "title": sq_title.strip(),
                        "description": sq_desc.strip(),
                        "search_queries": [
                            q.strip() for q in sq_queries_raw.splitlines() if q.strip()
                        ],
                    }
                )

            approved = st.form_submit_button("确认计划并继续执行", type="primary")
            if approved:
                updated_plan = {
                    "topic": plan.get("topic", detail["topic"]),
                    "refined_scope": edited_scope.strip(),
                    "sub_questions": edited_sqs,
                }
                _api_post(api_url, f"/api/tasks/{active_run_id}/approve", {"plan": updated_plan})
                st.session_state["pending_stream_run_id"] = active_run_id
                st.rerun()
        return

    # 结果展示四栏标签页
    tab_survey, tab_matrix, tab_reflect, tab_trace = st.tabs(
        [
            "综述初稿与引用溯源",
            "跨文献证据矩阵",
            "反思过程与引用核验",
            "入选文献与执行轨迹",
        ]
    )

    with tab_survey:
        col_report, col_prov = st.columns([2, 1])
        with col_report:
            st.download_button(
                "导出 Markdown 综述初稿",
                data=detail.get("draft_markdown", ""),
                file_name=f"{active_run_id}_report.md",
                mime="text/markdown",
            )
            st.markdown(detail.get("draft_markdown") or "_暂未生成综述正文。_")

        with col_prov:
            st.subheader("文献原文溯源检查器")
            papers = detail.get("selected_papers") or []
            if not papers:
                st.info("暂无入选论文。")
            else:
                paper_map = {f"[{p['paper_id']}] {p['title'][:50]}": p for p in papers}
                chosen_paper_label = st.selectbox("选择要核对原文的文献", list(paper_map.keys()))
                chosen_paper = paper_map[chosen_paper_label]
                st.markdown(f"**论文标题：** {chosen_paper['title']}")
                ingest_text = _INGEST_LABELS.get(
                    chosen_paper.get("ingest_status", ""), chosen_paper.get("ingest_status", "")
                )
                st.markdown(
                    f"**年份：** {chosen_paper.get('year')} | "
                    f"**来源：** {chosen_paper.get('source')} | "
                    f"**入库方式：** `{ingest_text}`"
                )
                st.markdown(f"**原文链接：** {chosen_paper.get('url')}")
                try:
                    chunks_resp = _api_get(
                        api_url,
                        f"/api/tasks/{active_run_id}/chunks",
                        params={"paper_id": chosen_paper["paper_id"]},
                    )
                    chunks = chunks_resp.get("chunks") or []
                    st.caption(f"共索引 {len(chunks)} 个原文分块（默认展示前 15 个）")
                    for chunk in chunks[:15]:
                        with st.expander(
                            f"{chunk['chunk_id']} | 章节：{chunk['section']}（第 {chunk['page']} 页）"
                        ):
                            st.write(chunk["text"])
                except Exception as exc:
                    st.warning(f"加载原文分块失败：{exc}")

    with tab_matrix:
        st.subheader("按子问题归类的跨文献结构化证据矩阵")
        plan = detail.get("plan") or {}
        ev_map = detail.get("evidence_by_subquestion") or {}
        for sq in plan.get("sub_questions") or []:
            sq_id = sq["id"]
            st.markdown(f"#### [{sq_id}] {sq['title']}")
            rows = ev_map.get(sq_id) or []
            if rows:
                formatted_rows = [
                    {
                        "论文编号": item["paper_id"],
                        "核心方法": item["method"],
                        "数据集 / 实验场景": item["dataset_or_setting"],
                        "关键结论与指标": item["key_findings"],
                        "主要局限性": item["limitations"],
                        "原文溯源分块": ", ".join(item.get("source_chunk_ids") or []),
                    }
                    for item in rows
                ]
                st.dataframe(formatted_rows, use_container_width=True, hide_index=True)
            else:
                st.info(f"子问题 {sq_id} 暂未抽取出结构化证据。")

    with tab_reflect:
        st.subheader("评估器（Evaluator）与检索词改写轨迹")
        eval_hist = detail.get("evaluation_history") or []
        if eval_hist:
            for ev in eval_hist:
                with st.container(border=True):
                    stop_label = _STOP_REASON_LABELS.get(
                        ev.get("stop_reason", ""), ev.get("stop_reason", "")
                    )
                    need_more_label = "是" if ev.get("needs_more_retrieval") else "否"
                    weak_label = ", ".join(ev.get("weak_subquestion_ids") or []) or "无"
                    st.markdown(
                        f"**第 {ev['iteration']} 轮评估** | 决策原因：`{stop_label}` | "
                        f"是否继续检索：`{need_more_label}` | 薄弱子问题：`{weak_label}`"
                    )
                    if ev.get("reasoning"):
                        st.caption(ev["reasoning"])
                    sq_evals = ev.get("subquestion_evaluations") or []
                    if sq_evals:
                        formatted_sq_evals = [
                            {
                                "子问题": item["sub_question_id"],
                                "证据条数": item.get("evidence_count", 0),
                                "涉及论文数": item.get("unique_paper_count", 0),
                                "覆盖度评分": item.get("coverage_score", 0.0),
                                "是否充分": "是" if item.get("sufficient") else "否",
                                "识别缺口": "；".join(item.get("gaps") or []) or "无",
                            }
                            for item in sq_evals
                        ]
                        st.dataframe(formatted_sq_evals, use_container_width=True, hide_index=True)
        else:
            st.info("无评估迭代记录（例如早期阶段一运行记录）。")

        rewrite_hist = detail.get("rewrite_history") or []
        if rewrite_hist:
            st.markdown("#### 改写后的补充检索词")
            for rw in rewrite_hist:
                st.markdown(f"**第 {rw['iteration']} 轮改写**：{rw.get('reasoning', '')}")
                if rw.get("rewritten_queries"):
                    formatted_rw = [
                        {
                            "目标子问题": item["sub_question_id"],
                            "新检索词": item["query"],
                            "改写理由": item.get("rationale", ""),
                        }
                        for item in rw["rewritten_queries"]
                    ]
                    st.dataframe(formatted_rw, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("引用核验（Citation Verifier）审计报告")
        ver = detail.get("verification_report")
        if ver:
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("参考文献真实率", f"{ver['reference_validity_rate'] * 100:.1f}%")
            v2.metric("论断原文支撑率", f"{ver['claim_support_rate'] * 100:.1f}%")
            v3.metric("已核验论断数", f"{ver['checked_claims']}/{ver['total_cited_claims']}")
            v4.metric("非法引用编号数", len(ver.get("invalid_citations") or []))
            if ver.get("details"):
                formatted_ver = [
                    {
                        "待核验论断": item["claim"],
                        "引用文献": ", ".join(item.get("cited_paper_ids") or []),
                        "匹配原文分块": ", ".join(item.get("matched_chunk_ids") or []),
                        "核验结论": _VERDICT_LABELS.get(item["verdict"], item["verdict"]),
                        "说明": item.get("reason", ""),
                    }
                    for item in ver["details"]
                ]
                st.dataframe(formatted_ver, use_container_width=True, hide_index=True)
        else:
            st.info("无引用核验记录。")

    with tab_trace:
        st.subheader("入选英文文献列表")
        papers = detail.get("selected_papers") or []
        if papers:
            paper_rows = [
                {
                    "编号": p["paper_id"],
                    "年份": p.get("year"),
                    "相关性得分": p.get("relevance_score"),
                    "来源": p.get("source"),
                    "入库状态": _INGEST_LABELS.get(
                        p.get("ingest_status", ""), p.get("ingest_status", "")
                    ),
                    "论文标题": p["title"],
                    "链接": p["url"],
                }
                for p in papers
            ]
            st.dataframe(paper_rows, use_container_width=True, hide_index=True)

        st.subheader("智能体节点执行轨迹")
        trace = detail.get("trace") or []
        if trace:
            formatted_trace = [
                {
                    "阶段": t.get("step", ""),
                    "执行节点": _NODE_LABELS.get(t.get("node", ""), t.get("node", "")),
                    "执行说明": t.get("message", ""),
                    "时间戳": t.get("timestamp", ""),
                }
                for t in trace
            ]
            st.dataframe(formatted_trace, use_container_width=True, hide_index=True)

        errors = detail.get("errors") or []
        if errors:
            st.subheader("非致命警告 / 降级记录")
            for err in errors:
                st.warning(err)


if __name__ == "__main__":
    main()
