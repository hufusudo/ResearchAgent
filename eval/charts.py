"""Pure-Python publication-grade Chinese SVG chart generator for graduation thesis."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

PALETTE = {
    "bg": "#FAF9F6",
    "card": "#FFFFFF",
    "border": "#E5E0D8",
    "text": "#1F2937",
    "muted": "#6B7280",
    "grid": "#ECE7DF",
    "slate": "#94A3B8",
    "amber": "#D97706",
    "teal": "#0D9488",
    "blue": "#2563EB",
    "indigo": "#4F46E5",
    "coral": "#E11D48",
    "emerald": "#059669",
}

FONT_FAMILY = "Microsoft YaHei, PingFang SC, SimHei, Noto Sans CJK SC, Segoe UI, sans-serif"


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _text_width(text: str, font_size: float = 11.5) -> float:
    """Estimate rendered text width in px accounting for full-width CJK characters."""
    width = 0.0
    for ch in text:
        if ord(ch) > 127:
            width += font_size * 1.02
        elif ch.isupper():
            width += font_size * 0.66
        else:
            width += font_size * 0.56
    return width


def _svg_header(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT_FAMILY}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{PALETTE["bg"]}" rx="10"/>',
        f'<rect x="12" y="12" width="{width - 24}" height="{height - 24}" fill="{PALETTE["card"]}" '
        f'stroke="{PALETTE["border"]}" stroke-width="1" rx="8"/>',
        f'<text x="32" y="42" font-size="18" font-weight="700" fill="{PALETTE["text"]}">{_esc(title)}</text>',
        f'<text x="32" y="63" font-size="12" fill="{PALETTE["muted"]}">{_esc(subtitle)}</text>',
    ]


def _draw_legend(
    lines: list[str],
    x: int,
    y: int,
    items: list[tuple[str, str]],
    max_x: int = 880,
) -> int:
    """Draw legend items with CJK-aware spacing and automatic line wrapping; returns final y."""
    cur_x = float(x)
    cur_y = float(y)
    for label, color in items:
        item_w = 18.0 + _text_width(label, 11.5) + 22.0
        if cur_x > x and cur_x + item_w > max_x:
            cur_x = float(x)
            cur_y += 20.0
        lines.append(
            f'<rect x="{cur_x:.1f}" y="{cur_y - 10:.1f}" width="12" height="12" fill="{color}" rx="2"/>'
        )
        lines.append(
            f'<text x="{cur_x + 17:.1f}" y="{cur_y:.1f}" font-size="11.5" fill="{PALETTE["text"]}">'
            f'{_esc(label)}</text>'
        )
        cur_x += item_w
    return int(cur_y)


def _draw_grouped_bars(
    lines: list[str],
    *,
    plot_x: int,
    plot_y: int,
    plot_w: int,
    plot_h: int,
    groups: list[str],
    series: list[tuple[str, str, list[float], list[str]]],
    y_max: float = 100.0,
    y_ticks: list[float] | None = None,
    y_suffix: str = "%",
) -> None:
    ticks = y_ticks or [0, 20, 40, 60, 80, 100]
    for tv in ticks:
        ty = plot_y + plot_h - (tv / y_max) * plot_h
        lines.append(
            f'<line x1="{plot_x}" y1="{ty:.1f}" x2="{plot_x + plot_w}" y2="{ty:.1f}" '
            f'stroke="{PALETTE["grid"]}" stroke-width="1"/>'
        )
        label = f"{tv:g}{y_suffix}"
        lines.append(
            f'<text x="{plot_x - 8}" y="{ty + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="{PALETTE["muted"]}">{_esc(label)}</text>'
        )

    lines.append(
        f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" '
        f'stroke="{PALETTE["border"]}" stroke-width="1.5"/>'
    )

    n_groups = max(1, len(groups))
    n_series = max(1, len(series))
    group_w = plot_w / n_groups
    inner_w = group_w * 0.72
    bar_w = max(10.0, min(36.0, (inner_w - (n_series - 1) * 4) / n_series))
    total_bars_w = n_series * bar_w + (n_series - 1) * 4

    for g_idx, g_label in enumerate(groups):
        gx = plot_x + g_idx * group_w + (group_w - total_bars_w) / 2.0
        for s_idx, (_, color, values, labels) in enumerate(series):
            val = max(0.0, min(y_max, values[g_idx]))
            bh = (val / y_max) * plot_h
            bx = gx + s_idx * (bar_w + 4)
            by = plot_y + plot_h - bh
            lines.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
                f'fill="{color}" rx="3"/>'
            )
            val_label = labels[g_idx] if g_idx < len(labels) else f"{val:.1f}"
            lines.append(
                f'<text x="{bx + bar_w / 2:.1f}" y="{max(plot_y + 10, by - 5):.1f}" '
                f'text-anchor="middle" font-size="10" font-weight="600" fill="{PALETTE["text"]}">'
                f'{_esc(val_label)}</text>'
            )
        center_x = plot_x + g_idx * group_w + group_w / 2.0
        lines.append(
            f'<text x="{center_x:.1f}" y="{plot_y + plot_h + 22}" text-anchor="middle" '
            f'font-size="11.5" font-weight="600" fill="{PALETTE["text"]}">{_esc(g_label)}</text>'
        )


def generate_fig1_system_comparison(summary: dict[str, Any], out_path: Path) -> None:
    sys_agg = summary.get("system_comparison_agg", {})
    systems = [
        ("direct_llm", "Direct LLM（直接生成）", PALETTE["slate"]),
        ("naive_rag", "Naive RAG（单轮向量检索）", PALETTE["amber"]),
        ("reacher_agent", "ReacherAgent（本文方法）", PALETTE["teal"]),
    ]
    groups = ["引用真实率", "声明支撑率", "子问题覆盖率", "盲评质量(×20)"]
    series = []
    for sys_key, label, color in systems:
        row = sys_agg.get(sys_key, {})
        rv = float(row.get("reference_validity_rate_mean", 0.0)) * 100.0
        cs = float(row.get("claim_support_rate_mean", 0.0)) * 100.0
        sq = float(row.get("subquestion_coverage_rate_mean", 0.0)) * 100.0
        rb = float(row.get("rubric_overall_mean", 0.0))
        rb_scaled = rb * 20.0
        vals = [rv, cs, sq, rb_scaled]
        lbls = [f"{rv:.1f}%", f"{cs:.1f}%", f"{sq:.1f}%", f"{rb:.2f}/5"]
        series.append((label, color, vals, lbls))

    n_topics = summary.get("topic_count", 0)
    lines = _svg_header(
        880,
        440,
        "系统级基线对比实验结果（Direct LLM vs. Naive RAG vs. ReacherAgent）",
        f"基于 {n_topics} 个中英文 CS/AI 基准主题均值；盲评综合质量按 ×20 映射至百分制并标注原始 5 分制得分",
    )
    _draw_legend(lines, 34, 90, [(lbl, col) for _, lbl, col in systems], max_x=840)
    _draw_grouped_bars(
        lines,
        plot_x=72,
        plot_y=120,
        plot_w=768,
        plot_h=258,
        groups=groups,
        series=series,
        y_max=100.0,
        y_ticks=[0, 20, 40, 60, 80, 100],
        y_suffix="%",
    )
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def generate_fig2_retrieval_ablation(summary: dict[str, Any], out_path: Path) -> None:
    ret_agg = summary.get("ablation_retrieval_agg", {})
    modes = [
        ("dense_only", "Dense-only（纯向量）", PALETTE["slate"]),
        ("bm25_only", "BM25-only（纯词法）", PALETTE["amber"]),
        ("hybrid_rrf", "Hybrid RRF（混合融合）", PALETTE["blue"]),
        ("hybrid_diversity", "Hybrid + 多样性上限（max=3）", PALETTE["teal"]),
    ]

    lines = _svg_header(
        960,
        450,
        "检索策略四联消融实验对比（Top-8 检索结果）",
        "左图：排序相关性与术语召回指标（nDCG@8 / Recall@8 / MRR / 术语命中率）｜右图：跨文献多样性与防垄断指标",
    )
    _draw_legend(lines, 34, 90, [(lbl, col) for _, lbl, col in modes], max_x=925)

    left_groups = ["nDCG@8", "Recall@8", "MRR", "术语命中率"]
    left_series = []
    for mk, lbl, col in modes:
        r = ret_agg.get(mk, {})
        vals = [
            float(r.get("ndcg_at_k_mean", 0.0)) * 100.0,
            float(r.get("recall_at_k_mean", 0.0)) * 100.0,
            float(r.get("mrr_mean", 0.0)) * 100.0,
            float(r.get("term_hit_rate_mean", 0.0)) * 100.0,
        ]
        left_series.append((lbl, col, vals, [f"{v / 100:.2f}" for v in vals]))

    _draw_grouped_bars(
        lines,
        plot_x=65,
        plot_y=124,
        plot_w=490,
        plot_h=254,
        groups=left_groups,
        series=left_series,
        y_max=100.0,
        y_ticks=[0, 25, 50, 75, 100],
        y_suffix="%",
    )

    right_groups = ["独立论文数(×10)", "单篇最高占比(%)"]
    right_series = []
    for mk, lbl, col in modes:
        r = ret_agg.get(mk, {})
        up = float(r.get("unique_papers_at_k_mean", 0.0))
        ms = float(r.get("max_single_paper_share_mean", 0.0)) * 100.0
        vals = [min(100.0, up * 10.0), ms]
        right_series.append((lbl, col, vals, [f"{up:.2f}", f"{ms:.1f}%"]))

    _draw_grouped_bars(
        lines,
        plot_x=615,
        plot_y=124,
        plot_w=308,
        plot_h=254,
        groups=right_groups,
        series=right_series,
        y_max=100.0,
        y_ticks=[0, 25, 50, 75, 100],
        y_suffix="",
    )
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def generate_fig3_reflection_ablation(summary: dict[str, Any], out_path: Path) -> None:
    ref_agg = summary.get("ablation_reflection_agg", {})
    it0 = ref_agg.get("iter0", {})
    it2 = ref_agg.get("iter2_full", {})

    groups = [
        "评估器均分(×10)",
        "多篇支撑比例",
        "正文覆盖率",
        "声明支撑率",
        "盲评质量(×20)",
    ]
    vals0 = [
        float(it0.get("evaluator_avg_coverage_mean", 0.0)) * 10.0,
        float(it0.get("multi_paper_sq_ratio_mean", 0.0)) * 100.0,
        float(it0.get("draft_sq_coverage_rate_mean", 0.0)) * 100.0,
        float(it0.get("claim_support_rate_mean", 0.0)) * 100.0,
        float(it0.get("rubric_overall_mean", 0.0)) * 20.0,
    ]
    lbls0 = [
        f"{vals0[0]/10:.2f}",
        f"{vals0[1]:.1f}%",
        f"{vals0[2]:.1f}%",
        f"{vals0[3]:.1f}%",
        f"{vals0[4]/20:.2f}",
    ]
    vals2 = [
        float(it2.get("evaluator_avg_coverage_mean", 0.0)) * 10.0,
        float(it2.get("multi_paper_sq_ratio_mean", 0.0)) * 100.0,
        float(it2.get("draft_sq_coverage_rate_mean", 0.0)) * 100.0,
        float(it2.get("claim_support_rate_mean", 0.0)) * 100.0,
        float(it2.get("rubric_overall_mean", 0.0)) * 20.0,
    ]
    lbls2 = [
        f"{vals2[0]/10:.2f}",
        f"{vals2[1]:.1f}%",
        f"{vals2[2]:.1f}%",
        f"{vals2[3]:.1f}%",
        f"{vals2[4]/20:.2f}",
    ]

    lines = _svg_header(
        880,
        440,
        "动态反思闭环消融实验对比（单轮无反思 vs. 有界反思迭代）",
        "基于相同首轮检索状态的配对控制变量实验（max_iterations=0 对比 max_iterations=2）",
    )
    _draw_legend(
        lines,
        34,
        90,
        [
            ("无反思单轮（max_iterations=0）", PALETTE["slate"]),
            ("完整反思闭环（max_iterations=2）", PALETTE["teal"]),
        ],
        max_x=840,
    )
    _draw_grouped_bars(
        lines,
        plot_x=72,
        plot_y=120,
        plot_w=768,
        plot_h=258,
        groups=groups,
        series=[
            ("max_iterations=0", PALETTE["slate"], vals0, lbls0),
            ("max_iterations=2", PALETTE["teal"], vals2, lbls2),
        ],
        y_max=100.0,
        y_ticks=[0, 20, 40, 60, 80, 100],
        y_suffix="",
    )
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def generate_fig4_matrix_and_revision(summary: dict[str, Any], out_path: Path) -> None:
    mat_agg = summary.get("ablation_matrix_agg", {})
    dc = mat_agg.get("direct_chunks", {})
    em = mat_agg.get("evidence_matrix", {})
    rev_agg = summary.get("ablation_verifier_revision_agg", {})

    lines = _svg_header(
        960,
        450,
        "结构化证据矩阵先行与核验驱动改写闭环消融结果",
        "左图：盲评四维质量对比（直接 Chunk 合成 vs. 证据矩阵先行，1–5 分）｜右图：Verifier 轻量改写前后忠实度跃升",
    )
    _draw_legend(
        lines,
        34,
        90,
        [
            ("基线组 / 改写前", PALETTE["slate"]),
            ("证据矩阵组 / 改写后", PALETTE["indigo"]),
        ],
        max_x=925,
    )

    dims = ["对比深度", "非流水账", "事实严谨", "结构完整", "综合得分"]
    dc_vals = [
        float(dc.get("comparative_depth_mean", 0.0)),
        float(dc.get("non_sequential_synthesis_mean", 0.0)),
        float(dc.get("factual_rigor_mean", 0.0)),
        float(dc.get("structural_completeness_mean", 0.0)),
        float(dc.get("overall_score_mean", 0.0)),
    ]
    em_vals = [
        float(em.get("comparative_depth_mean", 0.0)),
        float(em.get("non_sequential_synthesis_mean", 0.0)),
        float(em.get("factual_rigor_mean", 0.0)),
        float(em.get("structural_completeness_mean", 0.0)),
        float(em.get("overall_score_mean", 0.0)),
    ]
    _draw_grouped_bars(
        lines,
        plot_x=60,
        plot_y=124,
        plot_w=515,
        plot_h=254,
        groups=dims,
        series=[
            ("直接 Chunk 合成", PALETTE["slate"], dc_vals, [f"{v:.2f}" for v in dc_vals]),
            ("证据矩阵先行", PALETTE["indigo"], em_vals, [f"{v:.2f}" for v in em_vals]),
        ],
        y_max=5.0,
        y_ticks=[0, 1, 2, 3, 4, 5],
        y_suffix="",
    )

    b_sup = float(rev_agg.get("before_claim_support_rate_mean", 0.0)) * 100.0
    a_sup = float(rev_agg.get("after_claim_support_rate_mean", 0.0)) * 100.0
    b_val = float(rev_agg.get("before_validity_rate_mean", 0.0)) * 100.0
    a_val = float(rev_agg.get("after_validity_rate_mean", 0.0)) * 100.0

    _draw_grouped_bars(
        lines,
        plot_x=640,
        plot_y=124,
        plot_w=280,
        plot_h=254,
        groups=["引用真实率", "声明支撑率"],
        series=[
            ("改写前", PALETTE["slate"], [b_val, b_sup], [f"{b_val:.1f}%", f"{b_sup:.1f}%"]),
            ("改写后", PALETTE["indigo"], [a_val, a_sup], [f"{a_val:.1f}%", f"{a_sup:.1f}%"]),
        ],
        y_max=100.0,
        y_ticks=[0, 25, 50, 75, 100],
        y_suffix="%",
    )
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def generate_fig5_latency_and_cost(summary: dict[str, Any], out_path: Path) -> None:
    sys_agg = summary.get("system_comparison_agg", {})
    phase3_serial = float(summary.get("phase3_serial_avg_seconds", 565.6))

    d_sec = float(sys_agg.get("direct_llm", {}).get("elapsed_seconds_mean", 0.0))
    n_sec = float(sys_agg.get("naive_rag", {}).get("elapsed_seconds_mean", 0.0))
    r_sec = float(sys_agg.get("reacher_agent", {}).get("elapsed_seconds_mean", 0.0))

    node_means: dict[str, float] = summary.get("reacher_node_timings_mean", {})
    ordered_nodes = [
        ("planner", "任务规划", "#6366F1"),
        ("retriever_screener", "多源检索筛选", "#0EA5E9"),
        ("ingestion", "全文解析入库", "#14B8A6"),
        ("extractor", "证据抽取", "#10B981"),
        ("evaluator", "覆盖度评估", "#F59E0B"),
        ("query_rewriter", "检索词改写", "#F97316"),
        ("synthesizer", "综述合成", "#3B82F6"),
        ("verifier", "引用核验", "#8B5CF6"),
        ("reviser", "核验改写", "#EC4899"),
    ]

    lines = _svg_header(
        980,
        450,
        "系统端到端耗时对比与阶段四各节点开销剖析",
        "左图：三系统及阶段三串行基线端到端耗时对比（秒）｜右图：阶段四 ReacherAgent 各节点平均耗时分布（秒）",
    )

    max_sec = max(10.0, phase3_serial, r_sec, n_sec, d_sec)
    y_top = float(((int(max_sec) // 100) + 1) * 100)
    ticks = [round(y_top * i / 4.0) for i in range(5)]
    _draw_grouped_bars(
        lines,
        plot_x=72,
        plot_y=118,
        plot_w=430,
        plot_h=260,
        groups=["Direct LLM", "Naive RAG", "阶段四并发版", "阶段三串行版"],
        series=[
            (
                "平均耗时 (秒)",
                PALETTE["teal"],
                [d_sec, n_sec, r_sec, phase3_serial],
                [f"{d_sec:.1f}s", f"{n_sec:.1f}s", f"{r_sec:.1f}s", f"{phase3_serial:.1f}s"],
            )
        ],
        y_max=y_top,
        y_ticks=ticks,
        y_suffix="s",
    )

    rx = 640
    ry = 118
    rw = 300
    rh = 260
    present_nodes = [
        (lbl, col, float(node_means.get(key, 0.0)))
        for key, lbl, col in ordered_nodes
        if float(node_means.get(key, 0.0)) > 0.0
        or key in {"planner", "retriever_screener", "ingestion", "extractor", "synthesizer", "verifier"}
    ]
    max_node = max([1.0, *(v for _, _, v in present_nodes)])
    row_h = rh / max(1, len(present_nodes))
    for idx, (lbl, col, val) in enumerate(present_nodes):
        cy = ry + idx * row_h + row_h * 0.2
        bh = max(12.0, row_h * 0.56)
        bw = (val / max_node) * (rw - 60)
        lines.append(
            f'<text x="{rx - 10}" y="{cy + bh * 0.72:.1f}" text-anchor="end" font-size="11.5" '
            f'fill="{PALETTE["text"]}">{_esc(lbl)}</text>'
        )
        lines.append(
            f'<rect x="{rx}" y="{cy:.1f}" width="{max(2.0, bw):.1f}" height="{bh:.1f}" '
            f'fill="{col}" rx="3"/>'
        )
        lines.append(
            f'<text x="{rx + max(2.0, bw) + 6:.1f}" y="{cy + bh * 0.72:.1f}" font-size="11" '
            f'font-weight="600" fill="{PALETTE["text"]}">{val:.1f}s</text>'
        )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def generate_all_charts(summary: dict[str, Any], charts_dir: Path) -> list[Path]:
    """Render all 5 Phase 4 evaluation SVG charts and return their paths."""
    charts_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        (charts_dir / "fig1_system_comparison.svg", generate_fig1_system_comparison),
        (charts_dir / "fig2_retrieval_ablation.svg", generate_fig2_retrieval_ablation),
        (charts_dir / "fig3_reflection_ablation.svg", generate_fig3_reflection_ablation),
        (charts_dir / "fig4_matrix_and_revision.svg", generate_fig4_matrix_and_revision),
        (charts_dir / "fig5_latency_and_cost.svg", generate_fig5_latency_and_cost),
    ]
    for path, fn in targets:
        fn(summary, path)
    return [path for path, _ in targets]
