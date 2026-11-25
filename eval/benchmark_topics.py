"""10-topic bilingual CS/AI benchmark dataset for Phase 4 evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkTopic:
    """A standardized bilingual CS/AI literature survey benchmark topic."""

    topic_id: str
    language: str
    domain: str
    title: str
    english_query_hint: str
    expected_aspects: tuple[str, ...]


BENCHMARK_TOPICS: list[BenchmarkTopic] = [
    BenchmarkTopic(
        topic_id="T01",
        language="zh",
        domain="Agentic Literature Review",
        title="多智能体协同下的自动化科研文献综述与事实核验方法",
        english_query_hint="multi-agent automated scientific literature survey citation verification",
        expected_aspects=(
            "multi-agent orchestration",
            "literature retrieval and screening",
            "citation verification and faithfulness",
            "evaluation benchmarks and limitations",
        ),
    ),
    BenchmarkTopic(
        topic_id="T02",
        language="en",
        domain="Hybrid & Graph RAG",
        title="Hybrid Retrieval-Augmented Generation with Knowledge Graphs and Reranking",
        english_query_hint="hybrid retrieval augmented generation knowledge graph reranking",
        expected_aspects=(
            "dense and sparse hybrid retrieval",
            "knowledge graph integration",
            "reranking methods",
            "latency and scalability trade-offs",
        ),
    ),
    BenchmarkTopic(
        topic_id="T03",
        language="zh",
        domain="Test-Time Scaling & Reflection",
        title="大语言模型测试时扩展与自反思推理机制综述",
        english_query_hint="large language model test-time scaling self-reflection reasoning",
        expected_aspects=(
            "test-time compute scaling",
            "self-reflection and critique",
            "process reward and verification",
            "compute budget vs accuracy trade-offs",
        ),
    ),
    BenchmarkTopic(
        topic_id="T04",
        language="en",
        domain="Citation Faithfulness",
        title="Citation Faithfulness and Hallucination Mitigation in Retrieval-Augmented Scientific QA",
        english_query_hint="citation faithfulness hallucination mitigation retrieval augmented scientific QA",
        expected_aspects=(
            "hallucination taxonomy",
            "citation attribution and provenance",
            "automated verification metrics",
            "dataset benchmarks and open challenges",
        ),
    ),
    BenchmarkTopic(
        topic_id="T05",
        language="zh",
        domain="Multimodal Tool-Use Agents",
        title="多模态大语言模型驱动的自主智能体规划与工具调用综述",
        english_query_hint="multimodal large language model autonomous agent planning tool use",
        expected_aspects=(
            "multimodal perception and grounding",
            "hierarchical planning and decomposition",
            "tool invocation and execution feedback",
            "benchmarks and failure modes",
        ),
    ),
    BenchmarkTopic(
        topic_id="T06",
        language="en",
        domain="Agent Safety & Guardrails",
        title="Safety Alignment, Red-Teaming, and Guardrails for Autonomous LLM Agents",
        english_query_hint="safety alignment red-teaming guardrails autonomous LLM agents",
        expected_aspects=(
            "prompt injection and agent threats",
            "automated red-teaming",
            "runtime guardrails and policy enforcement",
            "safety-utility trade-offs",
        ),
    ),
    BenchmarkTopic(
        topic_id="T07",
        language="zh",
        domain="Software Engineering Agents",
        title="面向自动化软件工程与代码生成的大模型智能体研究进展",
        english_query_hint="LLM agents automated software engineering code generation program repair",
        expected_aspects=(
            "repository-level understanding",
            "iterative debugging and test feedback",
            "multi-agent SWE workflows",
            "SWE-bench evaluation and limitations",
        ),
    ),
    BenchmarkTopic(
        topic_id="T08",
        language="en",
        domain="Long-Context vs RAG",
        title="Long-Context Language Models versus Retrieval-Augmented Generation for Scientific Documents",
        english_query_hint="long-context language models versus retrieval augmented generation scientific documents",
        expected_aspects=(
            "long-context attention and needle retrieval",
            "RAG vs long-context accuracy",
            "cost and latency comparisons",
            "hybrid long-context RAG designs",
        ),
    ),
    BenchmarkTopic(
        topic_id="T09",
        language="zh",
        domain="Federated LLM Fine-Tuning",
        title="联邦学习与隐私保护下的大语言模型微调方法综述",
        english_query_hint="federated learning privacy preserving fine-tuning large language models",
        expected_aspects=(
            "federated LoRA and PEFT",
            "differential privacy and secure aggregation",
            "communication and heterogeneity overhead",
            "privacy-utility benchmarks",
        ),
    ),
    BenchmarkTopic(
        topic_id="T10",
        language="en",
        domain="AI4Science Literature Mining",
        title="AI for Science: Autonomous Literature Mining and Hypothesis Generation with Large Language Models",
        english_query_hint="AI for science autonomous literature mining hypothesis generation large language models",
        expected_aspects=(
            "scientific literature extraction",
            "hypothesis generation and novelty",
            "experimental validation loops",
            "reliability and ethical risks",
        ),
    ),
]


def get_topics_by_ids(topic_ids: list[str] | None = None) -> list[BenchmarkTopic]:
    """Return all benchmark topics or filter cleanly by requested topic IDs."""
    if not topic_ids:
        return list(BENCHMARK_TOPICS)
    wanted = {tid.strip().upper() for tid in topic_ids if tid.strip()}
    selected = [t for t in BENCHMARK_TOPICS if t.topic_id.upper() in wanted]
    if not selected:
        raise ValueError(f"No benchmark topics matched IDs: {topic_ids}")
    return selected
