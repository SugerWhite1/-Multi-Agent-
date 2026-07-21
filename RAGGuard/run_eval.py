#!/usr/bin/env python
"""RAGGuard CLI - One-click hallucination detection evaluation."""

import argparse
import json
import os
import sys
import time
from datetime import datetime
import yaml
from dotenv import load_dotenv

from src.engine import run_detection, LLMEngine
from src.runner import run_batch
from src.evaluator import evaluate, load_ground_truth
from src.patterns import analyze, format_for_report, save_profile, _make_json_safe
from src.visualization import generate_all_charts
from src.logging_config import setup_logging, get_logger

# Load .env file before any env var access
load_dotenv()

logger = get_logger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_replies(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def generate_report(results: list, metrics, output_path: str, chart_paths: dict = None):
    """Generate Markdown evaluation report."""
    # Count hallucination types
    type_counts = {}
    for r in results:
        t = r.get("hallucination_type", "无")
        type_counts[t] = type_counts.get(t, 0) + 1

    lines = [
        "# RAGGuard 客服回复幻觉检测报告",
        "",
        "## 1. 总体结果",
        "",
        f"- 检测样本: {metrics.total}",
        f"- 发现幻觉: {metrics.tp + metrics.fp}",
        f"- 实际幻觉: {metrics.tp + metrics.fn}",
        f"- 幻觉发生率: {(metrics.tp + metrics.fn) / metrics.total * 100:.0f}%",
        "",
        "## 2. 检测性能指标",
        "",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| Precision | {metrics.precision:.2%} |",
        f"| Recall | {metrics.recall:.2%} |",
        f"| F1-Score | {metrics.f1:.4f} |",
        f"| Type Accuracy | {metrics.type_accuracy:.2%} |",
        f"| Severity Accuracy | {metrics.severity_accuracy:.2%} |",
        "",
        f"| 统计 | 数值 |",
        f"|------|------|",
        f"| True Positive (正确检出) | {metrics.tp} |",
        f"| False Positive (误报) | {metrics.fp} |",
        f"| False Negative (漏检) | {metrics.fn} |",
        f"| True Negative (正确排除) | {metrics.tn} |",
        "",
        "## 3. 幻觉类型分布",
        "",
        "| 类型 | 数量 | 占比 |",
        "|------|------|------|",
    ]

    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    for t, c in sorted_types:
        lines.append(f"| {t} | {c} | {c/metrics.total*100:.0f}% |")

    # Embed charts
    if chart_paths:
        if "type_pie" in chart_paths:
            lines += [
                "",
                f"![{chart_paths['type_pie']}]({chart_paths['type_pie']})",
            ]
        if "type_bar" in chart_paths:
            lines += [
                "",
                f"![{chart_paths['type_bar']}]({chart_paths['type_bar']})",
            ]

    # Find worst 3 cases (most contradicted claims)
    results_sorted = sorted(
        results,
        key=lambda r: sum(1 for c in r.get("claims", []) if c.get("nli_status") == "CONTRADICTED"),
        reverse=True
    )
    worst_3 = [r for r in results_sorted if r.get("is_hallucination")][:3]

    lines += [
        "",
        "## 4. 最差 3 条案例",
        "",
    ]
    for i, case in enumerate(worst_3):
        n_contradicted = sum(1 for c in case.get("claims", []) if c.get("nli_status") == "CONTRADICTED")
        lines += [
            f"### Case {i+1}: {case['id']}",
            f"- 幻觉类型: {case.get('hallucination_type')}",
            f"- 严重度: {case.get('severity')}",
            f"- 矛盾声明数: {n_contradicted}",
            f"- 分析: {case.get('detail', '')[:300]}",
            "",
        ]

    # Error analysis
    lines += [
        "## 5. 误判分析",
        "",
        "### 误报 (False Positives)",
    ]
    if metrics.fp_cases:
        for c in metrics.fp_cases:
            lines.append(f"- {c}")
    else:
        lines.append("无误报")

    lines += ["", "### 漏检 (False Negatives)"]
    if metrics.fn_cases:
        for c in metrics.fn_cases:
            lines.append(f"- {c}")
    else:
        lines.append("无漏检")

    lines += ["", "### 类型误判"]
    if metrics.type_mismatches:
        for m in metrics.type_mismatches:
            lines.append(f"- {m['id']}: 预测={m['predicted']}, 实际={m['actual']}")
    else:
        lines.append("无类型误判")

    lines += [
        "",
        "## 6. 局限性讨论",
        "",
        "1. **边界案例判定困难**: 如 h20 的\"信息遗漏\"，Mock 模式下依赖关键词匹配可能误判",
        "2. **部分正确/部分错误**: 政策偏差类（如 h04）容易漏判或归类错误",
        "3. **寒暄语干扰**: 需在 Claim 提取阶段过滤非事实性表达",
        "4. **KB 语义理解**: Mock 模式无法理解语义相似但措辞不同的一致性",
        "5. **改进方向**: LLM 模式 + CoT 推理可显著提升边界 case 的判定质量",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="RAGGuard Hallucination Detection")
    parser.add_argument("--mode", choices=["mock", "llm"], default="mock",
                        help="Detection mode: mock (rule-based) or llm (OpenAI API)")
    parser.add_argument("--api-key", default=None, help="API key (or set RAGGUARD_API_KEY env var)")
    parser.add_argument("--base-url", default=None, help="API base URL (or set RAGGUARD_BASE_URL env var)")
    parser.add_argument("--model", default=None, help="LLM model name (or set RAGGUARD_MODEL env var)")
    parser.add_argument("--case-id", default="", help="Run single case by ID")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel processes (1=single-process async, >1=ProcessPool Map-Reduce)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max concurrent API calls across all workers")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Minimum log level for file output")
    args = parser.parse_args()

    # Create timestamped run directory
    run_dir = os.path.join("outputs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # Initialize logging into run directory
    setup_logging(level=args.log_level, log_file=os.path.join(run_dir, "ragguard.log"))
    t_start = time.time()

    # Load config
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.debug("Config loaded from %s", args.config)

    data_dir = config.get("data", {})
    config_output = config.get("output", {})
    replies_path = data_dir.get("replies_path", "data/replies.json")
    gt_path = data_dir.get("ground_truth_path", "data/ground_truth.json")
    results_path = os.path.join(run_dir, config_output.get("results_basename", "detection_results.json"))
    metrics_path = os.path.join(run_dir, config_output.get("metrics_basename", "evaluation_metrics.json"))
    report_path = os.path.join(run_dir, config_output.get("report_basename", "report.md"))

    # Initialize engine
    engine = None
    if args.mode == "llm":
        # Priority: CLI args > env vars > config.yaml
        api_key = (args.api_key
                   or os.getenv("RAGGUARD_API_KEY")
                   or os.getenv("OPENAI_API_KEY")
                   or config.get("llm", {}).get("api_key", ""))
        base_url = (args.base_url
                    or os.getenv("RAGGUARD_BASE_URL")
                    or os.getenv("OPENAI_BASE_URL")
                    or config.get("llm", {}).get("base_url", "https://api.openai.com/v1"))
        model = (args.model
                 or os.getenv("RAGGUARD_MODEL")
                 or config.get("llm", {}).get("model", "gpt-4.1-mini"))
        if not api_key:
            logger.error("LLM mode requires API key but none found")
            sys.exit(1)
        engine = LLMEngine(api_key=api_key, base_url=base_url, model=model)
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        logger.info("Engine: LLM mode | model=%s | base_url=%s | key=%s", model, base_url, masked)
    else:
        logger.info("Engine: Mock mode (rule-based)")

    # Load data
    replies = load_replies(replies_path)
    logger.info("Loaded %d cases from %s", len(replies), replies_path)

    # Run detection
    if args.case_id:
        replies = [r for r in replies if r["id"] == args.case_id]
        if not replies:
            logger.error("Case %s not found", args.case_id)
            sys.exit(1)

    logger.info("Starting detection: workers=%d concurrency=%d cases=%d",
                args.workers, args.concurrency, len(replies))

    batch_results = run_batch(
        replies, engine=engine,
        workers=args.workers,
        max_concurrency=args.concurrency,
    )
    results = [_make_json_safe(r.model_dump()) for r in batch_results]

    # Per-case summary
    hallucinations = [r for r in batch_results if r.is_hallucination]
    logger.info("Detection complete: %d/%d cases have hallucinations",
                len(hallucinations), len(batch_results))
    for r in batch_results:
        status = "HALLUCINATION" if r.is_hallucination else "OK"
        logger.info("  %s: %s (%s)", r.id, status, r.hallucination_type)

    # Save results
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Results saved to %s", results_path)

    # Evaluate against ground truth if available
    if os.path.exists(gt_path) and len(results) > 1:
        ground_truth = load_ground_truth(gt_path)
        metrics = evaluate(results, ground_truth)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(_make_json_safe(metrics.model_dump()), f, ensure_ascii=False, indent=2)
        logger.info("Metrics saved to %s", metrics_path)

        # Summary
        logger.info("")
        logger.info("=" * 50)
        logger.info("  Evaluation Results")
        logger.info("=" * 50)
        logger.info("  Precision:    %.2f%%", metrics.precision * 100)
        logger.info("  Recall:       %.2f%%", metrics.recall * 100)
        logger.info("  F1-Score:     %.4f", metrics.f1)
        logger.info("  Type Acc:     %.2f%%", metrics.type_accuracy * 100)
        logger.info("  Severity Acc: %.2f%%", metrics.severity_accuracy * 100)
        logger.info("  FP: %d, FN: %d", metrics.fp, metrics.fn)
        if metrics.type_mismatches:
            logger.warning("  Type mismatches: %d", len(metrics.type_mismatches))
            for m in metrics.type_mismatches:
                logger.warning("    %s: predicted=%s, actual=%s",
                               m["id"], m["predicted"], m["actual"])
    else:
        metrics = None

    # Pattern analysis (runs regardless of ground truth availability)
    profile = analyze(results)
    profile_path = os.path.join(run_dir, config_output.get("profile_basename", "pattern_profile.json"))
    save_profile(profile, profile_path)
    logger.info("Pattern profile saved to %s", profile_path)
    logger.info(
        "  Numeric conflicts: %d, Polarity conflicts: %d, Capability violations: %d",
        profile["tool_trace_stats"]["numeric_conflicts_detected"],
        profile["tool_trace_stats"]["polarity_conflicts_detected"],
        profile["tool_trace_stats"]["capability_violations_detected"],
    )

    # Generate charts
    type_counts = {}
    for r in results:
        t = r.get("hallucination_type", "无")
        type_counts[t] = type_counts.get(t, 0) + 1
    chart_paths = generate_all_charts(
        type_counts=type_counts,
        severity_counts=profile.get("severity_distribution", {}),
        nli_counts=profile.get("nli_distribution", {}),
        output_dir=run_dir,
    )
    logger.info("Charts saved: %s", ", ".join(chart_paths.values()))

    # Use basenames for markdown embeds (report.md is in same directory)
    chart_basenames = {k: os.path.basename(v) for k, v in chart_paths.items()}

    # Generate / update report with charts
    if os.path.exists(gt_path) and len(results) > 1 and metrics is not None:
        generate_report(results, metrics, report_path, chart_paths=chart_basenames)
    else:
        # Write minimal report header if no GT
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# RAGGuard 客服回复幻觉检测报告\n\n")

    # Append pattern report to markdown
    pattern_report = format_for_report(profile, chart_paths=chart_basenames)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(pattern_report)
    logger.info("Full report saved to %s", report_path)

    elapsed = time.time() - t_start
    logger.info("Total elapsed: %.1fs (%.2fs/case)", elapsed, elapsed / max(len(replies), 1))


if __name__ == "__main__":
    main()
