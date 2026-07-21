"""Cross-case hallucination pattern analysis.

Aggregates tool traces and detection results across all cases to produce
a structured "hallucination profile" — useful for:
  - Understanding systemic failure patterns
  - Tracking improvement across prompt / tool iterations
  - Generating interview-ready summary statistics
"""

import json
from typing import List, Dict
from collections import Counter
from .state import HallucinationResult, ClaimItem


def analyze(results: List[dict]) -> dict:
    """Analyze hallucination patterns across all detection results.

    Args:
        results: List of dicts from detection_results.json.
            Each dict has keys: id, is_hallucination, hallucination_type,
            severity, detail, claims[].

    Returns:
        A structured pattern profile dict.
    """
    total = len(results)
    hallucination_cases = [r for r in results if r.get("is_hallucination")]
    ok_cases = [r for r in results if not r.get("is_hallucination")]
    n_hallucination = len(hallucination_cases)

    # ── 1. Type & Severity Distribution ──────────────────────
    type_counter = Counter(
        r["hallucination_type"] for r in hallucination_cases
    )
    severity_counter = Counter(
        r["severity"] for r in hallucination_cases
    )

    # ── 2. Claim-level statistics ─────────────────────────────
    all_claims: List[dict] = []
    for r in results:
        for c in r.get("claims", []):
            c["_case_id"] = r["id"]
            all_claims.append(c)

    total_claims = len(all_claims)
    nli_counter = Counter(
        c.get("nli_status", "UNKNOWN") for c in all_claims
    )
    contradicted_claims = [
        c for c in all_claims if c.get("nli_status") == "CONTRADICTED"
    ]

    # ── 3. Tool trace aggregation ─────────────────────────────
    tool_trace_stats = _aggregate_tool_traces(all_claims)

    # ── 4. Contradiction pattern breakdown ────────────────────
    contradiction_patterns = _classify_contradictions(contradicted_claims)

    # ── 5. High-frequency contradiction topics ────────────────
    topic_counter = Counter()
    for c in contradicted_claims:
        trace = c.get("tool_trace") or {}
        numeric = trace.get("numeric", {})
        polarity = trace.get("polarity", {})
        capability = trace.get("capability", {})

        if numeric.get("has_conflict"):
            for conflict in numeric.get("conflicts", []):
                topic_counter[f"数值矛盾: {conflict.get('unit', '?')}"] += 1
            if numeric.get("amount_conflict"):
                topic_counter["数值矛盾: 金额"] += 1
        if polarity.get("has_conflict"):
            for conflict in polarity.get("conflicts", []):
                topic_counter[f"极性矛盾: {conflict.get('topic', '?')}"] += 1
        if capability.get("capability_violation"):
            topic_counter["能力越界: 声称执行了KB不具备的操作"] += 1

    # ── 6. Claims per case distribution ───────────────────────
    claims_per_case = Counter(
        sum(1 for c in r.get("claims", [])) for r in results
    )

    return {
        # Overview
        "total_cases": total,
        "hallucination_cases": n_hallucination,
        "ok_cases": len(ok_cases),
        "hallucination_rate": round(n_hallucination / max(total, 1), 3),
        "total_claims": total_claims,
        "claims_per_case_avg": round(total_claims / max(total, 1), 1),

        # Type & Severity
        "type_distribution": dict(type_counter.most_common()),
        "severity_distribution": dict(severity_counter.most_common()),

        # NLI status distribution
        "nli_distribution": dict(nli_counter.most_common()),
        "contradicted_count": len(contradicted_claims),

        # Tool trace
        "tool_trace_stats": tool_trace_stats,

        # Contradiction patterns
        "contradiction_patterns": contradiction_patterns,

        # High-frequency topics
        "top_contradiction_topics": topic_counter.most_common(10),

        # Claims per case
        "claims_per_case_distribution": {
            str(k): v for k, v in sorted(claims_per_case.items())
        },

        # KB overlap (semantic match quality)
        "kb_overlap_distribution": _kb_overlap_buckets(all_claims),
    }


def _aggregate_tool_traces(all_claims: List[dict]) -> dict:
    """Aggregate tool trace statistics across all claims."""
    claims_with_trace = [c for c in all_claims if c.get("tool_trace")]

    numeric_conflicts = 0
    polarity_conflicts = 0
    capability_violations = 0
    unmarked_matches = 0
    explicit_no_matches = 0
    total_overlap = 0.0
    overlap_count = 0

    for c in claims_with_trace:
        trace = c["tool_trace"]
        if trace.get("numeric", {}).get("has_conflict"):
            numeric_conflicts += 1
        if trace.get("polarity", {}).get("has_conflict"):
            polarity_conflicts += 1
        if trace.get("capability", {}).get("capability_violation"):
            capability_violations += 1
        if trace.get("capability", {}).get("unmarked_match"):
            unmarked_matches += 1
        if trace.get("capability", {}).get("explicit_no_match"):
            explicit_no_matches += 1

        ratio = trace.get("kb_location", {}).get("overlap_ratio", 0)
        if ratio > 0:
            total_overlap += ratio
            overlap_count += 1

    return {
        "claims_with_tool_trace": len(claims_with_trace),
        "numeric_conflicts_detected": numeric_conflicts,
        "polarity_conflicts_detected": polarity_conflicts,
        "capability_violations_detected": capability_violations,
        "unmarked_feature_matches": unmarked_matches,
        "explicit_no_matches": explicit_no_matches,
        "avg_kb_overlap_ratio": (
            round(total_overlap / max(overlap_count, 1), 3)
        ),
    }


def _classify_contradictions(contradicted_claims: List[dict]) -> dict:
    """Categorize contradicted claims by root cause."""
    patterns = {
        "numeric_only": 0,
        "polarity_only": 0,
        "capability_only": 0,
        "numeric_and_polarity": 0,
        "numeric_and_capability": 0,
        "polarity_and_capability": 0,
        "no_tool_match": 0,
    }

    for c in contradicted_claims:
        trace = c.get("tool_trace") or {}
        has_num = trace.get("numeric", {}).get("has_conflict", False)
        has_pol = trace.get("polarity", {}).get("has_conflict", False)
        has_cap = trace.get("capability", {}).get("capability_violation", False)

        if has_num and has_pol:
            patterns["numeric_and_polarity"] += 1
        elif has_num and has_cap:
            patterns["numeric_and_capability"] += 1
        elif has_pol and has_cap:
            patterns["polarity_and_capability"] += 1
        elif has_num:
            patterns["numeric_only"] += 1
        elif has_pol:
            patterns["polarity_only"] += 1
        elif has_cap:
            patterns["capability_only"] += 1
        else:
            patterns["no_tool_match"] += 1

    return patterns


def _kb_overlap_buckets(all_claims: List[dict]) -> dict:
    """Bucket KB overlap ratios for distribution view."""
    buckets = {"0-25%": 0, "25-50%": 0, "50-75%": 0, "75-100%": 0}

    for c in all_claims:
        trace = c.get("tool_trace")
        if not trace:
            continue
        ratio = trace.get("kb_location", {}).get("overlap_ratio", 0)
        if ratio < 0.25:
            buckets["0-25%"] += 1
        elif ratio < 0.50:
            buckets["25-50%"] += 1
        elif ratio < 0.75:
            buckets["50-75%"] += 1
        else:
            buckets["75-100%"] += 1

    return buckets


def format_for_report(profile: dict, chart_paths: dict = None) -> str:
    """Format a pattern profile as Markdown report sections."""
    lines = [
        "",
        "## 7. 幻觉模式画像",
        "",
        f"- 总 Case 数: {profile['total_cases']}",
        f"- 幻觉发生率: {profile['hallucination_rate'] * 100:.0f}% "
        f"({profile['hallucination_cases']}/{profile['total_cases']})",
        f"- 总 Claim 数: {profile['total_claims']} "
        f"(平均 {profile['claims_per_case_avg']}/case)",
        "",
        "### 7.1 幻觉类型分布",
        "",
        "| 类型 | 数量 | 占比 |",
        "|------|------|------|",
    ]
    for t, count in profile["type_distribution"].items():
        pct = count / max(profile["hallucination_cases"], 1) * 100
        lines.append(f"| {t} | {count} | {pct:.0f}% |")

    lines += [
        "",
        "### 7.2 严重度分布",
        "",
        "| 严重度 | 数量 |",
        "|------|------|",
    ]
    for s, count in profile["severity_distribution"].items():
        lines.append(f"| {s} | {count} |")

    if chart_paths and "severity_bar" in chart_paths:
        lines += [
            "",
            f"![{chart_paths['severity_bar']}]({chart_paths['severity_bar']})",
        ]

    lines += [
        "",
        "### 7.3 NLI 判定分布 (Claim 级)",
        "",
        "| 判定 | 数量 | 占比 |",
        "|------|------|------|",
    ]
    total_claims = max(profile["total_claims"], 1)
    for status, count in profile["nli_distribution"].items():
        lines.append(f"| {status} | {count} | {count / total_claims * 100:.0f}% |")

    if chart_paths and "nli_bar" in chart_paths:
        lines += [
            "",
            f"![{chart_paths['nli_bar']}]({chart_paths['nli_bar']})",
        ]

    lines += [
        "",
        "### 7.4 矛盾根因分析",
        "",
        "| 根因组合 | 数量 |",
        "|------|------|",
    ]
    for pattern_name, count in profile["contradiction_patterns"].items():
        if count > 0:
            label = {
                "numeric_only": "纯数值矛盾",
                "polarity_only": "纯极性矛盾",
                "capability_only": "纯能力越界(工具判定)",
                "numeric_and_polarity": "数值 + 极性",
                "numeric_and_capability": "数值 + 能力越界",
                "polarity_and_capability": "极性 + 能力越界",
                "no_tool_match": "LLM 独立判定(工具未捕获)",
            }.get(pattern_name, pattern_name)
            lines.append(f"| {label} | {count} |")

    lines += [
        "",
        "### 7.5 工具检测统计",
        "",
        f"- 数值矛盾检出: {profile['tool_trace_stats']['numeric_conflicts_detected']} 次",
        f"- 极性矛盾检出: {profile['tool_trace_stats']['polarity_conflicts_detected']} 次",
        f"- 能力越界检出: {profile['tool_trace_stats']['capability_violations_detected']} 次",
        f"- KB 未标注匹配: {profile['tool_trace_stats']['unmarked_feature_matches']} 次",
        f"- KB 明确否定匹配: {profile['tool_trace_stats']['explicit_no_matches']} 次",
        f"- 平均 KB 语义重叠率: {profile['tool_trace_stats']['avg_kb_overlap_ratio'] * 100:.0f}%",
        "",
        "### 7.6 高频矛盾话题 Top 10",
        "",
        "| 话题 | 出现次数 |",
        "|------|------|",
    ]
    for topic, count in profile["top_contradiction_topics"]:
        lines.append(f"| {topic} | {count} |")

    lines += [
        "",
        "### 7.7 Claim 数分布 (每 Case)",
        "",
        "| Claim 数 | Case 数 |",
        "|------|------|",
    ]
    for n_claims, count in profile["claims_per_case_distribution"].items():
        lines.append(f"| {n_claims} | {count} |")

    return "\n".join(lines)


def make_json_safe(obj):
    """Recursively convert sets and other non-JSON types to JSON-safe equivalents."""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    return obj


# Backwards-compatible alias
_make_json_safe = make_json_safe


def save_profile(profile: dict, path: str) -> None:
    """Save pattern profile as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(profile), f, ensure_ascii=False, indent=2)
