"""Unified tool registry for orchestrating all deterministic pre-compute tools.

Replaces scattered per-tool calls with a single run_all() → format_for_prompt()
pipeline.  Each tool is registered with its run() and format_for_prompt() methods
so callers (engine.py, skills/nli_verifier.py) don't need to know about individual
tool modules.
"""

from typing import Any, Callable, Dict, List

from .numeric import NumericExtractor
from .polarity import PolarityDetector
from .capability import CapabilityParser
from .locator import KBSectionLocator


# Tool descriptor: (name, icon, run_fn, format_fn, run_kwargs_builder)
# run_kwargs_builder: (claim_text, kb, claim_type) → kwargs dict for run()
def _tool_specs() -> List[dict]:
    return [
        {
            "name": "numeric",
            "icon": "📏 数值比对",
            "run": NumericExtractor.run,
            "format": NumericExtractor.format_for_prompt,
            "kwargs": lambda ct, kb, ctype: {"claim_text": ct, "knowledge_base": kb},
        },
        {
            "name": "polarity",
            "icon": "🔄 极性检测",
            "run": PolarityDetector.run,
            "format": PolarityDetector.format_for_prompt,
            "kwargs": lambda ct, kb, ctype: {"claim_text": ct, "knowledge_base": kb},
        },
        {
            "name": "capability",
            "icon": "🛡️ 能力边界",
            "run": CapabilityParser.run,
            "format": CapabilityParser.format_for_prompt,
            "kwargs": lambda ct, kb, ctype: {
                "knowledge_base": kb,
                "claim_text": ct,
                "claim_type": ctype,
            },
        },
        {
            "name": "kb_location",
            "icon": "📍 KB 相关段落",
            "run": KBSectionLocator.run,
            "format": KBSectionLocator.format_for_prompt,
            "kwargs": lambda ct, kb, ctype: {"claim_text": ct, "knowledge_base": kb},
        },
    ]


class ToolRegistry:
    """Unified runner for all deterministic pre-compute tools.

    Usage:
        registry = ToolRegistry()
        results = registry.run_all(claim_text, knowledge_base, claim_type)
        # results == {"numeric": {...}, "polarity": {...}, ...}

        prompt_block = registry.format_for_prompt(results)
        # Inject into LLM system prompt
    """

    def __init__(self):
        self._specs = _tool_specs()

    def run_all(self, claim_text: str, knowledge_base: str, claim_type: str = "fact") -> Dict[str, Any]:
        """Run all registered tools and return combined results keyed by tool name."""
        results = {}
        for spec in self._specs:
            kwargs = spec["kwargs"](claim_text, knowledge_base, claim_type)
            try:
                results[spec["name"]] = spec["run"](**kwargs)
            except Exception:
                results[spec["name"]] = {}
        return results

    def format_for_prompt(self, results: Dict[str, Any]) -> str:
        """Format all tool results as a unified prompt injection block."""
        lines = ["【工具预计算结果（请直接使用，无需自行判断）】", ""]
        for spec in self._specs:
            name = spec["name"]
            icon = spec["icon"]
            result = results.get(name, {})
            formatted = spec["format"](result)
            lines.append(f"{icon}:")
            lines.append(formatted)
            lines.append("")
        return "\n".join(lines)

    def build_tool_trace(self, results: Dict[str, Any]) -> dict:
        """Extract structured tool_trace dict for ClaimItem serialization."""
        numeric = results.get("numeric", {})
        polarity = results.get("polarity", {})
        capability = results.get("capability", {})
        kb_location = results.get("kb_location", {})

        return {
            "numeric": {
                "has_conflict": numeric.get("has_numeric_conflict"),
                "conflicts": numeric.get("conflicts", []),
                "amount_conflict": numeric.get("amount_conflict"),
                "version_conflict": numeric.get("version_conflict"),
            },
            "polarity": {
                "has_conflict": polarity.get("has_conflict"),
                "conflicts": polarity.get("conflicts", []),
            },
            "capability": {
                "has_no_capability": capability.get("has_no_capability"),
                "capability_violation": capability.get("capability_violation"),
                "unmarked_match": capability.get("unmarked_check", {}).get("has_match"),
                "explicit_no_match": capability.get("explicit_no_check", {}).get("has_match"),
            },
            "kb_location": {
                "overlap_ratio": round(kb_location.get("overlap_ratio", 0), 3),
                "best_section": kb_location.get("best_section", "")[:200],
            },
        }

    def has_capability_violation(self, results: Dict[str, Any]) -> bool:
        """Check if capability tool detected a violation (for fast path)."""
        return bool(results.get("capability", {}).get("capability_violation"))

    def has_numeric_conflict(self, results: Dict[str, Any]) -> bool:
        """Check if numeric tool detected a conflict (for post_validate override)."""
        return bool(results.get("numeric", {}).get("has_numeric_conflict"))

    def has_polarity_conflict(self, results: Dict[str, Any]) -> bool:
        """Check if polarity tool detected a conflict (for post_validate override)."""
        return bool(results.get("polarity", {}).get("has_conflict"))

    def get_filtered_kb(self, results: Dict[str, Any], knowledge_base: str,
                         min_overlap: float = 0.05) -> str:
        """Return a relevance-filtered KB using KBSectionLocator results.

        If the best section has meaningful overlap, return just the top sections
        instead of the full KB to reduce token consumption.
        """
        kb_loc = results.get("kb_location", {})
        overlap = kb_loc.get("overlap_ratio", 0)
        relevant = kb_loc.get("all_relevant", [])

        if overlap >= min_overlap and relevant:
            return "\n".join(relevant)
        # Fallback: return truncated full KB
        return knowledge_base[:1500]


# Module-level singleton for convenience
_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
