"""Deterministic pre-compute tools for hallucination detection.

Each tool extracts structured facts from claim/KB text that LLMs are
error-prone at handling — numerics, polarity, capability boundaries, and
relevant KB section location.

The ToolRegistry provides a unified run_all() → format_for_prompt() interface
so callers don't need to know about individual tool modules.
"""

from .numeric import NumericExtractor
from .polarity import PolarityDetector
from .capability import CapabilityParser
from .locator import KBSectionLocator
from .registry import ToolRegistry, get_registry

__all__ = [
    "NumericExtractor",
    "PolarityDetector",
    "CapabilityParser",
    "KBSectionLocator",
    "ToolRegistry",
    "get_registry",
]
