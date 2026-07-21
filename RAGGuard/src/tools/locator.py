"""Locate the most relevant KB section for a given claim using bigram overlap."""

import re
from typing import List


def _char_bigrams(text: str) -> set:
    """Extract character bigrams for fuzzy CJK matching.

    Preserves CJK Unified Ideographs (U+4E00-U+9FFF), Extension A
    (U+3400-U+4DBF), digits, and common CJK punctuation for better
    sentence-boundary awareness.
    """
    clean = re.sub(r'[^㐀-䶿一-鿿\d。，、：；]', '', text)
    return {clean[i:i + 2] for i in range(len(clean) - 1)} if len(clean) >= 2 else set()


class KBSectionLocator:
    """Find which part of KB is most relevant to a claim."""

    @staticmethod
    def split_sections(kb: str) -> List[str]:
        """Split KB into logical sections by newlines or semicolons."""
        parts = []
        for line in kb.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Further split long lines by Chinese semicolons
            sub = re.split(r'[；;]', line)
            parts.extend(s.strip() for s in sub if s.strip())
        return parts if parts else [kb]

    @classmethod
    def find_best_section(cls, claim_text: str, kb: str) -> dict:
        """Find the KB section with highest bigram overlap with the claim."""
        sections = cls.split_sections(kb)

        claim_bigrams = _char_bigrams(claim_text)
        if not claim_bigrams:
            return {"best_section": kb[:300], "overlap_ratio": 0.0, "all_relevant": []}

        # Handle single-section KB: still compute overlap
        if len(sections) <= 1:
            sec = sections[0] if sections else kb
            sec_bigrams = _char_bigrams(sec)
            overlap = claim_bigrams & sec_bigrams
            ratio = len(overlap) / max(len(claim_bigrams), 1) if sec_bigrams else 0.0
            return {
                "best_section": sec[:300],
                "overlap_ratio": ratio,
                "all_relevant": [sec[:200]] if ratio > 0 else [],
            }

        scored = []
        for sec in sections:
            sec_bigrams = _char_bigrams(sec)
            if not sec_bigrams:
                continue
            overlap = claim_bigrams & sec_bigrams
            ratio = len(overlap) / max(len(claim_bigrams), 1)
            scored.append((ratio, sec))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top sections with meaningful overlap
        relevant = [(r, s) for r, s in scored if r > 0.0][:3]

        return {
            "best_section": relevant[0][1][:300] if relevant else kb[:300],
            "overlap_ratio": relevant[0][0] if relevant else 0.0,
            "all_relevant": [s[:200] for _, s in relevant],
        }

    @classmethod
    def run(cls, claim_text: str, knowledge_base: str) -> dict:
        """Locate the KB passage most relevant to this claim."""
        result = cls.find_best_section(claim_text, knowledge_base)
        return result

    @classmethod
    def format_for_prompt(cls, result: dict) -> str:
        """Format KB location result as prompt injection."""
        ratio = result.get("overlap_ratio", 0)
        lines = [f"  相关性: {ratio:.0%}"]
        if result.get("all_relevant"):
            for i, sec in enumerate(result["all_relevant"]):
                lines.append(f"  段落{i + 1}: {sec}")
        return "\n".join(lines)
