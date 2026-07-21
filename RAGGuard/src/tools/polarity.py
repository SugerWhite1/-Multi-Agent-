"""Detect polarity conflicts: Claim affirms something KB explicitly denies."""

import re
from typing import List, Dict, Optional


# Claim says positive → KB says negative about the SAME topic
POLARITY_PAIRS = [
    # (claim_positive_pattern, claim_neg_reject, kb_negative_pattern, label)
    (r"(?<!不)支持.*货到付款", r"不支持.*货到付款", r"不支持.*货到付款", "货到付款"),
    (r"(?<!不)支持.*纸质发票", r"不支持.*纸质发票", r"(不支持|暂不支持).*纸质发票", "纸质发票"),
    (r"有.*线下", r"没有.*线下", r"(纯线上|无.*线下|无线下|纯线上电商)", "线下门店"),
    (r"(?<!不)支持.*线下", r"不支持.*线下", r"(纯线上|无.*线下|无线下|纯线上电商)", "线下门店"),
]

POSITIVE_WORDS = {"支持", "可以", "能够", "提供", "包含", "具有", "采用", "是"}
NEGATIVE_WORDS_IN_KB = {"不支持", "不可以", "不能够", "不提供", "不包含", "不具有", "暂不支持", "纯线上"}


class PolarityDetector:
    """Detect direct negation conflicts between claim and KB."""

    @classmethod
    def check_negation_pairs(cls, claim_text: str, kb: str) -> List[dict]:
        """Check predefined negation-pair rules."""
        conflicts = []
        for pos_pattern, neg_reject, kb_neg_pattern, desc in cls.PAIR_RULES:
            if re.search(neg_reject, claim_text):
                continue
            if re.search(pos_pattern, claim_text):
                if re.search(kb_neg_pattern, kb):
                    conflicts.append({
                        "topic": desc,
                        "claim_stance": "肯定",
                        "kb_stance": "否定",
                        "rule": "negation_pair",
                    })
        return conflicts

    PAIR_RULES = POLARITY_PAIRS

    @staticmethod
    def _char_bigrams(text: str) -> set:
        """Extract character bigrams (sliding window) for fuzzy CJK matching."""
        clean = re.sub(r'[^一-鿿\d]', '', text)
        return {clean[i:i + 2] for i in range(len(clean) - 1)} if len(clean) >= 2 else set()

    @classmethod
    def check_contextual_polarity(cls, claim_text: str, kb: str) -> List[dict]:
        """Broader keyword-level polarity check with context overlap.

        Only flags a contradiction when the positive word in the claim and the
        negative word in KB refer to the SAME topic (>=2 overlapping bigrams
        and the KB negation word is not present in the claim itself).
        This prevents false positives from unrelated positive/negative pairs.
        """
        conflicts = []
        claim_bigrams = cls._char_bigrams(claim_text)

        for pw in POSITIVE_WORDS:
            if pw not in claim_text or claim_text.startswith("不" + pw):
                continue

            for nw in NEGATIVE_WORDS_IN_KB:
                if nw not in kb:
                    continue
                kb_neg_idx = kb.find(nw)
                kb_context = kb[kb_neg_idx:kb_neg_idx + len(nw) + 20]
                kb_bigrams = cls._char_bigrams(kb_context)

                overlap = claim_bigrams & kb_bigrams
                if len(overlap) >= 2 and nw not in claim_text:
                    conflicts.append({
                        "topic": ", ".join(list(overlap)[:3]),
                        "claim_stance": f"肯定({pw})",
                        "kb_stance": f"否定({nw})",
                        "rule": "contextual",
                    })
        return conflicts

    @classmethod
    def run(cls, claim_text: str, knowledge_base: str) -> dict:
        """Run all polarity checks.

        Returns:
            dict with conflicts, has_conflict
        """
        conflicts = (
            cls.check_negation_pairs(claim_text, knowledge_base) +
            cls.check_contextual_polarity(claim_text, knowledge_base)
        )
        # Deduplicate by topic
        seen_topics = set()
        unique = []
        for c in conflicts:
            if c["topic"] not in seen_topics:
                seen_topics.add(c["topic"])
                unique.append(c)

        return {
            "conflicts": unique,
            "has_conflict": len(unique) > 0,
        }

    @classmethod
    def format_for_prompt(cls, result: dict) -> str:
        """Format polarity results as human-readable prompt injection."""
        if not result.get("conflicts"):
            return "无极性问题"

        lines = []
        for c in result["conflicts"]:
            lines.append(
                f"  ⚠️ Claim{c['claim_stance']} vs KB{c['kb_stance']}"
                f" — 主题: {c['topic']}"
            )
        return "\n".join(lines)
