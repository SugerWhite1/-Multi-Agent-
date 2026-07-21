"""Extract and compare numeric values between claim and KB."""

import re
from typing import Dict, List


# Standard units: (pattern, unit_name)
NUMERIC_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(天|工作日|小时|元|块|折|%|版本|年|月|ms|毫秒)')
AMOUNT_PATTERN = re.compile(r'[满减]\s*(\d+(?:\.\d+)?)')
VERSION_PATTERN = re.compile(r'(?:蓝牙|版本)\s*(\d+(?:\.\d+)?)')


class NumericExtractor:
    """Extract numeric claims and compare with KB values."""

    @staticmethod
    def extract(text: str) -> Dict[str, list]:
        """Extract standard-unit numerics from text (preserves duplicates)."""
        result = {}
        for m in NUMERIC_PATTERN.finditer(text):
            unit = m.group(2)
            value = float(m.group(1))
            if unit not in result:
                result[unit] = []
            result[unit].append(value)
        return result

    @staticmethod
    def extract_amounts(text: str) -> set:
        """Extract promotional amounts (满X减Y patterns)."""
        return {int(m) for m in AMOUNT_PATTERN.findall(text)}

    @staticmethod
    def extract_versions(text: str) -> set:
        """Extract version numbers (蓝牙X.X, 版本X.X)."""
        return set(VERSION_PATTERN.findall(text))

    @classmethod
    def run(cls, claim_text: str, knowledge_base: str) -> dict:
        """Compare numerics between claim and KB.

        Returns:
            dict with claim_nums, kb_nums, conflicts, has_conflict
        """
        claim_nums = cls.extract(claim_text)
        kb_nums = cls.extract(knowledge_base)

        conflicts = []
        for unit, claim_vals in claim_nums.items():
            kb_vals = kb_nums.get(unit, [])
            if kb_vals and not any(cv in kb_vals for cv in claim_vals):
                conflicts.append({
                    "unit": unit,
                    "claim_value": claim_vals,
                    "kb_value": kb_vals,
                })

        # Amount conflicts
        claim_amounts = cls.extract_amounts(claim_text)
        kb_amounts = cls.extract_amounts(knowledge_base)
        amount_conflict = (
            claim_amounts and kb_amounts and claim_amounts != kb_amounts
        )

        # Version conflicts
        claim_versions = cls.extract_versions(claim_text)
        kb_versions = cls.extract_versions(knowledge_base)
        version_conflict = (
            claim_versions and kb_versions and claim_versions != kb_versions
        )

        return {
            "claim_nums": {k: v for k, v in claim_nums.items()},
            "kb_nums": {k: v for k, v in kb_nums.items()},
            "conflicts": conflicts,
            "amount_conflict": amount_conflict,
            "claim_amounts": list(claim_amounts),
            "kb_amounts": list(kb_amounts),
            "version_conflict": version_conflict,
            "claim_versions": list(claim_versions),
            "kb_versions": list(kb_versions),
            "has_numeric_conflict": bool(conflicts or amount_conflict or version_conflict),
        }

    @classmethod
    def format_for_prompt(cls, result: dict) -> str:
        """Format numeric comparison as human-readable prompt injection."""
        if not result.get("claim_nums") and not result.get("claim_amounts") and not result.get("claim_versions"):
            return "无待比对的数值"

        lines = []
        for unit, vals in result.get("claim_nums", {}).items():
            kb_vals = result.get("kb_nums", {}).get(unit, [])
            claim_str = str(vals[0]) if len(vals) == 1 else str(vals)
            kb_str = str(kb_vals[0]) if len(kb_vals) == 1 else str(kb_vals)
            if kb_vals and not any(cv in kb_vals for cv in vals):
                lines.append(f"  ⚠️ Claim={claim_str}{unit}, KB={kb_str}{unit} — 数值矛盾")
            elif kb_vals:
                lines.append(f"  ✅ Claim=KB={claim_str}{unit} — 一致")
            else:
                lines.append(f"  ❓ Claim={claim_str}{unit}, KB无此数值")

        if result.get("amount_conflict"):
            lines.append(
                f"  ⚠️ 金额矛盾: Claim金额={result['claim_amounts']}, "
                f"KB金额={result['kb_amounts']}"
            )
        if result.get("version_conflict"):
            lines.append(
                f"  ⚠️ 版本矛盾: Claim={result['claim_versions']}, "
                f"KB={result['kb_versions']}"
            )

        return "\n".join(lines) if lines else "数值一致，无矛盾"
