"""Parse KB for capability boundaries — when KB says the system can't do something."""

import re
from typing import List


KB_NO_CAPABILITY_PATTERNS = [
    r"无[（(]",               # 无（系统未接入...）
    r"未接入",
    r"不具备",
    r"需转人工",
    r"需人工",
    r"无权限",
    r"系统不支持",
    r"暂不提供",
    r"无法(?:直接)?操作",
    r"手动处理",
    r"不可.*口头告知",
]

# Action verb patterns in claims — detecting capability claims
ACTION_CLAIM_PATTERNS = [
    r"已(?:帮|为)您(?:查询|修改|发送|升级|处理|安排|投诉|核实|备注|标记|取消|退款)",
    r"系统(?:已|显示|查到|查询到)",
    r"(?:已经|已经帮您|为您)(?:查询|修改|发送|处理)",
]

# Match KB phrases like "无关联品牌信息", "无满减活动" — KB explicitly says something doesn't exist
KB_EXPLICIT_NO_PATTERN = re.compile(
    r'无[一-鿿\d]+(?:的)?(?:活动|政策|优惠|功能|接口|门店|记录|入口|信息|关联|能力|权限|服务)'
)
# Match KB phrases like "未标注XX功能", "不支持XX" — features KB says are not documented
KB_UNMARKED_PATTERN = re.compile(
    r'(?:未标注|未提及|未接入|未开通|不支持|无(?!.*接口)[（(])[^）)]+'
)
# Match parenthetical negations like "无（客服系统未接入物流查询接口）"
KB_PARENTHETICAL_NO = re.compile(r'无[（(]([^）)]*)[）)]')


class CapabilityParser:
    """Parse KB to determine if the system lacks a claimed capability."""

    @classmethod
    def has_no_capability(cls, kb: str) -> bool:
        """Check if KB indicates the system lacks operational capability."""
        for pat in KB_NO_CAPABILITY_PATTERNS:
            if re.search(pat, kb):
                return True
        return False

    @classmethod
    def extract_no_descriptions(cls, kb: str) -> List[str]:
        """Extract descriptions of what the system cannot do."""
        # Parenthetical: 无（客服系统未接入物流查询接口）
        paren = KB_PARENTHETICAL_NO.findall(kb)
        # Explicit: 无关联品牌信息
        explicit = KB_EXPLICIT_NO_PATTERN.findall(kb)
        return [p.strip() for p in paren] + [e.strip() for e in explicit]

    @classmethod
    def extract_unmarked_features(cls, kb: str) -> List[str]:
        """Extract features KB says are '未标注' or '未提及'."""
        return [m.strip() for m in KB_UNMARKED_PATTERN.findall(kb)]

    @classmethod
    def check_claim_against_unmarked(cls, claim_text: str, kb: str) -> dict:
        """Check if a claim asserts a feature KB says is unmarked."""
        unmarked = cls.extract_unmarked_features(kb)
        claim_words = set(re.findall(r'[一-鿿]{2,}', claim_text))
        matched = []
        for item in unmarked:
            item_words = set(re.findall(r'[一-鿿]{2,}', item))
            if claim_words & item_words:
                matched.append(item[:80])
        return {
            "matched": matched,
            "has_match": len(matched) > 0,
        }

    @classmethod
    def check_claim_against_explicit_no(cls, claim_text: str, kb: str) -> dict:
        """Check if a claim asserts something KB explicitly says doesn't exist."""
        explicit = cls.extract_no_descriptions(kb)
        claim_words = set(re.findall(r'[一-鿿\d]{2,}', claim_text))
        matched = []
        for item in explicit:
            item_words = set(re.findall(r'[一-鿿\d]{2,}', item))
            if item_words & claim_words:
                matched.append(item[:80])
        return {
            "matched": matched,
            "has_match": len(matched) > 0,
        }

    @classmethod
    def run(cls, knowledge_base: str, claim_text: str = "", claim_type: str = "fact") -> dict:
        """Parse KB capability boundaries.

        Returns:
            dict with has_no_capability, descriptions, unmarked_match, explicit_no_match
        """
        result = {
            "has_no_capability": cls.has_no_capability(knowledge_base),
            "no_descriptions": cls.extract_no_descriptions(knowledge_base),
            "unmarked_features": cls.extract_unmarked_features(knowledge_base),
        }

        if claim_text:
            result["unmarked_check"] = cls.check_claim_against_unmarked(claim_text, knowledge_base)
            result["explicit_no_check"] = cls.check_claim_against_explicit_no(claim_text, knowledge_base)

        # Action tool + no capability = certain hallucination
        if claim_type == "action_tool" and result["has_no_capability"]:
            result["capability_violation"] = True
        else:
            result["capability_violation"] = False

        return result

    @classmethod
    def format_for_prompt(cls, result: dict) -> str:
        """Format capability results as human-readable prompt injection."""
        lines = []
        if result.get("has_no_capability"):
            lines.append(f"  🛡️ KB标注系统能力受限: {result.get('no_descriptions', [])}")
            if result.get("capability_violation"):
                lines.append("  ⚠️ 能力越界: Claim声称执行了KB明确不具备的操作")
        if result.get("unmarked_features"):
            lines.append(f"  📋 KB未标注的功能: {result['unmarked_features']}")
        if result.get("unmarked_check", {}).get("has_match"):
            lines.append(f"  ⚠️ Claim声称的功能在KB中标记为未标注: {result['unmarked_check']['matched']}")
        if result.get("explicit_no_check", {}).get("has_match"):
            lines.append(f"  ⚠️ Claim声称的内容在KB中被明确否定: {result['explicit_no_check']['matched']}")
        return "\n".join(lines) if lines else "无能力边界问题"
