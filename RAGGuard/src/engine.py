"""Dual-engine: Mock (rule-based) and LLM (OpenAI API) hallucination detection."""

import re
import json
import time
from typing import List, Optional
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .state import ClaimItem, HallucinationResult, NLIVerdict
from .prompts import (
    CLAIM_EXTRACTION_SYSTEM, CLAIM_EXTRACTION_USER,
    NLI_VERIFY_SYSTEM, NLI_VERIFY_USER,
    CLASSIFY_SYSTEM, CLASSIFY_USER,
)
from .validators import LLMOutputValidator
from .tools import get_registry
from .logging_config import get_logger

logger = get_logger(__name__)


# ── Mock Engine ──────────────────────────────────────────────

ACTION_KEYWORDS = [
    "已帮您", "已为您", "已帮", "已升级", "已处理", "已修改",
    "已发送", "已安排", "已备注", "已核实", "已确认",
    "直接发到", "我帮您查", "帮您查了", "帮您查询", "帮您确认",
    "升级为", "已将", "转人工", "已转接",
]

GREETING_PATTERNS = [
    r"^(您好|你好|亲爱的用户|尊敬的客户)[，,！!]?$",
    r"^(非常)?抱歉(给[您你]带来(了)?(不便|不好的体验|困扰))?[，,！!]?$",
    r"^(感谢|谢谢)[您你]的?.*[！!。.]?$",
    r"^(如有|如果).*(问题|疑问|需要).*(请联系|随时联系|咨询).*$",
    r"^祝您.*$",
]

KB_NO_CAPABILITY_PATTERNS = [
    r"无[（(]",
    r"未接入", r"不具备", r"需转人工", r"需人工",
    r"无权限", r"系统不支持", r"暂不提供", r"手动处理",
    r"不可.*口头告知",
]

SAFETY_RISK_PATTERNS = [
    (r"(放心使用|绝对安全|完全安全|没有任何问题|肯定没问题)", ["咨询医生", "慎用", "禁用", "不建议", "遵医嘱"]),
]

NUMERIC_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(天|工作日|小时|元|块|折|%|版本|年|月|ms|毫秒)'
)
# Amount patterns: "满300减50", "满200减20" etc.
AMOUNT_PATTERN = re.compile(r'[满减]\s*(\d+(?:\.\d+)?)')
# Version pattern: "蓝牙5.3", "版本5.0"
VERSION_PATTERN = re.compile(r'(?:蓝牙|版本)\s*(\d+(?:\.\d+)?)')


def _is_greeting(text: str) -> bool:
    for pat in GREETING_PATTERNS:
        if re.match(pat, text.strip()):
            return True
    return False


def _is_action_claim(text: str) -> bool:
    for kw in ACTION_KEYWORDS:
        if kw in text:
            return True
    return False


def _kb_has_no_capability(kb: str) -> bool:
    for pat in KB_NO_CAPABILITY_PATTERNS:
        if re.search(pat, kb):
            return True
    return False


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, handling Chinese punctuation."""
    parts = re.split(r'[。！!；;？?\n]', text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]


def _extract_numerics(text: str) -> dict:
    """Extract numeric claims from text. Returns dict of {unit: [values]}."""
    result = {}
    for m in NUMERIC_PATTERN.finditer(text):
        value = float(m.group(1))
        unit = m.group(2)
        if unit not in result:
            result[unit] = []
        result[unit].append(value)
    return result


def _check_negation_conflict(claim: str, kb: str) -> Optional[str]:
    """Check for direct negation conflicts between claim and KB."""
    claim_lower = claim.lower()
    kb_lower = kb.lower()

    negation_pairs = [
        # (claim_positive_pattern, claim_negative_reject, kb_negative_pattern, description)
        # claim_negative_reject: if this pattern is found before the positive match, skip
        (r"(?<!不)支持.*货到付款", r"不支持.*货到付款", r"不支持.*货到付款", "货到付款"),
        (r"(?<!不)支持.*纸质发票", r"不支持.*纸质发票", r"(不支持|暂不支持).*纸质发票", "纸质发票"),
        (r"有.*线下", r"没有.*线下", r"(纯线上|无.*线下|无线下|纯线上电商)", "线下门店"),
        (r"(?<!不)支持.*线下", r"不支持.*线下", r"(纯线上|无.*线下|无线下|纯线上电商)", "线下门店"),
    ]

    for pos_pattern, neg_reject_pattern, kb_neg_pattern, desc in negation_pairs:
        # Skip if claim actually negates (e.g., "不支持货到付款")
        if re.search(neg_reject_pattern, claim):
            continue
        if re.search(pos_pattern, claim):
            if re.search(kb_neg_pattern, kb):
                return f"Claim肯定但KB否定: {desc}"
    return None


def mock_extract_claims(system_reply: str) -> List[ClaimItem]:
    """Extract atomic claims from reply using rules."""
    sentences = _split_sentences(system_reply)
    claims = []

    for sent in sentences:
        if _is_greeting(sent):
            continue

        claim_type = "action_tool" if _is_action_claim(sent) else "fact"
        claims.append(ClaimItem(claim_text=sent, claim_type=claim_type))

    return claims


def mock_capability_guard(claims: List[ClaimItem], knowledge_base: str) -> List[ClaimItem]:
    """Check action_tool claims against KB capability and build tool_trace.

    Builds the tool_trace for every action_tool claim (even non-violations)
    so the pattern profile can report per-claim tool detection statistics.
    """
    from .tools.registry import get_registry
    registry = get_registry()
    for claim in claims:
        if claim.claim_type == "action_tool":
            results = registry.run_all(claim.claim_text, knowledge_base, claim.claim_type)
            claim.tool_trace = registry.build_tool_trace(results)
            if registry.has_capability_violation(results):
                claim.nli_status = "CONTRADICTED"
                claim.reasoning = "能力越界：知识库表明系统不具备该操作能力"
    return claims


def mock_verify_claim(claim: ClaimItem, knowledge_base: str) -> ClaimItem:
    """Verify a single claim against KB using rules.

    Checks are ordered from most specific/high-confidence to most general,
    so we return early on the strongest signal rather than a weaker one.
    """
    if claim.nli_status is not None:
        return claim  # Already judged by capability guard

    claim_text = claim.claim_text
    kb = knowledge_base

    # 1. Check safety risk first — highest priority, can't be overridden by weaker signals
    for safe_pattern, risk_keywords in SAFETY_RISK_PATTERNS:
        if re.search(safe_pattern, claim_text):
            for rk in risk_keywords:
                if rk in kb:
                    claim.nli_status = "CONTRADICTED"
                    claim.reasoning = f"安全误导：回复宣称安全({safe_pattern})，但KB标注风险({rk})"
                    return claim

    # 2. Check negation conflicts
    conflict_reason = _check_negation_conflict(claim_text, kb)
    if conflict_reason:
        claim.nli_status = "CONTRADICTED"
        claim.reasoning = conflict_reason
        return claim

    # 3. Compare numeric values (standard units)
    claim_nums = _extract_numerics(claim_text)
    kb_nums = _extract_numerics(kb)

    for unit, claim_vals in claim_nums.items():
        kb_vals = kb_nums.get(unit, [])
        if kb_vals and not any(cv in kb_vals for cv in claim_vals):
            claim_str = str(claim_vals[0]) if len(claim_vals) == 1 else str(claim_vals)
            kb_str = str(kb_vals[0]) if len(kb_vals) == 1 else str(kb_vals)
            claim.nli_status = "CONTRADICTED"
            claim.reasoning = f"数值矛盾：Claim {claim_str}{unit} vs KB {kb_str}{unit}"
            return claim

    # 3b. Compare amounts (满X减Y patterns)
    claim_amounts = set(int(m) for m in AMOUNT_PATTERN.findall(claim_text))
    kb_amounts = set(int(m) for m in AMOUNT_PATTERN.findall(kb))
    if claim_amounts and kb_amounts and claim_amounts != kb_amounts:
        claim.nli_status = "CONTRADICTED"
        claim.reasoning = f"金额矛盾：Claim金额 {claim_amounts} vs KB金额 {kb_amounts}"
        return claim

    # 3c. Compare version numbers
    claim_versions = set(VERSION_PATTERN.findall(claim_text))
    kb_versions = set(VERSION_PATTERN.findall(kb))
    if claim_versions and kb_versions and claim_versions != kb_versions:
        claim.nli_status = "CONTRADICTED"
        claim.reasoning = f"版本矛盾：Claim {claim_versions} vs KB {kb_versions}"
        return claim

    # 4. KB "未标注/未提及" check: KB says feature not documented, claim asserts it
    kb_unmarked = re.findall(r'(?:未标注|未提及|未接入|无(?!.*接口)[（(])[^）)]+', kb)
    for unmarked in kb_unmarked:
        # Extract the feature being negated
        feature_words = set(re.findall(r'[一-鿿]{2,}', unmarked))
        claim_words = set(re.findall(r'[一-鿿]{2,}', claim_text))
        if feature_words & claim_words:
            claim.nli_status = "CONTRADICTED"
            claim.reasoning = f"KB未标注/未提及该功能: {unmarked[:50]}"
            return claim

    # 4b. KB explicit negation: "无X的活动/政策/功能" while claim asserts X
    claim_words_temp = set(re.findall(r'[一-鿿\d]{2,}', claim_text))
    kb_explicit_no = re.findall(r'无[一-鿿\d]+(?:的)?(?:活动|政策|优惠|功能|接口|门店|记录|入口|信息|关联)', kb)
    for no_item in kb_explicit_no:
        no_kws = set(re.findall(r'[一-鿿\d]{2,}', no_item))
        if no_kws & claim_words_temp:
            claim.nli_status = "CONTRADICTED"
            claim.reasoning = f"KB明确否定: {no_item[:50]}"
            return claim

    # 5. Check if KB says "无" about a capability while claim asserts it
    kb_no_cap = re.findall(r'无[（(]([^）)]*)[）)]', kb)
    for cap_desc in kb_no_cap:
        cap_words = set(re.findall(r'[一-鿿]{2,}', cap_desc))
        claim_words = set(re.findall(r'[一-鿿]{2,}', claim_text))
        if cap_words & claim_words:
            claim.nli_status = "CONTRADICTED"
            claim.reasoning = f"KB标注无此能力: {cap_desc[:50]}"
            return claim

    # 6. Keyword negation check — only flag contradiction when the positive/negative
    # words in claim and KB refer to the SAME topic (>=2 overlapping CJK bigrams).
    # This prevents false positives from unrelated positive/negative word pairs.
    claim_kws = set(re.findall(r'[一-鿿]{2,}', claim_text))
    kb_kws = set(re.findall(r'[一-鿿]{2,}', kb))

    # Claim says positive, KB says negative about same topic
    positive_words = {"支持", "可以", "能够", "提供", "包含", "具有", "采用", "是"}
    negative_words_in_kb = {"不支持", "不可以", "不能够", "不提供", "不包含", "不具有", "暂不支持", "纯线上"}

    for pw in positive_words:
        # Skip if claim's use of positive word is preceded by negation
        if pw in claim_text and not claim_text.startswith("不" + pw):
            for nw in negative_words_in_kb:
                if nw in kb:
                    # Check if KB negation is about the SAME thing claim affirms
                    # Extract what follows the negation in KB
                    kb_neg_idx = kb.find(nw)
                    kb_neg_context = kb[kb_neg_idx:kb_neg_idx + len(nw) + 20] if kb_neg_idx >= 0 else ""
                    kb_neg_kws = set(re.findall(r'[一-鿿]{2,}', kb_neg_context))

                    # Extract what follows the positive in claim
                    claim_pos_idx = claim_text.find(pw)
                    claim_pos_context = claim_text[claim_pos_idx:claim_pos_idx + len(pw) + 30] if claim_pos_idx >= 0 else ""
                    claim_pos_kws = set(re.findall(r'[一-鿿]{2,}', claim_pos_context))

                    specific_overlap = kb_neg_kws & claim_pos_kws
                    # Also check: if the full claim already contains the KB negation,
                    # it means the claim correctly states the negation, skip
                    if len(specific_overlap) >= 2 and nw not in claim_text:
                        claim.nli_status = "CONTRADICTED"
                        claim.reasoning = f"极性矛盾：Claim肯定({pw})，KB否定({nw})，重叠: {specific_overlap}"
                        return claim

    # 7. Default: use character bigram overlap for robust matching
    def _char_bigrams(text: str) -> set:
        """Extract character bigrams for fuzzy CJK matching."""
        # Keep only CJK chars and digits
        clean = re.sub(r'[^一-鿿\d]', '', text)
        return {clean[i:i+2] for i in range(len(clean)-1)} if len(clean) >= 2 else set()

    claim_bigrams = _char_bigrams(claim_text)
    kb_bigrams = _char_bigrams(kb)
    bigram_overlap = claim_bigrams & kb_bigrams
    overlap_ratio = len(bigram_overlap) / max(len(claim_bigrams), 1)

    if overlap_ratio >= 0.3:
        claim.nli_status = "ENTAILED"
        claim.reasoning = f"语义匹配：bigram重叠率 {overlap_ratio:.0%}"
    elif len(bigram_overlap) >= 3:
        claim.nli_status = "ENTAILED"
        claim.reasoning = f"部分语义匹配：{len(bigram_overlap)}个bigram重叠"
    else:
        claim.nli_status = "UNMENTIONED"
        claim.reasoning = f"KB未充分覆盖该声明 (bigram重叠率 {overlap_ratio:.0%})"

    return claim


def mock_classify(case_id: str, user_question: str, system_reply: str,
                  knowledge_base: str, claims: List[ClaimItem]) -> HallucinationResult:
    """Classify hallucination type and severity based on claim verdicts."""
    contradicted = [c for c in claims if c.nli_status == "CONTRADICTED"]
    unmentioned = [c for c in claims if c.nli_status == "UNMENTIONED"]

    has_action_hallucination = any(
        c.claim_type == "action_tool" and c.nli_status == "CONTRADICTED"
        for c in claims
    )

    if not contradicted and not unmentioned:
        return HallucinationResult(
            id=case_id, is_hallucination=False,
            hallucination_type="无", severity="None",
            detail="所有声明均通过验证", claims=claims
        )

    # Determine type
    # Determine type based on claim content and verdict patterns
    all_claim_text = " ".join(c.claim_text for c in claims)

    # Capability overreach (action_tool claims that are contradicted)
    if has_action_hallucination:
        h_type = "能力越界"
        severity = "Critical"
    # Safety misleading
    elif any("安全" in (c.reasoning or "") for c in contradicted):
        h_type = "安全误导"
        severity = "Critical"
    # Policy/rule contradictions
    elif any(kw in all_claim_text for kw in ["退货", "退款", "运费", "货到付款", "发票", "换货",
                                                "无理由", "质保", "保修", "发货"]):
        # Check if it's total fabrication or partial deviation
        if contradicted and any("数值矛盾" in (c.reasoning or "") for c in contradicted):
            h_type = "参数编造"
            severity = "High"
        elif contradicted and any("极性矛盾" in (c.reasoning or "") for c in contradicted):
            h_type = "政策编造"
            severity = "High"
        elif contradicted:
            h_type = "政策偏差"
            severity = "Medium"
        else:
            h_type = "政策偏差"
            severity = "Medium"
    # Product parameter contradictions
    elif any(kw in all_claim_text for kw in ["蓝牙", "NFC", "Type-C", "USB", "接口", "材质",
                                                "牛皮", "PU", "硅胶", "版本", "参数"]):
        h_type = "参数编造"
        severity = "High"
    # Promotion/coupon contradictions
    elif any(kw in all_claim_text for kw in ["优惠", "满减", "折扣", "学生", "优惠券", "减"]):
        h_type = "优惠编造"
        severity = "High"
    # Address/store/brand fabrications
    elif any(kw in all_claim_text for kw in ["地址", "门店", "线下", "品牌", "旗下"]):
        h_type = "信息编造"
        severity = "High"
    # Specific location/entity fabrications
    elif contradicted and any("地址" in c.claim_text or "门店" in c.claim_text
                              or "品牌" in c.claim_text for c in contradicted):
        h_type = "信息编造"
        severity = "High"
    # Number contradictions are parameter fabrication
    elif contradicted and any("数值" in (c.reasoning or "") for c in contradicted):
        h_type = "参数编造"
        severity = "High"
    # Remaining contradicted -> parameter fabrication by default
    elif contradicted:
        h_type = "参数编造"
        severity = "High"
    # Information omission (UNMENTIONED but important context missing)
    elif unmentioned and any("标准" in c.claim_text or "偏" in c.claim_text for c in unmentioned):
        h_type = "信息遗漏"
        severity = "Medium"
    elif unmentioned:
        h_type = "信息遗漏"
        severity = "Medium"
    else:
        h_type = "信息遗漏"
        severity = "Medium"

    detail_parts = []
    if contradicted:
        detail_parts.append(f"矛盾声明({len(contradicted)}条):")
        for c in contradicted:
            detail_parts.append(f"  - [{c.claim_text}] → {c.reasoning}")
    if unmentioned:
        detail_parts.append(f"未验证声明({len(unmentioned)}条):")
        for c in unmentioned:
            detail_parts.append(f"  - [{c.claim_text}]")

    return HallucinationResult(
        id=case_id, is_hallucination=True,
        hallucination_type=h_type, severity=severity,
        detail="\n".join(detail_parts), claims=claims,
    )


# ── LLM Engine ───────────────────────────────────────────────

class LLMEngine:
    """LLM-based hallucination detection using OpenAI API."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4.1-mini", temperature: float = 0.0,
                 timeout: float = 60.0, max_retries: int = 3):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self._api_key = api_key  # stored for ProcessPool worker serialization

    def _call(self, system: str, user: str, response_format: Optional[dict] = None) -> str:
        return self._call_with_retry(system, user, response_format)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((Exception,)),
    )
    def _call_with_retry(self, system: str, user: str,
                         response_format: Optional[dict] = None) -> str:
        kwargs = dict(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            timeout=self.timeout,
        )
        if response_format:
            kwargs["response_format"] = response_format

        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(**kwargs)
            elapsed = time.time() - t0
            content = resp.choices[0].message.content
            logger.debug("API call OK | model=%s | elapsed=%.2fs | response_len=%d",
                         self.model, elapsed, len(content or ""))
            return content
        except Exception as e:
            elapsed = time.time() - t0
            logger.warning("API call FAILED | model=%s | elapsed=%.2fs | error=%s",
                          self.model, elapsed, str(e)[:200])
            raise

    def extract_claims(self, user_question: str, system_reply: str) -> List[ClaimItem]:
        user_prompt = CLAIM_EXTRACTION_USER.format(
            user_question=user_question, system_reply=system_reply
        )
        content = self._call(CLAIM_EXTRACTION_SYSTEM, user_prompt)
        try:
            data = LLMOutputValidator.parse_json(content)
        except json.JSONDecodeError:
            logger.warning("extract_claims: JSON parse failed, using mock fallback")
            return mock_extract_claims(system_reply)

        if not isinstance(data, list):
            logger.warning("extract_claims: output is not a list, using mock fallback")
            return mock_extract_claims(system_reply)

        claims = []
        for item in data:
            try:
                claims.append(ClaimItem.model_validate(item))
            except Exception as e:
                logger.debug("extract_claims: skipping invalid claim item: %s", str(e)[:100])
                continue

        if not claims:
            logger.warning("extract_claims: no valid claims parsed, using mock fallback")
            return mock_extract_claims(system_reply)

        logger.debug("extract_claims: extracted %d claims (fact=%d, action_tool=%d)",
                     len(claims),
                     sum(1 for c in claims if c.claim_type == "fact"),
                     sum(1 for c in claims if c.claim_type == "action_tool"))
        return claims

    def verify_claim(self, claim: ClaimItem, knowledge_base: str) -> ClaimItem:
        # Pre-compute all tools via unified registry
        registry = get_registry()
        tool_results = registry.run_all(
            claim.claim_text, knowledge_base, claim.claim_type
        )

        # Record tool trace for auditability
        claim.tool_trace = registry.build_tool_trace(tool_results)

        logger.debug("verify_claim tools: numeric_conflict=%s polarity_conflict=%s cap_violation=%s kb_overlap=%.0f%%",
                     registry.has_numeric_conflict(tool_results),
                     registry.has_polarity_conflict(tool_results),
                     registry.has_capability_violation(tool_results),
                     tool_results.get("kb_location", {}).get("overlap_ratio", 0) * 100)

        # Deterministic fast path: capability violation
        if registry.has_capability_violation(tool_results):
            logger.info("verify_claim: capability violation — skipping LLM, CONTRADICTED")
            claim.nli_status = "CONTRADICTED"
            claim.reasoning = "能力越界：知识库表明系统不具备该操作能力（工具判定）"
            return claim

        # Inject tool results into system prompt
        tool_block = registry.format_for_prompt(tool_results)
        system_with_tools = NLI_VERIFY_SYSTEM + "\n\n" + tool_block

        # Use relevance-filtered KB to reduce token consumption
        filtered_kb = registry.get_filtered_kb(tool_results, knowledge_base)

        user_prompt = NLI_VERIFY_USER.format(
            claim_text=claim.claim_text,
            claim_type=claim.claim_type,
            knowledge_base=filtered_kb,
        )
        content = self._call(system_with_tools, user_prompt)

        # Parse with validator
        try:
            verdict = LLMOutputValidator.parse(content, NLIVerdict)
            nli_status = verdict.nli_status
            reasoning = verdict.reasoning
            logger.debug("verify_claim NLI: %s", nli_status)
        except json.JSONDecodeError:
            logger.warning("verify_claim: JSON parse failed, using text fallback")
            if "CONTRADICTED" in content:
                nli_status = "CONTRADICTED"
            elif "ENTAILED" in content:
                nli_status = "ENTAILED"
            else:
                nli_status = "UNMENTIONED"
            reasoning = content[:200]
        except Exception as e:
            logger.warning("verify_claim: validation error — %s", str(e)[:100])
            if "CONTRADICTED" in content:
                nli_status = "CONTRADICTED"
            elif "ENTAILED" in content:
                nli_status = "ENTAILED"
            else:
                nli_status = "UNMENTIONED"
            reasoning = content[:200]

        # Post-validate: deterministic tools are authoritative for numerics and polarity.
        # If the tool detected a conflict but the LLM missed it (or was uncertain),
        # we override to CONTRADICTED to prevent false negatives from LLM errors.
        if registry.has_numeric_conflict(tool_results) and nli_status != "CONTRADICTED":
            logger.warning("verify_claim: TOOL OVERRIDE — numeric conflict, LLM said %s → CONTRADICTED",
                          nli_status)
            nli_status = "CONTRADICTED"
            reasoning = (
                f"[工具覆写] 数值矛盾，LLM 未识别。"
                f"原reasoning: {reasoning}"
            )[:300]

        if registry.has_polarity_conflict(tool_results) and nli_status != "CONTRADICTED":
            logger.warning("verify_claim: TOOL OVERRIDE — polarity conflict, LLM said %s → CONTRADICTED",
                          nli_status)
            nli_status = "CONTRADICTED"
            reasoning = (
                f"[工具覆写] 极性矛盾，LLM 未识别。"
                f"原reasoning: {reasoning}"
            )[:300]

        claim.nli_status = nli_status
        claim.reasoning = reasoning
        return claim

    def classify(self, case_id: str, user_question: str, system_reply: str,
                 knowledge_base: str, claims: List[ClaimItem]) -> HallucinationResult:
        n_contradicted = sum(1 for c in claims if c.nli_status == "CONTRADICTED")
        n_unmentioned = sum(1 for c in claims if c.nli_status == "UNMENTIONED")
        logger.debug("classify: case=%s | contradicted=%d unmentioned=%d total_claims=%d",
                     case_id, n_contradicted, n_unmentioned, len(claims))

        claims_summary = "\n".join(
            f"  [{i+1}] {c.claim_text} (type={c.claim_type}, nli={c.nli_status}, reason={c.reasoning})"
            for i, c in enumerate(claims)
        )
        user_prompt = CLASSIFY_USER.format(
            case_id=case_id,
            user_question=user_question,
            system_reply=system_reply,
            knowledge_base=knowledge_base,
            claims_summary=claims_summary,
        )
        content = self._call(CLASSIFY_SYSTEM, user_prompt)

        # Parse with Pydantic validator (includes cross-field model_validator)
        # LLM doesn't output id/claims — inject them before validation
        try:
            data = LLMOutputValidator.parse_json(content)
            data = LLMOutputValidator.auto_fix(data, HallucinationResult)
            data["id"] = case_id
            data["claims"] = claims
            result = LLMOutputValidator.validate(data, HallucinationResult)
            logger.debug("classify: case=%s type=%s severity=%s",
                         case_id, result.hallucination_type, result.severity)
            return result
        except json.JSONDecodeError:
            logger.warning("classify: JSON parse failed for %s, using mock fallback", case_id)
            return mock_classify(case_id, user_question, system_reply, knowledge_base, claims)
        except Exception as e:
            logger.warning("classify: validation error for %s — %s, using mock fallback",
                          case_id, str(e)[:100])
            return mock_classify(case_id, user_question, system_reply, knowledge_base, claims)


# ── Unified Pipeline Runner ──────────────────────────────────

def run_detection(case: dict, engine: Optional[LLMEngine] = None) -> HallucinationResult:
    """Run hallucination detection on a single case.

    Args:
        case: dict with id, user_question, system_reply, knowledge_base
        engine: LLMEngine instance for LLM mode, None for mock mode

    Returns:
        HallucinationResult
    """
    case_id = case["id"]
    user_question = case["user_question"]
    system_reply = case["system_reply"]
    knowledge_base = case["knowledge_base"]

    logger.debug("=== Processing %s ===", case_id)
    logger.debug("  user_question len=%d, reply len=%d, kb len=%d",
                 len(user_question), len(system_reply), len(knowledge_base))

    # Step 1: Extract claims
    if engine:
        claims = engine.extract_claims(user_question, system_reply)
    else:
        claims = mock_extract_claims(system_reply)

    if not claims:
        logger.info("%s: no verifiable claims extracted → OK", case_id)
        return HallucinationResult(
            id=case_id, is_hallucination=False,
            hallucination_type="无", severity="None",
            detail="未提取到可验证的声明", claims=[]
        )

    # Step 2: Capability guard for action_tool claims
    # In LLM mode, let verify_claim handle it (builds tool_trace + fast path)
    # In mock mode, use mock_capability_guard
    if engine is None:
        claims = mock_capability_guard(claims, knowledge_base)

    # Step 3: Verify each claim (serial, concurrency handled by outer async layer)
    for i, claim in enumerate(claims):
        if claim.nli_status is not None:
            continue
        if engine:
            claims[i] = engine.verify_claim(claim, knowledge_base)
        else:
            claims[i] = mock_verify_claim(claim, knowledge_base)

    # Step 4: Classify
    if engine:
        result = engine.classify(case_id, user_question, system_reply, knowledge_base, claims)
    else:
        result = mock_classify(case_id, user_question, system_reply, knowledge_base, claims)

    logger.info("%s: %s | type=%s severity=%s claims=%d",
                case_id,
                "HALLUCINATION" if result.is_hallucination else "OK",
                result.hallucination_type, result.severity, len(claims))
    return result
