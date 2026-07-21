"""Pydantic data models and LangGraph state definition."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator


class ClaimItem(BaseModel):
    """An atomic claim extracted from a system reply."""
    claim_text: str = Field(
        description="Atomic claim text", min_length=1, max_length=500
    )
    claim_type: Literal["fact", "action_tool"] = Field(
        description="fact=factual assertion, action_tool=claimed tool execution"
    )
    nli_status: Optional[Literal["ENTAILED", "CONTRADICTED", "UNMENTIONED"]] = Field(
        default=None, description="NLI verification result"
    )
    reasoning: Optional[str] = Field(
        default=None, max_length=500, description="Verification reasoning"
    )
    tool_trace: Optional[dict] = Field(
        default=None,
        description="Deterministic tool execution trace: numeric, polarity, capability, kb_location"
    )


class NLIVerdict(BaseModel):
    """Structured NLI verification result for a single claim."""
    nli_status: Literal["ENTAILED", "CONTRADICTED", "UNMENTIONED"] = Field(
        description="NLI three-class verdict"
    )
    reasoning: str = Field(default="", description="Verification reasoning")


class HallucinationResult(BaseModel):
    """Final hallucination detection result for one reply."""
    id: str
    is_hallucination: bool
    hallucination_type: Optional[
        Literal[
            "能力越界", "安全误导", "参数编造", "信息编造",
            "政策编造", "优惠编造", "政策偏差", "信息遗漏", "无"
        ]
    ] = "无"
    severity: Literal["Critical", "High", "Medium", "None"] = "None"
    detail: str = Field(description="Analysis summary", max_length=800)
    claims: List[ClaimItem] = []

    @model_validator(mode='after')
    def enforce_consistency(self):
        """Cross-field consistency: hallucination type/severity must match is_hallucination.

        The severity-to-type mapping is intentionally strict — each hallucination
        type has exactly one valid severity level per the taxonomy definition.
        This prevents the LLM from producing inconsistent type/severity combinations.
        """
        valid_types = {
            "能力越界", "安全误导", "参数编造", "信息编造",
            "政策编造", "优惠编造", "政策偏差", "信息遗漏", "无"
        }
        if self.hallucination_type is not None and self.hallucination_type not in valid_types:
            raise ValueError(
                f"Invalid hallucination_type '{self.hallucination_type}'. "
                f"Must be one of: {valid_types}"
            )

        if not self.is_hallucination:
            if self.hallucination_type and self.hallucination_type != "无":
                raise ValueError(
                    f"is_hallucination=False but type is '{self.hallucination_type}'. "
                    f"Must be '无'."
                )
            if self.severity != "None":
                raise ValueError(
                    f"is_hallucination=False but severity is '{self.severity}'. "
                    f"Must be 'None'."
                )

        if self.is_hallucination and self.hallucination_type == "无":
            raise ValueError(
                "is_hallucination=True but type is '无'. "
                "Must be specific hallucination type."
            )

        # Severity-to-type mapping check (non-blocking — just ensure plausibility)
        critical_types = {"能力越界", "安全误导"}
        high_types = {"参数编造", "信息编造", "政策编造", "优惠编造"}
        medium_types = {"政策偏差", "信息遗漏"}

        if self.is_hallucination:
            if self.hallucination_type in critical_types and self.severity != "Critical":
                raise ValueError(
                    f"Type '{self.hallucination_type}' requires severity 'Critical', "
                    f"got '{self.severity}'"
                )
            if self.hallucination_type in high_types and self.severity != "High":
                raise ValueError(
                    f"Type '{self.hallucination_type}' requires severity 'High', "
                    f"got '{self.severity}'"
                )
            if self.hallucination_type in medium_types and self.severity != "Medium":
                raise ValueError(
                    f"Type '{self.hallucination_type}' requires severity 'Medium', "
                    f"got '{self.severity}'"
                )

        return self


class EvalMetrics(BaseModel):
    """Evaluation metrics comparing predictions to ground truth."""
    total: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    type_accuracy: float
    severity_accuracy: float
    fp_cases: List[str] = []
    fn_cases: List[str] = []
    type_mismatches: List[dict] = []
