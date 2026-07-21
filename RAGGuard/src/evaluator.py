"""Evaluation module: compare detection results with ground truth."""

import json
from typing import List
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from .state import HallucinationResult, EvalMetrics


def load_ground_truth(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(results: List[dict], ground_truth: List[dict]) -> EvalMetrics:
    """Compute evaluation metrics for hallucination detection.

    Args:
        results: List of detection result dicts
        ground_truth: List of ground truth dicts

    Returns:
        EvalMetrics with precision, recall, f1, type accuracy, severity accuracy
    """
    gt_dict = {item["id"]: item for item in ground_truth}
    total = len(results)

    y_true = []
    y_pred = []
    fp_cases = []
    fn_cases = []
    type_mismatches = []

    for pred in results:
        pid = pred["id"]
        gt = gt_dict.get(pid)
        if gt is None:
            continue

        y_true.append(gt["is_hallucination"])
        y_pred.append(pred["is_hallucination"])

        # Track false positives
        if pred["is_hallucination"] and not gt["is_hallucination"]:
            fp_cases.append(f"{pid}: predicted {pred.get('hallucination_type')}, actual non-hallucination")

        # Track false negatives
        if not pred["is_hallucination"] and gt["is_hallucination"]:
            fn_cases.append(f"{pid}: missed {gt.get('hallucination_type')}")

        # Track type mismatches (only for true positives)
        if pred["is_hallucination"] and gt["is_hallucination"]:
            pred_type = pred.get("hallucination_type", "")
            gt_type = gt.get("hallucination_type", "")
            if pred_type != gt_type:
                type_mismatches.append({
                    "id": pid, "predicted": pred_type, "actual": gt_type,
                    "detail_pred": pred.get("detail", ""), "detail_gt": gt.get("detail", ""),
                })

    if len(y_true) < 2 or len(set(y_true)) < 2:
        # Edge case: all same class
        p = r = f = 1.0 if (sum(y_true) == sum(y_pred)) else 0.0
    else:
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Type accuracy (among hallucination cases)
    hallucination_count = sum(1 for r in results if r["is_hallucination"]
                              and gt_dict.get(r["id"], {}).get("is_hallucination"))
    if hallucination_count > 0:
        type_correct = hallucination_count - len(type_mismatches)
        type_acc = type_correct / hallucination_count
    else:
        type_acc = 0.0

    # Severity accuracy (approximate)
    severity_map = {"Critical": 3, "High": 2, "Medium": 1, "None": 0}
    sev_correct = 0
    sev_total = 0
    for pred in results:
        pid = pred["id"]
        gt = gt_dict.get(pid)
        if gt is None or not gt.get("is_hallucination"):
            continue
        sev_total += 1
        # Compare severity: Critical/High = high agreement
        pred_sev = pred.get("severity", "None")
        # We consider "High" and "Critical" as similar tier for severity accuracy
        if pred_sev in ("Critical", "High") and gt.get("hallucination_type") in (
            "能力越界", "安全误导", "参数编造", "信息编造", "政策编造", "优惠编造"
        ):
            sev_correct += 1
        elif pred_sev == "Medium" and gt.get("hallucination_type") in ("政策偏差", "信息遗漏"):
            sev_correct += 1
    severity_acc = sev_correct / sev_total if sev_total > 0 else 0.0

    return EvalMetrics(
        total=total,
        tp=int(sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)),
        fp=int(sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)),
        fn=int(sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)),
        tn=int(sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)),
        precision=p, recall=r, f1=f,
        type_accuracy=type_acc, severity_accuracy=severity_acc,
        fp_cases=fp_cases, fn_cases=fn_cases, type_mismatches=type_mismatches,
    )
