"""RAGGuard REST API Server — FastAPI backend for hallucination detection."""

import json
import os
import sys
import time
import threading
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from src.engine import run_detection, LLMEngine
from src.runner import run_batch
from src.evaluator import evaluate, load_ground_truth
from src.patterns import analyze, format_for_report, save_profile, _make_json_safe
from src.visualization import generate_all_charts

os.makedirs("outputs", exist_ok=True)

app = FastAPI(title="RAGGuard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount static files for chart access ──────────────────────
if os.path.isdir("outputs"):
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


# ── Request / Response Models ────────────────────────────────

class SingleDetectRequest(BaseModel):
    id: str
    user_question: str
    system_reply: str
    knowledge_base: str


class BatchEvalRequest(BaseModel):
    mode: str = "mock"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    concurrency: int = 8


class RunListItem(BaseModel):
    run_id: str
    timestamp: str
    case_count: int
    hallucination_count: int


# ── Helpers ──────────────────────────────────────────────────

def _load_replies() -> list:
    with open("data/replies.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _discover_runs() -> list:
    """List timestamped run directories in outputs/."""
    runs = []
    if not os.path.isdir("outputs"):
        return runs
    for name in os.listdir("outputs"):
        path = os.path.join("outputs", name)
        if not os.path.isdir(path):
            continue
        results_file = os.path.join(path, "detection_results.json")
        if os.path.isfile(results_file):
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                h_count = sum(1 for r in results if r.get("is_hallucination"))
                runs.append({
                    "run_id": name,
                    "path": path,
                    "case_count": len(results),
                    "hallucination_count": h_count,
                })
            except Exception:
                pass
    runs.sort(key=lambda r: r["run_id"], reverse=True)
    return runs


# ── Background task tracker ──────────────────────────────────
# task_id → {"status": "running"|"done"|"error", "progress": (done, total),
#            "result": dict|None, "error": str|None}
_tasks: dict = {}
_tasks_lock = threading.Lock()


def _run_eval_background(task_id: str, replies: list, engine, concurrency: int,
                         run_dir: str):
    """Run batch evaluation in a background thread, updating _tasks progress."""
    try:
        def progress_callback(done, total):
            with _tasks_lock:
                if task_id in _tasks:
                    _tasks[task_id]["progress"] = (done, total)

        with _tasks_lock:
            _tasks[task_id]["progress"] = (0, len(replies))

        batch_results = run_batch(
            replies, engine=engine, workers=1,
            max_concurrency=concurrency,
            progress_callback=progress_callback,
        )
        results = [_make_json_safe(r.model_dump()) for r in batch_results]

        # Save results
        results_path = os.path.join(run_dir, "detection_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Evaluate against ground truth
        gt_path = "data/ground_truth.json"
        metrics = None
        if os.path.exists(gt_path):
            ground_truth = load_ground_truth(gt_path)
            metrics = evaluate(results, ground_truth)
            metrics_path = os.path.join(run_dir, "evaluation_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(_make_json_safe(metrics.model_dump()), f, ensure_ascii=False, indent=2)

        # Pattern analysis
        profile = analyze(results)
        profile_path = os.path.join(run_dir, "pattern_profile.json")
        save_profile(profile, profile_path)

        # Charts
        type_counts = {}
        for r in results:
            t = r.get("hallucination_type", "无")
            type_counts[t] = type_counts.get(t, 0) + 1
        generate_all_charts(
            type_counts=type_counts,
            severity_counts=profile.get("severity_distribution", {}),
            nli_counts=profile.get("nli_distribution", {}),
            output_dir=run_dir,
        )

        # Report
        report_path = os.path.join(run_dir, "report.md")
        if metrics:
            from run_eval import generate_report
            generate_report(results, metrics, report_path, chart_paths={})
        else:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("# RAGGuard 客服回复幻觉检测报告\n\n")
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(format_for_report(profile, chart_paths={}))

        h_count = sum(1 for r in batch_results if r.is_hallucination)
        with _tasks_lock:
            _tasks[task_id]["status"] = "done"
            _tasks[task_id]["result"] = {
                "run_id": os.path.basename(run_dir),
                "total_cases": len(results),
                "hallucination_count": h_count,
                "metrics": {
                    "precision": metrics.precision if metrics else None,
                    "recall": metrics.recall if metrics else None,
                    "f1": metrics.f1 if metrics else None,
                    "type_accuracy": metrics.type_accuracy if metrics else None,
                } if metrics else None,
            }
    except Exception as e:
        with _tasks_lock:
            _tasks[task_id]["status"] = "error"
            _tasks[task_id]["error"] = str(e)


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
def detect_single(req: SingleDetectRequest):
    """Run hallucination detection on a single case."""
    case = {
        "id": req.id,
        "user_question": req.user_question,
        "system_reply": req.system_reply,
        "knowledge_base": req.knowledge_base,
    }
    result = run_detection(case, engine=None)
    return _make_json_safe(result.model_dump())


@app.post("/evaluate")
def evaluate_batch(req: BatchEvalRequest):
    """Run batch hallucination detection on all 20 cases from replies.json."""
    replies = _load_replies()

    engine = None
    if req.mode == "llm":
        api_key = req.api_key or os.getenv("RAGGUARD_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = req.base_url or os.getenv("RAGGUARD_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model = req.model or os.getenv("RAGGUARD_MODEL") or "gpt-4.1-mini"
        if not api_key:
            raise HTTPException(400, "LLM mode requires API key")
        engine = LLMEngine(api_key=api_key, base_url=base_url, model=model)

    # Create run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("outputs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Run detection
    t_start = time.time()
    batch_results = run_batch(
        replies, engine=engine,
        workers=1,
        max_concurrency=req.concurrency,
    )
    results = [_make_json_safe(r.model_dump()) for r in batch_results]

    # Save results
    results_path = os.path.join(run_dir, "detection_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Evaluate against ground truth
    gt_path = "data/ground_truth.json"
    metrics = None
    if os.path.exists(gt_path):
        ground_truth = load_ground_truth(gt_path)
        metrics = evaluate(results, ground_truth)
        metrics_path = os.path.join(run_dir, "evaluation_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(_make_json_safe(metrics.model_dump()), f, ensure_ascii=False, indent=2)

    # Pattern analysis
    profile = analyze(results)
    profile_path = os.path.join(run_dir, "pattern_profile.json")
    save_profile(profile, profile_path)

    # Charts
    type_counts = {}
    for r in results:
        t = r.get("hallucination_type", "无")
        type_counts[t] = type_counts.get(t, 0) + 1
    chart_paths = generate_all_charts(
        type_counts=type_counts,
        severity_counts=profile.get("severity_distribution", {}),
        nli_counts=profile.get("nli_distribution", {}),
        output_dir=run_dir,
    )

    # Report
    chart_basenames = {k: os.path.basename(v) for k, v in chart_paths.items()}
    report_path = os.path.join(run_dir, "report.md")
    if metrics:
        from run_eval import generate_report
        generate_report(results, metrics, report_path, chart_paths=chart_basenames)
    else:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# RAGGuard 客服回复幻觉检测报告\n\n")
    pattern_report = format_for_report(profile, chart_paths=chart_basenames)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(pattern_report)

    elapsed = time.time() - t_start
    h_count = sum(1 for r in batch_results if r.is_hallucination)

    return {
        "run_id": run_id,
        "total_cases": len(results),
        "hallucination_count": h_count,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": {
            "precision": metrics.precision if metrics else None,
            "recall": metrics.recall if metrics else None,
            "f1": metrics.f1 if metrics else None,
            "type_accuracy": metrics.type_accuracy if metrics else None,
        } if metrics else None,
    }


@app.post("/evaluate/start")
def evaluate_start(req: BatchEvalRequest):
    """Start async batch evaluation, return task_id immediately for progress polling."""
    replies = _load_replies()

    engine = None
    if req.mode == "llm":
        api_key = req.api_key or os.getenv("RAGGUARD_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = req.base_url or os.getenv("RAGGUARD_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model = req.model or os.getenv("RAGGUARD_MODEL") or "gpt-4.1-mini"
        if not api_key:
            raise HTTPException(400, "LLM mode requires API key")
        engine = LLMEngine(api_key=api_key, base_url=base_url, model=model)

    task_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("outputs", task_id)
    os.makedirs(run_dir, exist_ok=True)

    with _tasks_lock:
        _tasks[task_id] = {"status": "running", "progress": (0, len(replies)),
                           "result": None, "error": None}

    thread = threading.Thread(
        target=_run_eval_background,
        args=(task_id, replies, engine, req.concurrency, run_dir),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id, "total": len(replies)}


@app.get("/evaluate/{task_id}")
def evaluate_status(task_id: str):
    """Get progress and (if done) results for an async evaluation task."""
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(404, f"Task {task_id} not found")

    progress = task["progress"]
    response = {
        "task_id": task_id,
        "status": task["status"],
        "done": progress[0] if progress else 0,
        "total": progress[1] if progress else 0,
    }
    if task["status"] == "done":
        response.update(task["result"])
        # Fetch full results
        results_path = os.path.join("outputs", task_id, "detection_results.json")
        if os.path.isfile(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                response["results"] = json.load(f)
        metrics_path = os.path.join("outputs", task_id, "evaluation_metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                response["metrics_data"] = json.load(f)
    elif task["status"] == "error":
        response["error"] = task["error"]

    return response


@app.get("/runs")
def list_runs():
    """List all previous evaluation runs."""
    return _discover_runs()


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    """Get full results for a specific run."""
    results_path = os.path.join("outputs", run_id, "detection_results.json")
    if not os.path.isfile(results_path):
        raise HTTPException(404, f"Run {run_id} not found")

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    h_count = sum(1 for r in results if r.get("is_hallucination"))

    # Load metrics if available
    metrics = None
    metrics_path = os.path.join("outputs", run_id, "evaluation_metrics.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    # Load report
    report = ""
    report_path = os.path.join("outputs", run_id, "report.md")
    if os.path.isfile(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report = f.read()

    # Discover chart files
    charts = {}
    run_dir = os.path.join("outputs", run_id)
    for fname in os.listdir(run_dir):
        if fname.endswith(".png"):
            charts[fname.replace(".png", "")] = f"/outputs/{run_id}/{fname}"

    return {
        "run_id": run_id,
        "results": results,
        "metrics": metrics,
        "report": report,
        "charts": charts,
        "total_cases": len(results),
        "hallucination_count": h_count,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
