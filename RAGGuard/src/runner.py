"""Async batch runner with two-tier concurrency control.

Two modes controlled by the ``workers`` parameter:

  workers=1 (default) — Single-process async
      All cases run in one process via asyncio + Semaphore.
      Best for development, debugging, and small datasets (< 100 cases).

  workers>1 — Multi-process Map-Reduce
      Cases are sharded across ``workers`` independent processes.
      Each process runs its own asyncio loop + Semaphore.
      Best for production throughput (1000+ cases).

  Total API concurrency = workers × (max_concurrency // workers), capped
  by ``max_concurrency`` to avoid rate limits.

Usage::

    from src.runner import run_batch

    # Single-process async (default)
    results = run_batch(cases, engine=engine)

    # 4 workers, 8 total API concurrency
    results = run_batch(cases, engine=engine, workers=4, max_concurrency=8)
"""

import asyncio
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

from .engine import run_detection, LLMEngine
from .state import HallucinationResult
from .logging_config import get_logger

logger = get_logger(__name__)


# ── Single-process async core ──────────────────────────────────

async def _process_case(case: dict, engine, semaphore: asyncio.Semaphore) -> HallucinationResult:
    """Process a single case under the global concurrency semaphore."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, run_detection, case, engine)


async def _run_async(cases: List[dict], engine, max_concurrency: int,
                     progress_callback=None) -> List[HallucinationResult]:
    """Process all cases in one process with asyncio + Semaphore.

    When progress_callback is provided, it is called as progress_callback(done, total)
    after each case completes.
    """
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [_process_case(c, engine, sem) for c in cases]

    # Use as_completed to report progress, then re-sort to input order
    results: List[HallucinationResult] = []
    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        if progress_callback:
            progress_callback(completed, len(cases))

    # Re-sort to match original case order
    id_to_result = {r.id: r for r in results}
    return [id_to_result[c["id"]] for c in cases]


# ── Multi-process Map-Reduce ────────────────────────────────────

def _worker_fn(args: tuple) -> List[HallucinationResult]:
    """Worker entry point for ProcessPoolExecutor.

    Args:
        args: (chunk_cases, engine_config, max_concurrency)
            engine_config is a dict with api_key, base_url, model.

    Returns:
        List of HallucinationResult (already model_dump'd for pickling safety).
    """
    chunk_cases, engine_config, max_concurrency = args
    worker_id = os.getpid()

    # Load .env in worker process (belt-and-suspenders)
    from dotenv import load_dotenv
    load_dotenv()

    engine = None
    if engine_config:
        engine = LLMEngine(
            api_key=engine_config["api_key"],
            base_url=engine_config.get("base_url", "https://api.openai.com/v1"),
            model=engine_config.get("model", "gpt-4.1-mini"),
            timeout=engine_config.get("timeout", 60.0),
            max_retries=engine_config.get("max_retries", 3),
        )

    t0 = time.time()
    logger.info("Worker %d: processing %d cases (concurrency=%d)",
                worker_id, len(chunk_cases), max_concurrency)
    results = asyncio.run(_run_async(chunk_cases, engine, max_concurrency))
    logger.info("Worker %d: done in %.1fs", worker_id, time.time() - t0)
    return results


# ── Public API ──────────────────────────────────────────────────

def run_batch(
    cases: List[dict],
    engine: Optional[LLMEngine] = None,
    workers: int = 1,
    max_concurrency: int = 8,
    progress_callback=None,
) -> List[HallucinationResult]:
    """Process cases with configurable concurrency.

    Args:
        cases: List of case dicts (id, user_question, system_reply, knowledge_base).
        engine: LLMEngine instance. None for Mock mode.
        workers: Number of parallel processes.
            - 1: Single-process asyncio (all cases in one process).
            - >1: ProcessPool Map-Reduce (cases sharded across workers).
        max_concurrency: Maximum concurrent API calls across all workers.
        progress_callback: Optional callable(done: int, total: int) per case.

    Returns:
        List of HallucinationResult, one per case in input order.
    """
    if not cases:
        logger.warning("run_batch: no cases provided")
        return []

    total = len(cases)
    t0 = time.time()

    if workers <= 1:
        # Single-process async
        logger.info("Batch start: %d cases | mode=asyncio | concurrency=%d",
                    total, max_concurrency)
        results = asyncio.run(_run_async(cases, engine, max_concurrency, progress_callback))
        elapsed = time.time() - t0
        logger.info("Batch done: %d cases in %.1fs (%.2fs/case)",
                    len(results), elapsed, elapsed / total)
        return results

    # Multi-process Map-Reduce
    # OpenAI client objects are not pickle-safe, so we serialize engine
    # config as a plain dict and reconstruct the engine in each worker process.
    engine_config = None
    if engine is not None:
        engine_config = {
            "api_key": getattr(engine, "_api_key", "") or "",
            "base_url": getattr(engine, "base_url", "https://api.openai.com/v1"),
            "model": getattr(engine, "model", "gpt-4.1-mini"),
            "timeout": getattr(engine, "timeout", 60.0),
            "max_retries": getattr(engine, "max_retries", 3),
        }

    # Shard cases across workers
    chunk_size = max(1, total // workers)
    chunks = []
    for i in range(0, total, chunk_size):
        chunks.append(cases[i:i + chunk_size])

    # Per-worker concurrency: divide total slots evenly
    per_worker_concurrency = max(1, max_concurrency // len(chunks))

    logger.info("Batch start: %d cases | mode=ProcessPool | workers=%d | "
                "chunks=%s | per_worker_concurrency=%d | total_concurrency=%d",
                total, workers,
                [len(c) for c in chunks],
                per_worker_concurrency, max_concurrency)

    # Spawn workers
    worker_args = [(chunk, engine_config, per_worker_concurrency) for chunk in chunks]
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        chunk_results = list(pool.map(_worker_fn, worker_args))

    # Flatten and preserve original order
    results = []
    for cr in chunk_results:
        results.extend(cr)

    # Re-sort to match input order (sharding may have reordered)
    id_to_result = {r.id: r for r in results}
    ordered = [id_to_result[c["id"]] for c in cases]

    elapsed = time.time() - t0
    logger.info("Batch done: %d cases in %.1fs (%.2fs/case) | "
                "hallucinations=%d",
                len(ordered), elapsed, elapsed / total,
                sum(1 for r in ordered if r.is_hallucination))
    return ordered
