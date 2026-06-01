"""
main.py — Jatayu Fraud Detection API

Endpoints:
  POST /upload              — accept CSV, return task_id, start background processing
  WS   /ws/{task_id}        — real-time progress stream (WebSocket)
  GET  /results/{task_id}   — fetch processed transaction results
  GET  /summary/{task_id}   — high-level metrics for dashboard
  GET  /audit               — audit trail (all tasks or filtered by task_id)
  GET  /tasks               — list all uploaded tasks

Serve the frontend from /static and / -> login.html.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import logging
import uuid
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database import (
    create_task,
    get_audit_records,
    get_results,
    get_task,
    init_db,
    list_tasks,
    save_audit_record,
    save_transaction,
    update_task,
)

# Import new routers for real-time system
from routers import (
    auth_router,
    users_router,
    transactions_router,
    admin_router,
    merchant_router,
    support_router,
)

# ══════════════════════════════════════════════════════════════════════════════
# Logging Configuration - Enhanced for fraud detection observability
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    """Configure comprehensive logging for the fraud detection system."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    log_format = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []
    root_logger.addHandler(console_handler)
    
    # Set specific log levels for agent modules (more verbose)
    logging.getLogger("agents").setLevel(logging.INFO)
    logging.getLogger("agents.pattern_detection_agent").setLevel(logging.INFO)
    logging.getLogger("agents.transaction_monitoring_agent").setLevel(logging.INFO)
    logging.getLogger("agents.risk_assessment_agent").setLevel(logging.INFO)
    logging.getLogger("agents.alert_block_agent").setLevel(logging.INFO)
    logging.getLogger("agents.compliance_logging_agent").setLevel(logging.INFO)
    
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()
logger.info("=" * 70)
logger.info("🛡️  JATAYU FRAUD DETECTION SYSTEM - Starting Up")
logger.info("=" * 70)

# ── In-memory task state ──────────────────────────────────────────────────────
# task_queues: streams WebSocket events to connected clients
task_queues: dict[str, asyncio.Queue] = {}
# intelligence_cache: caches LLM results per task so re-visits don't re-generate
intelligence_cache: dict[str, dict] = {}


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialised.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Jatayu AFDRN", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register API routers for real-time system ─────────────────────────────────
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(transactions_router)
app.include_router(admin_router)
app.include_router(merchant_router)
app.include_router(support_router)

STATIC_DIR = PROJECT_ROOT / "frontend"
if not STATIC_DIR.exists():
    STATIC_DIR = BACKEND_DIR / "static"


# ── CSV column normalisation ──────────────────────────────────────────────────

PAYSIM_COLUMNS = {
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
}


def _row_to_msg(row: pd.Series):
    """Convert a CSV row to a TransactionMessage. Returns None on failure."""
    try:
        from agents.models import TransactionMessage, TransactionType, TrafficMode

        type_map = {t.value: t for t in TransactionType}
        raw_type = str(row.get("type", "PAYMENT")).upper().replace("-", "_")
        txn_type = type_map.get(raw_type, TransactionType.PAYMENT)

        return TransactionMessage(
            step=int(row.get("step", 0)),
            type=txn_type,
            amount=float(row.get("amount", 0.0)),
            nameOrig=str(row.get("nameOrig", "")),
            nameDest=str(row.get("nameDest", "")),
            oldbalanceOrg=float(row.get("oldbalanceOrg", 0.0)),
            newbalanceOrig=float(row.get("newbalanceOrig", 0.0)),
            oldbalanceDest=float(row.get("oldbalanceDest", 0.0)),
            newbalanceDest=float(row.get("newbalanceDest", 0.0)),
            ip_address=str(row.get("ip_address", f"ip_{uuid.uuid4().hex[:4]}")),
            device_id=str(row.get("device_id", f"device_{uuid.uuid4().hex[:4]}")),
            ground_truth_label=bool(row.get("isFraud", 0)) if "isFraud" in row.index else None,
        )
    except Exception as exc:
        logger.warning("Row conversion failed: %s", exc)
        return None


# ── Background processing ─────────────────────────────────────────────────────

async def _emit(task_id: str, event: dict) -> None:
    """Push an event to the task's WebSocket queue."""
    q = task_queues.get(task_id)
    if q:
        await q.put(event)


async def _process_csv(task_id: str, csv_bytes: bytes, filename: str) -> None:
    """
    Full pipeline:
      1. Parse CSV with pandas (up to 500 rows for demo responsiveness)
      2. Route each row through the Orchestrator on a thread-pool executor
      3. Stream progress events via the WebSocket queue
      4. Persist results to SQLite
    """
    loop = asyncio.get_event_loop()

    # ── 1. Parse CSV ──────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes), nrows=500)
    except Exception as exc:
        await _emit(task_id, {"status": "error", "message": str(exc)})
        update_task(task_id, status="error")
        return

    total = len(df)
    update_task(task_id, total_rows=total, status="processing")
    await _emit(task_id, {"status": "started", "total": total, "filename": filename})

    # ── 2. Initialise orchestrator (once, on thread pool) ─────────────────────
    try:
        from agents.orchestrator import Orchestrator
        orchestrator = await loop.run_in_executor(None, Orchestrator)
        use_real_agents = True
        logger.info("[%s] Real orchestrator loaded.", task_id)
    except Exception as exc:
        logger.warning("[%s] Orchestrator unavailable (%s). Using mock.", task_id, exc)
        orchestrator = None
        use_real_agents = False

    # ── 3. Process rows ───────────────────────────────────────────────────────
    AGENT_STAGES = [
        ("Agent 1", "Transaction Monitoring", 0.20),
        ("Agent 2", "Pattern Detection",      0.40),
        ("Agent 3", "Risk Assessment",         0.60),
        ("Agent 4", "Alert & Block",           0.80),
        ("Agent 5", "Compliance Logging",      1.00),
    ]

    fraud_count = 0
    processed = 0

    for idx_raw, row in df.iterrows():
        idx = int(idx_raw) if isinstance(idx_raw, (int, float)) else processed
        # ── Stage progress messages ──────────────────────────────────────────
        stage_idx = min(int((processed / max(total, 1)) * 5), 4)
        agent_name, stage_label, _ = AGENT_STAGES[stage_idx]
        await _emit(task_id, {
            "status": "progress",
            "progress": round(processed / max(total, 1) * 100, 1),
            "message": f"{agent_name} — {stage_label}",
            "processed": processed,
            "total": total,
        })

        # ── Run one row through pipeline ──────────────────────────────────────
        if use_real_agents:
            msg = _row_to_msg(row)
            if msg is not None:
                try:
                    def _run_one(m):
                        if orchestrator is None:
                            raise RuntimeError("Orchestrator unavailable")
                        return orchestrator._process_one(m)
                    msg = await loop.run_in_executor(None, _run_one, msg)
                    result = msg.to_dict()
                except Exception as exc:
                    logger.error("[%s] Pipeline error row %s: %s", task_id, idx, exc)
                    result = _mock_result(row, idx)
            else:
                result = _mock_result(row, idx)
        else:
            await asyncio.sleep(0.01)   # simulate work
            result = _mock_result(row, idx)

        # ── Persist ───────────────────────────────────────────────────────────
        txn_id = result.get("transaction_id", str(uuid.uuid4()))
        save_transaction(task_id, txn_id, result)

        if result.get("fraud_label") or result.get("fraud_score", 0) > 0.5:
            fraud_count += 1

        audit = result.get("audit_log")
        if audit:
            save_audit_record(task_id, audit)

        processed += 1
        update_task(task_id, processed=processed, fraud_count=fraud_count)

        # Yield control so WS handler can send queued events
        if processed % 5 == 0:
            await asyncio.sleep(0)

    # ── 4. Completion ─────────────────────────────────────────────────────────
    update_task(
        task_id,
        status="complete",
        fraud_count=fraud_count,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    await _emit(task_id, {
        "status": "complete",
        "progress": 100,
        "message": "Analysis complete",
        "processed": processed,
        "total": total,
        "fraud_count": fraud_count,
    })


def _mock_result(row: pd.Series, idx: int) -> dict:
    """Fallback mock result when the real orchestrator is unavailable."""
    import random
    fraud_score = random.uniform(0.0, 1.0)
    fraud_label = fraud_score > 0.5
    patterns = ["NONE", "NONE", "NONE", "MULE_NETWORK", "VELOCITY_SPIKE", "ACCOUNT_TAKEOVER"]
    return {
        "transaction_id": str(uuid.uuid4()),
        "step": int(row.get("step", idx)),
        "type": str(row.get("type", "PAYMENT")),
        "amount": float(row.get("amount", 0.0)),
        "nameOrig": str(row.get("nameOrig", f"C{idx:07d}")),
        "nameDest": str(row.get("nameDest", f"M{idx:07d}")),
        "fraud_score": round(fraud_score, 4),
        "fraud_label": fraud_label,
        "pattern_type": random.choice(patterns),
        "pattern_confidence": round(random.uniform(0.3, 0.95), 3) if fraud_label else 0.0,
        "risk_level": random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]) if fraud_label else "LOW",
        "recommended_action": random.choice(["PASS", "SILENT_FLAG", "HOLD", "BLOCK"]),
        "action_taken": random.choice(["PASS", "SILENT_FLAG", "HOLD", "BLOCK"]),
        "explanation": "Mock analysis — real agents unavailable.",
        "oldbalanceOrg": float(row.get("oldbalanceOrg", 0.0)),
        "newbalanceOrig": float(row.get("newbalanceOrig", 0.0)),
        "oldbalanceDest": float(row.get("oldbalanceDest", 0.0)),
        "newbalanceDest": float(row.get("newbalanceDest", 0.0)),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/batch-api/upload")
async def upload_csv(file: UploadFile = File(...)):
    filename: Optional[str] = file.filename
    if not filename or not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    csv_bytes = await file.read()
    if len(csv_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    task_id = str(uuid.uuid4())
    task_queues[task_id] = asyncio.Queue()
    create_task(task_id, filename)

    asyncio.create_task(_process_csv(task_id, csv_bytes, filename))

    return {"task_id": task_id, "filename": filename}


@app.websocket("/batch-api/ws/{task_id}")
async def websocket_endpoint(ws: WebSocket, task_id: str):
    await ws.accept()

    # Create queue if client connects before upload response is processed
    if task_id not in task_queues:
        task_queues[task_id] = asyncio.Queue()

    task = get_task(task_id)
    if not task:
        await ws.send_json({"status": "error", "message": "Task not found."})
        await ws.close()
        return

    # If already complete, send immediate completion event
    if task["status"] == "complete":
        await ws.send_json({
            "status": "complete",
            "progress": 100,
            "message": "Analysis already complete.",
            "processed": task["processed"],
            "total": task["total_rows"],
            "fraud_count": task["fraud_count"],
        })
        await ws.close()
        return

    q = task_queues[task_id]
    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws.send_json({"status": "ping"})
                continue

            await ws.send_json(event)

            if event.get("status") in ("complete", "error"):
                break
    except WebSocketDisconnect:
        logger.info("[%s] WebSocket disconnected.", task_id)
    finally:
        task_queues.pop(task_id, None)


@app.get("/batch-api/results/{task_id}")
async def get_task_results(task_id: str, limit: int = 1000, offset: int = 0):
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    rows = get_results(task_id)
    return {"task": task, "results": rows[offset: offset + limit], "total": len(rows)}


@app.get("/batch-api/summary/{task_id}")
async def get_summary(task_id: str):
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    rows = get_results(task_id)
    if not rows:
        return {"task": task, "metrics": {}}

    total = len(rows)
    fraud = sum(1 for r in rows if r.get("fraud_label"))
    total_amount = sum(r.get("amount", 0) for r in rows)
    fraud_amount = sum(r.get("amount", 0) for r in rows if r.get("fraud_label"))

    scores = [r.get("fraud_score", 0) for r in rows if r.get("fraud_score") is not None]
    avg_risk = round(sum(scores) / len(scores), 4) if scores else 0
    first_influence = next(
        (r.get("dataset_influence") for r in rows if r.get("dataset_influence")),
        {},
    )
    dataset_artifact = {
        "model_version": first_influence.get("model_version"),
        "artifact_mode": first_influence.get("artifact_mode"),
        "decision_threshold": first_influence.get("decision_threshold"),
        "best_val_auc": first_influence.get("best_val_auc"),
        "embedding_dim": first_influence.get("embedding_dim"),
        "feature_count": (first_influence.get("feature_contract") or {}).get("feature_count"),
    }

    pattern_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}

    for r in rows:
        p = r.get("pattern_type", "NONE") or "NONE"
        pattern_counts[p] = pattern_counts.get(p, 0) + 1

        a = r.get("action_taken", "PASS") or "PASS"
        action_counts[a] = action_counts.get(a, 0) + 1

        rl = r.get("risk_level", "LOW") or "LOW"
        risk_counts[rl] = risk_counts.get(rl, 0) + 1

    # Score histogram (10 buckets)
    buckets = [0] * 10
    for s in scores:
        idx = min(int(s * 10), 9)
        buckets[idx] += 1

    # Exception queue (HIGH + CRITICAL + BLOCK/HOLD)
    exceptions = [
        r for r in rows
        if r.get("risk_level") in ("HIGH", "CRITICAL")
        or r.get("action_taken") in ("BLOCK", "HOLD")
    ]

    # Network graph data (sampled, ≤200 nodes)
    # csv_ids: set of account/merchant IDs that appear in the uploaded CSV rows
    csv_ids: set[str] = set()
    sample = rows[:200]
    for r in sample:
        orig = r.get("nameOrig", "")
        dest = r.get("nameDest", "")
        if orig:
            csv_ids.add(orig)
        if dest:
            csv_ids.add(dest)

    # Build a richer node set: CSV nodes + synthetic NPC background nodes
    nodes: dict[str, dict] = {}
    links = []
    for r in sample:
        orig = r.get("nameOrig", "")
        dest = r.get("nameDest", "")
        if orig:
            nodes[orig] = {
                "id": orig,
                "type": "account",
                "fraud": r.get("fraud_label", False),
                "in_csv": True,
            }
        if dest:
            nodes[dest] = {
                "id": dest,
                "type": "merchant" if dest.startswith("M") else "account",
                "fraud": False,
                "in_csv": True,
            }
        if orig and dest:
            links.append({
                "source": orig,
                "target": dest,
                "amount": r.get("amount", 0),
                "fraud": r.get("fraud_label", False),
            })

    # Add NPC background nodes (synthetic accounts not in CSV) for visual depth
    import random as _rnd
    _rng = _rnd.Random(42)  # deterministic
    npc_count = min(40, max(10, len(nodes) // 3))
    for i in range(npc_count):
        npc_id = f"NPC_{i:04d}"
        if npc_id not in nodes:
            nodes[npc_id] = {"id": npc_id, "type": "account", "fraud": False, "in_csv": False}
        # Optionally add a sparse NPC-to-NPC or NPC-to-existing link
        if nodes and _rng.random() < 0.3:
            target_id = _rng.choice(list(nodes.keys()))
            links.append({"source": npc_id, "target": target_id, "amount": 0, "fraud": False, "npc": True})

    return {
        "task": task,
        "metrics": {
            "total_transactions": total,
            "total_amount": round(total_amount, 2),
            "fraud_count": fraud,
            "fraud_rate": round(fraud / max(total, 1) * 100, 2),
            "fraud_amount": round(fraud_amount, 2),
            "avg_risk_score": avg_risk,
            "pattern_counts": pattern_counts,
            "action_counts": action_counts,
            "risk_counts": risk_counts,
            "score_histogram": buckets,
            "dataset_artifact": dataset_artifact,
        },
        "exceptions": exceptions[:50],
        "graph": {"nodes": list(nodes.values()), "links": links},
    }


@app.get("/batch-api/audit")
async def get_audit(task_id: str | None = None):
    records = get_audit_records(task_id)
    return {"records": records, "total": len(records)}


@app.get("/batch-api/tasks")
async def get_tasks():
    return {"tasks": list_tasks()}


@app.get("/batch-api/agent-outputs/{task_id}")
async def get_agent_outputs(task_id: str, limit: int = 200):
    """
    Return per-agent sliced outputs for all transactions in a task.
    Shaped so the frontend can populate each agent's accordion panel directly.

    Shape:
    {
      "agent1": [ { txn_id, nameOrig, nameDest, amount, type, fraud_score, fraud_label, top_features, model_version }, ... ],
      "agent2": [ { txn_id, nameOrig, nameDest, amount, pattern_type, pattern_confidence, pattern_reasoning, window_snapshot }, ... ],
      "agent3": [ { txn_id, nameOrig, amount, risk_level, recommended_action, rl_mode }, ... ],
      "agent4": [ { txn_id, nameOrig, nameDest, amount, action_taken, risk_level, explanation }, ... ],
      "agent5": [ { txn_id, nameOrig, action_taken, risk_level, fraud_score, decision_latency_ms, audit_overhead_ms, pipeline_metadata }, ... ],
    }
    """
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    rows = get_results(task_id)[:limit]

    agent1, agent2, agent3, agent4, agent5 = [], [], [], [], []

    for r in rows:
        txn_id = (r.get("transaction_id") or "")[:8]
        nameOrig = r.get("nameOrig") or "—"
        nameDest = r.get("nameDest") or "—"
        amount = r.get("amount", 0)

        # Agent 1 — Transaction Monitoring
        agent1.append({
            "txn_id": txn_id,
            "nameOrig": nameOrig,
            "nameDest": nameDest,
            "type": r.get("type", "—"),
            "amount": amount,
            "fraud_score": r.get("fraud_score"),
            "fraud_label": r.get("fraud_label", False),
            "top_features": r.get("top_features") or [],
            "model_version": r.get("model_version") or "—",
            "dataset_influence": r.get("dataset_influence") or {},
        })

        # Agent 2 — Pattern Detection
        agent2.append({
            "txn_id": txn_id,
            "nameOrig": nameOrig,
            "nameDest": nameDest,
            "amount": amount,
            "pattern_type": r.get("pattern_type") or "NONE",
            "pattern_confidence": r.get("pattern_confidence"),
            "pattern_reasoning": r.get("pattern_reasoning") or "",
            "window_snapshot": r.get("window_snapshot") or [],
        })

        # Agent 3 — Risk Assessment (PPO)
        agent3.append({
            "txn_id": txn_id,
            "nameOrig": nameOrig,
            "amount": amount,
            "fraud_score": r.get("fraud_score"),
            "pattern_type": r.get("pattern_type") or "NONE",
            "risk_level": r.get("risk_level") or "LOW",
            "recommended_action": r.get("recommended_action") or "PASS",
            "account_context": r.get("account_context") or {},
        })

        # Agent 4 — Alert & Block
        agent4.append({
            "txn_id": txn_id,
            "nameOrig": nameOrig,
            "nameDest": nameDest,
            "amount": amount,
            "action_taken": r.get("action_taken") or "PASS",
            "risk_level": r.get("risk_level") or "LOW",
            "explanation": r.get("explanation") or "—",
            "dataset_influence": r.get("dataset_influence") or {},
        })

        # Agent 5 — Compliance Logging
        audit = r.get("audit_log") or {}
        pipeline_meta = r.get("pipeline_metadata") or []
        latency = None
        if audit:
            latency = audit.get("decision_latency_ms")
        agent5.append({
            "txn_id": txn_id,
            "nameOrig": nameOrig,
            "action_taken": r.get("action_taken") or "PASS",
            "risk_level": r.get("risk_level") or "LOW",
            "fraud_score": r.get("fraud_score"),
            "decision_latency_ms": latency,
            "pipeline_metadata": pipeline_meta,
        })

    return {
        "task_id": task_id,
        "total": len(rows),
        "agent1": agent1,
        "agent2": agent2,
        "agent3": agent3,
        "agent4": agent4,
        "agent5": agent5,
    }


@app.get("/batch-api/intelligence-stream/{task_id}")
async def get_intelligence_stream(task_id: str):
    """
    Server-Sent Events (SSE) streaming endpoint for LLM intelligence.
    Streams raw text tokens from the llama-server as they arrive.
    On completion, emits a final 'complete' event with the full structured JSON.
    Also populates the intelligence_cache so the regular tab re-visit is instant.
    """
    import httpx

    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    # Return cached result immediately if already computed
    if task_id in intelligence_cache:
        cached = intelligence_cache[task_id]

        async def _cached_stream() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps({'type': 'cached', 'payload': cached})}\n\n"

        return StreamingResponse(_cached_stream(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    rows = get_results(task_id)
    if not rows:
        async def _empty() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps({'type': 'error', 'message': 'No results available yet.'})}\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    # Build compact summary
    total = len(rows)
    fraud_rows = [r for r in rows if r.get("fraud_label")]
    fraud_count = len(fraud_rows)
    fraud_rate = round(fraud_count / max(total, 1) * 100, 2)

    pattern_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for r in rows:
        p = r.get("pattern_type", "NONE") or "NONE"
        pattern_counts[p] = pattern_counts.get(p, 0) + 1
        rl = r.get("risk_level", "LOW") or "LOW"
        risk_counts[rl] = risk_counts.get(rl, 0) + 1
        a = r.get("action_taken", "PASS") or "PASS"
        action_counts[a] = action_counts.get(a, 0) + 1

    high_risk_samples = [
        {
            "transaction_id": r.get("transaction_id", "")[:12],
            "amount": r.get("amount", 0),
            "pattern_type": r.get("pattern_type"),
            "risk_level": r.get("risk_level"),
            "action_taken": r.get("action_taken"),
            "fraud_score": r.get("fraud_score"),
            "dataset_threshold_ratio": (r.get("dataset_influence") or {}).get("threshold_ratio"),
            "top_features": r.get("top_features") or [],
            "explanation": r.get("explanation") or r.get("pattern_reasoning") or "",
        }
        for r in rows
        if r.get("risk_level") in ("HIGH", "CRITICAL")
    ][:5]

    meta = {
        "total_transactions": total,
        "fraud_count": fraud_count,
        "fraud_rate": fraud_rate,
        "pattern_counts": pattern_counts,
        "risk_counts": risk_counts,
        "action_counts": action_counts,
    }

    summary_payload = {
        "dataset_summary": {
            "total_transactions": total,
            "fraud_count": fraud_count,
            "fraud_rate_pct": fraud_rate,
            "pattern_distribution": pattern_counts,
            "risk_distribution": risk_counts,
            "action_distribution": action_counts,
        },
        "high_risk_samples": high_risk_samples,
    }

    system_prompt = (
        "You are an expert financial fraud analyst AI. You have been given a summary of "
        "transaction analysis results from the Jatayu Autonomous Fraud Detection Network. "
        "Provide a concise, insightful intelligence report covering: "
        "1) Key fraud patterns observed, "
        "2) Risk assessment summary, "
        "3) Notable high-risk transactions, "
        "4) Recommended focus areas for the fraud team. "
        "Give weight to the PaySim-trained XGBoost/GNN threshold ratios and top features. "
        "Be specific and analytical. Format your response as structured JSON."
    )

    user_prompt = (
        "Analyse the following fraud detection results and generate an intelligence report. "
        "Respond ONLY with a JSON object with these exact keys: "
        '{"headline": "<2-sentence executive summary>", '
        '"key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"], '
        '"threat_assessment": "<paragraph about detected threats>", '
        '"recommendations": ["<rec 1>", "<rec 2>", "<rec 3>"], '
        '"confidence": "<LOW|MEDIUM|HIGH based on data quality>"} \n\n'
        f"Data:\n{json.dumps(summary_payload, separators=(',', ':'))}"
    )

    async def _stream_llm() -> AsyncGenerator[str, None]:
        full_text = ""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    "http://localhost:8080/v1/chat/completions",
                    json={
                        "model": "LFM2.5-1.2B-Instruct",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 600,
                        "stream": True,
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        chunk_data = line[5:].strip()
                        if chunk_data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(chunk_data)
                            token = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if token:
                                full_text += token
                                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                        except json.JSONDecodeError:
                            continue

            # Parse structured JSON from accumulated text
            start = full_text.find("{")
            end = full_text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    parsed = json.loads(full_text[start:end])
                except json.JSONDecodeError:
                    parsed = {"headline": full_text, "key_findings": [], "threat_assessment": "", "recommendations": [], "confidence": "LOW"}
            else:
                parsed = {"headline": full_text, "key_findings": [], "threat_assessment": "", "recommendations": [], "confidence": "LOW"}

            result = {"analysis": parsed, "meta": meta, "high_risk_samples": high_risk_samples}
            intelligence_cache[task_id] = result
            yield f"data: {json.dumps({'type': 'complete', 'payload': result})}\n\n"

        except Exception as exc:
            logger.warning("[Intelligence Stream] LLM call failed: %s", exc)
            fallback = {
                "analysis": None,
                "error": f"LLM service unavailable: {exc}",
                "meta": meta,
                "high_risk_samples": high_risk_samples,
            }
            intelligence_cache[task_id] = fallback
            yield f"data: {json.dumps({'type': 'error_fallback', 'payload': fallback})}\n\n"

    return StreamingResponse(
        _stream_llm(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/batch-api/intelligence/{task_id}")
async def get_intelligence(task_id: str):
    """
    Call the local LLM (llama.cpp server on port 8080) to generate a
    high-level intelligence analysis of the processed transaction data.
    Returns structured reasoning about fraud patterns, risk distribution,
    and recommended actions based on the actual analysis results.
    """
    import httpx

    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    rows = get_results(task_id)
    if not rows:
        return {"analysis": None, "error": "No results available yet."}

    # Build a compact summary for the LLM
    total = len(rows)
    fraud_rows = [r for r in rows if r.get("fraud_label")]
    fraud_count = len(fraud_rows)
    fraud_rate = round(fraud_count / max(total, 1) * 100, 2)

    pattern_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for r in rows:
        p = r.get("pattern_type", "NONE") or "NONE"
        pattern_counts[p] = pattern_counts.get(p, 0) + 1
        rl = r.get("risk_level", "LOW") or "LOW"
        risk_counts[rl] = risk_counts.get(rl, 0) + 1
        a = r.get("action_taken", "PASS") or "PASS"
        action_counts[a] = action_counts.get(a, 0) + 1

    # Sample up to 5 high-risk transactions with their LLM reasoning
    high_risk_samples = [
        {
            "transaction_id": r.get("transaction_id", "")[:12],
            "amount": r.get("amount", 0),
            "pattern_type": r.get("pattern_type"),
            "risk_level": r.get("risk_level"),
            "action_taken": r.get("action_taken"),
            "fraud_score": r.get("fraud_score"),
            "dataset_threshold_ratio": (r.get("dataset_influence") or {}).get("threshold_ratio"),
            "top_features": r.get("top_features") or [],
            "explanation": r.get("explanation") or r.get("pattern_reasoning") or "",
        }
        for r in rows
        if r.get("risk_level") in ("HIGH", "CRITICAL")
    ][:5]

    summary_payload = {
        "dataset_summary": {
            "total_transactions": total,
            "fraud_count": fraud_count,
            "fraud_rate_pct": fraud_rate,
            "pattern_distribution": pattern_counts,
            "risk_distribution": risk_counts,
            "action_distribution": action_counts,
        },
        "high_risk_samples": high_risk_samples,
    }

    system_prompt = (
        "You are an expert financial fraud analyst AI. You have been given a summary of "
        "transaction analysis results from the Jatayu Autonomous Fraud Detection Network. "
        "Provide a concise, insightful intelligence report covering: "
        "1) Key fraud patterns observed, "
        "2) Risk assessment summary, "
        "3) Notable high-risk transactions, "
        "4) Recommended focus areas for the fraud team. "
        "Give weight to the PaySim-trained XGBoost/GNN threshold ratios and top features. "
        "Be specific and analytical. Format your response as structured JSON."
    )

    user_prompt = (
        "Analyse the following fraud detection results and generate an intelligence report. "
        "Respond ONLY with a JSON object with these exact keys: "
        '{"headline": "<2-sentence executive summary>", '
        '"key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"], '
        '"threat_assessment": "<paragraph about detected threats>", '
        '"recommendations": ["<rec 1>", "<rec 2>", "<rec 3>"], '
        '"confidence": "<LOW|MEDIUM|HIGH based on data quality>"} \n\n'
        f"Data:\n{json.dumps(summary_payload, separators=(',', ':'))}"
    )

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "http://localhost:8080/v1/chat/completions",
                json={
                    "model": "LFM2.5-1.2B-Instruct",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 600,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            # Extract JSON object from content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(content[start:end])
            else:
                parsed = {"headline": content, "key_findings": [], "threat_assessment": "", "recommendations": [], "confidence": "LOW"}

            result = {
                "analysis": parsed,
                "meta": {
                    "total_transactions": total,
                    "fraud_count": fraud_count,
                    "fraud_rate": fraud_rate,
                    "pattern_counts": pattern_counts,
                    "risk_counts": risk_counts,
                    "action_counts": action_counts,
                },
                "high_risk_samples": high_risk_samples,
            }
            intelligence_cache[task_id] = result
            return result

    except Exception as exc:
        logger.warning("[Intelligence] LLM call failed: %s", exc)
        # Return structured fallback without LLM
        fallback = {
            "analysis": None,
            "error": f"LLM service unavailable: {exc}",
            "meta": {
                "total_transactions": total,
                "fraud_count": fraud_count,
                "fraud_rate": fraud_rate,
                "pattern_counts": pattern_counts,
                "risk_counts": risk_counts,
                "action_counts": action_counts,
            },
            "high_risk_samples": high_risk_samples,
        }
        intelligence_cache[task_id] = fallback
        return fallback


# ── Static file serving ───────────────────────────────────────────────────────

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def _serve_html_no_cache(filename: str) -> FileResponse:
        response = FileResponse(STATIC_DIR / filename)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.get("/")
    async def serve_index():
        """Serve login page as default."""
        return _serve_html_no_cache("login.html")

    @app.get("/login")
    async def serve_login():
        """Serve login page."""
        return _serve_html_no_cache("login.html")

    @app.get("/dashboard")
    async def serve_dashboard(request: Request):
        """Serve customer dashboard - requires authentication."""
        from auth.jwt_handler import verify_token
        
        # Check if authenticated
        auth_header = request.headers.get("Authorization")
        token = request.cookies.get("jatayu_token") or (auth_header.split(" ")[1] if auth_header and " " in auth_header else None)
        
        if not token:
            return FileResponse(STATIC_DIR / "login.html")
        
        try:
            token_data = verify_token(token)
            if not token_data or token_data.user_type not in ["CUSTOMER", "MERCHANT"]:
                return _serve_html_no_cache("login.html")
        except:
            return _serve_html_no_cache("login.html")
            
        return _serve_html_no_cache("dashboard.html")

    @app.get("/merchant")
    async def serve_merchant(request: Request):
        """Serve merchant portal - requires merchant auth."""
        from auth.jwt_handler import verify_token
        
        auth_header = request.headers.get("Authorization")
        token = request.cookies.get("jatayu_token") or (auth_header.split(" ")[1] if auth_header and " " in auth_header else None)
        
        if not token:
            return _serve_html_no_cache("login.html")
        
        try:
            token_data = verify_token(token)
            if not token_data or token_data.user_type != "MERCHANT":
                return _serve_html_no_cache("login.html")
        except:
            return _serve_html_no_cache("login.html")
            
        return _serve_html_no_cache("merchant.html")

    @app.get("/admin")
    async def serve_admin(request: Request):
        """Serve admin console - requires admin auth."""
        from auth.jwt_handler import verify_token
        from fastapi.responses import Response
        
        auth_header = request.headers.get("Authorization")
        token = request.cookies.get("jatayu_token") or (auth_header.split(" ")[1] if auth_header and " " in auth_header else None)
        
        if not token:
            return _serve_html_no_cache("login.html")
        
        try:
            token_data = verify_token(token)
            if not token_data or token_data.user_type != "ADMIN":
                return _serve_html_no_cache("login.html")
        except:
            return _serve_html_no_cache("login.html")
            
        return _serve_html_no_cache("admin.html")

    @app.get("/batch")
    async def serve_batch():
        """Serve batch CSV upload page — fully public, no auth required."""
        return _serve_html_no_cache("batch.html")

    @app.get("/suspended")
    @app.get("/suspended/")
    @app.get("/account-restricted")
    async def serve_suspended():
        """Serve the account-restricted page for SUSPENDED/BLOCKED users.
        
        These users have a valid JWT but limited API access.
        They can only use /support/* endpoints from here.
        """
        return _serve_html_no_cache("suspended.html")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
