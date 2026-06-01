"""
compliance_logging_agent.py — Agent 5: Compliance Logging

Generates a structured audit log entry from the full decision chain accumulated
across all previous agents. By the time this agent runs, TransactionMessage
contains the complete pipeline record.

Implementation: Persistent append-only JSONL file + in-memory backup.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from agents.base_agent import BaseAgent
from agents.models import TransactionMessage
from config import backend_path_from_env, load_environment

logger = logging.getLogger(__name__)

# Default audit file path (can be overridden via environment variable)
load_environment()
AUDIT_FILE_PATH = str(backend_path_from_env("JATAYU_AUDIT_FILE", "data/audit.jsonl"))


class ComplianceLoggingAgent(BaseAgent):
    """
    Agent 5 — Compliance Logging.

    Reads: the entire TransactionMessage (all fields from Agents 1–4).

    Writes to TransactionMessage:
        msg.audit_log → structured dict with the full decision chain

    Persists each entry to an append-only JSONL file for compliance.
    Also maintains in-memory audit_trail for batch operations.
    """

    name = "ComplianceLoggingAgent"

    def __init__(self, audit_file_path: str = AUDIT_FILE_PATH) -> None:
        self.audit_file_path = audit_file_path
        # In-memory backup for batch operations
        self.audit_trail: list[dict[str, Any]] = []

        logger.info("[%s] Initialized (persisting to %s)", self.name, self.audit_file_path)

    def _process(self, msg: TransactionMessage) -> TransactionMessage:
        entry = self._build_audit_entry(msg)
        msg.audit_log = entry

        # Append to in-memory trail
        self.audit_trail.append(entry)

        # Persist to JSONL file immediately (append mode, line-buffered)
        self._persist_entry(entry)

        logger.info(
            "[AUDIT] txn=%s | action=%s | risk=%s | score=%.4f",
            entry["transaction_id"],
            entry["action_taken"],
            entry["risk_level"],
            entry["fraud_score"] or 0.0,
        )
        return msg

    def _persist_entry(self, entry: dict[str, Any]) -> None:
        """Append a single audit entry to the JSONL file."""
        try:
            with open(self.audit_file_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error("[%s] Failed to persist audit entry: %s", self.name, e)

    def _build_audit_entry(self, msg: TransactionMessage) -> dict[str, Any]:
        """
        Build the structured audit log entry from the full decision chain.

        Telemetry fields:
          - pipeline_end_ms     — set by the Orchestrator before Agent 5 is
                                  called; marks end of the decision pipeline.
                                  Agent 5 reads this value directly instead of
                                  stamping its own time so that decision_latency_ms
                                  excludes compliance logging overhead.
          - logged_at_ms        — when this audit entry was written (Agent 5 time).
          - decision_latency_ms — pipeline_end_ms - pipeline_start_ms (Agents 1–4).
          - audit_overhead_ms   — logged_at_ms - pipeline_end_ms (Agent 5 only).

        Regulatory fields included for compliance frameworks:
          - model_version       — ML model version used for fraud scoring
          - human_review_flag   — True if action requires human review (HOLD)
          - agent_versions      — Version info for audit trail
        """
        logged_at_ms = time.time() * 1000
        # Use the Orchestrator-stamped pipeline_end_ms as the authoritative
        # boundary between decision-making and audit logging.  Fall back to
        # logged_at_ms only if the field was not set (e.g. in unit tests).
        pipeline_end_ms = msg.pipeline_end_ms if msg.pipeline_end_ms is not None else logged_at_ms

        # Determine action_taken value
        action_taken = msg.action_taken.value if hasattr(msg.action_taken, "value") else msg.action_taken

        return {
            # ── Identity ──────────────────────────────────────────────────────
            "transaction_id":   msg.transaction_id,
            "generated_at":     msg.generated_at,
            "logged_at_ms":     logged_at_ms,

            # ── Raw transaction ───────────────────────────────────────────────
            "step":             msg.step,
            "type":             msg.type.value if hasattr(msg.type, "value") else msg.type,
            "amount":           msg.amount,
            "nameOrig":         msg.nameOrig,
            "nameDest":         msg.nameDest,
            "oldbalanceOrg":    msg.oldbalanceOrg,
            "newbalanceOrig":   msg.newbalanceOrig,
            "oldbalanceDest":   msg.oldbalanceDest,
            "newbalanceDest":   msg.newbalanceDest,
            "ip_address":       msg.ip_address,
            "device_id":        msg.device_id,

            # ── Agent 1: Fraud Scoring ────────────────────────────────────────
            "fraud_score":      msg.fraud_score,
            "fraud_label":      msg.fraud_label,
            "top_features":     msg.top_features,
            "model_version":    msg.model_version,
            "dataset_influence": msg.dataset_influence,

            # ── Agent 2: Pattern Detection ────────────────────────────────────
            "pattern_type":         msg.pattern_type.value
                                    if hasattr(msg.pattern_type, "value") else msg.pattern_type,
            "pattern_confidence":   msg.pattern_confidence,
            "window_snapshot":      msg.window_snapshot,

            # ── Agent 3: Risk Assessment ──────────────────────────────────────
            "risk_level":           msg.risk_level.value
                                    if hasattr(msg.risk_level, "value") else msg.risk_level,
            "recommended_action":   msg.recommended_action.value
                                    if hasattr(msg.recommended_action, "value") else msg.recommended_action,
            "account_context":      msg.account_context,

            # ── Agent 4: Alert & Block ────────────────────────────────────────
            "action_taken":     action_taken,
            "explanation":      msg.explanation,

            # ── Regulatory fields ─────────────────────────────────────────────
            "human_review_flag":    action_taken == "HOLD",
            "agent_versions": {
                "fraud_scoring": "1.0",
                "pattern_detection": "1.0",
                "risk_assessment": "1.0",
                "alert_block": "1.0",
                "compliance_logging": "1.0",
            },

            # ── Pipeline telemetry ────────────────────────────────────────────
            "pipeline_start_ms":    msg.pipeline_start_ms,
            "pipeline_end_ms":      pipeline_end_ms,
            "decision_latency_ms":  (
                round(pipeline_end_ms - msg.pipeline_start_ms, 2)
                if msg.pipeline_start_ms is not None else None
            ),
            "audit_overhead_ms":    round(logged_at_ms - pipeline_end_ms, 2),

            # ── Per-agent provenance ──────────────────────────────────────────
            # Each entry is an AgentMeta dict: {agent_name, status, latency_ms,
            # error, confidence}.  Enables per-agent fault attribution and
            # latency profiling without a separate side-channel.
            "pipeline_metadata": [
                {
                    "agent_name": m.agent_name,
                    "status":     m.status,
                    "latency_ms": m.latency_ms,
                    "error":      m.error,
                    "confidence": m.confidence,
                }
                for m in msg.pipeline_metadata
            ],
        }

    def export_jsonl(self, path: str) -> None:
        """
        Dump the full in-memory audit trail to a JSONL file.

        Usage:
            compliance_agent.export_jsonl("audit_trail_backup.jsonl")

        Note: Primary persistence is now via _persist_entry() to self.audit_file_path.
        This method is useful for exporting to a different location or creating backups.
        """
        with open(path, "w", encoding="utf-8") as fh:
            for entry in self.audit_trail:
                fh.write(json.dumps(entry, default=str) + "\n")
        logger.info("[%s] Exported %d entries → %s", self.name, len(self.audit_trail), path)
