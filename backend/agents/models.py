"""
models.py — Shared TransactionMessage schema for the Jatayu agent pipeline.

A TransactionMessage starts with raw transaction fields and is progressively
enriched by each agent as it flows through the pipeline. By Agent 5 it
contains the full decision chain.

Serialization: use dataclasses.asdict(msg) to get a JSON-ready dict.
"""

from __future__ import annotations

import uuid
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Per-agent execution metadata (provenance)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentMeta:
    """
    Execution metadata recorded by BaseAgent.process() for every agent run.

    Stored in TransactionMessage.pipeline_metadata as an ordered list.
    Enables per-agent latency profiling and fault attribution without
    requiring a separate logging side-channel.

    Fields:
        agent_name  — e.g. "PatternDetectionAgent"
        status      — "ok" | "fallback" | "error"
        latency_ms  — wall-clock time spent in _process()
        error       — exception message when status == "error", else None
        confidence  — optional agent-reported confidence (e.g. pattern_confidence)
    """
    agent_name: str
    status: str                        # "ok" | "fallback" | "error"
    latency_ms: float
    error: Optional[str] = None
    confidence: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class TrafficMode(str, Enum):
    """The synthetic traffic mode under which this transaction was generated."""
    NORMAL           = "NORMAL"
    MULE_NETWORK     = "MULE_NETWORK"
    ACCOUNT_TAKEOVER = "ACCOUNT_TAKEOVER"


class TransactionType(str, Enum):
    """PaySim transaction types."""
    PAYMENT  = "PAYMENT"
    TRANSFER = "TRANSFER"
    CASH_IN  = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT    = "DEBIT"


class PatternType(str, Enum):
    """Coordinated attack patterns detected by Agent 2."""
    NONE             = "NONE"
    MULE_NETWORK     = "MULE_NETWORK"
    ACCOUNT_TAKEOVER = "ACCOUNT_TAKEOVER"
    VELOCITY_SPIKE   = "VELOCITY_SPIKE"
    CIRCULAR_FLOW    = "CIRCULAR_FLOW"


class RiskLevel(str, Enum):
    """Risk tier assigned by Agent 3."""
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class Action(str, Enum):
    """Enforcement action decided by Agent 3 and executed by Agent 4."""
    PASS         = "PASS"
    SILENT_FLAG  = "SILENT_FLAG"
    HOLD         = "HOLD"
    BLOCK        = "BLOCK"


# ─────────────────────────────────────────────────────────────────────────────
# TransactionMessage — the accumulating pipeline message
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransactionMessage:
    """
    The central data object that flows through all 5 agents.

    Sections:
      [RAW]        Set by the SyntheticTrafficGenerator before entering the pipeline.
      [AGENT 1]    Set by TransactionMonitoringAgent (fraud score + top features).
      [AGENT 2]    Set by PatternDetectionAgent (coordinated pattern analysis).
      [AGENT 3]    Set by RiskAssessmentAgent (risk tier + recommended action).
      [AGENT 4]    Set by AlertBlockAgent (executed action + explanation).
      [AGENT 5]    Set by ComplianceLoggingAgent (structured audit log entry).

    JSON serialization:
        import dataclasses, json
        json.dumps(dataclasses.asdict(msg), default=str)
    """

    # ── [RAW] Generator-populated identity fields ─────────────────────────────
    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    traffic_mode: TrafficMode = TrafficMode.NORMAL

    # ── [RAW] PaySim core fields ──────────────────────────────────────────────
    step: int = 0                  # Hour of the simulation (1..743)
    type: TransactionType = TransactionType.PAYMENT
    amount: float = 0.0
    nameOrig: str = ""             # Sender account ID (Cxxxxxxxx format)
    nameDest: str = ""             # Receiver account / merchant ID
    oldbalanceOrg: float = 0.0    # Sender balance before transaction
    newbalanceOrig: float = 0.0   # Sender balance after transaction
    oldbalanceDest: float = 0.0   # Receiver balance before transaction
    newbalanceDest: float = 0.0   # Receiver balance after transaction

    # ── [RAW] Extended fields added by the PaySim enrichment (beyond base CSV) ─
    ip_address: str = ""           # IP used for this transaction (ip_XXXX format)
    device_id: str = ""            # Device used (device_XXXX format)

    # ── [SUPERVISION] Ground-truth fraud label ────────────────────────────────
    # Set ONLY when a confirmed label is available: e.g. from the synthetic
    # simulator, a chargeback feed, or a human review queue.
    # Always None during real-time inference.  Used exclusively by the reward
    # function so that compute_reward() is not dependent on Agent 1's output.
    ground_truth_label: Optional[bool] = None

    # ── [AGENT 1] TransactionMonitoringAgent outputs ──────────────────────────
    fraud_score: Optional[float] = None     # Raw probability from XGBoost ∈ [0, 1]
    fraud_label: Optional[bool] = None     # fraud_score >= threshold (0.0224)
    top_features: Optional[list[str]] = None  # Top contributing XGBoost features
    model_version: Optional[str] = None    # Artifact timestamp used for scoring
    dataset_influence: Optional[dict[str, Any]] = None
    # Compact provenance for the PaySim-trained artifacts used by Agent 1:
    # threshold, XGBoost score components, GNN embedding lookup, and artifact files.

    # ── [AGENT 2] PatternDetectionAgent outputs ───────────────────────────────
    pattern_type: Optional[PatternType] = None
    pattern_confidence: Optional[float] = None   # ∈ [0, 1]
    window_snapshot: Optional[list[str]] = None  # Last ≤5 txn_ids in rolling buffer
    pattern_reasoning: Optional[str] = None      # One-sentence explanation

    # ── [AGENT 3] RiskAssessmentAgent outputs ────────────────────────────────
    risk_level: Optional[RiskLevel] = None
    recommended_action: Optional[Action] = None
    account_context: Optional[dict[str, Any]] = None  # e.g. {"known_devices": [...]}

    # ── [AGENT 4] AlertBlockAgent outputs ────────────────────────────────────
    action_taken: Optional[Action] = None
    explanation: Optional[str] = None   # Plain-English decision rationale
    suspend_sender: bool = False
    suspend_receiver: bool = False
    suspend_mule_network: bool = False

    # ── [AGENT 5] ComplianceLoggingAgent outputs ─────────────────────────────
    audit_log: Optional[dict[str, Any]] = None  # Structured audit log entry

    # ── Timing / telemetry (set by Orchestrator) ─────────────────────────────
    pipeline_start_ms: Optional[float] = None  # epoch ms when pipeline started
    pipeline_end_ms: Optional[float] = None    # epoch ms when decision pipeline finished (before Agent 5)

    # ── Per-agent execution provenance ────────────────────────────────────────
    # Populated by BaseAgent.process() for every agent in the pipeline.
    # Ordered by execution sequence.  Each entry is an AgentMeta instance.
    pipeline_metadata: list = field(default_factory=list)  # list[AgentMeta]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (enums converted to string values)."""
        raw = dataclasses.asdict(self)
        # Convert any Enum instances the default asdict doesn't touch at nested depth
        def _coerce(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, dict):
                return {k: _coerce(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_coerce(i) for i in obj]
            return obj
        return _coerce(raw)
