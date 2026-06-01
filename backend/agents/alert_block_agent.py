"""
alert_block_agent.py — Agent 4: Alert & Block

Executes the recommended action from Agent 3 and generates a plain-English
explanation of the decision for downstream systems / human review.

Current implementation: stub explanation generator.
TODO: Replace explanation with an LLM-generated narrative.
"""

from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from agents.models import (
    Action,
    PatternType,
    RiskLevel,
    TransactionMessage,
)

logger = logging.getLogger(__name__)


class AlertBlockAgent(BaseAgent):
    """
    Agent 4 — Alert & Block.

    Reads from TransactionMessage:
        msg.recommended_action  (from Agent 3)
        msg.fraud_score         (from Agent 1)
        msg.top_features        (from Agent 1)
        msg.pattern_type        (from Agent 2)
        msg.risk_level          (from Agent 3)

    Writes to TransactionMessage:
        msg.action_taken   → Action enum (mirrors recommended_action in stub)
        msg.explanation    → plain-English decision rationale
    """

    name = "AlertBlockAgent"

    def __init__(self) -> None:
        # TODO: Initialize LLM client here (e.g. google.generativeai or openai).
        # self.llm_client = genai.GenerativeModel("gemini-1.5-flash")
        # Keep the client alive across transactions to avoid repeated auth overhead.
        logger.info("[%s] Initialized (stub explanation mode)", self.name)

    def _process(self, msg: TransactionMessage) -> TransactionMessage:
        # Execute the recommended action
        # TODO: If action == BLOCK, call payment gateway API to actually block.
        # TODO: If action == HOLD, push to a human review queue.
        # TODO: If action == SILENT_FLAG, write to a fraud ops dashboard topic.
        msg.action_taken = msg.recommended_action or Action.PASS

        # Dynamic suspension policy mapping
        msg.suspend_sender = False
        msg.suspend_receiver = False
        msg.suspend_mule_network = False

        if msg.action_taken == Action.BLOCK:
            pattern = msg.pattern_type or PatternType.NONE
            if pattern == PatternType.MULE_NETWORK:
                msg.suspend_sender = True
                msg.suspend_receiver = True
                msg.suspend_mule_network = True
            elif pattern == PatternType.ACCOUNT_TAKEOVER:
                msg.suspend_sender = True
                is_merchant = False
                if msg.nameDest:
                    try:
                        import database as db
                        dest_user = db.get_user_by_id(msg.nameDest)
                        if dest_user and dest_user.get("user_type") == "MERCHANT":
                            is_merchant = True
                    except Exception:
                        is_merchant = msg.nameDest.startswith("M")
                if not is_merchant:
                    msg.suspend_receiver = True
            elif pattern == PatternType.CIRCULAR_FLOW:
                msg.suspend_sender = True
                msg.suspend_receiver = True
            elif pattern == PatternType.VELOCITY_SPIKE:
                # Velocity spike rejects the transaction but leaves accounts active
                msg.suspend_sender = False
            else:
                # Default fallback block enforcement
                msg.suspend_sender = True

        # Generate explanation
        msg.explanation = self._explain(msg)
        return msg

    def _explain(self, msg: TransactionMessage) -> str:
        """
        Generate a plain-English explanation of the fraud decision.

        ── CURRENT STATE: STUB (template-based) ─────────────────────────────────
        The stub builds a readable sentence from structured fields.

        ── TODO: Replace with LLM call ──────────────────────────────────────────
        Replace the template string below with a call like:

            prompt = f\"\"\"
            You are a fraud analyst AI. Explain the following decision in 2-3 sentences
            for a compliance officer. Be specific about the evidence.

            Transaction: {msg.amount:.2f} from {msg.nameOrig} to {msg.nameDest}
            Fraud score: {msg.fraud_score:.4f} (threshold: 0.0224)
            Top contributing features: {', '.join(msg.top_features or [])}
            Detected pattern: {msg.pattern_type}
            Risk level: {msg.risk_level}
            Action taken: {msg.action_taken}
            \"\"\"
            response = self.llm_client.generate_content(prompt)
            return response.text.strip()

        Keep LLM latency in mind — for real-time blocking this should run
        asynchronously or use a streaming response.
        """

        # ── STUB explanation template ─────────────────────────────────────────
        score_pct  = f"{(msg.fraud_score or 0) * 100:.1f}%"
        action     = (msg.action_taken or Action.PASS).value
        pattern    = (msg.pattern_type or PatternType.NONE).value
        risk       = (msg.risk_level or RiskLevel.LOW).value
        top        = ", ".join(msg.top_features or []) or "N/A"
        amount     = f"{msg.amount:,.2f}"
        dataset = msg.dataset_influence or {}
        threshold = dataset.get("decision_threshold")
        threshold_ratio = dataset.get("threshold_ratio")
        gnn_profile = dataset.get("gnn_embedding") or {}
        artifact_mode = dataset.get("artifact_mode") or "unknown"
        model_detail = ""
        if threshold is not None and threshold_ratio is not None:
            model_detail = (
                f" PaySim artifact signal: score is {float(threshold_ratio):.2f}x "
                f"the trained threshold {float(threshold):.4f}; "
                f"GNN embedding {'matched' if gnn_profile.get('used') else 'not matched'}."
            )

        template = (
            f"Transaction of {msg.type.value if hasattr(msg.type, 'value') else msg.type} "
            f"${amount} from {msg.nameOrig} to {msg.nameDest} received a fraud probability "
            f"of {score_pct}. "
            f"Top contributing signals from the {artifact_mode} PaySim-trained XGBoost/GNN stack: [{top}]."
            f"{model_detail} "
            f"Pattern analysis flagged this as {pattern} with risk level {risk}. "
            f"Action taken: {action}."
        )
        return template
        # ── END STUB ──────────────────────────────────────────────────────────
