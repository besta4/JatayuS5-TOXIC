"""
pattern_detection_agent.py — Agent 2: Pattern Detection

Analyzes a rolling window of recent transactions to detect
coordinated attack patterns (mule networks, ATO, velocity spikes).

The rolling buffer is real (deque of up to 75 TransactionMessages).
Pattern detection is rule-based, with an optional LLM used only to generate
natural-language reasoning for detected patterns.

Enhanced with:
  - DB-backed velocity (survives server restarts, covers ALL transactions)
  - Time-based velocity (inter-transaction timing, not just count)
  - Receiver-side velocity (multiple senders → one receiver)
  - Redis dynamic graph context (real-time counterparty + burst detection)
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque, defaultdict
from typing import Deque, Dict, Optional, Set

import requests

from agents.base_agent import BaseAgent
from agents.models import PatternType, TransactionMessage

logger = logging.getLogger(__name__)

WINDOW_SIZE = 75  # Rolling buffer depth (must match orchestrator config)

# ── Dynamic Fraud Rule Constants ──────────────────────────────────────────────
VELOCITY_MIN_AMOUNT_THRESHOLD = 100.0   # Exempt transactions below this from velocity blocking
VELOCITY_SPIKE_COUNT_THRESHOLD = 6      # Number of recent transactions in window indicating velocity anomaly
MULE_MICRO_AMOUNT_THRESHOLD = 500.0     # Threshold to distinguish micro-structuring from normal transfers
MULE_LARGE_TOTAL_THRESHOLD = 20000.0    # Cumulative volume threshold indicating high-risk mule activity
DORMANT_LARGE_AMOUNT_THRESHOLD = 10000.0 # Threshold for sudden large transfers on dormant accounts
ATO_FRAUD_SCORE_THRESHOLD = 0.7         # XGBoost score threshold for Account Takeover suspicion


class PatternDetectionAgent(BaseAgent):
    """
    Agent 2 — Pattern Detection.

    Reads from TransactionMessage:
        msg.fraud_label       → only flagged transactions are analysed
        msg.nameOrig          → sender identity
        msg.nameDest          → receiver / collector identity
        msg.step              → for timing-based pattern matching
        msg.ip_address        → for ATO detection
        msg.device_id         → for ATO detection
        msg.traffic_mode      → (used in stub; real model ignores this)

    Writes to TransactionMessage:
        msg.pattern_type         → PatternType enum
        msg.pattern_confidence   → float ∈ [0, 1]
        msg.window_snapshot      → list of last ≤5 txn_ids from the buffer
    """

    name = "PatternDetectionAgent"

    def __init__(
        self,
        generate_reasoning: bool = True,
        graph_cache=None,
    ) -> None:
        # ── Real rolling window buffer ────────────────────────────────────────
        self.buffer: Deque[TransactionMessage] = deque(maxlen=WINDOW_SIZE)

        # Per-user device/IP history tracker for ATO detection.
        self._user_device_history: Dict[str, Set[str]] = defaultdict(set)
        self._user_ip_history: Dict[str, Set[str]] = defaultdict(set)

        # Redis-backed dynamic graph cache (optional, graceful degradation)
        self._graph_cache = graph_cache

        # Set to False during offline training to skip LLM calls entirely.
        self._generate_reasoning = generate_reasoning

        logger.info("[%s] Initialized with window_size=%d, graph_cache=%s",
                    self.name, WINDOW_SIZE, "enabled" if graph_cache else "disabled")

    def _process(self, msg: TransactionMessage) -> TransactionMessage:
        try:
            # Take a snapshot of current buffer state (up to last 5 IDs)
            snapshot = [m.transaction_id for m in list(self.buffer)[-5:]]
            msg.window_snapshot = snapshot

            # Log buffer state for observability
            logger.info(
                "[%s] Processing txn=%s | Buffer size=%d | Sender=%s | Receiver=%s | Amount=₹%.0f",
                self.name, msg.transaction_id[:12] if msg.transaction_id else "?",
                len(self.buffer), msg.nameOrig, msg.nameDest, msg.amount or 0
            )

            pattern, confidence = self._detect_pattern(msg)
            msg.pattern_type = pattern
            msg.pattern_confidence = confidence

            # Log detection result
            if pattern is not None and pattern != PatternType.NONE:
                logger.warning(
                    "[%s] 🚨 PATTERN DETECTED: %s (confidence=%.2f) | txn=%s",
                    self.name, pattern.value, confidence, msg.transaction_id[:12] if msg.transaction_id else "?"
                )
            else:
                logger.debug(
                    "[%s] No pattern detected for txn=%s",
                    self.name, msg.transaction_id[:12] if msg.transaction_id else "?"
                )

            # Update the most recent AgentMeta entry with detected confidence.
            for meta in reversed(msg.pipeline_metadata):
                if meta.agent_name == self.name:
                    meta.confidence = confidence
                    break

            # Use the LLM only to generate human-readable reasoning for
            # patterns that the rule-based detector has already identified.
            if self._generate_reasoning and pattern is not None and pattern is not PatternType.NONE:
                logger.info("[%s] Generating LLM reasoning for %s pattern...", self.name, pattern.value)
                msg.pattern_reasoning = self._call_llm(msg, pattern, confidence)
                if msg.pattern_reasoning:
                    logger.info("[%s] LLM reasoning: %s", self.name, msg.pattern_reasoning[:100])
            else:
                msg.pattern_reasoning = None

        except Exception as exc:
            logger.error("[%s] Pattern detection failed: %s", self.name, exc, exc_info=True)
            msg.pattern_type = PatternType.NONE
            msg.pattern_confidence = 0.0
            msg.pattern_reasoning = None
            self._record_fallback(msg, str(exc))

        finally:
            # History update must run regardless of detection outcome
            if msg.nameOrig:
                if msg.device_id:
                    self._user_device_history[msg.nameOrig].add(msg.device_id)
                if msg.ip_address:
                    self._user_ip_history[msg.nameOrig].add(msg.ip_address)

        return msg

    def _detect_pattern(
        self, msg: TransactionMessage
    ) -> tuple[PatternType, float]:
        """
        Inspect the rolling buffer, DB records, and Redis graph context
        to detect coordinated fraud patterns.

        Rule priority (highest confidence wins):

        1. MULE_NETWORK — Buffer + DB + Redis receiver convergence
        2. ACCOUNT_TAKEOVER — Novel IP/device for known user + high score
        3. VELOCITY_SPIKE — Buffer + DB + Redis time-based sender velocity
        """

        pattern: PatternType = PatternType.NONE
        confidence: float = 0.0

        # Fetch time-based velocity metrics early to share across multiple rules
        db_txn_count = 0
        db_total_amount = 0.0
        db_unique_receivers = 0
        db_avg_gap = float("inf")
        db_min_gap = float("inf")
        if msg.nameOrig:
            try:
                import database as db
                db_velocity = db.get_sender_time_velocity(
                    msg.nameOrig, lookback_minutes=10
                )
                db_txn_count = db_velocity["txn_count"]
                db_total_amount = db_velocity["total_amount"]
                db_unique_receivers = db_velocity["unique_receivers"]
                db_avg_gap = db_velocity["avg_inter_txn_seconds"]
                db_min_gap = db_velocity["min_inter_txn_seconds"]
            except Exception as e:
                logger.debug("[%s] Early velocity query failed: %s", self.name, e)

        # ══════════════════════════════════════════════════════════════════════
        # RULE 1: MULE_NETWORK — Multi-source convergence detection
        # ══════════════════════════════════════════════════════════════════════
        # Layer A: In-memory buffer check (original rule, enhanced)
        # Layer B: DB-backed receiver convergence (survives restarts)
        # Layer C: Redis real-time inbound stats (sub-second latency)
        # ══════════════════════════════════════════════════════════════════════

        # Verify if the destination is a registered and active merchant in the database
        is_merchant = False
        if msg.nameDest:
            try:
                import database as db
                dest_user = db.get_user_by_id(msg.nameDest)
                if dest_user and dest_user.get("user_type") == "MERCHANT" and dest_user.get("account_status") == "ACTIVE":
                    is_merchant = True
            except Exception as e:
                logger.debug("[%s] Database merchant check failed: %s", self.name, e)
                # Fallback to naming prefix check
                is_merchant = msg.nameDest.startswith("M")

        receiver_has_prior_flow = False
        if msg.nameDest:
            try:
                import database as db
                receiver_has_prior_flow = (
                    db.get_user_transaction_count(
                        msg.nameDest,
                        exclude_transaction_id=msg.transaction_id,
                        lookback_days=90,
                    )
                    > 0
                )
            except Exception as e:
                logger.debug("[%s] Receiver history check failed: %s", self.name, e)

        current_is_micro = msg.amount is not None and msg.amount < MULE_MICRO_AMOUNT_THRESHOLD
        is_receiver_dormant = (
            bool(msg.nameDest)
            and (msg.oldbalanceDest or 0.0) <= 0.0
            and not receiver_has_prior_flow
        )
        origin_looks_like_disburser = False
        if msg.nameOrig:
            try:
                import database as db
                origin_user = db.get_user_with_profile(msg.nameOrig) or db.get_user_by_id(msg.nameOrig)
                if origin_user:
                    origin_text = " ".join(
                        str(origin_user.get(key) or "")
                        for key in ("user_type", "email", "display_name", "business_name")
                    ).lower()
                    origin_looks_like_disburser = (
                        origin_user.get("user_type") == "MERCHANT"
                        or any(
                            token in origin_text
                            for token in ("corp", "company", "business", "merchant", "employer", "payroll", "salary")
                        )
                    )
            except Exception as e:
                logger.debug("[%s] Origin disbursement profile check failed: %s", self.name, e)

        amount_value = float(msg.amount or 0.0)
        sender_balance_before = float(msg.oldbalanceOrg or 0.0)
        sender_drain_ratio = (
            amount_value / sender_balance_before
            if sender_balance_before > 0
            else 1.0
        )
        looks_like_funded_disbursement = (
            origin_looks_like_disburser
            and amount_value > 0
            and sender_balance_before >= amount_value * 5
            and sender_drain_ratio <= 0.25
        )

        if msg.nameDest and msg.step is not None and not is_merchant:
            # ── Layer A: Buffer-based mule detection ──────────────────────────
            recent_window = [
                m for m in self.buffer
                if m.nameDest == msg.nameDest and (msg.step - m.step) <= 5
            ]
            unique_senders_buffer = {m.nameOrig for m in recent_window if m.nameOrig}
            n_buffer_senders = len(unique_senders_buffer)

            logger.debug(
                "[%s] MULE_NETWORK buffer check: nameDest=%s, unique_senders=%d (need ≥3)",
                self.name, msg.nameDest, n_buffer_senders
            )

            if n_buffer_senders >= 3:
                avg_amt = sum(m.amount or 0 for m in recent_window) / len(recent_window)
                is_micro = avg_amt < MULE_MICRO_AMOUNT_THRESHOLD  # Micro-structuring threshold
                
                # Behavioral Check: Dormant-to-Active or Pass-Through
                # If the destination account had 0 balance before this, or has very low balance relative to flow
                is_dormant_active = is_receiver_dormant
                
                mule_confidence = min(1.0, 0.50 + 0.10 * (n_buffer_senders - 3))
                
                if current_is_micro and not is_dormant_active:
                    # A single small UPI transfer to an already-active receiver
                    # must not inherit the receiver's old convergence history.
                    mule_confidence = 0.0
                elif is_micro:
                    # Contextual whitelist: Don't assume micro-structuring is fraud without downstream extraction proof
                    # unless it's a dormant account suddenly waking up.
                    if is_dormant_active:
                        mule_confidence = min(1.0, mule_confidence + 0.15)
                    else:
                        mule_confidence = 0.0  # Whitelist normal split bills
                else:
                    if is_dormant_active:
                        mule_confidence += 0.15
                    else:
                        # Relationship intelligence: cap confidence so it only HOLDs, never BLOCKs
                        mule_confidence = min(0.55, mule_confidence)
                    
                if mule_confidence > confidence and mule_confidence >= 0.50:
                    pattern = PatternType.MULE_NETWORK
                    confidence = mule_confidence
                    logger.info(
                        "[%s] ✓ MULE_NETWORK (buffer): %d unique senders → %s (conf=%.2f, micro=%s)",
                        self.name, n_buffer_senders, msg.nameDest, mule_confidence, is_micro
                    )

            # ── Layer B: DB-backed receiver convergence ───────────────────────
            # Queries the transaction table directly — covers ALL transactions,
            # not just the last 100 in the buffer. Catches the "10 friends
            # send ₹50" scenario that the buffer alone might miss.
            try:
                import database as db
                db_convergence = db.get_receiver_recent_convergence(
                    msg.nameDest, lookback_minutes=60
                )
                db_unique_senders = db_convergence["unique_senders"]
                db_txn_count = db_convergence["txn_count"]
                db_total_amount = db_convergence["total_amount"]

                logger.debug(
                    "[%s] MULE_NETWORK DB check: receiver=%s, unique_senders=%d, "
                    "txn_count=%d, total=₹%.0f",
                    self.name, msg.nameDest, db_unique_senders, db_txn_count,
                    db_total_amount
                )

                if db_unique_senders >= 3:
                    avg_amt = db_total_amount / max(1, db_txn_count)
                    is_micro = avg_amt < MULE_MICRO_AMOUNT_THRESHOLD
                    is_dormant_active = is_receiver_dormant

                    db_mule_conf = min(1.0, 0.50 + 0.08 * (db_unique_senders - 3))
                    
                    if current_is_micro and not is_dormant_active:
                        # Historical receiver convergence can explain why a
                        # receiver deserves monitoring, but it should not
                        # block a new user's tiny payment on its own.
                        db_mule_conf = 0.0
                    elif is_micro:
                        if is_dormant_active:
                            db_mule_conf = min(1.0, db_mule_conf + 0.15)
                        else:
                            db_mule_conf = 0.0
                    else:
                        if is_dormant_active:
                            db_mule_conf += 0.15
                        else:
                            db_mule_conf = min(0.55, db_mule_conf)

                    # Boost if cumulative amount is significant
                    if db_total_amount > MULE_LARGE_TOTAL_THRESHOLD and not (current_is_micro and not is_dormant_active):
                        db_mule_conf = min(1.0, db_mule_conf + 0.10)
                        
                    if db_mule_conf > confidence and db_mule_conf >= 0.50:
                        pattern = PatternType.MULE_NETWORK
                        confidence = db_mule_conf
                        logger.info(
                            "[%s] ✓ MULE_NETWORK (DB): %d senders → %s, "
                            "total=₹%.0f (conf=%.2f, micro=%s)",
                            self.name, db_unique_senders, msg.nameDest,
                            db_total_amount, db_mule_conf, is_micro
                        )
            except Exception as exc:
                logger.debug("[%s] DB mule check failed: %s", self.name, exc)

            # ── Layer C: Redis real-time inbound convergence ──────────────────
            if self._graph_cache and self._graph_cache.available:
                try:
                    redis_stats = self._graph_cache.get_receiver_inbound_stats(
                        msg.nameDest, window_seconds=3600
                    )
                    redis_senders = redis_stats["unique_senders"]
                    redis_burst = redis_stats["burst_5min"]

                    logger.debug(
                        "[%s] MULE_NETWORK Redis check: receiver=%s, "
                        "unique_senders=%d, burst_5min=%d",
                        self.name, msg.nameDest, redis_senders, redis_burst
                    )

                    if redis_senders >= 3:
                        redis_mule_conf = min(1.0, 0.55 + 0.08 * (redis_senders - 3))
                        
                        # Graph Analytics: Pass-Through Behavior (Fan-in -> Fan-out)
                        # Does this receiver immediately send the money elsewhere?
                        pass_through_bonus = 0.0
                        try:
                            redis_out = self._graph_cache.get_sender_outbound_stats(msg.nameDest, 3600)
                            if redis_out["txn_count"] > 0:
                                # High fan-out or rapid forwarding
                                if redis_out["unique_receivers"] >= 2 or redis_out["burst_5min"] >= 1:
                                    pass_through_bonus = 0.25
                                    logger.debug("[%s] Graph Analytics: Pass-through behavior detected for %s", self.name, msg.nameDest)
                        except Exception:
                            pass
                            
                        # Micro-structuring check
                        is_micro = msg.amount is not None and msg.amount < MULE_MICRO_AMOUNT_THRESHOLD
                        if pass_through_bonus == 0.0:
                            if is_micro:
                                # Contextual Whitelist: It's a birthday/split bill.
                                redis_mule_conf = 0.0
                            else:
                                # Relationship intelligence: Legitimate large split without proof of extraction
                                redis_mule_conf = min(0.55, redis_mule_conf)
                        else:
                            redis_mule_conf = min(1.0, redis_mule_conf + pass_through_bonus)

                        # Burst bonus: many inbound txns in last 5 minutes
                        if redis_burst >= 5:
                            redis_mule_conf = min(1.0, redis_mule_conf + 0.15)
                            
                        if redis_mule_conf > confidence and redis_mule_conf >= 0.50:
                            pattern = PatternType.MULE_NETWORK
                            confidence = redis_mule_conf
                            logger.info(
                                "[%s] ✓ MULE_NETWORK (Redis/Graph): %d senders → %s, "
                                "burst=%d, pass_through=%.2f (conf=%.2f)",
                                self.name, redis_senders, msg.nameDest,
                                redis_burst, pass_through_bonus, redis_mule_conf
                            )
                except Exception as exc:
                    logger.debug("[%s] Redis mule check failed: %s", self.name, exc)

        # ══════════════════════════════════════════════════════════════════════
        # RULE 5: MULE_EXTRACTION (Suspicion Propagation + Decay-Aware)
        # ══════════════════════════════════════════════════════════════════════
        # If msg.nameOrig recently received funds from many unique senders, and is now sending it out.
        # Enhanced: Also queries the decay engine for historical suspicion on the sender.
        if msg.nameOrig and msg.amount is not None and msg.amount > 0 and not is_merchant:
            try:
                if self._graph_cache and self._graph_cache.available:
                    inbound_stats = self._graph_cache.get_receiver_inbound_stats(msg.nameOrig, 3600)
                    if inbound_stats["unique_senders"] >= 3:
                        ext_conf = min(1.0, 0.70 + 0.10 * (inbound_stats["unique_senders"] - 3))

                        # Decay-aware amplifier: if sender already has historical
                        # pattern-tier suspicion, boost extraction confidence.
                        # This models "repeat offender" behavior where a previously
                        # flagged mule account is more likely to be extracting.
                        try:
                            decayed = self._graph_cache.get_decayed_risk(msg.nameOrig, tier="pattern")
                            hist_pattern_risk = decayed.get("pattern", 0.0)
                            if hist_pattern_risk > 0.1:
                                decay_boost = min(0.15, hist_pattern_risk * 0.2)
                                ext_conf = min(1.0, ext_conf + decay_boost)
                                logger.debug(
                                    "[%s] MULE_EXTRACTION decay boost: user=%s, "
                                    "hist_pattern=%.3f, boost=+%.3f",
                                    self.name, msg.nameOrig, hist_pattern_risk, decay_boost
                                )
                        except Exception:
                            pass  # graceful degradation — decay unavailable

                        if ext_conf > confidence:
                            pattern = PatternType.MULE_NETWORK
                            confidence = ext_conf
                            logger.info("[%s] ✓ MULE_EXTRACTION: %s is extracting funds (received from %d senders) (conf=%.2f)",
                                        self.name, msg.nameOrig, inbound_stats["unique_senders"], ext_conf)
            except Exception as exc:
                logger.debug("[%s] MULE_EXTRACTION check failed: %s", self.name, exc)

        # ══════════════════════════════════════════════════════════════════════
        # RULE 6: DORMANT_ACCOUNT_HIJACK
        # ══════════════════════════════════════════════════════════════════════
        # If a dormant account receives a massive sum, OR suddenly empties out
        is_large = msg.amount is not None and msg.amount >= DORMANT_LARGE_AMOUNT_THRESHOLD
        if is_large:
            if is_receiver_dormant and not is_merchant and not looks_like_funded_disbursement:
                # Sudden massive inbound to a dormant user account
                dorm_conf = 0.85
                if dorm_conf > confidence:
                    pattern = PatternType.ACCOUNT_TAKEOVER
                    confidence = dorm_conf
                    logger.info("[%s] ✓ DORMANT_HIJACK (Inbound): %s received %.0f to 0-balance (conf=%.2f)",
                                self.name, msg.nameDest, msg.amount, dorm_conf)
            elif is_receiver_dormant and looks_like_funded_disbursement:
                logger.info(
                    "[%s] Dormant inbound treated as funded disbursement: sender=%s receiver=%s amount=%.0f",
                    self.name, msg.nameOrig, msg.nameDest, msg.amount or 0.0
                )
            
            if msg.oldbalanceOrg > 0 and msg.amount >= 0.90 * msg.oldbalanceOrg:
                # Almost completely emptying an account with a single large transfer (cashout)
                dorm_conf = 0.85
                if dorm_conf > confidence:
                    pattern = PatternType.ACCOUNT_TAKEOVER
                    confidence = dorm_conf
                    logger.info("[%s] ✓ DORMANT_HIJACK (Cashout): %s emptying %.0f (conf=%.2f)",
                                self.name, msg.nameOrig, msg.amount, dorm_conf)

        # ══════════════════════════════════════════════════════════════════════
        # RULE 7: P2P_SCAM_SWEEP_DRAINAGE (Account Takeover / Mule drainage)
        # ══════════════════════════════════════════════════════════════════════
        # If a sender is rapidly transferring high-value sums to 3 or more unique
        # destinations in P2P transfers (not merchant), they are likely compromised
        # and being swept/drained by a scam/hacker.
        if msg.nameOrig and not is_merchant and db_unique_receivers >= 3:
            # Check if this is a high-value sweep (either current amount >= 5000 or total >= 15000)
            is_high_value_sweep = (msg.amount is not None and msg.amount >= 5000.0) or db_total_amount >= 15000.0
            if is_high_value_sweep:
                sweep_conf = 0.95
                if sweep_conf > confidence:
                    pattern = PatternType.ACCOUNT_TAKEOVER
                    confidence = sweep_conf
                    logger.warning(
                        "[%s] ✓ P2P_SCAM_SWEEP_DRAINAGE: user=%s is draining funds to %d unique receivers (total=₹%.2f, conf=%.2f)",
                        self.name, msg.nameOrig, db_unique_receivers, db_total_amount, sweep_conf
                    )

        # ══════════════════════════════════════════════════════════════════════
        # RULE 2: ACCOUNT_TAKEOVER — Novel IP/device for known user
        # ══════════════════════════════════════════════════════════════════════

        fraud_score = msg.fraud_score or 0.0
        if msg.nameOrig:
            known_ips = self._user_ip_history[msg.nameOrig]
            known_devices = self._user_device_history[msg.nameOrig]
            has_history = bool(known_ips) or bool(known_devices)
            new_ip = bool(msg.ip_address and msg.ip_address not in known_ips)
            new_device = bool(msg.device_id and msg.device_id not in known_devices)

            logger.debug(
                "[%s] ATO check: user=%s, has_history=%s, new_ip=%s, new_device=%s, fraud_score=%.2f",
                self.name, msg.nameOrig, has_history, new_ip, new_device, fraud_score
            )

            if has_history and (new_ip or new_device) and fraud_score >= ATO_FRAUD_SCORE_THRESHOLD:
                ato_confidence = min(1.0, 0.5 + 0.5 * fraud_score)
                if ato_confidence > confidence:
                    pattern = PatternType.ACCOUNT_TAKEOVER
                    confidence = ato_confidence
                    logger.info(
                        "[%s] ✓ ACCOUNT_TAKEOVER: user=%s, new_ip=%s, new_device=%s (conf=%.2f)",
                        self.name, msg.nameOrig, new_ip, new_device, ato_confidence
                    )

        # ══════════════════════════════════════════════════════════════════════
        # RULE 3: VELOCITY_SPIKE — Multi-layer velocity detection
        # ══════════════════════════════════════════════════════════════════════
        # Layer A: Buffer-based count (original rule)
        # Layer B: DB-backed time-based velocity (inter-txn timing)
        # Layer C: Redis real-time burst + outbound stats
        # ══════════════════════════════════════════════════════════════════════

        if msg.nameOrig:
            # ── Layer A: Buffer-based velocity ────────────────────────────────
            recent_history = list(self.buffer)[-WINDOW_SIZE:]
            count_same_origin = sum(1 for m in recent_history if m.nameOrig == msg.nameOrig)

            logger.debug(
                "[%s] VELOCITY buffer check: user=%s, occurrences_in_last_75=%d (need >=%d)",
                self.name, msg.nameOrig, count_same_origin, VELOCITY_SPIKE_COUNT_THRESHOLD
            )

            if count_same_origin >= VELOCITY_SPIKE_COUNT_THRESHOLD:
                velocity_confidence = min(1.0, 0.62 + 0.04 * (count_same_origin - VELOCITY_SPIKE_COUNT_THRESHOLD))
                if velocity_confidence > confidence:
                    if pattern in (PatternType.MULE_NETWORK, PatternType.ACCOUNT_TAKEOVER, PatternType.CIRCULAR_FLOW) and confidence >= 0.60:
                        pass
                    else:
                        pattern = PatternType.VELOCITY_SPIKE
                        confidence = velocity_confidence
                        logger.info(
                            "[%s] VELOCITY_SPIKE (buffer): user=%s, %d txns in last 75 (conf=%.2f)",
                            self.name, msg.nameOrig, count_same_origin, velocity_confidence
                        )

            # ── Layer B: DB-backed time-based velocity ────────────────────────
            # Queries the transaction table for sender activity in last 10 min.
            # Key advantage: detects inter-transaction timing, not just count.
            logger.debug(
                "[%s] VELOCITY DB check: user=%s, txn_count=%d, "
                "avg_gap=%.1fs, min_gap=%.1fs, unique_receivers=%d",
                self.name, msg.nameOrig, db_txn_count,
                db_avg_gap if db_avg_gap != float("inf") else -1,
                db_min_gap if db_min_gap != float("inf") else -1,
                db_unique_receivers
            )

            # Trigger if ≥5 txns in 10 minutes
            if db_txn_count >= 5:
                db_vel_conf = min(1.0, 0.65 + 0.05 * (db_txn_count - 5))
                # Boost if inter-txn gap is very small (rapid-fire)
                if db_avg_gap != float("inf") and db_avg_gap < 30:
                    db_vel_conf = min(1.0, db_vel_conf + 0.15)
                elif db_avg_gap != float("inf") and db_avg_gap < 60:
                    db_vel_conf = min(1.0, db_vel_conf + 0.10)
                # Boost if sending to many unique receivers (smurfing pattern)
                if db_unique_receivers >= 3:
                    db_vel_conf = min(1.0, db_vel_conf + 0.05)
                if db_vel_conf > confidence:
                    if pattern in (PatternType.MULE_NETWORK, PatternType.ACCOUNT_TAKEOVER, PatternType.CIRCULAR_FLOW) and confidence >= 0.60:
                        pass
                    else:
                        pattern = PatternType.VELOCITY_SPIKE
                        confidence = db_vel_conf
                        logger.info(
                            "[%s] ✓ VELOCITY_SPIKE (DB): user=%s, %d txns in 10min, "
                            "avg_gap=%.1fs (conf=%.2f)",
                            self.name, msg.nameOrig, db_txn_count, db_avg_gap,
                            db_vel_conf
                        )

            # Also trigger on extremely rapid bursts (≥3 txns with <10s gaps)
            elif db_txn_count >= 3 and db_min_gap != float("inf") and db_min_gap < 10:
                burst_conf = min(1.0, 0.70 + 0.10 * (3 - db_min_gap))
                if burst_conf > confidence:
                    if pattern in (PatternType.MULE_NETWORK, PatternType.ACCOUNT_TAKEOVER, PatternType.CIRCULAR_FLOW) and confidence >= 0.60:
                        pass
                    else:
                        pattern = PatternType.VELOCITY_SPIKE
                        confidence = burst_conf
                        logger.info(
                            "[%s] ✓ VELOCITY_SPIKE (DB burst): user=%s, min_gap=%.1fs (conf=%.2f)",
                            self.name, msg.nameOrig, db_min_gap, burst_conf
                        )

            # ── Layer C: Redis real-time burst detection ──────────────────────
            if self._graph_cache and self._graph_cache.available:
                try:
                    redis_stats = self._graph_cache.get_sender_outbound_stats(
                        msg.nameOrig, window_seconds=3600
                    )
                    redis_txn_count = redis_stats["txn_count"]
                    redis_burst = redis_stats["burst_5min"]
                    redis_avg_gap = redis_stats["avg_inter_txn_seconds"]

                    logger.debug(
                        "[%s] VELOCITY Redis check: user=%s, txn_1h=%d, "
                        "burst_5min=%d, avg_gap=%.1fs",
                        self.name, msg.nameOrig, redis_txn_count,
                        redis_burst, redis_avg_gap if redis_avg_gap != float("inf") else -1
                    )

                    # 5-minute burst: ≥4 txns in 5 minutes is extremely suspicious
                    if redis_burst >= 4:
                        redis_vel_conf = min(1.0, 0.70 + 0.08 * (redis_burst - 4))
                        if redis_vel_conf > confidence:
                            if pattern in (PatternType.MULE_NETWORK, PatternType.ACCOUNT_TAKEOVER, PatternType.CIRCULAR_FLOW) and confidence >= 0.60:
                                pass
                            else:
                                pattern = PatternType.VELOCITY_SPIKE
                                confidence = redis_vel_conf
                                logger.info(
                                    "[%s] ✓ VELOCITY_SPIKE (Redis burst): user=%s, "
                                    "%d txns in 5min (conf=%.2f)",
                                    self.name, msg.nameOrig, redis_burst, redis_vel_conf
                                )

                    # Hourly volume: ≥10 txns in 1 hour
                    elif redis_txn_count >= 10:
                        redis_vel_conf = min(1.0, 0.65 + 0.05 * (redis_txn_count - 10))
                        if redis_vel_conf > confidence:
                            if pattern in (PatternType.MULE_NETWORK, PatternType.ACCOUNT_TAKEOVER, PatternType.CIRCULAR_FLOW) and confidence >= 0.60:
                                pass
                            else:
                                pattern = PatternType.VELOCITY_SPIKE
                                confidence = redis_vel_conf
                                logger.info(
                                    "[%s] ✓ VELOCITY_SPIKE (Redis hourly): user=%s, "
                                    "%d txns in 1h (conf=%.2f)",
                                    self.name, msg.nameOrig, redis_txn_count, redis_vel_conf
                                )
                except Exception as exc:
                    logger.debug("[%s] Redis velocity check failed: %s", self.name, exc)

            # Apply Merchant-Safe Velocity Rule
            if pattern == PatternType.VELOCITY_SPIKE and is_merchant:
                confidence = max(0.0, confidence - 0.35)
                confidence = min(0.55, confidence)
                if confidence < 0.50:
                    pattern = PatternType.NONE
                    confidence = 0.0
                logger.info("[%s] Merchant-safe velocity applied to %s. New conf=%.2f", self.name, msg.nameOrig, confidence)

            # Apply Low-Value Exemption to avoid false positives on small peer-to-peer transfers (e.g., split bills, minor transfers)
            if pattern == PatternType.VELOCITY_SPIKE and msg.amount is not None and msg.amount < VELOCITY_MIN_AMOUNT_THRESHOLD:
                # Bypass low-value exemption if the user is sending to multiple unique customer receivers (bot probe/velocity script pattern)
                recent_history = list(self.buffer)[-WINDOW_SIZE:]
                unique_customer_receivers = set()
                for m in recent_history:
                    if m.nameOrig == msg.nameOrig and m.nameDest:
                        is_dest_merchant = False
                        try:
                            import database as db
                            dest_user = db.get_user_by_id(m.nameDest)
                            if dest_user and dest_user.get("user_type") == "MERCHANT":
                                is_dest_merchant = True
                        except Exception:
                            is_dest_merchant = m.nameDest.startswith("M")
                        if not is_dest_merchant:
                            unique_customer_receivers.add(m.nameDest)

                redis_unique_receivers = 0
                if self._graph_cache and self._graph_cache.available:
                    try:
                        redis_stats = self._graph_cache.get_sender_outbound_stats(msg.nameOrig, window_seconds=3600)
                        redis_unique_receivers = redis_stats.get("unique_receivers", 0)
                    except Exception:
                        pass

                if len(unique_customer_receivers) >= 3 or redis_unique_receivers >= 4:
                    logger.info("[%s] Low-value velocity exemption BYPASSED for %s due to rapid counterparty burst (receivers=%d, redis_receivers=%d)",
                                self.name, msg.nameOrig, len(unique_customer_receivers), redis_unique_receivers)
                else:
                    confidence = max(0.0, confidence - 0.40)
                    confidence = min(0.50, confidence)
                    if confidence < 0.50:
                        pattern = PatternType.NONE
                        confidence = 0.0
                    logger.info("[%s] Low-value velocity exemption applied to %s for amount ₹%.2f. New conf=%.2f", self.name, msg.nameOrig, msg.amount, confidence)

        # ══════════════════════════════════════════════════════════════════════
        # RULE 4: CIRCULAR_FLOW — Ping-Pong or Triangle loops via Redis
        # ══════════════════════════════════════════════════════════════════════
        if self._graph_cache and self._graph_cache.available and msg.nameOrig and msg.nameDest:
            try:
                cycle_data = self._graph_cache.check_circular_flow(msg.nameOrig, msg.nameDest)
                if cycle_data["is_circular"]:
                    # High confidence for ping-pong, even higher for complex triangles
                    cycle_type = cycle_data["type"]
                    cycle_conf = 0.85 if cycle_type == "ping_pong" else 0.95
                    
                    if cycle_conf > confidence:
                        pattern = PatternType.CIRCULAR_FLOW
                        confidence = cycle_conf
                        logger.info(
                            "[%s] ✓ CIRCULAR_FLOW (Redis): %s detected between %s and %s (conf=%.2f)",
                            self.name, cycle_type, msg.nameOrig, msg.nameDest, cycle_conf
                        )
            except Exception as exc:
                logger.debug("[%s] Redis circular flow check failed: %s", self.name, exc)

        return pattern, confidence

    # ── Observable signal helpers (used by Orchestrator → Agent 3) ───────────

    def is_new_device(self, nameOrig: str, device_id: str) -> bool:
        """
        Return True iff device_id has never been seen for this account AND
        the account has prior device history (so we are not flagging first-time
        customers as suspicious simply because we have no record of them).
        """
        if not nameOrig or not device_id:
            return False
        history = self._user_device_history.get(nameOrig)
        if not history:
            return False  # no history → cannot classify as new vs known
        return device_id not in history

    def is_new_ip(self, nameOrig: str, ip_address: str) -> bool:
        """
        Return True iff ip_address has never been seen for this account AND
        the account has prior IP history.
        """
        if not nameOrig or not ip_address:
            return False
        history = self._user_ip_history.get(nameOrig)
        if not history:
            return False
        return ip_address not in history

    def _call_llm(
        self,
        msg: TransactionMessage,
        pattern: PatternType,
        confidence: float,
        timeout: int = 15,  # Reduced timeout - 15 seconds max
    ) -> str:
        """
        Call the local LFM2.5-1.2B-Instruct model via llama.cpp server
        to generate natural-language reasoning for an already-detected pattern.

        The server exposes an OpenAI-compatible chat API at:
            http://localhost:8080/v1/chat/completions

        Start the server with: python run_llama_server.py

        The model must respond ONLY with:
            {"reasoning": "<one sentence>"}

        Robust to timeouts / parse errors — falls back to pattern-specific default.
        """
        try:
            # Build compact JSON summary of last 10 transactions in the buffer.
            window = list(self.buffer)[-10:]

            def _txn_summary(t: TransactionMessage) -> dict:
                return {
                    "transaction_id": t.transaction_id,
                    "nameOrig": t.nameOrig,
                    "nameDest": t.nameDest,
                    "amount": t.amount,
                    "step": t.step,
                    "ip_address": t.ip_address,
                    "device_id": t.device_id,
                    "fraud_score": t.fraud_score,
                }

            window_payload = [_txn_summary(t) for t in window]
            current_payload = _txn_summary(msg)

            user_content = json.dumps(
                {
                    "pattern_type": pattern.value,
                    "pattern_confidence": confidence,
                    "window": window_payload,
                    "current_transaction": current_payload,
                },
                separators=(",", ":"),
            )

            system_message = (
                "You are a fraud pattern explanation model. "
                "Given a detected pattern label and supporting evidence from a "
                "rolling transaction window, explain in one concise sentence "
                "why this pattern label makes sense. Do not change the label."
            )

            user_instruction = (
                "Here is a JSON summary of the detected pattern, the last 10 "
                "transactions in the buffer, and the current transaction. "
                "Respond ONLY with a single JSON object of the form: "
                '{"reasoning": "<one sentence>"}.\n'
                "Do not include any extra text outside the JSON object.\n"
                "Data:\n"
                f"{user_content}"
            )

            payload = {
                "model": "LFM2.5-1.2B-Instruct",  # LiquidAI model via llama.cpp
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_instruction},
                ],
                "temperature": 0.2,
                "max_tokens": 256,
                "stream": False,
            }

            response = requests.post(
                "http://localhost:8080/v1/chat/completions",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            # Some local LLMs may wrap the JSON in extra text or code fences.
            def _extract_json_object(raw: str) -> str:
                start = raw.find("{")
                if start == -1:
                    raise ValueError("No JSON object found in LLM response.")
                depth = 0
                for idx, ch in enumerate(raw[start:], start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return raw[start : idx + 1]
                raise ValueError("Unbalanced JSON object in LLM response.")

            content_stripped = content.strip()
            json_str = _extract_json_object(content_stripped)
            parsed = json.loads(json_str)
            reasoning = str(parsed.get("reasoning", "")).strip()

            logger.debug(
                "[%s] LLM reasoning for pattern=%s confidence=%.3f: %s",
                self.name,
                pattern.value,
                confidence,
                reasoning,
            )

            return reasoning

        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning(
                "[%s] LLM reasoning generation failed (%s). Using fallback reasoning.",
                self.name,
                exc,
            )
            # Provide pattern-specific fallback reasoning
            fallback_reasons = {
                PatternType.MULE_NETWORK: f"Multiple unique senders converging on same destination detected with {confidence:.0%} confidence.",
                PatternType.VELOCITY_SPIKE: f"Rapid transaction velocity from same sender detected with {confidence:.0%} confidence.",
                PatternType.ACCOUNT_TAKEOVER: f"Suspicious activity from new device/IP for established user detected with {confidence:.0%} confidence.",
            }
            return fallback_reasons.get(pattern, f"Pattern {pattern.value} detected with {confidence:.0%} confidence.")
