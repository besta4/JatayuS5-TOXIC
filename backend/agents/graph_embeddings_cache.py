"""
graph_embeddings_cache.py — Redis-backed Dynamic Graph Context Engine

Provides real-time graph awareness that supplements static GNN embeddings.
Instead of relying solely on precomputed embeddings from training time, this
module tracks live transaction relationships, velocity windows, and
account behavior via Redis, producing a dynamic feature vector that Agent 1
uses as a bounded post-model risk adjustment.

Redis data structures used:
    - Sorted Sets (ZSET):  time-indexed transaction logs per user
    - Hash Maps (HMAP):    per-user aggregate counters (total_in, total_out, etc.)
    - Sets:                unique counterparty tracking

All keys have TTL of 72 hours to auto-expire stale data.

Graceful degradation: if Redis is unavailable, all methods return
zero-vectors / safe defaults — the pipeline never crashes.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict

import numpy as np
from config import load_environment

logger = logging.getLogger(__name__)
load_environment()

# Dimensions of the dynamic feature vector produced by this module. Keep this
# at 16 unless the XGBoost training pipeline is retrained for a wider vector.
DYNAMIC_FEATURE_DIM = 16

# TTL for all Redis keys (144 hours = 6 days = 2x storage retention)
KEY_TTL_SECONDS = 144 * 3600

# ── Exponential Risk Score Decay Constants ────────────────────────────────────
#
# R(t) = R₀ × e^{-λt}   where λ = ln(2) / half_life
#
# Three decay tiers with different half-lives:
#   SHORT  — velocity/burst signals (fade within hours)
#   MEDIUM — pattern-level suspicion: mule, ATO, circular flow (fade within a day)
#   LONG   — structural network risk: graph topology suspicion (fade slowly)
#
# Half-lives are in SECONDS for computational efficiency.
# ──────────────────────────────────────────────────────────────────────────────

DECAY_HALF_LIFE_SHORT  = 1 * 3600       # 1 hour  → λ ≈ 0.000193/s
DECAY_HALF_LIFE_MEDIUM = 12 * 3600      # 12 hours → λ ≈ 0.0000161/s
DECAY_HALF_LIFE_LONG   = 72 * 3600      # 72 hours → λ ≈ 0.00000267/s

DECAY_LAMBDA_SHORT  = math.log(2) / DECAY_HALF_LIFE_SHORT
DECAY_LAMBDA_MEDIUM = math.log(2) / DECAY_HALF_LIFE_MEDIUM
DECAY_LAMBDA_LONG   = math.log(2) / DECAY_HALF_LIFE_LONG

# Minimum score below which we consider the risk "fully decayed" (saves Redis space)
DECAY_FLOOR = 0.01

# Risk score tiers mapped to decay rates
DECAY_TIER_MAP = {
    "velocity":   DECAY_LAMBDA_SHORT,    # VELOCITY_SPIKE signals
    "burst":      DECAY_LAMBDA_SHORT,    # Burst activity signals
    "pattern":    DECAY_LAMBDA_MEDIUM,   # MULE_NETWORK, ATO, CIRCULAR_FLOW
    "network":    DECAY_LAMBDA_LONG,     # Structural graph-based risk
}


class DynamicGraphCache:
    """
    Redis-backed dynamic graph context engine.

    Tracks per-user transaction patterns in real time and produces a
    feature vector capturing:
      - Outbound velocity (txn count + total amount in last 1h, 24h)
      - Inbound velocity (txn count + total amount in last 1h, 24h)
      - Unique counterparties (senders-to-me, receivers-from-me) in 24h
      - Fan-in ratio (unique senders / total inbound txns)
      - Fan-out ratio (unique receivers / total outbound txns)
      - Account activity age (seconds since first seen txn)
      - Burst score (txns in last 5 minutes)
    """

    def __init__(self, redis_url: str | None = None, prefix: str | None = None) -> None:
        self._redis: Any = None
        self._available: bool = False
        self.prefix = prefix or os.environ.get("JATAYU_REDIS_PREFIX", "jatayu")
        redis_url = redis_url or os.environ.get("JATAYU_REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis
            self._redis = redis.Redis.from_url(
                redis_url, decode_responses=True, socket_timeout=2
            )
            self._redis.ping()
            self._available = True
            logger.info("[DynamicGraphCache] Connected to Redis at %s (prefix: %s)", redis_url, self.prefix)
        except Exception as exc:
            logger.warning(
                "[DynamicGraphCache] Redis unavailable (%s). "
                "Dynamic graph features disabled — using zero vectors.",
                exc,
            )

    @property
    def available(self) -> bool:
        return self._available

    # ── Record a transaction (call AFTER Agent 1, BEFORE Agent 2) ─────────

    def record_transaction(
        self,
        sender_id: str,
        receiver_id: str,
        amount: float,
        transaction_id: str,
    ) -> None:
        """
        Record a new transaction edge in the dynamic graph.
        Updates all velocity counters and counterparty sets.
        """
        if not self._available or not self._redis:
            return

        now_ms = time.time()
        txn_value = f"{transaction_id}:{amount}"

        pipe = self._redis.pipeline(transaction=False)
        try:
            # ── Sender outbound log (sorted set, score=timestamp) ─────────
            sender_out_key = f"{self.prefix}:out:{sender_id}"
            pipe.zadd(sender_out_key, {txn_value: now_ms})
            pipe.expire(sender_out_key, KEY_TTL_SECONDS)

            # ── Receiver inbound log (sorted set, score=timestamp) ────────
            receiver_in_key = f"{self.prefix}:in:{receiver_id}"
            pipe.zadd(receiver_in_key, {txn_value: now_ms})
            pipe.expire(receiver_in_key, KEY_TTL_SECONDS)

            # ── Sender's unique receiver set (24h window) ─────────────────
            sender_receivers_key = f"{self.prefix}:out_peers:{sender_id}"
            pipe.sadd(sender_receivers_key, receiver_id)
            pipe.expire(sender_receivers_key, KEY_TTL_SECONDS)

            # ── Receiver's unique sender set (24h window) ─────────────────
            receiver_senders_key = f"{self.prefix}:in_peers:{receiver_id}"
            pipe.sadd(receiver_senders_key, sender_id)
            pipe.expire(receiver_senders_key, KEY_TTL_SECONDS)

            # ── Aggregate counters ────────────────────────────────────────
            sender_agg_key = f"{self.prefix}:agg:{sender_id}"
            pipe.hincrby(sender_agg_key, "total_out_count", 1)
            pipe.hincrbyfloat(sender_agg_key, "total_out_amount", amount)
            pipe.hsetnx(sender_agg_key, "first_seen", str(now_ms))
            pipe.expire(sender_agg_key, KEY_TTL_SECONDS)

            receiver_agg_key = f"{self.prefix}:agg:{receiver_id}"
            pipe.hincrby(receiver_agg_key, "total_in_count", 1)
            pipe.hincrbyfloat(receiver_agg_key, "total_in_amount", amount)
            pipe.hsetnx(receiver_agg_key, "first_seen", str(now_ms))
            pipe.expire(receiver_agg_key, KEY_TTL_SECONDS)

            pipe.execute()
        except Exception as exc:
            logger.warning("[DynamicGraphCache] Failed to record transaction: %s", exc)

    # ── Compute dynamic feature vector for a user ─────────────────────────

    def get_dynamic_features(self, user_id: str) -> np.ndarray:
        """
        Build a DYNAMIC_FEATURE_DIM-dimensional feature vector for the given
        user based on their real-time transaction graph context.

        Vector layout (16 dims):
          [0]  out_count_1h          - outbound txn count in last 1 hour
          [1]  out_amount_1h         - outbound total amount in last 1 hour (log-scaled)
          [2]  out_count_24h         - outbound txn count in last 24 hours
          [3]  out_amount_24h        - outbound total amount in last 24 hours (log-scaled)
          [4]  in_count_1h           - inbound txn count in last 1 hour
          [5]  in_amount_1h          - inbound total amount in last 1 hour (log-scaled)
          [6]  in_count_24h          - inbound txn count in last 24 hours
          [7]  in_amount_24h         - inbound total amount in last 24 hours (log-scaled)
          [8]  unique_receivers_24h  - unique receivers sent to in 24h
          [9]  unique_senders_24h    - unique senders received from in 24h
          [10] fan_out_ratio         - unique_receivers / max(out_count_24h, 1)
          [11] fan_in_ratio          - unique_senders / max(in_count_24h, 1)
          [12] account_age_hours     - hours since first seen (capped at 720 = 30 days)
          [13] burst_5min            - outbound txn count in last 5 minutes
          [14] in_burst_5min         - inbound txn count in last 5 minutes
          [15] amount_velocity_ratio - out_amount_1h / max(out_amount_24h, 1)
        """
        if not self._available or not self._redis:
            return np.zeros(DYNAMIC_FEATURE_DIM, dtype=np.float32)

        now_ms = time.time()
        one_hour_ago = now_ms - 3600
        twenty_four_hours_ago = now_ms - 86400
        five_min_ago = now_ms - 300

        try:
            pipe = self._redis.pipeline(transaction=False)

            # Outbound velocity
            out_key = f"{self.prefix}:out:{user_id}"
            pipe.zrangebyscore(out_key, one_hour_ago, "+inf", withscores=True)       # [0]
            pipe.zrangebyscore(out_key, twenty_four_hours_ago, "+inf", withscores=True)  # [1]
            pipe.zrangebyscore(out_key, five_min_ago, "+inf", withscores=True)        # [2]

            # Inbound velocity
            in_key = f"{self.prefix}:in:{user_id}"
            pipe.zrangebyscore(in_key, one_hour_ago, "+inf", withscores=True)         # [3]
            pipe.zrangebyscore(in_key, twenty_four_hours_ago, "+inf", withscores=True)  # [4]
            pipe.zrangebyscore(in_key, five_min_ago, "+inf", withscores=True)          # [5]

            # Unique counterparties
            pipe.scard(f"{self.prefix}:out_peers:{user_id}")   # [6]
            pipe.scard(f"{self.prefix}:in_peers:{user_id}")    # [7]

            # Aggregate
            pipe.hgetall(f"{self.prefix}:agg:{user_id}")       # [8]

            results = pipe.execute()

            # Parse outbound
            out_1h = results[0] or []
            out_24h = results[1] or []
            out_5min = results[2] or []

            out_count_1h = len(out_1h)
            out_amount_1h = sum(self._parse_amount(v) for v, _ in out_1h)
            out_count_24h = len(out_24h)
            out_amount_24h = sum(self._parse_amount(v) for v, _ in out_24h)

            # Parse inbound
            in_1h = results[3] or []
            in_24h = results[4] or []
            in_5min = results[5] or []

            in_count_1h = len(in_1h)
            in_amount_1h = sum(self._parse_amount(v) for v, _ in in_1h)
            in_count_24h = len(in_24h)
            in_amount_24h = sum(self._parse_amount(v) for v, _ in in_24h)

            # Counterparties
            unique_receivers = results[6] or 0
            unique_senders = results[7] or 0

            # Aggregate
            agg = results[8] or {}
            first_seen = float(agg.get("first_seen", str(now_ms)))
            account_age_hours = min((now_ms - first_seen) / 3600, 720.0)

            # Derived features
            fan_out_ratio = unique_receivers / max(out_count_24h, 1)
            fan_in_ratio = unique_senders / max(in_count_24h, 1)
            burst_5min = len(out_5min)
            in_burst_5min = len(in_5min)
            amount_velocity_ratio = out_amount_1h / max(out_amount_24h, 1.0)

            # Build vector (log-scale amounts to keep magnitudes manageable)
            vec = np.array([
                float(out_count_1h),
                np.log1p(out_amount_1h),
                float(out_count_24h),
                np.log1p(out_amount_24h),
                float(in_count_1h),
                np.log1p(in_amount_1h),
                float(in_count_24h),
                np.log1p(in_amount_24h),
                float(unique_receivers),
                float(unique_senders),
                fan_out_ratio,
                fan_in_ratio,
                account_age_hours / 720.0,  # normalize to [0,1]
                float(burst_5min),
                float(in_burst_5min),
                amount_velocity_ratio,
            ], dtype=np.float32)

            return vec

        except Exception as exc:
            logger.warning("[DynamicGraphCache] Failed to compute features: %s", exc)
            return np.zeros(DYNAMIC_FEATURE_DIM, dtype=np.float32)

    # ── Receiver convergence query (used by Agent 2) ──────────────────────

    def get_receiver_inbound_stats(
        self, receiver_id: str, window_seconds: int = 3600
    ) -> Dict[str, Any]:
        """
        Get inbound convergence stats for a receiver within the time window.

        Returns:
            {
                "unique_senders": int,
                "txn_count": int,
                "total_amount": float,
                "burst_5min": int,
            }
        """
        if not self._available or not self._redis:
            return {"unique_senders": 0, "txn_count": 0, "total_amount": 0.0, "burst_5min": 0}

        now_ms = time.time()
        window_start = now_ms - window_seconds
        five_min_ago = now_ms - 300

        try:
            pipe = self._redis.pipeline(transaction=False)
            in_key = f"{self.prefix}:in:{receiver_id}"
            pipe.zrangebyscore(in_key, window_start, "+inf", withscores=True)
            pipe.zrangebyscore(in_key, five_min_ago, "+inf", withscores=True)
            pipe.scard(f"{self.prefix}:in_peers:{receiver_id}")
            results = pipe.execute()

            txns_in_window = results[0] or []
            txns_5min = results[1] or []
            unique_senders = results[2] or 0

            return {
                "unique_senders": int(unique_senders),
                "txn_count": len(txns_in_window),
                "total_amount": sum(self._parse_amount(v) for v, _ in txns_in_window),
                "burst_5min": len(txns_5min),
            }
        except Exception as exc:
            logger.warning("[DynamicGraphCache] Receiver stats failed: %s", exc)
            return {"unique_senders": 0, "txn_count": 0, "total_amount": 0.0, "burst_5min": 0}

    # ── Sender velocity query (used by Agent 2) ──────────────────────────

    def get_sender_outbound_stats(
        self, sender_id: str, window_seconds: int = 3600
    ) -> Dict[str, Any]:
        """
        Get outbound velocity stats for a sender within the time window.

        Returns:
            {
                "txn_count": int,
                "total_amount": float,
                "unique_receivers": int,
                "burst_5min": int,
                "avg_inter_txn_seconds": float,   # average gap between txns
            }
        """
        if not self._available or not self._redis:
            return {
                "txn_count": 0, "total_amount": 0.0,
                "unique_receivers": 0, "burst_5min": 0,
                "avg_inter_txn_seconds": float("inf"),
            }

        now_ms = time.time()
        window_start = now_ms - window_seconds
        five_min_ago = now_ms - 300

        try:
            pipe = self._redis.pipeline(transaction=False)
            out_key = f"{self.prefix}:out:{sender_id}"
            pipe.zrangebyscore(out_key, window_start, "+inf", withscores=True)
            pipe.zrangebyscore(out_key, five_min_ago, "+inf", withscores=True)
            pipe.scard(f"{self.prefix}:out_peers:{sender_id}")
            results = pipe.execute()

            txns_in_window = results[0] or []
            txns_5min = results[1] or []
            unique_receivers = results[2] or 0

            # Compute average inter-transaction time
            timestamps = sorted([score for _, score in txns_in_window])
            if len(timestamps) >= 2:
                gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                avg_gap = sum(gaps) / len(gaps)
            else:
                avg_gap = float("inf")

            return {
                "txn_count": len(txns_in_window),
                "total_amount": sum(self._parse_amount(v) for v, _ in txns_in_window),
                "unique_receivers": int(unique_receivers),
                "burst_5min": len(txns_5min),
                "avg_inter_txn_seconds": avg_gap,
            }
        except Exception as exc:
            logger.warning("[DynamicGraphCache] Sender stats failed: %s", exc)
            return {
                "txn_count": 0, "total_amount": 0.0,
                "unique_receivers": 0, "burst_5min": 0,
                "avg_inter_txn_seconds": float("inf"),
            }

    # ── Graph Cyclicality Query (used by Agent 2) ─────────────────────────

    def check_circular_flow(self, sender_id: str, receiver_id: str) -> dict[str, Any]:
        """
        Check for cyclical money flow (Ping-Pong or 3-hop triangles) in O(1) time.
        Evaluates the current edge (sender -> receiver) against the historical graph.

        Returns:
            {"is_circular": bool, "type": str, "nodes": list}
        """
        if not self._available or not self._redis:
            return {"is_circular": False, "type": "none"}
            
        try:
            # 1. 2-hop loop (Ping-Pong): Has B sent to A recently?
            # i.e., is A in B's out_peers?
            b_out_peers_key = f"{self.prefix}:out_peers:{receiver_id}"
            is_ping_pong = self._redis.sismember(b_out_peers_key, sender_id)
            if is_ping_pong:
                return {"is_circular": True, "type": "ping_pong"}
                
            # 2. 3-hop loop (Triangle): Has B sent to C, who has sent to A?
            # This is the intersection of B's out_peers and A's in_peers.
            a_in_peers_key = f"{self.prefix}:in_peers:{sender_id}"
            common_nodes = self._redis.sinter(b_out_peers_key, a_in_peers_key)
            if common_nodes:
                return {
                    "is_circular": True, 
                    "type": "triangle", 
                    "nodes": [n.decode('utf-8') if isinstance(n, bytes) else n for n in common_nodes]
                }
                
        except Exception as exc:
            logger.warning("[DynamicGraphCache] Circular flow check failed: %s", exc)
            
        return {"is_circular": False, "type": "none"}

    # ══════════════════════════════════════════════════════════════════════
    # EXPONENTIAL RISK SCORE DECAY ENGINE
    # ══════════════════════════════════════════════════════════════════════
    #
    # Implements: R(t) = R₀ × e^{-λt}
    #
    # Strategy: "Decay-on-Read" (lazy evaluation)
    #   - When a risk event occurs, we STORE the raw score + timestamp.
    #   - When any system component READS the risk, we compute the
    #     time-decayed value on the fly.
    #   - This avoids expensive background decay jobs and ensures
    #     O(1) complexity per read.
    #
    # Redis structure per user:
    #   Hash: jatayu:risk:{user_id}
    #     velocity_score      → float (raw score at recording time)
    #     velocity_ts          → float (epoch seconds when recorded)
    #     pattern_score        → float
    #     pattern_ts           → float
    #     network_score        → float
    #     network_ts           → float
    # ══════════════════════════════════════════════════════════════════════

    def record_risk_score(
        self,
        user_id: str,
        tier: str,
        score: float,
        accumulate: bool = True,
    ) -> None:
        """
        Record a risk score for a user in a specific decay tier.

        Args:
            user_id:     The account/user to tag.
            tier:        One of 'velocity', 'burst', 'pattern', 'network'.
            score:       The raw suspicion score ∈ [0, 1].
            accumulate:  If True, add to existing (decayed) score instead of
                         replacing. This models "suspicion stacking" where
                         repeated bad behavior compounds risk.
        """
        if not self._available or not self._redis or tier not in DECAY_TIER_MAP:
            return

        now = time.time()
        risk_key = f"{self.prefix}:risk:{user_id}"
        score_field = f"{tier}_score"
        ts_field = f"{tier}_ts"

        try:
            if accumulate:
                # Read existing score and apply decay before adding
                existing = self._redis.hmget(risk_key, score_field, ts_field)
                old_score = float(existing[0]) if existing[0] else 0.0
                old_ts = float(existing[1]) if existing[1] else now

                # Decay the old score to present time
                dt = max(0.0, now - old_ts)
                lam = DECAY_TIER_MAP[tier]
                decayed_old = old_score * math.exp(-lam * dt)

                # Accumulate: old (decayed) + new, capped at 1.0
                new_score = min(1.0, decayed_old + score)
            else:
                new_score = min(1.0, score)

            pipe = self._redis.pipeline(transaction=False)
            pipe.hset(risk_key, score_field, str(new_score))
            pipe.hset(risk_key, ts_field, str(now))
            pipe.expire(risk_key, KEY_TTL_SECONDS)
            pipe.execute()

            logger.debug(
                "[DynamicGraphCache] Risk recorded: user=%s tier=%s score=%.4f "
                "(accumulated=%s)",
                user_id, tier, new_score, accumulate,
            )
        except Exception as exc:
            logger.warning(
                "[DynamicGraphCache] Failed to record risk score: %s", exc
            )

    def get_decayed_risk(
        self, user_id: str, tier: str | None = None
    ) -> Dict[str, float]:
        """
        Get time-decayed risk scores for a user.

        Args:
            user_id: The account/user to query.
            tier:    If specified, return only that tier. Otherwise return all.

        Returns:
            Dict mapping tier names to their current decayed scores.
            Example: {"velocity": 0.12, "pattern": 0.67, "network": 0.89}
        """
        result: Dict[str, float] = {}
        if not self._available or not self._redis:
            for t in (DECAY_TIER_MAP if tier is None else {tier: 0}):
                result[t] = 0.0
            return result

        now = time.time()
        risk_key = f"{self.prefix}:risk:{user_id}"
        tiers_to_query = [tier] if tier else list(DECAY_TIER_MAP.keys())

        try:
            # Batch-read all score/ts pairs in a single pipeline
            pipe = self._redis.pipeline(transaction=False)
            for t in tiers_to_query:
                pipe.hmget(risk_key, f"{t}_score", f"{t}_ts")
            raw_results = pipe.execute()

            for i, t in enumerate(tiers_to_query):
                pair = raw_results[i]
                raw_score = float(pair[0]) if pair[0] else 0.0
                raw_ts = float(pair[1]) if pair[1] else now

                if raw_score <= DECAY_FLOOR:
                    result[t] = 0.0
                    continue

                dt = max(0.0, now - raw_ts)
                lam = DECAY_TIER_MAP.get(t, DECAY_LAMBDA_MEDIUM)
                decayed = raw_score * math.exp(-lam * dt)

                # Floor check: if decayed below threshold, treat as zero
                result[t] = round(decayed, 6) if decayed > DECAY_FLOOR else 0.0

        except Exception as exc:
            logger.warning(
                "[DynamicGraphCache] Failed to read decayed risk: %s", exc
            )
            for t in tiers_to_query:
                result[t] = 0.0

        return result

    def get_composite_risk(self, user_id: str) -> float:
        """
        Compute a single composite risk score by combining all decay tiers.

        Weighting:
          - velocity:  0.15  (transient, fades fast)
          - burst:     0.15  (transient)
          - pattern:   0.40  (most important — confirmed pattern suspicion)
          - network:   0.30  (structural risk from graph topology)

        Returns:
            Composite risk ∈ [0, 1], already time-decayed.
        """
        tier_weights = {
            "velocity": 0.15,
            "burst":    0.15,
            "pattern":  0.40,
            "network":  0.30,
        }

        decayed = self.get_decayed_risk(user_id)
        composite = sum(
            decayed.get(t, 0.0) * w for t, w in tier_weights.items()
        )
        return min(1.0, round(composite, 6))

    def clear_risk(
        self, user_id: str, tier: str | None = None
    ) -> None:
        """
        Manually clear risk scores for a user (e.g., after admin review).

        Args:
            user_id: The user to clear.
            tier:    If specified, clear only that tier. Otherwise clear all.
        """
        if not self._available or not self._redis:
            return

        risk_key = f"{self.prefix}:risk:{user_id}"
        try:
            if tier:
                self._redis.hdel(risk_key, f"{tier}_score", f"{tier}_ts")
            else:
                self._redis.delete(risk_key)
            logger.info(
                "[DynamicGraphCache] Risk cleared: user=%s tier=%s",
                user_id, tier or "ALL",
            )
        except Exception as exc:
            logger.warning(
                "[DynamicGraphCache] Failed to clear risk: %s", exc
            )

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _parse_amount(txn_value: str) -> float:
        """Parse amount from stored value format 'txn_id:amount'."""
        try:
            return float(txn_value.split(":")[-1])
        except (ValueError, IndexError):
            return 0.0
