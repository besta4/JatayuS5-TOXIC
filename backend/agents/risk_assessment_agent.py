"""
risk_assessment_agent.py — Agent 3: Risk Assessment

Combines fraud_score (Agent 1) + pattern_type (Agent 2) + account context
to decide a risk tier (LOW/MEDIUM/HIGH/CRITICAL) and a recommended action
(PASS / SILENT_FLAG / HOLD / BLOCK).

Current implementation: rule-based stub.
TODO: Replace rules with RL policy or learnable threshold optimization.
"""

from __future__ import annotations

import logging
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple
import typing
import math

if typing.TYPE_CHECKING:  # pragma: no cover
    import torch
    import torch.nn as nn
    import torch.optim as optim

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful degradation
    torch = typing.cast(typing.Any, None)
    nn = typing.cast(typing.Any, None)
    optim = typing.cast(typing.Any, None)
    _TORCH_AVAILABLE = False

from agents.base_agent import BaseAgent
from agents.models import (
    Action,
    PatternType,
    RiskLevel,
    TransactionMessage,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PPO core
# ─────────────────────────────────────────────────────────────────────────────

ACTION_SPACE: List[Action] = [
    Action.PASS,
    Action.SILENT_FLAG,
    Action.HOLD,
    Action.BLOCK,
]

PATTERN_SPACE: List[PatternType] = [
    PatternType.NONE,
    PatternType.MULE_NETWORK,
    PatternType.ACCOUNT_TAKEOVER,
    PatternType.VELOCITY_SPIKE,
]

PPO_PATTERN_ALIASES: dict[PatternType, PatternType] = {
    PatternType.CIRCULAR_FLOW: PatternType.MULE_NETWORK,
}


def _safe_float(x: Optional[float], default: float = 0.0) -> float:
    try:
        return float(x) if x is not None and not math.isnan(float(x)) else default
    except Exception:
        return default


def _one_hot(index: int, size: int) -> List[float]:
    return [1.0 if i == index else 0.0 for i in range(size)]


def build_state_vector(msg: TransactionMessage, account_context: Optional[dict[str, Any]] = None) -> List[float]:
    """
    Construct the RL state vector for risk assessment.

    State design (contextual bandit style, per-transaction), 11 dimensions:
        [0]  fraud_score                  ∈ [0, 1]           (from Agent 1)
        [1]  pattern_confidence           ∈ [0, 1]           (from Agent 2, 0 if None)
        [2:6] pattern_type one-hot        (NONE, MULE, ATO, VELOCITY_SPIKE)
        [6]  log10(amount + 1) / 6        (scale typical PaySim magnitudes)
        [7]  is_flagged_by_model          (fraud_label from Agent 1)
        [8]  account_age_days_normalized  (if available in account_context)
        [9]  is_new_device                (from account_context, data-driven)
        [10] is_new_ip                    (from account_context, data-driven)

    Note: traffic_mode (MULE_NETWORK / ACCOUNT_TAKEOVER) is intentionally
    excluded.  It is a synthetic scenario label that does not exist at
    inference time and would cause a train/serve distribution shift.
    The pattern one-hot at [2:6] already encodes the same signal from
    observable evidence (Agent 2 output).

    CIRCULAR_FLOW is a newer Agent 2 pattern, while the bundled PPO checkpoint
    was trained on the original 11-dimensional state. For PPO only, circular
    flow maps to the MULE_NETWORK bucket because both represent coordinated
    laundering behavior. Rule-based risk logic still sees CIRCULAR_FLOW as a
    distinct pattern.
    """

    fraud_score = _safe_float(msg.fraud_score, 0.0)
    pattern_confidence = _safe_float(msg.pattern_confidence, 0.0)

    pattern = msg.pattern_type or PatternType.NONE
    pattern = PPO_PATTERN_ALIASES.get(pattern, pattern)
    try:
        pattern_index = PATTERN_SPACE.index(pattern)
    except ValueError:
        pattern_index = 0

    pattern_oh = _one_hot(pattern_index, len(PATTERN_SPACE))

    # Amount scaling: PaySim uses amounts up to ~10^7. Use log scaling.
    amount_scaled = math.log10(max(0.0, _safe_float(msg.amount, 0.0)) + 1.0) / 6.0

    is_flagged = 1.0 if msg.fraud_label else 0.0

    ctx = account_context or {}
    age_days = ctx.get("account_age_days")
    age_days_norm = 0.0
    if isinstance(age_days, (int, float)) and age_days >= 0:
        # Assume 10 years as rough upper bound for normalization.
        age_days_norm = min(1.0, float(age_days) / (365.0 * 10.0))

    is_new_device = bool(ctx.get("is_new_device") or False)
    is_new_ip = bool(ctx.get("is_new_ip") or False)

    return [
        fraud_score,           # [0]
        pattern_confidence,    # [1]
        *pattern_oh,           # [2:6]
        amount_scaled,         # [6]
        is_flagged,            # [7]
        age_days_norm,         # [8]
        1.0 if is_new_device else 0.0,  # [9]
        1.0 if is_new_ip else 0.0,      # [10]
    ]


def compute_reward(msg: TransactionMessage, action: Action) -> float:
    """
    Reward function for risk assessment.

    Uses msg.ground_truth_label as the supervision signal.  Returns 0.0
    when no confirmed label is available so that unsupervised transitions
    are silently skipped by train_on_messages().

    Reward table:
        If fraud (ground_truth_label=True):
            BLOCK        → +1.0   (correct enforcement)
            HOLD         → +0.3   (too weak — fraud confirmed, should have blocked)
            SILENT_FLAG  → -0.5   (flagged but not stopped — unacceptable for fraud)
            PASS         → -2.0   (missed fraud, heavy penalty)

        If not fraud (ground_truth_label=False):
            PASS         → +0.5   (correct pass-through)
            SILENT_FLAG  → -0.2   (minor friction on legitimate transaction)
            HOLD         → -1.0   (unnecessary hold)
            BLOCK        → -2.0   (false positive, heavy penalty)

        If ground_truth_label is None:
            → 0.0        (no supervision signal — skip this transition)
    """
    if msg.ground_truth_label is None:
        return 0.0  # no confirmed label — caller should skip this transition

    is_fraud = bool(msg.ground_truth_label)

    if is_fraud:
        if action == Action.BLOCK:
            return 1.0
        if action == Action.HOLD:
            return 0.3
        if action == Action.SILENT_FLAG:
            return -0.5
        return -2.0

    # Non-fraudulent — penalize unnecessary friction.
    if action == Action.PASS:
        return 0.5
    if action == Action.SILENT_FLAG:
        return -0.2
    if action == Action.HOLD:
        return -1.0
    return -2.0


if _TORCH_AVAILABLE:

    class _PPOActorCritic(nn.Module):
        """
        Simple MLP actor-critic for contextual bandit style PPO.
        """

        def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
            self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: Any) -> Tuple[Any, Any]:
            logits = self.actor(x)
            value = self.critic(x).squeeze(-1)
            return logits, value


    @dataclass
    class _PPOConfig:
        state_dim: int
        action_dim: int = len(ACTION_SPACE)
        hidden_dim: int = 64
        lr: float = 3e-4
        gamma: float = 0.0  # contextual bandit – no temporal credit assignment
        clip_epsilon: float = 0.2
        entropy_coef: float = 0.01
        value_coef: float = 0.5
        max_grad_norm: float = 0.5
        epochs: int = 4
        batch_size: int = 64


    @dataclass
    class _Transition:
        state: List[float]
        action_idx: int
        log_prob: float
        reward: float
        value: float


    class PPORiskPolicy:
        """
        PPO training + inference wrapper for the risk assessment policy.

        This class is intentionally generic so that it can be re-used from
        external training scripts without going through the full agent pipeline.
        """

        def __init__(self, config: _PPOConfig, device: str = "cpu") -> None:
            self.config = config
            self.device = torch.device(device)
            self.model = _PPOActorCritic(
                state_dim=config.state_dim,
                hidden_dim=config.hidden_dim,
                action_dim=config.action_dim,
            ).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        # ── Inference ──────────────────────────────────────────────────────────

        def act(
            self, state: Sequence[float], deterministic: bool = False
        ) -> Tuple[int, float, float]:
            """
            Return (action_index, log_prob, value_estimate).
            """
            self.model.eval()
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits, value = self.model(s)
                dist = torch.distributions.Categorical(logits=logits)
                if deterministic:
                    action = torch.argmax(dist.probs, dim=-1)
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
            return int(action.item()), float(log_prob.item()), float(value.squeeze(0).item())

        # ── Training ───────────────────────────────────────────────────────────

        def update(self, transitions: Sequence[_Transition]) -> None:
            """
            Run PPO updates over a batch of transitions.
            """
            if not transitions:
                return

            states = torch.tensor(
                [t.state for t in transitions],
                dtype=torch.float32,
                device=self.device,
            )
            actions = torch.tensor(
                [t.action_idx for t in transitions],
                dtype=torch.int64,
                device=self.device,
            )
            old_log_probs = torch.tensor(
                [t.log_prob for t in transitions],
                dtype=torch.float32,
                device=self.device,
            )
            rewards = torch.tensor(
                [t.reward for t in transitions],
                dtype=torch.float32,
                device=self.device,
            )
            old_values = torch.tensor(
                [t.value for t in transitions],
                dtype=torch.float32,
                device=self.device,
            )

            # Contextual bandit: return = reward (no temporal discounting).
            returns = rewards
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            dataset_size = states.size(0)
            batch_size = min(self.config.batch_size, dataset_size)

            for _ in range(self.config.epochs):
                indices = torch.randperm(dataset_size, device=self.device)
                for start in range(0, dataset_size, batch_size):
                    end = start + batch_size
                    batch_idx = indices[start:end]

                    batch_states = states[batch_idx]
                    batch_actions = actions[batch_idx]
                    batch_old_log_probs = old_log_probs[batch_idx]
                    batch_returns = returns[batch_idx]
                    batch_advantages = advantages[batch_idx]

                    logits, values = self.model(batch_states)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    ) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = (batch_returns - values).pow(2).mean()

                    loss = (
                        policy_loss
                        + self.config.value_coef * value_loss
                        - self.config.entropy_coef * entropy
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

        # ── Persistence ────────────────────────────────────────────────────────

        def save(self, path: str) -> None:
            torch.save(
                {
                    "state_dict": self.model.state_dict(),
                    "config": dataclasses.asdict(self.config),
                },
                path,
            )

        @classmethod
        def load(cls, path: str, device: str = "cpu") -> "PPORiskPolicy":
            payload = torch.load(path, map_location=device)
            cfg_dict = payload.get("config", {})
            config = _PPOConfig(**cfg_dict)
            policy = cls(config=config, device=device)
            policy.model.load_state_dict(payload["state_dict"])
            return policy


class RiskAssessmentAgent(BaseAgent):
    """
    Agent 3 — Risk Assessment with PPO-based reinforcement learning.

    Reads from TransactionMessage:
        msg.fraud_score      (from Agent 1)
        msg.fraud_label      (from Agent 1)
        msg.pattern_type     (from Agent 2)
        msg.pattern_confidence

    Writes to TransactionMessage:
        msg.risk_level           → RiskLevel enum
        msg.recommended_action   → Action enum
        msg.account_context      → dict with supporting context

    PPO integration:
        - When explicitly enabled and PyTorch is available, the agent
          instantiates a PPORiskPolicy and uses it for action selection
          (inference mode).
        - If PyTorch is unavailable, the agent falls back to the existing
          rule-based decision table.

    Training:
        - Use `train_on_messages()` to run PPO updates on a batch of
          TransactionMessage objects (e.g. from a simulated dataset).
        - Use `save_policy()` / `load_policy()` for offline training scripts.
    """

    name = "RiskAssessmentAgent"

    def __init__(
        self,
        use_rl: bool = False,
        device: str = "auto",
        policy_path: Optional[str] = None,
    ) -> None:
        # TODO: At init, load account context source (e.g. a JSON file or DB connection)
        # that maps nameOrig → {"account_age_days": ..., "tier": ..., "velocity_limit": ...}.
        # This is needed so the RL policy / rule engine can factor in account standing.

        # Auto-detect device if set to "auto"
        if device == "auto" and _TORCH_AVAILABLE:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("[%s] Auto-detected device: %s", self.name, device)
        elif device == "auto":
            device = "cpu"

        # Safety default:
        # - We keep PPO integrated, but do NOT enable it unless the caller opts in
        #   or provides a trained policy_path. This avoids an untrained random
        #   policy taking real actions by default.
        enable_rl = bool(use_rl or policy_path)
        self._use_rl = enable_rl and _TORCH_AVAILABLE
        self._device = device
        self._policy: Optional[PPORiskPolicy] = None

        # Compute state dimension from a dummy message for convenience.
        dummy_msg = TransactionMessage()
        dummy_state = build_state_vector(dummy_msg, account_context=None)
        self._state_dim = len(dummy_state)

        if self._use_rl and _TORCH_AVAILABLE:
            try:
                if policy_path:
                    self._policy = PPORiskPolicy.load(policy_path, device=device)
                    logger.info(
                        "[%s] Loaded PPO policy from %s (state_dim=%d)",
                        self.name,
                        policy_path,
                        self._state_dim,
                    )
                else:
                    cfg = _PPOConfig(state_dim=self._state_dim)
                    self._policy = PPORiskPolicy(config=cfg, device=device)
                    logger.info(
                        "[%s] Initialized new PPO policy (state_dim=%d)",
                        self.name,
                        self._state_dim,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[%s] Failed to initialize PPO policy (%s). Falling back to rules.",
                    self.name,
                    exc,
                )
                self._use_rl = False
                self._policy = None
        elif use_rl and not _TORCH_AVAILABLE:
            logger.warning(
                "[%s] PyTorch is not available. Falling back to rule-based logic.",
                self.name,
            )
        else:
            logger.info("[%s] Initialized in rule-based mode.", self.name)

    # ── BaseAgent interface ────────────────────────────────────────────────────

    def process(
        self,
        msg: TransactionMessage,
        account_hints: Optional[dict[str, Any]] = None,
    ) -> TransactionMessage:
        """
        Override BaseAgent.process() to accept pre-computed account hints
        from the Orchestrator (sourced from Agent 2's device/IP history).

        The hints are stashed on the instance before the BaseAgent provenance
        wrapper calls _process(), then cleared afterwards.  This keeps the
        BaseAgent interface intact while threading Orchestrator-owned context
        into the agent without storing it on the message itself.
        """
        self._pending_account_hints = account_hints
        try:
            return super().process(msg)
        finally:
            self._pending_account_hints = None

    def _process(self, msg: TransactionMessage) -> TransactionMessage:
        account_hints = getattr(self, "_pending_account_hints", None)
        account_context = self._build_account_context(msg, hints=account_hints)

        if self._use_rl and self._policy is not None:
            try:
                state = build_state_vector(msg, account_context=account_context)
                action_idx, _, _ = self._policy.act(state, deterministic=True)
                action = ACTION_SPACE[action_idx]
                risk_level = self._map_action_to_risk(action)
            except Exception as exc:
                logger.warning(
                    "[%s] PPO inference failed (%s). Falling back to rules.",
                    self.name,
                    exc,
                )
                self._record_fallback(msg, f"PPO inference failed: {exc}")
                fraud_score = msg.fraud_score or 0.0
                pattern_type = msg.pattern_type or PatternType.NONE
                pattern_confidence = msg.pattern_confidence or 0.0
                historical_risk = account_context.get("historical_risk", 0.0)
                model_threshold = account_context.get("model_threshold")
                risk_level, action = self._decide_rule_based(
                    fraud_score,
                    pattern_type,
                    pattern_confidence,
                    historical_risk,
                    model_threshold=model_threshold,
                    fraud_label=bool(msg.fraud_label),
                )
        else:
            fraud_score = msg.fraud_score or 0.0
            pattern_type = msg.pattern_type or PatternType.NONE
            pattern_confidence = msg.pattern_confidence or 0.0
            historical_risk = account_context.get("historical_risk", 0.0)
            model_threshold = account_context.get("model_threshold")
            risk_level, action = self._decide_rule_based(
                fraud_score,
                pattern_type,
                pattern_confidence,
                historical_risk,
                model_threshold=model_threshold,
                fraud_label=bool(msg.fraud_label),
            )

        msg.risk_level = risk_level
        msg.recommended_action = action
        msg.account_context = account_context
        return msg

    # ── Rule-based fallback (original behavior) ───────────────────────────────

    def _decide_rule_based(
        self, fraud_score: float, pattern_type: PatternType,
        pattern_confidence: float = 0.0,
        historical_risk: float = 0.0,
        model_threshold: Optional[float] = None,
        fraud_label: bool = False,
    ) -> tuple[RiskLevel, Action]:
        """
        Rule-based risk decision table combining ML fraud score + pattern detection
        + time-decayed historical risk.

        Pattern detection (Agent 2) operates independently of ML scoring (Agent 1):
        - ML score: transaction-level fraud probability from XGBoost+GNN model
        - Pattern detection: buffer analysis for coordinated fraud (MULE_NETWORK, VELOCITY_SPIKE, ATO)
        - Historical risk: composite time-decayed suspicion from Redis decay engine R(t) = R₀ × e^{-λt}

        Historical risk amplification:
        - Acts as a "memory" that upgrades borderline decisions
        - A user with decayed risk ≥ 0.30 gets stricter treatment on borderline cases
        - A user with decayed risk ≥ 0.50 gets even stricter treatment
        - This prevents repeat offenders from slipping through with "just under threshold" txns

        Decision logic:
        | fraud_score | pattern_type                       | pattern_confidence | risk     | action      |
        |-------------|------------------------------------|--------------------|----------|-------------|
        | ≥ 0.80      | any                                | any                | CRITICAL | BLOCK       |
        | any         | MULE_NETWORK/ATO                   | ≥ 0.60             | HIGH     | BLOCK       |
        | any         | VELOCITY_SPIKE                     | ≥ 0.60             | HIGH     | BLOCK       |
        | ≥ 0.50      | MULE_NETWORK/ATO                   | any                | HIGH     | BLOCK       |
        | ≥ 0.50      | any other                          | any                | HIGH     | HOLD        |
        | any         | MULE_NETWORK/ATO/VELOCITY_SPIKE    | any (detected)     | MEDIUM   | HOLD        |
        | ≥ 0.20      | NONE                               | 0                  | MEDIUM   | SILENT_FLAG |
        | < 0.20      | NONE                               | 0                  | LOW      | PASS        |
        """

        coordinated = pattern_type in (
            PatternType.MULE_NETWORK,
            PatternType.ACCOUNT_TAKEOVER,
            PatternType.CIRCULAR_FLOW,
        )
        
        velocity = pattern_type == PatternType.VELOCITY_SPIKE
        pattern_detected = pattern_type != PatternType.NONE

        # ── Historical risk amplifier ─────────────────────────────────────────
        # Effective fraud score is boosted by decayed historical risk.
        # This models "suspicion memory" — a user who was flagged yesterday
        # gets slightly stricter treatment today, but a user flagged last week
        # gets almost none (exponential decay).
        hist_boost = min(0.15, historical_risk * 0.3)  # Max +0.15 boost
        effective_score = min(1.0, fraud_score + hist_boost)
        threshold = float(model_threshold or 0.0)
        threshold_ratio = fraud_score / threshold if threshold > 0 else 0.0
        calibrated_flag = bool(fraud_label or (threshold > 0 and fraud_score >= threshold))

        # Very high ML score - always block
        if effective_score >= 0.80:
            return RiskLevel.CRITICAL, Action.BLOCK
        
        # High-confidence coordinated pattern detection - block
        if coordinated and pattern_confidence >= 0.60:
            return RiskLevel.HIGH, Action.BLOCK
            
        # Velocity detection - reduce binary blocking. Only BLOCK if extremely high confidence.
        if velocity:
            if pattern_confidence >= 0.85:
                return RiskLevel.HIGH, Action.BLOCK
            elif pattern_confidence >= 0.60:
                return RiskLevel.HIGH, Action.HOLD

        # PaySim-trained XGBoost threshold calibration.
        # The saved model threshold is intentionally low for imbalanced fraud
        # detection, so a numerically small score can still be model-positive.
        # Threshold multiples make the trained dataset influence enforcement
        # even when Agent 2 has not found a coordinated pattern yet.
        if calibrated_flag:
            if threshold_ratio >= 12.0 or fraud_score >= 0.35:
                if coordinated or historical_risk >= 0.30:
                    return RiskLevel.CRITICAL, Action.BLOCK
                return RiskLevel.HIGH, Action.HOLD
            if coordinated and (threshold_ratio >= 4.0 or fraud_score >= 0.10):
                return RiskLevel.HIGH, Action.BLOCK
        
        # Moderate ML score with coordinated pattern - block
        if effective_score >= 0.50 and coordinated:
            return RiskLevel.HIGH, Action.BLOCK
        
        # Moderate ML score without coordination - hold for review
        if effective_score >= 0.50:
            return RiskLevel.HIGH, Action.HOLD
        
        # Pattern detected but lower confidence - hold for review
        if pattern_detected:
            return RiskLevel.MEDIUM, Action.HOLD

        # ── Historical risk upgrade for borderline cases ──────────────────────
        # If no pattern detected and ML score is low, but historical risk is
        # significant, upgrade from PASS to SILENT_FLAG (flag for monitoring).
        if historical_risk >= 0.50 and fraud_score >= 0.10:
            return RiskLevel.MEDIUM, Action.HOLD
        if historical_risk >= 0.30 and fraud_score >= 0.15:
            return RiskLevel.MEDIUM, Action.SILENT_FLAG
        
        # Low ML score, no pattern - normal processing
        if effective_score >= 0.20:
            return RiskLevel.MEDIUM, Action.SILENT_FLAG
        
        return RiskLevel.LOW, Action.PASS

    # ── Account context stub (unchanged, but used by RL state) ────────────────

    def _build_account_context(
        self,
        msg: TransactionMessage,
        hints: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Build an account context dict for downstream agents and RL state.

        The `hints` dict carries pre-computed, data-driven signals from the
        Orchestrator (sourced from Agent 2's device/IP history tracker).
        This replaces the previous hard-coded True stubs that always signalled
        "new device" and "new IP", biasing the policy toward ATO patterns.

        Real implementation should also query:
          - Account creation date (for age calculation)
          - Transaction velocity limits per account tier
          - Recent flagged transaction count (for recidivism check)
        """
        hints = hints or {}
        dataset = msg.dataset_influence or {}
        model_threshold = dataset.get("decision_threshold")
        try:
            model_threshold = float(model_threshold) if model_threshold is not None else None
        except Exception:
            model_threshold = None
        if model_threshold and model_threshold > 0:
            model_score_ratio = float(msg.fraud_score or 0.0) / model_threshold
        else:
            model_score_ratio = 0.0
        gnn_profile = dataset.get("gnn_embedding") or {}
        return {
            # TODO: Look up real account_age from account database.
            "account_age_days": None,
            # TODO: Look up known good devices from device history store.
            "known_devices": [msg.device_id],
            # TODO: Look up known good IPs from IP history store.
            "known_ips": [msg.ip_address],
            # TODO: Look up velocity limit from account tier config.
            "daily_txn_limit": None,
            # Data-driven: False unless Agent 2 confirms a novel device for a
            # known user.  Never hard-coded True.
            "is_new_device": hints.get("is_new_device", False),
            "is_new_ip":     hints.get("is_new_ip",     False),
            # Exponential decay risk — time-weighted historical suspicion
            # from the Redis-backed decay engine. Ranges [0, 1].
            "historical_risk": hints.get("historical_risk", 0.0),
            "decayed_risk_tiers": hints.get("decayed_risk_tiers", {}),
            "model_threshold": model_threshold,
            "model_score_ratio": round(model_score_ratio, 4),
            "dataset_artifact_mode": dataset.get("artifact_mode"),
            "gnn_embedding_used": bool(gnn_profile.get("used", False)),
            "gnn_structural_signal": gnn_profile.get("structural_signal", 0.0),
        }

    # ── Public hooks for other agents / training scripts ──────────────────────

    @property
    def rl_enabled(self) -> bool:
        """Return True if PPO is active for decision-making."""
        return self._use_rl and self._policy is not None

    def map_action_to_risk(self, action: Action) -> RiskLevel:
        """
        Public wrapper for mapping an Action to a canonical RiskLevel.
        """
        return self._map_action_to_risk(action)

    def build_state(self, msg: TransactionMessage) -> List[float]:
        """
        Public helper to build the PPO state vector for a given message.
        """
        ctx = self._build_account_context(msg, hints=None)
        return build_state_vector(msg, account_context=ctx)

    def _map_action_to_risk(self, action: Action) -> RiskLevel:
        """
        Map chosen Action → RiskLevel tier.

        This mapping keeps semantics close to the original rule-based policy:
            PASS         → LOW
            SILENT_FLAG  → MEDIUM
            HOLD         → HIGH
            BLOCK        → CRITICAL
        """
        if action == Action.BLOCK:
            return RiskLevel.CRITICAL
        if action == Action.HOLD:
            return RiskLevel.HIGH
        if action == Action.SILENT_FLAG:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    # ── PPO training interface ────────────────────────────────────────────────

    def train_on_messages(
        self,
        messages: Iterable[TransactionMessage],
        reward_fn: Optional[Callable[[TransactionMessage, Action], float]] = None,
        deterministic: bool = False,
    ) -> None:
        """
        Run one PPO training epoch over a collection of messages.

        Only messages with a confirmed ground_truth_label are used to form
        training transitions.  Messages where ground_truth_label is None
        (i.e. unlabeled, real-time inference) return reward=0.0 from the
        default reward function and are skipped to avoid noise.

        Args:
            messages:
                Iterable of TransactionMessage objects (e.g. from a synthetic
                generator or logged production data with confirmed labels).
            reward_fn:
                Optional custom reward function. If None, `compute_reward`
                is used, which requires msg.ground_truth_label to be set.
            deterministic:
                If True, uses greedy actions for training (mainly for
                offline fine-tuning). Otherwise, samples from the policy.
        """
        if not self.rl_enabled:
            logger.warning(
                "[%s] train_on_messages called but RL is disabled. No-op.",
                self.name,
            )
            return

        assert self._policy is not None  # for type-checkers

        reward_fn = reward_fn or compute_reward
        transitions: List[_Transition] = []
        skipped = 0

        for msg in messages:
            ctx = self._build_account_context(msg, hints=None)
            state = build_state_vector(msg, account_context=ctx)
            action_idx, log_prob, value = self._policy.act(
                state, deterministic=deterministic
            )
            action = ACTION_SPACE[action_idx]
            reward = reward_fn(msg, action)

            # Skip transitions with no supervision signal (ground_truth_label=None
            # causes compute_reward to return 0.0 as a sentinel).
            if reward == 0.0 and msg.ground_truth_label is None:
                skipped += 1
                continue

            transitions.append(
                _Transition(
                    state=state,
                    action_idx=action_idx,
                    log_prob=log_prob,
                    reward=reward,
                    value=value,
                )
            )

        if skipped:
            logger.warning(
                "[%s] Skipped %d unlabeled transitions (ground_truth_label=None).",
                self.name,
                skipped,
            )

        if transitions:
            self._policy.update(transitions)

    def save_policy(self, path: str) -> None:
        """
        Save the current PPO policy to disk.
        """
        if not self.rl_enabled or self._policy is None:
            logger.warning(
                "[%s] save_policy called but RL is disabled. No-op.",
                self.name,
            )
            return
        self._policy.save(path)

    def load_policy(self, path: str) -> None:
        """
        Load a PPO policy from disk and activate RL-based decision-making.
        """
        if not _TORCH_AVAILABLE:
            logger.error(
                "[%s] Cannot load PPO policy: PyTorch is not available.",
                self.name,
            )
            return
        try:
            self._policy = PPORiskPolicy.load(path, device=self._device)
            self._use_rl = True
            logger.info(
                "[%s] Loaded PPO policy from %s.",
                self.name,
                path,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "[%s] Failed to load PPO policy from %s (%s). Keeping existing policy.",
                self.name,
                path,
                exc,
            )
