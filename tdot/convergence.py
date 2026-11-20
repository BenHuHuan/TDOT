"""Adaptive two-stage convergence & early-stop monitor for fast dynamic trajectory optimization.

Pure Python stdlib (zero PyTorch/JAX imports) so it runs identically inside both the
persistent NablaEngine daemon (Python 3.12 JAX) and the TDOT web server / MLAB4_Fast runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

PresetName = Literal["fast", "standard", "tight"]


@dataclass(frozen=True)
class ConvergenceConfig:
  """Thresholds and cadence controlling adaptive early stopping."""

  preset: PresetName = "standard"
  target_return: float = 0.86
  target_mpjpe_mm: float = 48.0
  target_root_mm: float = 65.0
  target_joint_rad: float = 0.09
  max_action_rate_rms: float = 0.65
  confirm_streak: int = 2

  # Plateau early-exit for physically impossible / noisy input trajectories.
  min_feasible_return: float = 0.42
  plateau_patience: int = 6
  plateau_delta: float = 0.004

  # Two-stage adaptive evaluation cadence.
  min_iters: int = 10
  coarse_eval_every: int = 15
  fine_eval_every: int = 5
  arm_term_rate: float = 0.08
  arm_alive_frac: float = 0.70

  # Hard safety caps.
  max_seconds: float = 180.0
  max_iters: int = 2000

  @classmethod
  def from_preset(
    cls,
    preset: PresetName = "standard",
    **overrides: Any,
  ) -> ConvergenceConfig:
    base = dict(PRESET_CONFIGS[preset])
    base["preset"] = preset
    for k, v in overrides.items():
      if v is not None:
        base[k] = v
    return cls(**base)


PRESET_CONFIGS: dict[PresetName, dict[str, Any]] = {
  "fast": {
    "target_return": 0.80,
    "target_mpjpe_mm": 70.0,
    "target_root_mm": 90.0,
    "target_joint_rad": 0.14,
    "max_action_rate_rms": 0.85,
    "confirm_streak": 2,
    "min_feasible_return": 0.38,
    "plateau_patience": 4,
    "plateau_delta": 0.005,
    "min_iters": 10,
    "coarse_eval_every": 15,
    "fine_eval_every": 5,
    "max_seconds": 120.0,
  },
  "standard": {
    "target_return": 0.86,
    "target_mpjpe_mm": 48.0,
    "target_root_mm": 65.0,
    "target_joint_rad": 0.09,
    "max_action_rate_rms": 0.65,
    "confirm_streak": 2,
    "min_feasible_return": 0.42,
    "plateau_patience": 6,
    "plateau_delta": 0.004,
    "min_iters": 10,
    "coarse_eval_every": 15,
    "fine_eval_every": 5,
    "max_seconds": 180.0,
  },
  "tight": {
    "target_return": 0.91,
    "target_mpjpe_mm": 32.0,
    "target_root_mm": 45.0,
    "target_joint_rad": 0.06,
    "max_action_rate_rms": 0.50,
    "confirm_streak": 2,
    "min_feasible_return": 0.48,
    "plateau_patience": 8,
    "plateau_delta": 0.0025,
    "min_iters": 15,
    "coarse_eval_every": 15,
    "fine_eval_every": 5,
    "max_seconds": 240.0,
  },
}


@dataclass
class EvalMetrics:
  """Full-clip deterministic rollout quality snapshot."""

  iteration: int
  wall_s: float
  candidate: str
  alive_frac: float
  failed: bool
  fall_s: float | None
  fall_cause: str | None
  full_norm_return: float
  return_per_alive_step: float
  mpjpe_mm: float
  max_body_err_mm: float
  root_pos_err_mm: float
  joint_mae_rad: float
  action_rate_rms: float
  min_bin_return: float | None = None
  eval_ms: float = 0.0

  def rank_key(self) -> tuple[int, float, float, float]:
    """Lexicographic preference: 100% survival > higher alive_frac > return > lower MPJPE."""
    survived = 1 if (not self.failed and self.alive_frac >= 0.9999) else 0
    return (survived, round(self.alive_frac, 4), self.full_norm_return, -self.mpjpe_mm)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass
class ConvergenceMonitor:
  """Tracks training progress and triggers instant early stop upon trajectory sufficiency."""

  cfg: ConvergenceConfig = field(default_factory=ConvergenceConfig)
  armed: bool = False
  last_eval_iter: int = -10_000
  confirm_count: int = 0
  stale_feasible_evals: int = 0
  best_feasible_return: float = -1e9
  best_metrics: EvalMetrics | None = None
  stop_reason: str | None = None
  history: list[dict[str, Any]] = field(default_factory=list)

  def should_eval(
    self,
    iteration: int,
    wall_s: float,
    *,
    train_term_rate: float | None = None,
    force: bool = False,
  ) -> bool:
    if force:
      return True
    if wall_s >= self.cfg.max_seconds or iteration >= self.cfg.max_iters:
      return True
    if iteration < self.cfg.min_iters:
      return False
    best_alive = self.best_metrics.alive_frac if self.best_metrics is not None else 0.0
    if (
      train_term_rate is not None
      and train_term_rate <= self.cfg.arm_term_rate
      and best_alive >= 0.50
    ):
      self.armed = True
    cadence = self.cfg.fine_eval_every if self.armed else self.cfg.coarse_eval_every
    return (iteration - self.last_eval_iter) >= cadence

  def meets_target(self, m: EvalMetrics) -> bool:
    return (
      not m.failed
      and m.alive_frac >= 0.9999
      and m.full_norm_return >= self.cfg.target_return
      and m.mpjpe_mm <= self.cfg.target_mpjpe_mm
      and m.root_pos_err_mm <= self.cfg.target_root_mm
      and m.joint_mae_rad <= self.cfg.target_joint_rad
      and m.action_rate_rms <= self.cfg.max_action_rate_rms
    )

  def update(self, metrics: EvalMetrics) -> tuple[bool, str, bool]:
    """Ingest one deterministic full-clip eval. Returns (should_stop, reason, is_new_best)."""
    self.last_eval_iter = metrics.iteration
    if metrics.alive_frac >= self.cfg.arm_alive_frac:
      self.armed = True

    is_new_best = (
      self.best_metrics is None or metrics.rank_key() > self.best_metrics.rank_key()
    )
    if is_new_best:
      self.best_metrics = metrics

    target_ok = self.meets_target(metrics)
    if target_ok:
      self.confirm_count += 1
    else:
      self.confirm_count = 0

    feasible_survived = (
      not metrics.failed
      and metrics.alive_frac >= 0.9999
      and metrics.full_norm_return >= self.cfg.min_feasible_return
    )
    if feasible_survived:
      if metrics.full_norm_return > self.best_feasible_return + self.cfg.plateau_delta:
        self.best_feasible_return = metrics.full_norm_return
        self.stale_feasible_evals = 0
      else:
        self.stale_feasible_evals += 1
    else:
      self.stale_feasible_evals = 0

    reason = "CONTINUE"
    should_stop = False
    if self.confirm_count >= self.cfg.confirm_streak:
      should_stop = True
      reason = "TARGET_MET"
    elif (
      feasible_survived
      and self.stale_feasible_evals >= self.cfg.plateau_patience
      and self.best_metrics is not None
      and not self.best_metrics.failed
    ):
      should_stop = True
      reason = "PLATEAU_FEASIBLE"
    elif metrics.wall_s >= self.cfg.max_seconds:
      should_stop = True
      reason = "MAX_SECONDS"
    elif metrics.iteration >= self.cfg.max_iters:
      should_stop = True
      reason = "MAX_ITERS"

    entry = metrics.to_dict()
    entry["armed"] = self.armed
    entry["is_new_best"] = is_new_best
    entry["target_ok"] = target_ok
    entry["confirm_count"] = self.confirm_count
    entry["stale_feasible_evals"] = self.stale_feasible_evals
    entry["decision"] = reason
    self.history.append(entry)

    if should_stop:
      self.stop_reason = reason
    return should_stop, reason, is_new_best
