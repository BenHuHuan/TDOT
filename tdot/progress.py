"""Physics-driven monotonic progress bar & live ETA estimator for adaptive early-stop TO.

Because TDOT stops dynamically when the trajectory becomes physically sufficient
(`TARGET_MET` or `PLATEAU_FEASIBLE`) rather than running fixed epochs, a naive
`iter / max_iters` bar would jump from 6% straight to 100%.

Instead, `ProgressEstimator` maps actual physical convergence signals onto `[0, 100]`:
  - 0% ->  4%: Input sanitization & reference kinematics
  - 4% -> 12%: Bucket JIT check (instant 0ms jump to 12% when preloaded!)
  - 12% -> 56%: Whole-body balance discovery (`alive_frac` 0%->100% + `train_term_rate`)
  - 56% -> 93%: Tracking fidelity (`return`, `MPJPE`, `joint_mae`, `confirm_streak`)
                OR feasible plateau convergence (`stale_feasible_evals / plateau_patience`)
                so even physically impossible weird motions advance smoothly to 100%!
  - 93% -> 100%: Preloaded MuJoCo 4.0 closed-loop verification & unified export
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tdot.convergence import ConvergenceConfig, EvalMetrics


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
  return lo if x < lo else (hi if x > hi else x)


@dataclass
class ProgressSnapshot:
  percent: float
  phase: str
  phase_label: str
  eta_s: float | None
  wall_s: float
  iteration: int
  alive_pct: float
  full_norm_return: float
  mpjpe_mm: float
  root_mm: float
  joint_mae_rad: float
  confirm_count: int
  confirm_streak: int
  plateau_count: int
  plateau_patience: int
  detail: str = ""

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


class ProgressEstimator:
  """Strictly monotonic physics-based progress estimator with smoothed velocity ETA."""

  def __init__(self, cfg: ConvergenceConfig) -> None:
    self.cfg = cfg
    self.last_pct: float = 0.0
    self.last_wall_s: float = 0.0
    self.ema_pct_per_s: float = 3.5  # optimistic prior (~25s total)
    self.best_alive: float = 0.0
    self.last_metrics: EvalMetrics | None = None

  def _advance(self, raw_pct: float, wall_s: float) -> tuple[float, float | None]:
    pct = max(self.last_pct, min(100.0, raw_pct))
    dt = max(wall_s - self.last_wall_s, 1e-3)
    if pct > self.last_pct + 1e-4 and wall_s > 0.2:
      inst_rate = (pct - self.last_pct) / dt
      self.ema_pct_per_s = 0.72 * self.ema_pct_per_s + 0.28 * _clamp(inst_rate, 0.2, 25.0)
    self.last_pct = pct
    self.last_wall_s = max(self.last_wall_s, wall_s)
    if pct >= 99.9:
      return 100.0, 0.0
    rem_pct = max(0.0, 96.0 - pct)
    eta_by_vel = rem_pct / max(self.ema_pct_per_s, 0.35)
    eta_by_cap = max(0.0, self.cfg.max_seconds - wall_s)
    return round(pct, 1), round(min(eta_by_vel, eta_by_cap), 1)

  def stage_event(
    self,
    phase: str,
    raw_pct: float,
    wall_s: float,
    detail: str,
    *,
    iteration: int = 0,
  ) -> ProgressSnapshot:
    pct, eta = self._advance(raw_pct, wall_s)
    m = self.last_metrics
    labels = {
      "SANITIZE": "Sanitizing Input Trajectory",
      "COMPILE": "Warming GPU Bucket Cache",
      "BALANCE": "Learning Whole-Body Balance",
      "REFINE": "Refining Tracking & Dynamics",
      "VERIFY": "MuJoCo 4.0 Closed-Loop Verification",
      "DONE": "Dynamically Feasible Trajectory Ready",
    }
    return ProgressSnapshot(
      percent=pct,
      phase=phase,
      phase_label=labels.get(phase, phase),
      eta_s=eta,
      wall_s=round(wall_s, 2),
      iteration=iteration,
      alive_pct=round((m.alive_frac if m else self.best_alive) * 100.0, 1),
      full_norm_return=round(m.full_norm_return if m else 0.0, 4),
      mpjpe_mm=round(m.mpjpe_mm if m else 0.0, 1),
      root_mm=round(m.root_pos_err_mm if m else 0.0, 1),
      joint_mae_rad=round(m.joint_mae_rad if m else 0.0, 3),
      confirm_count=0,
      confirm_streak=self.cfg.confirm_streak,
      plateau_count=0,
      plateau_patience=self.cfg.plateau_patience,
      detail=detail,
    )

  def on_train_beat(
    self,
    *,
    iteration: int,
    wall_s: float,
    train_term_rate: float,
    train_rew_mean: float,
    min_bin_q: float,
  ) -> ProgressSnapshot:
    """Smooth progress tick between full-clip evaluations so the UI bar never stalls."""
    # Map training survival + worst-bin return smoothly without overshooting Phase 2 ceiling.
    term_score = _clamp(1.0 - train_term_rate / 0.25)
    bin_score = _clamp((min_bin_q - 0.15) / 0.70)
    if self.best_alive < 0.999:
      p_bal = 0.70 * self.best_alive + 0.20 * term_score + 0.10 * bin_score
      raw = 12.0 + 42.0 * _clamp(p_bal)
      phase, label = "BALANCE", "Learning Whole-Body Balance"
    else:
      # Gentle creep inside Phase 3 toward time cap while waiting for next full-clip eval.
      t_frac = _clamp(wall_s / max(self.cfg.max_seconds, 1.0))
      raw = max(self.last_pct + 0.25, 56.0 + 34.0 * t_frac)
      raw = min(raw, 91.5)
      phase, label = "REFINE", "Refining Tracking & Dynamics"

    pct, eta = self._advance(raw, wall_s)
    m = self.last_metrics
    return ProgressSnapshot(
      percent=pct,
      phase=phase,
      phase_label=label,
      eta_s=eta,
      wall_s=round(wall_s, 2),
      iteration=iteration,
      alive_pct=round((m.alive_frac if m else self.best_alive) * 100.0, 1),
      full_norm_return=round(m.full_norm_return if m else train_rew_mean, 4),
      mpjpe_mm=round(m.mpjpe_mm if m else 0.0, 1),
      root_mm=round(m.root_pos_err_mm if m else 0.0, 1),
      joint_mae_rad=round(m.joint_mae_rad if m else 0.0, 3),
      confirm_count=0,
      confirm_streak=self.cfg.confirm_streak,
      plateau_count=0,
      plateau_patience=self.cfg.plateau_patience,
      detail=f"iter {iteration} · term_rate {train_term_rate:.3f} · worst_bin {min_bin_q:.2f}",
    )

  def on_eval(
    self,
    metrics: EvalMetrics,
    *,
    confirm_count: int,
    stale_feasible_evals: int,
    decision: str,
  ) -> ProgressSnapshot:
    """Update progress from an authoritative deterministic full-clip evaluation."""
    self.last_metrics = metrics
    self.best_alive = max(self.best_alive, metrics.alive_frac)

    if decision in ("TARGET_MET", "PLATEAU_FEASIBLE", "MAX_SECONDS", "MAX_ITERS", "CANCELLED"):
      raw = 93.0
      phase = "VERIFY"
      label = f"Early-Stop ({decision}) -> Verifying in MuJoCo 4.0"
    elif metrics.alive_frac < 0.999:
      p_bal = 0.85 * self.best_alive + 0.15 * _clamp(metrics.full_norm_return / 0.80)
      raw = 12.0 + 44.0 * _clamp(p_bal)
      phase = "BALANCE"
      label = "Learning Whole-Body Balance"
    else:
      r_ret = _clamp(
        (metrics.full_norm_return - 0.50) / max(self.cfg.target_return - 0.50, 0.05)
      )
      r_mpjpe = _clamp(
        (160.0 - metrics.mpjpe_mm) / max(160.0 - self.cfg.target_mpjpe_mm, 10.0)
      )
      r_jmae = _clamp(
        (0.25 - metrics.joint_mae_rad) / max(0.25 - self.cfg.target_joint_rad, 0.02)
      )
      r_streak = _clamp(confirm_count / max(self.cfg.confirm_streak, 1))
      p_target = 0.82 * (0.45 * r_ret + 0.35 * r_mpjpe + 0.20 * r_jmae) + 0.18 * r_streak
      p_plateau = _clamp(stale_feasible_evals / max(self.cfg.plateau_patience, 1))
      p_time = _clamp(metrics.wall_s / max(self.cfg.max_seconds, 1.0), 0.0, 0.98)
      p_refine = max(p_target, p_plateau, 0.85 * p_time)
      raw = 56.0 + 37.0 * _clamp(p_refine)
      phase = "REFINE"
      label = (
        f"Confirming Target ({confirm_count}/{self.cfg.confirm_streak})"
        if confirm_count > 0
        else "Refining Tracking & Dynamics"
      )

    pct, eta = self._advance(raw, metrics.wall_s)
    return ProgressSnapshot(
      percent=pct,
      phase=phase,
      phase_label=label,
      eta_s=eta,
      wall_s=round(metrics.wall_s, 2),
      iteration=metrics.iteration,
      alive_pct=round(metrics.alive_frac * 100.0, 1),
      full_norm_return=round(metrics.full_norm_return, 4),
      mpjpe_mm=round(metrics.mpjpe_mm, 1),
      root_mm=round(metrics.root_pos_err_mm, 1),
      joint_mae_rad=round(metrics.joint_mae_rad, 3),
      confirm_count=confirm_count,
      confirm_streak=self.cfg.confirm_streak,
      plateau_count=stale_feasible_evals,
      plateau_patience=self.cfg.plateau_patience,
      detail=(
        f"iter {metrics.iteration} ({metrics.candidate}) · "
        f"alive {metrics.alive_frac * 100:.0f}% · "
        f"ret {metrics.full_norm_return:.3f} · mpjpe {metrics.mpjpe_mm:.1f}mm"
      ),
    )
