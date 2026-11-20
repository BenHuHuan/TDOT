"""Unit and integration tests for TDOT."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from tdot.convergence import ConvergenceConfig, ConvergenceMonitor, EvalMetrics
from tdot.engine_manager import TDOTEngineManager
from tdot.io_bridge import (
  G1_JOINT_NAMES,
  compute_qvel,
  export_lafan1_csv_bytes,
  export_unified_npz,
  parse_motion_bytes,
  qpos_to_editor_payload,
  sanitize_qpos_array,
)
from tdot.progress import ProgressEstimator
from tdot.server import create_app
from tdot.verifier_pool import PreloadedMuJoCoVerifier

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "static" / "samples"


def _make_metrics(
  iteration: int,
  wall_s: float,
  alive_frac: float,
  ret: float,
  mpjpe_mm: float,
  *,
  root_mm: float = 30.0,
  joint_rad: float = 0.06,
  action_rms: float = 0.40,
) -> EvalMetrics:
  return EvalMetrics(
    iteration=iteration,
    wall_s=wall_s,
    candidate="ema",
    alive_frac=alive_frac,
    failed=alive_frac < 0.999,
    fall_s=None if alive_frac >= 0.999 else wall_s * alive_frac,
    fall_cause=None if alive_frac >= 0.999 else "root_height",
    full_norm_return=ret,
    return_per_alive_step=ret / max(alive_frac, 1e-3),
    mpjpe_mm=mpjpe_mm,
    max_body_err_mm=mpjpe_mm * 1.8,
    root_pos_err_mm=root_mm,
    joint_mae_rad=joint_rad,
    action_rate_rms=action_rms,
  )


def test_progress_estimator_monotonic_balance_to_refine() -> None:
  cfg = ConvergenceConfig.from_preset("standard", max_seconds=45.0)
  est = ProgressEstimator(cfg)
  p0 = est.stage_event("SANITIZE", 4.0, 0.05, "Sanitized")
  p1 = est.stage_event("COMPILE", 12.0, 0.06, "Preloaded bucket hit")
  assert p1.percent > p0.percent
  assert p1.percent == pytest.approx(12.0)

  prev = p1.percent
  # Balance climb via train beats + evals
  for i, alive in enumerate([0.15, 0.45, 0.80, 0.95], start=1):
    beat = est.on_train_beat(
      iteration=i * 3,
      wall_s=i * 1.0,
      train_term_rate=max(0.02, 0.35 - 0.08 * i),
      train_rew_mean=0.2 + 0.12 * i,
      min_bin_q=0.2 + 0.12 * i,
    )
    assert beat.percent >= prev
    prev = beat.percent
    ev = est.on_eval(
      _make_metrics(i * 3 + 1, i * 1.2, alive, 0.2 + 0.15 * i, 130.0 - 15.0 * i),
      confirm_count=0,
      stale_feasible_evals=0,
      decision="CONTINUE",
    )
    assert ev.phase == "BALANCE"
    assert ev.percent >= prev
    prev = ev.percent

  # Refine climb + plateau
  for i, (ret, mpj, no_imp, conf) in enumerate(
    [(0.72, 62.0, 0, 0), (0.81, 48.0, 0, 0), (0.86, 42.0, 2, 1), (0.89, 36.0, 0, 2)], start=1
  ):
    ev = est.on_eval(
      _make_metrics(15 + i * 5, 6.0 + i * 2.0, 1.0, ret, mpj),
      confirm_count=conf,
      stale_feasible_evals=no_imp,
      decision="TARGET_MET" if conf >= 2 else "CONTINUE",
    )
    assert ev.percent >= prev
    if conf < 2:
      assert ev.phase == "REFINE"
      assert ev.percent <= 93.0
    else:
      assert ev.phase == "VERIFY"
    prev = ev.percent

  p_done = est.stage_event("DONE", 100.0, 15.4, "Complete")
  assert p_done.percent == 100.0


def test_convergence_monitor_arming_and_target_met() -> None:
  cfg = ConvergenceConfig.from_preset("standard", confirm_streak=2, min_iters=6, coarse_eval_every=10, fine_eval_every=3)
  mon = ConvergenceMonitor(cfg=cfg)
  # Early noisy state should NOT arm fine eval yet
  assert not mon.should_eval(iteration=3, wall_s=1.0, train_term_rate=0.05)
  assert mon.should_eval(iteration=10, wall_s=3.0, train_term_rate=0.25)

  # Confirm streak 1 -> CONTINUE; streak 2 -> TARGET_MET
  stop1, reason1, best1 = mon.update(_make_metrics(10, 4.0, 1.0, 0.90, 34.0))
  assert best1 is True
  assert stop1 is False
  assert reason1 == "CONTINUE"
  assert mon.armed is True

  stop2, reason2, _ = mon.update(_make_metrics(13, 5.2, 1.0, 0.91, 32.0))
  assert stop2 is True
  assert reason2 == "TARGET_MET"


def test_verifier_boot_and_unified_npz_roundtrip() -> None:
  verifier = PreloadedMuJoCoVerifier(solver_iterations=10)
  assert verifier.boot_ms < 2000.0  # typically ~50-200ms

  sample_path = SAMPLES_DIR / "weird_dance_6s.npz"
  assert sample_path.exists()
  with open(sample_path, "rb") as f:
    qpos, fps, fmt = parse_motion_bytes(sample_path.name, f)
  assert qpos.shape == (301, 36)
  assert fps == pytest.approx(50.0)

  # Sanitize
  san = sanitize_qpos_array(
    qpos, fps, verifier.mj_model, name="weird", floor_align=True, clamp_joints=True, smooth_window=7
  )
  assert san.qpos.shape == (301, 36)
  assert san.qvel.shape == (301, 35)
  assert san.sanitized_audit.joint_limit_violation_max_rad <= 1e-4

  # Unified export (uncompressed) contains BOTH MLAB4_Fast and robot-motion-editor keys
  buf = io.BytesIO()
  export_unified_npz(buf, san.qpos, san.qvel, verifier.mj_model, fps=50.0, compressed=False)
  buf.seek(0)
  z = np.load(buf)
  for k in (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "base_pos_w",
    "base_quat_w",
    "joint_names",
    "framerate",
    "qpos",
    "qvel",
  ):
    assert k in z.files, f"Missing key {k}"
  assert z["joint_pos"].shape == (301, 29)
  assert z["body_pos_w"].shape == (301, 30, 3)

  # LAFAN1 CSV round-trip
  csv_bytes = export_lafan1_csv_bytes(san.qpos, fps_in=50.0, fps_out=30.0)
  qpos_csv, fps_csv, fmt_csv = parse_motion_bytes("clip.csv", io.BytesIO(csv_bytes))
  assert fmt_csv == "lafan1_csv"
  assert fps_csv == pytest.approx(30.0)
  assert qpos_csv.shape[1] == 36
  assert len(qpos_csv) > 150


def test_flask_routes_without_daemon() -> None:
  eng = TDOTEngineManager(auto_preload=False)
  app = create_app(engine=eng)
  client = app.test_client()

  r = client.get("/")
  assert r.status_code == 200
  assert b"TDOT 3D Editor" in r.data

  r = client.get("/api/preload_status")
  assert r.status_code == 200
  st = r.get_json()
  assert "mujoco_boot_ms" in st

  r = client.get("/api/samples")
  assert r.status_code == 200
  samples = r.get_json()["samples"]
  names = {s["filename"] for s in samples}
  assert "weird_dance_6s.npz" in names
  assert "dance_6s_clean.npz" in names

  r = client.get("/api/load_sample/weird_dance_6s.npz")
  assert r.status_code == 200
  motion = r.get_json()
  assert motion["num_frames"] == 301
  assert len(motion["joint_names"]) == len(G1_JOINT_NAMES)
  assert "audit" in motion

  # Round-trip /save_motion -> unified .npz
  r = client.post("/save_motion", json=motion)
  assert r.status_code == 200
  z = np.load(io.BytesIO(r.data))
  assert "body_pos_w" in z.files and "base_pos_w" in z.files

  # Round-trip /save_motion -> LAFAN1 .csv
  r = client.post("/save_motion", json={**motion, "export_format": "csv"})
  assert r.status_code == 200
  assert len(r.data) > 5000

  # Round-trip /upload_motion with generated unified .npz
  qvel = compute_qvel(eng.verifier.mj_model, np.asarray(z["qpos"]), fps=50.0)
  payload = qpos_to_editor_payload(z["qpos"], fps=50.0)
  assert payload["num_frames"] == 301
  assert qvel.shape == (301, 35)
