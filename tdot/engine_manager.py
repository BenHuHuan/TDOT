"""Thread-safe TDOT EngineManager orchestrating Preloaded NablaDaemon + MuJoCo 4.0 Verifier.

Responsibilities:
- Starts `PreloadedMuJoCoVerifier` immediately at server launch (~40ms).
- Launches persistent `NablaDaemon` subprocess in background so GPU bucket 512 is hot
  in RAM before the user clicks "Optimize".
- Streams live progress snapshots to browser clients via SSE (`subscribe(job_id)`).
- Supports instant mid-run cancellation (`cancel(job_id)`) returning best trajectory so far.
- Caches verified trajectories (`qpos`, `qvel`, `ctrl`, `torques`) for instant 0ms export!
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tdot.io_bridge import (
  G1_JOINT_NAMES,
  qpos_to_editor_payload,
  reorder_joints_to_g1,
  sanitize_qpos_array,
)
from tdot.verifier_pool import PreloadedMuJoCoVerifier


def find_nabla_python(explicit: str | None = None) -> Path | None:
  cands: list[Path] = []
  if explicit:
    cands.append(Path(explicit))
  env_py = os.environ.get("NABLA_PYTHON")
  if env_py:
    cands.append(Path(env_py))
  cands.extend([
    Path.home() / "NablaMimic" / ".venv" / "bin" / "python",
    Path(__file__).resolve().parents[1] / ".venv-nabla" / "bin" / "python",
    Path(__file__).resolve().parents[1] / "third_party" / "NablaMimic" / ".venv" / "bin" / "python",
  ])
  for c in cands:
    if c.exists():
      return c
  return None


@dataclass
class TOJob:
  job_id: str
  status: str = "queued"  # queued | running | done | error
  latest_progress: dict[str, Any] | None = None
  result: dict[str, Any] | None = None
  error: str | None = None
  subscribers: list[queue.Queue[dict[str, Any] | None]] = field(default_factory=list)
  optimized_arrays: dict[str, np.ndarray] | None = None


class TDOTEngineManager:
  """Singleton coordinator for preloaded NablaEngine daemon & MuJoCo 4.0 verifier."""

  def __init__(
    self,
    *,
    nabla_python: str | None = None,
    preload_buckets: list[int] | None = None,
    arch: str = "lite",
    matmul_precision: str = "highest",
    auto_preload: bool = True,
  ) -> None:
    self.verifier = PreloadedMuJoCoVerifier(solver_iterations=10)
    self.nabla_python = find_nabla_python(nabla_python)
    self.preload_buckets = preload_buckets if preload_buckets is not None else [512]
    self.arch = arch
    self.matmul_precision = matmul_precision

    self._proc: subprocess.Popen[str] | None = None
    self._proc_lock = threading.Lock()
    self._jobs_lock = threading.Lock()
    self._jobs: dict[str, TOJob] = {}
    self._active_job_id: str | None = None
    self._work_root = Path(tempfile.mkdtemp(prefix="tdot_jobs_"))

    self.preload_status: dict[str, Any] = {
      "state": "idle",
      "mujoco_version": self.verifier.mj_model and __import__("mujoco").__version__,
      "mujoco_boot_ms": self.verifier.boot_ms,
      "nabla_available": self.nabla_python is not None,
      "preloaded_buckets": [],
      "gpu": None,
      "detail": "MuJoCo 4.0 preloaded; NablaEngine idle",
    }
    if auto_preload and self.nabla_python is not None:
      self.start_daemon()

  def start_daemon(self) -> None:
    with self._proc_lock:
      if self._proc is not None and self._proc.poll() is None:
        return
      if self.nabla_python is None:
        self.preload_status.update(
          state="unavailable",
          detail="NablaMimic/.venv not found (set $NABLA_PYTHON or run scripts/setup_tdot.sh)",
        )
        return
      daemon_script = Path(__file__).resolve().parent / "nabla_daemon.py"
      cmd = [
        str(self.nabla_python),
        "-u",
        str(daemon_script),
        "--arch",
        self.arch,
        "--matmul-precision",
        self.matmul_precision,
        "--preload-buckets",
        *[str(b) for b in self.preload_buckets],
      ]
      env = os.environ.copy()
      env.pop("PYTHONPATH", None)
      env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
      if "/usr/local/cuda/bin" not in env.get("PATH", "") and Path("/usr/local/cuda/bin/nvcc").exists():
        env["PATH"] = f"/usr/local/cuda/bin:{env.get('PATH', '')}"

      self.preload_status.update(state="starting", detail="Launching NablaEngine daemon...")
      self._proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
      )
      threading.Thread(target=self._stdout_loop, daemon=True).start()
      threading.Thread(target=self._stderr_loop, daemon=True).start()

  def _stderr_loop(self) -> None:
    proc = self._proc
    if proc is None or proc.stderr is None:
      return
    for line in proc.stderr:
      # Forward daemon diagnostics cleanly to server stderr without breaking protocol.
      if "[Nabla" in line or "Error" in line or "Traceback" in line:
        print(f"[nabla-daemon] {line}", end="", flush=True)

  def _stdout_loop(self) -> None:
    proc = self._proc
    if proc is None or proc.stdout is None:
      return
    for raw in proc.stdout:
      line = raw.strip()
      if not line:
        continue
      try:
        msg = json.loads(line)
      except Exception:
        continue
      mtype = msg.get("type")
      if mtype == "PRELOAD_STATE":
        self.preload_status.update({k: v for k, v in msg.items() if k != "type"})
      elif mtype == "PROGRESS":
        self._publish_progress(str(msg["job_id"]), msg)
      elif mtype == "RESULT":
        threading.Thread(target=self._finalize_job, args=(msg,), daemon=True).start()
      elif mtype == "ERROR":
        self._fail_job(str(msg.get("job_id", "")), str(msg.get("error", "Unknown error")))

    # Daemon exited unexpectedly -> fail any running job cleanly.
    with self._jobs_lock:
      if self._active_job_id and self._active_job_id in self._jobs:
        j = self._jobs[self._active_job_id]
        if j.status == "running":
          self._fail_job_unlocked(j, "NablaEngine daemon exited unexpectedly")
      self._active_job_id = None
    self.preload_status.update(state="stopped", detail="NablaEngine daemon stopped")

  def _publish_progress(self, job_id: str, payload: dict[str, Any]) -> None:
    with self._jobs_lock:
      job = self._jobs.get(job_id)
      if job is None:
        return
      job.latest_progress = payload
      for q in job.subscribers:
        q.put(payload)

  def _fail_job_unlocked(self, job: TOJob, err: str) -> None:
    job.status = "error"
    job.error = err
    ev = {"type": "ERROR", "job_id": job.job_id, "error": err}
    for q in job.subscribers:
      q.put(ev)
      q.put(None)

  def _fail_job(self, job_id: str, err: str) -> None:
    with self._jobs_lock:
      job = self._jobs.get(job_id)
      if job is not None:
        self._fail_job_unlocked(job, err)
      if self._active_job_id == job_id:
        self._active_job_id = None

  def _finalize_job(self, msg: dict[str, Any]) -> None:
    """Run preloaded MuJoCo 4.0 closed-loop verification (~0.35s) & deliver final motion."""
    job_id = str(msg["job_id"])
    try:
      self._publish_progress(
        job_id,
        {
          "type": "PROGRESS",
          "job_id": job_id,
          "percent": 95.0,
          "phase": "VERIFY",
          "phase_label": "MuJoCo 4.0 Closed-Loop Verification",
          "eta_s": 0.4,
          "wall_s": msg["timing"]["optimization_s"],
          "iteration": (msg.get("best_metrics") or {}).get("iteration", 0),
          "alive_pct": round(((msg.get("best_metrics") or {}).get("alive_frac", 0.0)) * 100.0, 1),
          "full_norm_return": round((msg.get("best_metrics") or {}).get("full_norm_return", 0.0), 4),
          "mpjpe_mm": round((msg.get("best_metrics") or {}).get("mpjpe_mm", 0.0), 1),
          "root_mm": round((msg.get("best_metrics") or {}).get("root_pos_err_mm", 0.0), 1),
          "joint_mae_rad": round((msg.get("best_metrics") or {}).get("joint_mae_rad", 0.0), 3),
          "detail": "Running preloaded MuJoCo 4.0 IEEE FP32 rollout...",
        },
      )

      z = np.load(msg["nabla_result_npz"])
      ver = self.verifier.verify_closed_loop(
        ref_qpos=z["ref_qpos"],
        ref_qvel=z["ref_qvel"],
        actor_params_npz=msg["actor_params_npz"],
        nabla_qpos=z["qpos"],
        nabla_qvel=z["qvel"],
        nabla_ctrl=z["ctrl"],
      )
      opt_qpos = ver["qpos"]
      opt_qvel = ver["qvel"]
      opt_ctrl = ver["ctrl"]
      opt_tau = ver["torques"]

      editor_motion = qpos_to_editor_payload(
        opt_qpos,
        fps=50.0,
        name=f"tdot_optimized_{job_id[:6]}",
        source_format="tdot_mujoco4",
      )
      editor_motion["audit"] = ver["optimized_audit"]

      total_s = round(float(msg["timing"]["daemon_total_s"]) + float(ver["verify_s"]), 2)
      final_result = {
        "type": "DONE",
        "job_id": job_id,
        "percent": 100.0,
        "phase": "DONE",
        "phase_label": f"Ready ({msg['stop_reason']})",
        "stop_reason": msg["stop_reason"],
        "used_warm_start": bool(msg.get("used_warm_start", False)),
        "bucket": msg.get("bucket"),
        "gpu": msg.get("gpu"),
        "timing": {
          **msg["timing"],
          "verify_s": ver["verify_s"],
          "end_to_end_s": total_s,
        },
        "nabla_best": msg.get("best_metrics"),
        "mujoco_verify": {
          k: v for k, v in ver.items() if k not in ("qpos", "qvel", "ctrl", "torques", "mpjpe_per_frame_mm")
        },
        "mpjpe_per_frame_mm": [round(float(x), 1) for x in ver["mpjpe_per_frame_mm"].tolist()],
        "eval_history": msg.get("eval_history", []),
        "motion": editor_motion,
      }

      with self._jobs_lock:
        job = self._jobs.get(job_id)
        if job is not None:
          job.status = "done"
          job.result = final_result
          job.optimized_arrays = {
            "qpos": opt_qpos,
            "qvel": opt_qvel,
            "ctrl": opt_ctrl,
            **({"torques": opt_tau} if opt_tau is not None else {}),
          }
          for q in job.subscribers:
            q.put(final_result)
            q.put(None)
        if self._active_job_id == job_id:
          self._active_job_id = None
    except Exception as exc:
      self._fail_job(job_id, f"Verification failed: {exc}")

  def submit_editor_motion(
    self,
    motion_json: dict[str, Any],
    *,
    preset: str = "standard",
    arch: str | None = None,
    warm_start: bool = False,
    smooth_window: int = 0,
    floor_align: bool = True,
    clamp_joints: bool = True,
    max_seconds: float | None = None,
  ) -> tuple[str, dict[str, Any]]:
    """Sanitize current editor trajectory in-memory and submit to preloaded NablaDaemon."""
    if self.nabla_python is None:
      raise RuntimeError(
        "NablaEngine venv not found. Run scripts/setup_tdot.sh or set $NABLA_PYTHON."
      )
    self.start_daemon()

    with self._jobs_lock:
      if self._active_job_id is not None:
        raise RuntimeError(f"Another TO job ({self._active_job_id}) is already running")
      job_id = uuid.uuid4().hex[:10]
      job = TOJob(job_id=job_id, status="running")
      self._jobs[job_id] = job
      self._active_job_id = job_id

    try:
      rpos = np.asarray(motion_json["base_pos_w"], dtype=np.float64)
      rquat = np.asarray(motion_json["base_quat_w"], dtype=np.float64)
      jpos = reorder_joints_to_g1(
        np.asarray(motion_json["joint_pos"], dtype=np.float64),
        motion_json.get("joint_names", list(G1_JOINT_NAMES)),
      )
      raw_qpos = np.concatenate([rpos, rquat, jpos], axis=1)
      fps_in = float(motion_json.get("framerate", 50.0))

      san = sanitize_qpos_array(
        raw_qpos,
        fps_in,
        self.verifier.mj_model,
        name=str(motion_json.get("name", "edited_motion")),
        source_format="editor_memory",
        target_fps=50.0,
        floor_align=floor_align,
        clamp_joints=clamp_joints,
        smooth_window=int(smooth_window),
      )

      job_dir = self._work_root / job_id
      job_dir.mkdir(parents=True, exist_ok=True)
      in_npz = job_dir / "sanitized_input.npz"
      np.savez(in_npz, qpos=san.qpos, qvel=san.qvel)

      cmd = {
        "cmd": "OPTIMIZE",
        "job_id": job_id,
        "input_npz": str(in_npz),
        "out_dir": str(job_dir),
        "clip_name": san.name,
        "preset": preset,
        "arch": arch or self.arch,
        "warm_start": bool(warm_start),
        "max_seconds": max_seconds,
      }
      with self._proc_lock:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(cmd) + "\n")
        self._proc.stdin.flush()

      return job_id, {
        "raw_audit": san.raw_audit.to_dict(),
        "sanitized_audit": san.sanitized_audit.to_dict(),
        "n_frames": int(len(san.qpos)),
      }
    except Exception:
      with self._jobs_lock:
        self._jobs.pop(job_id, None)
        if self._active_job_id == job_id:
          self._active_job_id = None
      raise

  def cancel_job(self, job_id: str) -> bool:
    with self._proc_lock:
      if self._proc is None or self._proc.stdin is None or self._proc.poll() is not None:
        return False
      self._proc.stdin.write(json.dumps({"cmd": "CANCEL", "job_id": job_id}) + "\n")
      self._proc.stdin.flush()
      return True

  def subscribe(self, job_id: str) -> queue.Queue[dict[str, Any] | None] | None:
    with self._jobs_lock:
      job = self._jobs.get(job_id)
      if job is None:
        return None
      q: queue.Queue[dict[str, Any] | None] = queue.Queue()
      if job.status == "done" and job.result is not None:
        q.put(job.result)
        q.put(None)
      elif job.status == "error":
        q.put({"type": "ERROR", "job_id": job_id, "error": job.error})
        q.put(None)
      else:
        if job.latest_progress is not None:
          q.put(job.latest_progress)
        job.subscribers.append(q)
      return q

  def get_job(self, job_id: str) -> TOJob | None:
    with self._jobs_lock:
      return self._jobs.get(job_id)

  def shutdown(self) -> None:
    with self._proc_lock:
      if self._proc is not None and self._proc.poll() is None:
        try:
          if self._proc.stdin is not None:
            self._proc.stdin.write(json.dumps({"cmd": "SHUTDOWN"}) + "\n")
            self._proc.stdin.flush()
          self._proc.wait(timeout=3.0)
        except Exception:
          self._proc.kill()
        self._proc = None
