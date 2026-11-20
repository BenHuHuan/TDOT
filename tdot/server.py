"""TDOT Flask web server integrating 3D Motion Editor + Preloaded NablaEngine + MuJoCo 4.0."""

from __future__ import annotations

import atexit
import io
import json
import queue
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file, stream_with_context

from tdot.engine_manager import TDOTEngineManager
from tdot.io_bridge import (
  G1_JOINT_NAMES,
  audit_trajectory_kinematics,
  compute_qvel,
  export_lafan1_csv_bytes,
  export_unified_npz,
  parse_motion_bytes,
  qpos_to_editor_payload,
  reorder_joints_to_g1,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "static" / "samples"

SAMPLE_DESCRIPTIONS: dict[str, str] = {
  "weird_dance_6s.npz": "Corrupted 6s Dance (+18cm float, -12cm floor dip, +2.2rad limit violation, jitter)",
  "dance_6s_clean.npz": "Clean 6s LAFAN1 Dance (50Hz Unitree G1)",
}


def create_app(
  engine: TDOTEngineManager | None = None,
  *,
  auto_preload: bool = True,
  preload_buckets: list[int] | None = None,
  nabla_python: str | None = None,
  arch: str = "lite",
  matmul_precision: str = "highest",
) -> Flask:
  app = Flask(
    __name__,
    template_folder=str(REPO_ROOT / "templates"),
    static_folder=str(REPO_ROOT / "static"),
  )
  app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 256 MB

  owns_engine = engine is None
  eng = engine or TDOTEngineManager(
    nabla_python=nabla_python,
    preload_buckets=preload_buckets if preload_buckets is not None else [512],
    arch=arch,
    matmul_precision=matmul_precision,
    auto_preload=auto_preload,
  )
  app.config["TDOT_ENGINE"] = eng
  if owns_engine:
    atexit.register(eng.shutdown)

  @app.get("/")
  def index() -> str:
    return render_template("index.html")

  @app.get("/api/preload_status")
  def preload_status() -> Response:
    return jsonify(eng.preload_status)

  @app.post("/api/preload_start")
  def preload_start() -> Response:
    eng.start_daemon()
    return jsonify(eng.preload_status)

  @app.get("/api/samples")
  def list_samples() -> Response:
    items: list[dict[str, Any]] = []
    if SAMPLES_DIR.exists():
      for p in sorted(SAMPLES_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in (".npz", ".csv", ".npy"):
          items.append({
            "filename": p.name,
            "label": SAMPLE_DESCRIPTIONS.get(p.name, p.stem),
            "size_kb": round(p.stat().st_size / 1024.0, 1),
          })
    return jsonify({"samples": items})

  @app.get("/api/load_sample/<path:filename>")
  def load_sample(filename: str) -> tuple[Response, int] | Response:
    safe_name = Path(filename).name
    target = (SAMPLES_DIR / safe_name).resolve()
    if not target.exists() or target.parent != SAMPLES_DIR.resolve():
      return jsonify({"error": f"Sample not found: {safe_name}"}), 404
    try:
      with open(target, "rb") as f:
        raw_qpos, fps, fmt = parse_motion_bytes(safe_name, f)
      audit = audit_trajectory_kinematics(eng.verifier.mj_model, raw_qpos, fps)
      payload = qpos_to_editor_payload(
        raw_qpos,
        fps=fps,
        name=target.stem,
        source_format=fmt,
        audit=audit,
      )
      return jsonify(payload)
    except Exception as exc:
      return jsonify({"error": str(exc)}), 400

  @app.post("/upload_motion")
  def upload_motion() -> tuple[Response, int] | Response:
    if "file" not in request.files:
      return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename:
      return jsonify({"error": "Empty filename"}), 400
    try:
      raw_bytes = f.read()
      raw_qpos, fps, fmt = parse_motion_bytes(f.filename, io.BytesIO(raw_bytes))
      audit = audit_trajectory_kinematics(eng.verifier.mj_model, raw_qpos, fps)
      payload = qpos_to_editor_payload(
        raw_qpos,
        fps=fps,
        name=Path(f.filename).stem,
        source_format=fmt,
        audit=audit,
      )
      return jsonify(payload)
    except Exception as exc:
      return jsonify({"error": str(exc)}), 400

  @app.post("/api/to/start")
  def start_to() -> tuple[Response, int] | Response:
    body = request.get_json(silent=True) or {}
    motion = body.get("motion")
    if not isinstance(motion, dict) or "joint_pos" not in motion:
      return jsonify({"error": "Missing motion payload"}), 400
    try:
      job_id, meta = eng.submit_editor_motion(
        motion,
        preset=str(body.get("preset", "standard")),
        arch=body.get("arch"),
        warm_start=bool(body.get("warm_start", False)),
        smooth_window=int(body.get("smooth_window", 0)),
        floor_align=bool(body.get("floor_align", True)),
        clamp_joints=bool(body.get("clamp_joints", True)),
        max_seconds=float(body["max_seconds"]) if body.get("max_seconds") else None,
      )
      return jsonify({"job_id": job_id, **meta})
    except Exception as exc:
      return jsonify({"error": str(exc)}), 400

  @app.get("/api/to/stream/<job_id>")
  def stream_to(job_id: str) -> tuple[Response, int] | Response:
    q = eng.subscribe(job_id)
    if q is None:
      return jsonify({"error": f"Unknown job_id: {job_id}"}), 404

    def _gen():
      while True:
        try:
          ev = q.get(timeout=5.0)
        except queue.Empty:
          yield ": keepalive\n\n"
          continue
        if ev is None:
          break
        yield f"data: {json.dumps(ev)}\n\n"

    return Response(
      stream_with_context(_gen()),
      mimetype="text/event-stream",
      headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
      },
    )

  @app.post("/api/to/cancel/<job_id>")
  def cancel_to(job_id: str) -> Response:
    ok = eng.cancel_job(job_id)
    return jsonify({"ok": ok, "job_id": job_id})

  @app.get("/api/to/download/<job_id>")
  def download_to(job_id: str) -> tuple[Response, int] | Response:
    job = eng.get_job(job_id)
    if job is None or job.optimized_arrays is None:
      return jsonify({"error": "Optimized job not ready or not found"}), 404

    fmt = request.args.get("fmt", "npz").lower()
    arrs = job.optimized_arrays
    qpos = arrs["qpos"]
    qvel = arrs["qvel"]
    ctrl = arrs.get("ctrl")
    torques = arrs.get("torques")

    if fmt == "csv":
      csv_bytes = export_lafan1_csv_bytes(qpos, fps_in=50.0, fps_out=30.0)
      return send_file(
        io.BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"tdot_{job_id}_lafan1_30hz.csv",
      )

    compressed = fmt == "npz_compressed"
    buf = io.BytesIO()
    export_unified_npz(
      buf,
      qpos,
      qvel,
      eng.verifier.mj_model,
      fps=50.0,
      ctrl=ctrl,
      torques=torques,
      compressed=compressed,
    )
    return send_file(
      buf,
      mimetype="application/octet-stream",
      as_attachment=True,
      download_name=f"tdot_{job_id}_mlab4_editor.npz",
    )

  @app.post("/save_motion")
  def save_motion() -> tuple[Response, int] | Response:
    """Export current editor trajectory as unified MLAB4_Fast + Editor .npz or LAFAN1 .csv."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "joint_pos" not in data:
      return jsonify({"error": "Missing motion payload"}), 400
    try:
      rpos = np.asarray(data["base_pos_w"], dtype=np.float64)
      rquat = np.asarray(data["base_quat_w"], dtype=np.float64)
      jpos = reorder_joints_to_g1(
        np.asarray(data["joint_pos"], dtype=np.float64),
        data.get("joint_names", list(G1_JOINT_NAMES)),
      )
      qpos = np.concatenate([rpos, rquat, jpos], axis=1).astype(np.float32)
      fps = float(data.get("framerate", 50.0))
      fmt = str(data.get("export_format", "npz")).lower()
      name = str(data.get("name", "tdot_motion")).strip() or "tdot_motion"

      if fmt == "csv":
        csv_bytes = export_lafan1_csv_bytes(qpos, fps_in=fps, fps_out=30.0)
        return send_file(
          io.BytesIO(csv_bytes),
          mimetype="text/csv",
          as_attachment=True,
          download_name=f"{name}_30hz.csv",
        )

      qvel = compute_qvel(eng.verifier.mj_model, qpos, fps=fps)
      buf = io.BytesIO()
      export_unified_npz(
        buf,
        qpos,
        qvel,
        eng.verifier.mj_model,
        fps=fps,
        compressed=bool(data.get("compressed", False)),
      )
      return send_file(
        buf,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"{name}_mlab4_editor.npz",
      )
    except Exception as exc:
      return jsonify({"error": str(exc)}), 400

  return app
