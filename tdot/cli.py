"""TDOT Command-Line Interface (`tdot serve` and `tdot optimize`)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tdot.engine_manager import TDOTEngineManager
from tdot.io_bridge import (
  export_lafan1_csv_bytes,
  export_unified_npz,
  load_and_sanitize_file,
  qpos_to_editor_payload,
)
from tdot.server import create_app


def _cmd_serve(args: argparse.Namespace) -> int:
  buckets = [int(x) for x in args.preload_buckets.split(",") if x.strip()] if args.preload_buckets else []
  app = create_app(
    auto_preload=not args.no_preload,
    preload_buckets=buckets,
    nabla_python=args.nabla_python,
    arch=args.arch,
    matmul_precision=args.matmul_precision,
  )
  print(
    f"[TDOT] Serving 3D Editor + Dynamic TO at http://{args.host}:{args.port} "
    f"(preload_buckets={buckets}, matmul={args.matmul_precision})",
    flush=True,
  )
  app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
  return 0


def _cmd_optimize(args: argparse.Namespace) -> int:
  t0 = time.perf_counter()
  eng = TDOTEngineManager(
    nabla_python=args.nabla_python,
    preload_buckets=[],  # CLI runs one clip directly without pre-compiling extra buckets
    arch=args.arch,
    matmul_precision=args.matmul_precision,
    auto_preload=True,
  )
  try:
    san = load_and_sanitize_file(
      args.input,
      eng.verifier.mj_model,
      input_fps=args.input_fps,
      target_fps=50.0,
      start_s=args.start_s,
      duration_s=args.duration_s,
      floor_align=not args.no_floor_align,
      clamp_joints=not args.no_clamp_joints,
      smooth_window=args.smooth_window,
    )
    print(
      f"[TDOT] Loaded {san.name} ({san.source_format}, {len(san.qpos)}f @ 50Hz) | "
      f"raw_foot_min={san.raw_audit.foot_min_z_mm:.1f}mm -> san={san.sanitized_audit.foot_min_z_mm:.1f}mm",
      flush=True,
    )
    editor_payload = qpos_to_editor_payload(san.qpos, fps=50.0, name=san.name)
    job_id, _ = eng.submit_editor_motion(
      editor_payload,
      preset=args.preset,
      arch=args.arch,
      warm_start=False,
      smooth_window=0,
      floor_align=False,
      clamp_joints=False,
      max_seconds=args.max_seconds,
    )
    q = eng.subscribe(job_id)
    assert q is not None
    final_msg = None
    while True:
      ev = q.get()
      if ev is None:
        break
      etype = ev.get("type")
      if etype == "PROGRESS":
        pct = ev.get("percent", 0.0)
        phase = ev.get("phase", "")
        eta = ev.get("eta_s")
        eta_str = f"ETA {eta:.1f}s" if eta is not None else "ETA --"
        print(
          f"  [{pct:5.1f}%] {phase:<8s} | alive={ev.get('alive_pct', 0):5.1f}% "
          f"ret={ev.get('full_norm_return', 0):.4f} mpjpe={ev.get('mpjpe_mm', 0):5.1f}mm | {eta_str}",
          flush=True,
        )
      elif etype == "ERROR":
        print(f"[TDOT] ERROR: {ev.get('error')}", file=sys.stderr, flush=True)
        return 1
      elif etype == "DONE":
        final_msg = ev

    job = eng.get_job(job_id)
    if final_msg is None or job is None or job.optimized_arrays is None:
      print("[TDOT] Optimization did not produce an output.", file=sys.stderr)
      return 1

    out_path = Path(args.out).resolve()
    arrs = job.optimized_arrays
    export_unified_npz(
      out_path,
      arrs["qpos"],
      arrs["qvel"],
      eng.verifier.mj_model,
      fps=50.0,
      ctrl=arrs.get("ctrl"),
      torques=arrs.get("torques"),
      compressed=args.compressed,
    )
    if args.csv_out:
      csv_path = Path(args.csv_out).resolve()
      csv_path.parent.mkdir(parents=True, exist_ok=True)
      csv_path.write_bytes(export_lafan1_csv_bytes(arrs["qpos"], fps_in=50.0, fps_out=30.0))

    mv = final_msg["mujoco_verify"]
    tm = final_msg["timing"]
    print(
      f"[TDOT] DONE ({final_msg['stop_reason']}) in {time.perf_counter() - t0:.1f}s "
      f"(opt={tm['optimization_s']:.1f}s, verify={tm['verify_s']:.2f}s) | "
      f"MuJoCo4 alive={mv['alive_frac'] * 100:.1f}% ret={mv['full_norm_return']:.4f} "
      f"MPJPE={mv['mpjpe_mm']:.1f}mm -> {out_path}",
      flush=True,
    )
    if args.report_json:
      rpath = Path(args.report_json).resolve()
      rpath.parent.mkdir(parents=True, exist_ok=True)
      summary = {k: v for k, v in final_msg.items() if k != "motion"}
      rpath.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0
  finally:
    eng.shutdown()


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(
    prog="tdot",
    description="TDOT: Fast Unitree G1 Dynamic Trajectory Optimization Tool (NablaMimic + MuJoCo 4.0 + 3D Editor)",
  )
  sub = p.add_subparsers(dest="command", required=True)

  s = sub.add_parser("serve", help="Start the 3D Motion Editor & Preloaded Dynamic TO web server")
  s.add_argument("--host", default="127.0.0.1")
  s.add_argument("--port", type=int, default=5000)
  s.add_argument(
    "--preload-buckets",
    default="512",
    help="Comma-separated frame buckets to pre-compile on startup (default: 512; empty to disable)",
  )
  s.add_argument("--no-preload", action="store_true", help="Delay NablaEngine launch until first TO click")
  s.add_argument("--nabla-python", default=None, help="Path to NablaMimic/.venv/bin/python")
  s.add_argument("--arch", choices=["lite", "base"], default="lite")
  s.add_argument(
    "--matmul-precision",
    choices=["highest", "high", "default", "nets_tf32"],
    default="highest",
  )
  s.set_defaults(func=_cmd_serve)

  o = sub.add_parser("optimize", help="Headless CLI dynamic trajectory optimization")
  o.add_argument("--input", required=True, help="Input trajectory (.npz, .csv, .npy)")
  o.add_argument("--out", required=True, help="Output unified .npz path (MLAB4_Fast + Editor compatible)")
  o.add_argument("--csv-out", default=None, help="Optional LAFAN1 30Hz .csv export path")
  o.add_argument("--report-json", default=None, help="Optional JSON metrics report path")
  o.add_argument("--preset", choices=["fast", "standard", "tight"], default="standard")
  o.add_argument("--arch", choices=["lite", "base"], default="lite")
  o.add_argument("--matmul-precision", choices=["highest", "high", "default", "nets_tf32"], default="highest")
  o.add_argument("--input-fps", type=float, default=None)
  o.add_argument("--start-s", type=float, default=None)
  o.add_argument("--duration-s", type=float, default=None)
  o.add_argument("--smooth-window", type=int, default=0)
  o.add_argument("--max-seconds", type=float, default=None)
  o.add_argument("--no-floor-align", action="store_true")
  o.add_argument("--no-clamp-joints", action="store_true")
  o.add_argument("--compressed", action="store_true", help="Use compressed npz (slower, smaller)")
  o.add_argument("--nabla-python", default=None)
  o.set_defaults(func=_cmd_optimize)

  args = p.parse_args(argv)
  return int(args.func(args))


if __name__ == "__main__":
  raise SystemExit(main())
