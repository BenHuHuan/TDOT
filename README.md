# TDOT — Fast Unitree G1 Dynamic Trajectory Optimization & 3D Editor

**TDOT** combines an upgraded 3D URDF / 2D Curve Editor (built on [`project-instinct/robot-motion-editor`](https://github.com/project-instinct/robot-motion-editor)) with a preloaded GPU **NablaMimic + NablaEngine + SuperMimic-lite** differentiable trajectory optimizer and a preloaded **MuJoCo 4.0** strict IEEE FP32 closed-loop verifier ([`BenHuHuan/MLAB4_Fast`](https://github.com/BenHuHuan/MLAB4_Fast)).

Feed in **any** rough, retargeted, hand-edited, or physically infeasible Unitree G1 29-DoF trajectory (floating, ground-penetrating, joint-limit-violating, or jittery) — TDOT returns a **100% dynamically feasible, MuJoCo 4.0-verified trajectory** in seconds, ready for direct playback, curve comparison, or drop-in `MLAB4_Fast` training.

---

## Key Features

### 1. Preloading Pipeline (Zero-Compile Interactive TO & ~20 ms Export)
Cold-starting JAX/XLA + `mjlab` + MuJoCo on every run wastes **~30–35 s** before optimization even starts. TDOT eliminates startup & export overhead via four preload layers:

| Stage | Cold One-Shot CLI | TDOT Preloaded Server | How |
|---|---|---|---|
| **NablaEngine + Bucket JIT** | ~17–80 s | **0.0 s** on click | Persistent `NablaDaemon` pre-compiles frame bucket `[512]` into GPU RAM at server boot (disk XLA cache reload ~15 s) |
| **Variable-Length Eval** | Re-JIT per clip length | **0.0 s** | `dual_eval` traced once over `bucket - 1` with dynamic `n_steps_real` scalar |
| **Interactive Edit Re-TO** | Cold random weights | **~4–10 s** | Warm-start cache reuses `(actor, critic, cnorm)` weights across user curve edits |
| **MuJoCo 4.0 Verification** | ~9.3 s (`mjlab.tasks` import) + 0.4 s | **~0.4–0.65 s** | `PreloadedMuJoCoVerifier` compiles Menagerie G1 XML directly in **~190 ms** at startup and stays resident in RAM |
| **Unified `.npz` Export** | ~0.5–1.5 s (recompute + deflate) | **~19–25 ms** | Uncompressed unified `.npz` served directly from cached verified arrays (`qpos/qvel/ctrl/torques`) |

### 2. Physics-Driven Monotonic Progress Bar (Adaptive Early-Stopping)
Because TDOT stops dynamically via `ConvergenceMonitor` (`TARGET_MET` or `PLATEAU_FEASIBLE`) rather than running fixed epochs, a naive `iter / max_iters` bar would jump abruptly from `8%` to `100%`.

`ProgressEstimator` ([`tdot/progress.py`](tdot/progress.py)) maps real physical convergence signals onto a strictly monotonic `[0%, 100%]` bar with EMA-smoothed velocity ETA:
- **`0% → 4%` SANITIZE**: 50 Hz SLERP resampling, joint-limit clamping, floor alignment & kinematic audit.
- **`4% → 12%` COMPILE**: GPU bucket cache check (instant `0 ms` jump to `12%` when preloaded).
- **`12% → 56%` BALANCE**: Whole-body survival discovery driven by deterministic full-clip `alive_frac` (`0% → 100%`) and `train_term_rate`.
- **`56% → 93%` REFINE**: `max(p_target, p_plateau, 0.85 * p_time)` combining tracking fidelity (`return`, `MPJPE`, `joint_mae`, `confirm_streak`) **and** feasible plateau progress (`stale_feasible_evals / plateau_patience`) — so both clean dances and physically impossible motions advance smoothly to completion.
- **`93% → 100%` VERIFY / DONE**: Preloaded MuJoCo 4.0 IEEE FP32 closed-loop rollout & unified export.
- **Stop & Use Best**: Cancel at any time mid-run to immediately receive the highest-ranked trajectory found so far.

### 3. Upgraded 3D Motion Editor & Comparison UI
- **Auto-loaded Unitree G1 29-DoF URDF** + Three.js studio lighting + Play/Pause loop (`Space`), frame step (`←`/`→`), continuous keyframe value edit (`↑`/`↓`), and drag-to-edit 2D curve canvas.
- **Universal Input**: Upload `.npz` (`robot-motion-editor`, `MLAB4_Fast`, or raw `qpos`), LAFAN1 `.csv`, or raw `.npy` — or one-click load the bundled samples (`weird_dance_6s.npz`, `dance_6s_clean.npz`).
- **Before vs After Visual Diff**: Toggle between **Optimized (MuJoCo 4.0)** and **Original**, with an optional **Orange Dashed Ghost Curve** overlay in the 2D curve editor.
- **Unified Export**: One `.npz` file simultaneously satisfies both `robot-motion-editor` (`base_pos_w`, `base_quat_w`, `joint_pos`, `joint_names`, `framerate`) and `MLAB4_Fast` `MotionCommand` (`fps`, `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`, `qpos`, `qvel`, `ctrl`, `torques`), plus optional 30 Hz LAFAN1 `.csv`.

---

## Measured Benchmarks (Strict IEEE FP32, `matmul_precision=highest`)

### NVIDIA RTX PRO 6000 Blackwell (96 GB, Colab G4)
| Test Trajectory | Pure Opt Time | Stop Reason | MuJoCo 4.0 Survival | Return | MPJPE | Joint MAE |
|---|---:|---|---:|---:|---:|---:|
| **6s Clean LAFAN1 Dance** (`301f`) | **14.5 s** | `TARGET_MET` | **100.0%** | **0.9192** | **33.8 mm** | 0.066 rad |
| **15s Continuous LAFAN1 Dance** (`751f`) | **46.8 s** (peak 41.9 s) | `PLATEAU_FEASIBLE` | **100.0%** | **0.8941** | **33.9 mm** | 0.069 rad |
| **6s Corrupted "Weird" Dance** (+18cm float, -12cm dip, +2.2rad spike, jitter) | **15.1 s** (peak 11.6 s) | `PLATEAU_FEASIBLE` | **100.0%** | **0.6401** | 115 mm* | — |
| *Baseline: Pure `MLAB4_Fast fp32_fast` PPO (393k FPS, 22.6M steps, 61.5s) on Corrupted 6s* | *61.5 s* | *timeout* | *2.0%* | *0.011* | *—* | *—* |

*\*MPJPE on the corrupted motion is measured against the physically impossible corrupted input reference; MuJoCo 4.0 output is 100% balanced and floor-consistent.*

### Local Laptop GPU (NVIDIA RTX 4080 Laptop, Preloaded Daemon E2E)
| Run | Compile Time | Click → Done | Verify (MuJoCo 4.0) | `.npz` Download | MuJoCo 4.0 Survival | Return | MPJPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Background Preload (warm XLA disk cache) | — | **15.2 s** (once at server boot) | MuJoCo boot: **202 ms** | — | — | — | — |
| `dance_6s_clean.npz` (`warm_start=True`) | **0.0 s** | **10.53 s** (opt 9.71 s) | **0.653 s** | **19 ms** | **100.0%** | **0.8929** | **32.5 mm** |

---

## Quick Start

### 1. Install
```bash
git clone https://github.com/BenHuHuan/TDOT.git
cd TDOT
bash scripts/setup_tdot.sh
```
*(If you already have `~/NablaMimic/.venv` and/or `~/mjlab/.venv-mj4`, `scripts/launch_tdot.sh` auto-detects them without reinstalling.)*

### 2. Launch the 3D Editor + Preloaded Dynamic TO Web App
```bash
bash scripts/launch_tdot.sh --host 127.0.0.1 --port 5000 --preload-buckets 512
```
Open **http://127.0.0.1:5000**:
1. Wait for the top HUD pill to turn green (`GPU Ready • Bucket [512] Hot • MuJoCo 4.0`).
2. Click **Load** next to `weird_dance_6s.npz` (or upload your own `.npz` / `.csv` / `.npy`).
3. Click **Run Dynamic TO** — watch the physics-driven progress bar climb through `BALANCE → REFINE → VERIFY`, then toggle **Optimized / Original** or inspect the orange ghost curve!
4. Click **Download Unified NPZ** (drop-in ready for both `MLAB4_Fast` and `robot-motion-editor`).

### 3. Headless CLI Mode
```bash
python -m tdot.cli optimize \
  --input static/samples/weird_dance_6s.npz \
  --out outputs/fixed_dance.npz \
  --csv-out outputs/fixed_dance_30hz.csv \
  --preset standard
```

---

## Repository Structure
```
TDOT/
├── tdot/
│   ├── convergence.py      # Adaptive two-stage early-stopping (TARGET_MET / PLATEAU_FEASIBLE)
│   ├── progress.py         # Physics-driven strictly monotonic progress bar + ETA estimator
│   ├── io_bridge.py        # Universal G1 parser, sanitizer, kinematic audit, unified NPZ & CSV export
│   ├── nabla_daemon.py     # Persistent GPU NablaEngine + NablaMimic + SuperMimic-lite daemon (bucket preload + warm-start)
│   ├── verifier_pool.py    # Preloaded ~190ms MuJoCo 4.0 IEEE FP32 closed-loop verifier
│   ├── engine_manager.py   # Thread-safe job orchestration, SSE fan-out, cancel & cached export
│   ├── server.py           # Flask web server & SSE streaming endpoints
│   └── cli.py              # `tdot serve` & `tdot optimize` CLI
├── templates/index.html    # Upgraded Three.js 3D URDF Viewer + 2D Curve Editor + TDOT TO UI
├── static/
│   ├── unitree_g1/         # Unitree G1 29-DoF URDF & STL visual meshes
│   └── samples/            # Bundled unified .npz test motions (weird_dance_6s.npz, dance_6s_clean.npz)
├── scripts/
│   ├── setup_tdot.sh       # Automated dual-venv setup
│   └── launch_tdot.sh      # One-command launcher with auto-detected venvs & CUDA PATH
└── tests/test_tdot.py      # Unit + Flask route + round-trip export test suite
```

## License & Credits
Licensed under **Apache-2.0**. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attributions to `robot-motion-editor`, `NablaMimic`, `NablaEngine`, `MLAB4_Fast`, and `mujoco_menagerie`.
