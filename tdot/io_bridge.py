"""Universal Unitree G1 trajectory I/O, sanitizer, kinematic auditor & unified exporter.

Accepts ALL common Unitree G1 motion formats seamlessly:
  - `robot-motion-editor` `.npz` (`base_pos_w`, `base_quat_w`, `joint_pos`, `joint_names`, `framerate`)
  - `MLAB4_Fast` / `mjlab` `MotionCommand` `.npz` (`body_pos_w`, `body_quat_w`, `joint_pos`, `fps`)
  - LAFAN1 retargeting `.csv` (30 Hz: `xyz, quat_xyzw, 29 joints`)
  - Raw MuJoCo `qpos` `.npy` / `.npz` (`(T, 36)`: `xyz, quat_wxyz, 29 joints`)
  - In-memory JSON dict directly from the browser editor!

Exports ONE unified `.npz` containing BOTH `robot-motion-editor` keys AND native
`MLAB4_Fast` `MotionCommand` keys (30 bodies + 29 joints + `qpos`/`qvel`/`ctrl`/`torques`),
plus optional LAFAN1 `.csv`.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

import mujoco
import numpy as np
from scipy.signal import savgol_filter

G1_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

G1_BODY_NAMES: tuple[str, ...] = (
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "waist_roll_link",
  "torso_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
)

TRACKED_14_BODIES: tuple[str, ...] = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)


@dataclass
class InputAudit:
  """Physical/kinematic infeasibility diagnostics of a G1 trajectory."""

  num_frames: int
  fps: float
  duration_s: float
  foot_min_z_mm: float
  foot_p05_z_mm: float
  foot_penetration_max_mm: float
  foot_float_p50_mm: float
  foot_float_max_mm: float
  joint_limit_violation_max_rad: float
  joint_limit_violation_frac: float
  max_joint_vel_rad_s: float
  p99_joint_vel_rad_s: float
  max_joint_acc_rad_s2: float
  max_root_vel_m_s: float
  max_root_acc_m_s2: float
  floor_shift_mm: float = 0.0

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass
class SanitizedTrajectory:
  name: str
  source_format: str
  fps: float
  qpos: np.ndarray  # (T, 36) float32 [xyz, wxyz, 29 joints in G1_JOINT_NAMES order]
  qvel: np.ndarray  # (T, 35) float32
  raw_audit: InputAudit
  sanitized_audit: InputAudit


def resolve_body_id(m: mujoco.MjModel, name: str) -> int:
  for cand in (name, f"robot/{name}"):
    bid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cand))
    if bid >= 0:
      return bid
  suffix = f"/{name}"
  for bid in range(m.nbody):
    bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
    if bname.endswith(suffix):
      return bid
  return -1


def resolve_joint_id(m: mujoco.MjModel, name: str) -> int:
  for cand in (name, f"robot/{name}"):
    jid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, cand))
    if jid >= 0:
      return jid
  suffix = f"/{name}"
  for jid in range(m.njnt):
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
    if jname.endswith(suffix):
      return jid
  return -1


def get_g1_joint_limits(m: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
  jids = [resolve_joint_id(m, nm) for nm in G1_JOINT_NAMES]
  if all(j >= 0 for j in jids):
    lo = np.array([m.jnt_range[j, 0] for j in jids], dtype=np.float64)
    hi = np.array([m.jnt_range[j, 1] for j in jids], dtype=np.float64)
    return lo, hi
  lo = np.array([m.jnt_range[m.actuator_trnid[i, 0], 0] for i in range(m.nu)], dtype=np.float64)
  hi = np.array([m.jnt_range[m.actuator_trnid[i, 0], 1] for i in range(m.nu)], dtype=np.float64)
  return lo, hi


def reorder_joints_to_g1(
  joint_pos: np.ndarray, joint_names: list[str] | tuple[str, ...] | None
) -> np.ndarray:
  """Reorder arbitrary joint columns into canonical 29-DoF `G1_JOINT_NAMES` order."""
  jp = np.asarray(joint_pos, dtype=np.float64)
  if joint_names is None:
    assert jp.shape[1] == 29, f"Expected 29 G1 joints, got {jp.shape}"
    return jp
  names = [str(n).split("/")[-1] for n in joint_names]
  if tuple(names) == G1_JOINT_NAMES:
    return jp
  name_to_idx = {n: i for i, n in enumerate(names)}
  out = np.zeros((len(jp), 29), dtype=np.float64)
  for j, target in enumerate(G1_JOINT_NAMES):
    if target in name_to_idx:
      out[:, j] = jp[:, name_to_idx[target]]
  return out


def _normalize_quats_continuous(q: np.ndarray) -> np.ndarray:
  q = np.asarray(q, dtype=np.float64).copy()
  nrm = np.linalg.norm(q, axis=-1, keepdims=True)
  bad = nrm[:, 0] < 1e-8
  q[bad] = np.array([1.0, 0.0, 0.0, 0.0])
  nrm = np.linalg.norm(q, axis=-1, keepdims=True)
  q /= nrm
  for i in range(1, len(q)):
    if float(np.dot(q[i - 1], q[i])) < 0.0:
      q[i] = -q[i]
  return q


def _slerp_batch(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
  dot = np.sum(q0 * q1, axis=-1, keepdims=True)
  q1 = np.where(dot < 0.0, -q1, q1)
  dot = np.abs(dot)
  theta = np.arccos(np.clip(dot, -1.0, 1.0))
  small = theta < 1e-6
  s = np.where(small, 1.0, np.sin(theta))
  w0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / s)
  w1 = np.where(small, t, np.sin(t * theta) / s)
  q = w0 * q0 + w1 * q1
  return q / np.linalg.norm(q, axis=-1, keepdims=True)


def resample_qpos(qpos: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
  qpos = np.asarray(qpos, dtype=np.float64)
  n = len(qpos)
  if n < 2 or abs(fps_in - fps_out) < 1e-6:
    out = qpos.copy()
    out[:, 3:7] = _normalize_quats_continuous(out[:, 3:7])
    return out
  duration = (n - 1) / float(fps_in)
  t_out = np.arange(0.0, duration + 1e-9, 1.0 / float(fps_out))
  if t_out[-1] > duration:
    t_out = t_out[:-1]
  x = np.clip(t_out * float(fps_in), 0.0, n - 1.000001)
  i0 = np.floor(x).astype(int)
  i1 = np.minimum(i0 + 1, n - 1)
  w = (x - i0)[:, None]
  pos = (1.0 - w) * qpos[i0, :3] + w * qpos[i1, :3]
  quat = _slerp_batch(qpos[i0, 3:7], qpos[i1, 3:7], w)
  joints = (1.0 - w) * qpos[i0, 7:] + w * qpos[i1, 7:]
  return np.concatenate([pos, quat, joints], axis=1)


def _find_foot_geom_ids(m: mujoco.MjModel) -> list[int]:
  ids: list[int] = []
  for g in range(m.ngeom):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if "foot" in name and "collision" in name:
      ids.append(g)
  if not ids:
    for g in range(m.ngeom):
      b = int(m.geom_bodyid[g])
      bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
      if "ankle_roll" in bname and m.geom_type[g] in (
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_BOX),
      ):
        ids.append(g)
  return ids


def _foot_lowest_z(m: mujoco.MjModel, d: mujoco.MjData, foot_geom_ids: list[int]) -> float:
  zs: list[float] = []
  for g in foot_geom_ids:
    gtype = int(m.geom_type[g])
    pos_z = float(d.geom_xpos[g, 2])
    if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
      rad = float(m.geom_size[g, 0])
      half = float(m.geom_size[g, 1])
      axis_z = abs(float(d.geom_xmat[g].reshape(3, 3)[2, 2]))
      zs.append(pos_z - half * axis_z - rad)
    elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
      zs.append(pos_z - float(m.geom_size[g, 0]))
    elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
      rot = np.abs(d.geom_xmat[g].reshape(3, 3)[2, :])
      zs.append(pos_z - float(np.dot(rot, m.geom_size[g, :3])))
    else:
      zs.append(pos_z)
  return min(zs) if zs else float(d.qpos[2])


def compute_qvel(m: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
  dt = 1.0 / float(fps)
  n = len(qpos)
  qvel = np.zeros((n, m.nv), dtype=np.float64)
  for i in range(n):
    a, b = max(i - 1, 0), min(i + 1, n - 1)
    if b > a:
      mujoco.mj_differentiatePos(m, qvel[i], (b - a) * dt, qpos[a], qpos[b])
  return qvel


def audit_trajectory_kinematics(
  m: mujoco.MjModel,
  qpos: np.ndarray,
  fps: float,
  *,
  floor_shift_mm: float = 0.0,
) -> InputAudit:
  qpos64 = np.asarray(qpos, dtype=np.float64)
  n = len(qpos64)
  d = mujoco.MjData(m)
  foot_ids = _find_foot_geom_ids(m)
  low_z = np.empty(n, dtype=np.float64)
  for i in range(n):
    d.qpos[:] = qpos64[i]
    mujoco.mj_kinematics(m, d)
    low_z[i] = _foot_lowest_z(m, d, foot_ids)

  jnt_lo, jnt_hi = get_g1_joint_limits(m)
  jpos = qpos64[:, 7 : 7 + len(jnt_lo)]
  viol = np.maximum(jpos - jnt_hi[None, :], 0.0) + np.maximum(
    jnt_lo[None, :] - jpos, 0.0
  )

  qvel = compute_qvel(m, qpos64, fps)
  jvel = np.abs(qvel[:, 6:])
  rvel = np.linalg.norm(qvel[:, :3], axis=-1)
  if n >= 3:
    dt = 1.0 / float(fps)
    jacc = np.abs(np.diff(qvel[:, 6:], axis=0)) / dt
    racc = np.linalg.norm(np.diff(qvel[:, :3], axis=0), axis=-1) / dt
    max_jacc = float(jacc.max())
    max_racc = float(racc.max())
  else:
    max_jacc = 0.0
    max_racc = 0.0

  return InputAudit(
    num_frames=n,
    fps=float(fps),
    duration_s=float((n - 1) / float(fps)) if n > 1 else 0.0,
    foot_min_z_mm=float(low_z.min() * 1000.0),
    foot_p05_z_mm=float(np.percentile(low_z, 5) * 1000.0),
    foot_penetration_max_mm=float(max(0.0, -low_z.min()) * 1000.0),
    foot_float_p50_mm=float(np.percentile(np.maximum(low_z, 0.0), 50) * 1000.0),
    foot_float_max_mm=float(max(0.0, low_z.max()) * 1000.0),
    joint_limit_violation_max_rad=float(viol.max()),
    joint_limit_violation_frac=float((viol > 1e-4).mean()),
    max_joint_vel_rad_s=float(jvel.max()),
    p99_joint_vel_rad_s=float(np.percentile(jvel, 99)),
    max_joint_acc_rad_s2=max_jacc,
    max_root_vel_m_s=float(rvel.max()),
    max_root_acc_m_s2=max_racc,
    floor_shift_mm=float(floor_shift_mm),
  )


def parse_motion_bytes(
  filename: str,
  stream: BinaryIO,
  *,
  input_fps: float | None = None,
) -> tuple[np.ndarray, float, str]:
  """Parse uploaded motion (.npz editor/MLAB4_Fast, .csv LAFAN1, .npy qpos) into (qpos, fps, fmt)."""
  suffix = Path(filename).suffix.lower()
  if suffix == ".csv":
    raw = np.loadtxt(stream, delimiter=",", dtype=np.float64)
    if raw.ndim == 1:
      raw = raw[None, :]
    assert raw.shape[1] == 36, f"Expected 36 columns in LAFAN1 CSV, got {raw.shape}"
    qpos = np.concatenate([raw[:, :3], raw[:, [6, 3, 4, 5]], raw[:, 7:]], axis=1)
    return qpos, float(input_fps or 30.0), "lafan1_csv"

  if suffix == ".npy":
    arr = np.load(stream).astype(np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 36, f"Expected (T, 36) qpos, got {arr.shape}"
    return arr.copy(), float(input_fps or 50.0), "raw_qpos_npy"

  if suffix == ".npz":
    z = np.load(stream, allow_pickle=True)
    fps_val = float(input_fps) if input_fps is not None else None
    if fps_val is None:
      for fk in ("fps", "framerate"):
        if fk in z.files:
          fps_val = float(np.asarray(z[fk]).reshape(-1)[0])
          break
    jnames = z["joint_names"].tolist() if "joint_names" in z.files else None
    if "base_pos_w" in z.files and "base_quat_w" in z.files and "joint_pos" in z.files:
      rpos = np.asarray(z["base_pos_w"], dtype=np.float64)
      rquat = np.asarray(z["base_quat_w"], dtype=np.float64)
      jpos = reorder_joints_to_g1(z["joint_pos"], jnames)
      qpos = np.concatenate([rpos, rquat, jpos], axis=1)
      return qpos, float(fps_val or 50.0), "editor_npz"
    if "joint_pos" in z.files and "body_pos_w" in z.files and "body_quat_w" in z.files:
      rpos = np.asarray(z["body_pos_w"], dtype=np.float64)[:, 0, :]
      rquat = np.asarray(z["body_quat_w"], dtype=np.float64)[:, 0, :]
      jpos = reorder_joints_to_g1(z["joint_pos"], jnames)
      qpos = np.concatenate([rpos, rquat, jpos], axis=1)
      return qpos, float(fps_val or 50.0), "mlab4_npz"
    if "qpos" in z.files:
      qpos = np.asarray(z["qpos"], dtype=np.float64)
      assert qpos.ndim == 2 and qpos.shape[1] == 36, f"Unexpected qpos {qpos.shape}"
      return qpos, float(fps_val or 50.0), "qpos_npz"
    raise ValueError(
      f"Unsupported .npz keys {z.files}. Expected editor (`base_pos_w`, `base_quat_w`, `joint_pos`), "
      "MLAB4_Fast (`body_pos_w`, `body_quat_w`, `joint_pos`), or `qpos`."
    )

  raise ValueError(f"Unsupported file extension: {suffix} (use .npz, .csv, or .npy)")


def sanitize_qpos_array(
  raw_qpos: np.ndarray,
  fps_in: float,
  mj_model: mujoco.MjModel,
  *,
  name: str = "motion",
  source_format: str = "array",
  target_fps: float = 50.0,
  start_s: float | None = None,
  duration_s: float | None = None,
  floor_align: bool = True,
  clamp_joints: bool = True,
  smooth_window: int = 0,
) -> SanitizedTrajectory:
  qpos_arr = np.asarray(raw_qpos, dtype=np.float64)
  if start_s is not None or duration_s is not None:
    i_start = max(0, int(round((start_s or 0.0) * fps_in)))
    if duration_s is not None:
      i_end = min(len(qpos_arr), i_start + int(round(duration_s * fps_in)) + 1)
    else:
      i_end = len(qpos_arr)
    qpos_arr = qpos_arr[i_start:i_end]
  assert len(qpos_arr) >= 4, f"Trajectory too short ({len(qpos_arr)} frames)"

  qpos_50 = resample_qpos(qpos_arr, fps_in=fps_in, fps_out=target_fps)
  raw_audit = audit_trajectory_kinematics(mj_model, qpos_50, target_fps)

  clean = qpos_50.copy()
  clean[:, 3:7] = _normalize_quats_continuous(clean[:, 3:7])

  if smooth_window >= 5:
    win = int(smooth_window) | 1
    win = min(win, len(clean) if len(clean) % 2 == 1 else len(clean) - 1)
    if win >= 5:
      poly = min(3, win - 2)
      clean[:, :3] = savgol_filter(clean[:, :3], win, poly, axis=0)
      clean[:, 7:] = savgol_filter(clean[:, 7:], win, poly, axis=0)
      clean[:, 3:7] = _normalize_quats_continuous(
        savgol_filter(clean[:, 3:7], win, poly, axis=0)
      )

  if clamp_joints:
    jnt_lo, jnt_hi = get_g1_joint_limits(mj_model)
    clean[:, 7 : 7 + len(jnt_lo)] = np.clip(
      clean[:, 7 : 7 + len(jnt_lo)], jnt_lo[None, :], jnt_hi[None, :]
    )

  floor_shift_mm = 0.0
  if floor_align:
    d = mujoco.MjData(mj_model)
    foot_ids = _find_foot_geom_ids(mj_model)
    low = np.empty(len(clean), dtype=np.float64)
    for i in range(len(clean)):
      d.qpos[:] = clean[i]
      mujoco.mj_kinematics(mj_model, d)
      low[i] = _foot_lowest_z(mj_model, d, foot_ids)
    shift = float(np.percentile(low, 5))
    clean[:, 2] -= shift
    floor_shift_mm = -shift * 1000.0

  qvel_50 = compute_qvel(mj_model, clean, target_fps)
  sanitized_audit = audit_trajectory_kinematics(
    mj_model, clean, target_fps, floor_shift_mm=floor_shift_mm
  )
  return SanitizedTrajectory(
    name=name,
    source_format=source_format,
    fps=float(target_fps),
    qpos=clean.astype(np.float32),
    qvel=qvel_50.astype(np.float32),
    raw_audit=raw_audit,
    sanitized_audit=sanitized_audit,
  )


def load_and_sanitize_file(
  path: str | Path,
  mj_model: mujoco.MjModel,
  *,
  input_fps: float | None = None,
  target_fps: float = 50.0,
  start_s: float | None = None,
  duration_s: float | None = None,
  floor_align: bool = True,
  clamp_joints: bool = True,
  smooth_window: int = 0,
) -> SanitizedTrajectory:
  p = Path(path).resolve()
  with open(p, "rb") as f:
    raw_qpos, fps_in, fmt = parse_motion_bytes(p.name, f, input_fps=input_fps)
  return sanitize_qpos_array(
    raw_qpos,
    fps_in,
    mj_model,
    name=p.stem,
    source_format=fmt,
    target_fps=target_fps,
    start_s=start_s,
    duration_s=duration_s,
    floor_align=floor_align,
    clamp_joints=clamp_joints,
    smooth_window=smooth_window,
  )


def compute_full_body_kinematics(
  mj_model: mujoco.MjModel,
  qpos: np.ndarray,
  qvel: np.ndarray,
  body_names: tuple[str, ...] = G1_BODY_NAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  qpos64 = np.asarray(qpos, dtype=np.float64)
  qvel64 = np.asarray(qvel, dtype=np.float64)
  n = len(qpos64)
  body_ids = [resolve_body_id(mj_model, name) for name in body_names]
  if any(b < 0 for b in body_ids):
    missing = [name for name, b in zip(body_names, body_ids, strict=True) if b < 0]
    raise ValueError(f"Bodies not found in MuJoCo model: {missing}")

  nb = len(body_ids)
  body_pos = np.empty((n, nb, 3), dtype=np.float32)
  body_quat = np.empty((n, nb, 4), dtype=np.float32)
  body_lin = np.empty((n, nb, 3), dtype=np.float32)
  body_ang = np.empty((n, nb, 3), dtype=np.float32)
  d = mujoco.MjData(mj_model)
  vel6 = np.empty(6, dtype=np.float64)

  for i in range(n):
    d.qpos[:] = qpos64[i]
    d.qvel[:] = qvel64[i]
    mujoco.mj_kinematics(mj_model, d)
    mujoco.mj_comPos(mj_model, d)
    mujoco.mj_comVel(mj_model, d)
    body_pos[i] = d.xpos[body_ids]
    body_quat[i] = d.xquat[body_ids]
    for k, b in enumerate(body_ids):
      mujoco.mj_objectVelocity(mj_model, d, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0)
      body_ang[i, k] = vel6[:3]
      body_lin[i, k] = vel6[3:]
  return body_pos, body_quat, body_lin, body_ang


def build_unified_npz_dict(
  qpos: np.ndarray,
  qvel: np.ndarray,
  mj_model: mujoco.MjModel,
  *,
  fps: float = 50.0,
  ctrl: np.ndarray | None = None,
  torques: np.ndarray | None = None,
  extra: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
  """Build ONE unified array dict compatible with BOTH `MLAB4_Fast` and `robot-motion-editor`."""
  qpos32 = np.asarray(qpos, dtype=np.float32)
  qvel32 = np.asarray(qvel, dtype=np.float32)
  body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w = compute_full_body_kinematics(
    mj_model, qpos32, qvel32, body_names=G1_BODY_NAMES
  )
  payload: dict[str, np.ndarray] = {
    # MLAB4_Fast / mjlab MotionCommand keys:
    "fps": np.array([float(fps)], dtype=np.float64),
    "joint_pos": qpos32[:, 7:36].copy(),
    "joint_vel": qvel32[:, 6:35].copy(),
    "body_pos_w": body_pos_w,
    "body_quat_w": body_quat_w,
    "body_lin_vel_w": body_lin_vel_w,
    "body_ang_vel_w": body_ang_vel_w,
    # robot-motion-editor keys:
    "base_pos_w": qpos32[:, :3].copy(),
    "base_quat_w": qpos32[:, 3:7].copy(),
    "joint_names": np.array(G1_JOINT_NAMES),
    "framerate": np.array(float(fps), dtype=np.float64),
    # Full MuJoCo state & actuator telemetry:
    "qpos": qpos32,
    "qvel": qvel32,
  }
  if ctrl is not None:
    payload["ctrl"] = np.asarray(ctrl, dtype=np.float32)
  if torques is not None:
    payload["torques"] = np.asarray(torques, dtype=np.float32)
  if extra:
    payload.update(extra)
  return payload


def export_unified_npz(
  target: str | Path | io.BytesIO,
  qpos: np.ndarray,
  qvel: np.ndarray,
  mj_model: mujoco.MjModel,
  *,
  fps: float = 50.0,
  ctrl: np.ndarray | None = None,
  torques: np.ndarray | None = None,
  extra: dict[str, np.ndarray] | None = None,
  compressed: bool = False,
) -> Path | io.BytesIO:
  """Export unified .npz (uncompressed by default for 5x faster interactive export!)."""
  payload = build_unified_npz_dict(
    qpos, qvel, mj_model, fps=fps, ctrl=ctrl, torques=torques, extra=extra
  )
  save_fn: Any = cast(Any, np.savez_compressed if compressed else np.savez)
  if isinstance(target, (str, Path)):
    out = Path(target)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_fn(out, **payload)
    return out
  save_fn(target, **payload)
  target.seek(0)
  return target


def export_lafan1_csv_bytes(qpos: np.ndarray, *, fps_in: float = 50.0, fps_out: float = 30.0) -> bytes:
  q_out = resample_qpos(qpos, fps_in=fps_in, fps_out=fps_out)
  csv_mat = np.concatenate(
    [q_out[:, :3], q_out[:, [4, 5, 6, 3]], q_out[:, 7:36]], axis=1
  )
  buf = io.StringIO()
  np.savetxt(buf, csv_mat, delimiter=",", fmt="%.6f")
  return buf.getvalue().encode("utf-8")


def qpos_to_editor_payload(
  qpos: np.ndarray,
  *,
  fps: float = 50.0,
  name: str = "motion",
  source_format: str = "tdot",
  audit: InputAudit | None = None,
) -> dict[str, Any]:
  """Convert canonical (T, 36) qpos into the JSON structure consumed by the 3D Web Editor."""
  q32 = np.asarray(qpos, dtype=np.float32)
  return {
    "name": name,
    "source_format": source_format,
    "joint_names": list(G1_JOINT_NAMES),
    "joint_pos": np.round(q32[:, 7:36], 5).tolist(),
    "base_pos_w": np.round(q32[:, :3], 5).tolist(),
    "base_quat_w": np.round(q32[:, 3:7], 5).tolist(),
    "num_frames": int(len(q32)),
    "framerate": float(fps),
    "audit": audit.to_dict() if audit is not None else None,
  }
