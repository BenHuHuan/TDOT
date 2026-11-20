"""Preloaded strict-IEEE-FP32 MuJoCo 4.0 closed-loop verifier & unified exporter.

Compiles the exact mjlab-matched Unitree G1 MuJoCo model ONCE at server startup (~40 ms
directly from Menagerie XML, or from mjlab if available), eliminating the ~9.3s per-run
`mjlab.tasks` import + Scene.compile() overhead. Subsequent closed-loop verification +
30-body FK + unified `.npz` export completes in ~0.25–0.45 seconds!
"""

from __future__ import annotations

import itertools
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from tdot.io_bridge import (
  G1_JOINT_NAMES,
  TRACKED_14_BODIES,
  audit_trajectory_kinematics,
  compute_full_body_kinematics,
  get_g1_joint_limits,
  resolve_body_id,
  resolve_joint_id,
)

# Exact mjlab / NablaMimic Unitree G1 sole capsules & contact parameters.
_FOOT_CAPSULES = {
  "left": [
    ((0.01, 0.025005), (0.075, -0.0265, -0.025), (0.707107, -0.014139, 0.706965, 0.0)),
    ((0.01, 0.0835), (0.0395, -0.018, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
    ((0.01, 0.091), (0.039, -0.01, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
    ((0.01, 0.093), (0.039, 0.0, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
    ((0.01, 0.091), (0.039, 0.01, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
    ((0.01, 0.0835), (0.0395, 0.018, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
    ((0.01, 0.025), (0.075, 0.026, -0.025), (0.707107, 0.0, 0.707107, 0.0)),
  ],
  "right": [
    ((0.01, 0.025), (0.075, -0.026, -0.025), (0.707107, 0.0, 0.707107, 0.0)),
    ((0.01, 0.0835), (0.0395, -0.018, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
    ((0.01, 0.091), (0.039, -0.01, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
    ((0.01, 0.093), (0.039, 0.0, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
    ((0.01, 0.091), (0.039, 0.01, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
    ((0.01, 0.0835), (0.0395, 0.018, -0.025), (0.707107, 0.0, -0.707107, 0.0)),
    ((0.01, 0.025), (0.075, 0.026, -0.025), (0.707107, 0.0, 0.707107, 0.0)),
  ],
}
_EXCLUDE_PAIRS = {
  ("pelvis", "left_hip_roll_link"),
  ("pelvis", "right_hip_roll_link"),
  ("left_elbow_link", "left_wrist_pitch_link"),
  ("right_elbow_link", "right_wrist_pitch_link"),
}
_SOLREF, _SOLIMP = (0.02, 1.0), (0.9, 0.95, 0.001, 0.5, 2.0)
_HIST, _FUTURE = 10, 10
_GOAL_ANCHOR_SLICE = slice(3, 9)
_OMEGA = 10.0 * 2.0 * math.pi
_Q0_SUBSTR: dict[str, float] = {
  "hip_pitch": -0.312,
  "knee": 0.669,
  "ankle_pitch": -0.363,
  "elbow": 0.6,
  "left_shoulder_roll": 0.2,
  "right_shoulder_roll": -0.2,
  "shoulder_pitch": 0.2,
}
_SCALE_JVEL = 1.0 / 5.0
_SCALE_AVEL = 1.0 / 2.0
_STD = dict(body_pos=0.3, body_ori=0.4, body_lin_vel=1.0, body_ang_vel=3.14, root_pos=0.3, root_ori=0.4)
_W = dict(body_pos=1.0, body_ori=1.0, body_lin_vel=1.0, body_ang_vel=1.0, root_pos=0.5, root_ori=0.5, action_rate=-0.1, joint_limit=-2.0)
_DT = 0.02
_MAX_REWARD = _DT * 5.0


def _find_menagerie_scene_xml() -> Path | None:
  cands = []
  env_dir = os.environ.get("MENAGERIE_DIR")
  if env_dir:
    cands.append(Path(env_dir) / "unitree_g1/scene_mjx.xml")
  cands.extend([
    Path.home() / "NablaMimic/third_party/mujoco_menagerie/unitree_g1/scene_mjx.xml",
    Path(__file__).resolve().parents[1] / "third_party/mujoco_menagerie/unitree_g1/scene_mjx.xml",
  ])
  for c in cands:
    if c.exists():
      return c
  return None


def _build_g1_model_fast(solver_iterations: int = 10) -> mujoco.MjModel:
  """Compile mjlab-matched Unitree G1 in ~40ms from Menagerie XML (fallback to mjlab Scene)."""
  scene_xml = _find_menagerie_scene_xml()
  if scene_xml is not None:
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    for g in list(spec.geoms):
      if re.fullmatch(r"(left|right)_foot(\d|_box)_collision", g.name or ""):
        spec.delete(g)
    for side, geoms in _FOOT_CAPSULES.items():
      body = spec.body(f"{side}_ankle_roll_link")
      for k, (size, pos, quat) in enumerate(geoms, 1):
        body.add_geom(
          name=f"{side}_foot{k}_collision",
          type=mujoco.mjtGeom.mjGEOM_CAPSULE,
          size=[*size, 0.0],
          pos=pos,
          quat=quat,
          contype=0,
          conaffinity=0,
          group=3,
          condim=3,
          friction=[0.6, 0.005, 0.0001],
        )
    for p in list(spec.pairs):
      spec.delete(p)
    m0 = spec.compile()
    geoms = [
      g for g in range(m0.ngeom)
      if ((m0.geom(g).name or "").endswith("_collision") or m0.geom(g).name == "floor")
    ]

    def _add(g1: int, g2: int, name: str) -> None:
      n1, n2 = m0.geom(g1).name, m0.geom(g2).name
      f1 = 0.6 if "_foot" in n1 else 1.0
      f2 = 0.6 if "_foot" in n2 else 1.0
      fr = 1.0 if (n1 == "floor" or n2 == "floor") else max(f1, f2)
      condim = 3 if ("_foot" in n1 or "_foot" in n2 or "floor" in (n1, n2)) else 1
      spec.add_pair(
        name=name,
        geomname1=n1,
        geomname2=n2,
        condim=condim,
        friction=[fr, fr, 0.005, 0.0001, 0.0001],
        solref=list(_SOLREF),
        solimp=list(_SOLIMP),
      )

    floor = next(g for g in geoms if m0.geom(g).name == "floor")
    robot_g = [g for g in geoms if g != floor]
    for g in robot_g:
      nm = m0.geom(g).name
      _add(g, floor, nm.replace("_collision", "_floor") if "_foot" in nm else f"{nm}__floor")
    for g1, g2 in itertools.combinations(robot_g, 2):
      b1, b2 = int(m0.geom_bodyid[g1]), int(m0.geom_bodyid[g2])
      if b1 == b2 or int(m0.body_parentid[b1]) == b2 or int(m0.body_parentid[b2]) == b1:
        continue
      bn1, bn2 = m0.body(b1).name, m0.body(b2).name
      if (bn1, bn2) in _EXCLUDE_PAIRS or (bn2, bn1) in _EXCLUDE_PAIRS:
        continue
      _add(g1, g2, f"{m0.geom(g1).name}__{m0.geom(g2).name}")
    m = spec.compile()
    m.dof_frictionloss[:] = 0.0
    m.jnt_solref[:] = _SOLREF
    m.jnt_solimp[:] = _SOLIMP
    m.opt.timestep = 0.005
    m.opt.iterations = int(solver_iterations)
    return m

  # Fallback: compile directly via MLAB4_Fast Scene
  import mjlab.tasks  # noqa: F401
  from mjlab.scene import Scene
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1")
  cfg.scene.num_envs = 1
  m = Scene(cfg.scene, device="cpu").compile()
  m.opt.timestep = 0.005
  m.opt.iterations = int(solver_iterations)
  return m


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
  w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
  w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
  return np.stack([
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ], axis=-1)


def _quat_inv(q: np.ndarray) -> np.ndarray:
  out = q.copy()
  out[..., 1:] = -out[..., 1:]
  return out


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return np.stack([
    np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
    np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
    np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
  ], axis=-2)


def _rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
  return np.einsum("...ij,...j->...i", _quat_to_mat(q), v)


def _rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  return np.einsum("...ji,...j->...i", _quat_to_mat(q), v)


def _yaw_quat(q: np.ndarray) -> np.ndarray:
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return np.stack([np.cos(yaw / 2.0), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2.0)], axis=-1)


def _gravity_b(q: np.ndarray) -> np.ndarray:
  return _rotate_inv(q, np.array([0.0, 0.0, -1.0], dtype=np.float64))


def _quat_angle_sq(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
  e = _quat_mul(q1, _quat_inv(q2))
  vn = np.sqrt(np.sum(e[..., 1:] ** 2, axis=-1) + 1e-12)
  return (2.0 * np.arctan2(vn, np.abs(e[..., 0]))) ** 2


class NablaActorTorch:
  """Zero-JAX strict-IEEE-FP32 PyTorch evaluator for NablaMimic actor weights."""

  def __init__(self, params_npz_path: str | Path, device: str = "cpu") -> None:
    z = np.load(params_npz_path)
    n_enc, n_dec = int(z["n_enc"][0]), int(z["n_dec"][0])
    self.device = torch.device(device)
    self.enc = [
      (
        torch.from_numpy(z[f"enc_w{i}"]).to(self.device, dtype=torch.float32),
        torch.from_numpy(z[f"enc_b{i}"]).to(self.device, dtype=torch.float32),
      )
      for i in range(n_enc)
    ]
    self.dec = [
      (
        torch.from_numpy(z[f"dec_w{i}"]).to(self.device, dtype=torch.float32),
        torch.from_numpy(z[f"dec_b{i}"]).to(self.device, dtype=torch.float32),
      )
      for i in range(n_dec)
    ]
    self.anchor_k = int(z["anchor_dec"].shape[0]) if "anchor_dec" in z.files else 0

  @staticmethod
  def _mlp(layers: list[tuple[torch.Tensor, torch.Tensor]], x: torch.Tensor) -> torch.Tensor:
    for w, b in layers[:-1]:
      x = F.elu(x @ w + b)
    w, b = layers[-1]
    return x @ w + b

  @torch.inference_mode()
  def __call__(self, hist: np.ndarray, goal: np.ndarray) -> np.ndarray:
    ht = torch.from_numpy(hist.astype(np.float32, copy=False)).to(self.device)
    gt = torch.from_numpy(goal.astype(np.float32, copy=False)).to(self.device)
    if self.anchor_k > 0:
      z = torch.tanh(self._mlp(self.enc, gt[:-self.anchor_k]))
      out = self._mlp(self.dec, torch.cat([z, ht, gt[-self.anchor_k :]], dim=-1))
    else:
      z = torch.tanh(self._mlp(self.enc, gt))
      out = self._mlp(self.dec, torch.cat([z, ht], dim=-1))
    return out.cpu().numpy()


class PreloadedMuJoCoVerifier:
  """Pre-compiled MuJoCo 4.0 G1 physics verifier resident in server memory (~0.35s/verify)."""

  def __init__(self, solver_iterations: int = 10) -> None:
    t0 = time.time()
    # Enforce strict IEEE FP32 in PyTorch
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
      torch.backends.cuda.matmul.fp32_precision = "ieee"
    self.mj_model = _build_g1_model_fast(solver_iterations=solver_iterations)
    m = self.mj_model
    nu = len(G1_JOINT_NAMES)
    jid_to_act = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
    self.joint_to_act = np.empty(nu, dtype=np.int64)
    self.kp = np.empty(nu, dtype=np.float64)
    self.kd = np.empty(nu, dtype=np.float64)
    self.effort = np.empty(nu, dtype=np.float64)
    self.q0 = np.zeros(nu, dtype=np.float64)
    self.ctrl_lo = np.empty(nu, dtype=np.float64)
    self.ctrl_hi = np.empty(nu, dtype=np.float64)
    for j, jname in enumerate(G1_JOINT_NAMES):
      jid = resolve_joint_id(m, jname)
      a_idx = jid_to_act[jid]
      self.joint_to_act[j] = a_idx
      arm = float(m.dof_armature[int(m.jnt_dofadr[jid])])
      self.kp[j] = arm * (_OMEGA**2)
      self.kd[j] = 2.0 * 2.0 * arm * _OMEGA
      eff_a = float(m.actuator_forcerange[a_idx, 1])
      eff_j = float(m.jnt_actfrcrange[jid, 1])
      self.effort[j] = eff_a if eff_a > 0.0 else eff_j
      for sub, val in _Q0_SUBSTR.items():
        if sub in jname:
          self.q0[j] = val
      ext = self.effort[j] / self.kp[j]
      self.ctrl_lo[j] = float(m.jnt_range[jid, 0]) - ext
      self.ctrl_hi[j] = float(m.jnt_range[jid, 1]) + ext
      m.actuator_gainprm[a_idx, 0] = self.kp[j]
      m.actuator_biasprm[a_idx, 1] = -self.kp[j]
      m.actuator_biasprm[a_idx, 2] = -self.kd[j]
      m.actuator_ctrlrange[a_idx, 0] = self.ctrl_lo[j]
      m.actuator_ctrlrange[a_idx, 1] = self.ctrl_hi[j]

    self.action_scale = 0.25 * self.effort / self.kp
    self.body_ids_14 = [resolve_body_id(m, nm) for nm in TRACKED_14_BODIES]
    self.anchor_k = TRACKED_14_BODIES.index("torso_link")
    self.torso_id = self.body_ids_14[self.anchor_k]
    jnt_lo, jnt_hi = get_g1_joint_limits(m)
    mid = 0.5 * (jnt_lo + jnt_hi)
    half = 0.5 * (jnt_hi - jnt_lo) * 0.9
    self.soft_lo, self.soft_hi = mid - half, mid + half
    self.boot_ms = round((time.time() - t0) * 1000.0, 1)

  def verify_closed_loop(
    self,
    ref_qpos: np.ndarray,
    ref_qvel: np.ndarray,
    actor_params_npz: str | Path,
    *,
    nabla_qpos: np.ndarray | None = None,
    nabla_qvel: np.ndarray | None = None,
    nabla_ctrl: np.ndarray | None = None,
    term_height: float = 0.30,
    term_tilt: float = 0.80,
  ) -> dict[str, Any]:
    """Run closed-loop strict-IEEE-FP32 rollout in preloaded MuJoCo 4.0 (~0.2-0.4s!)."""
    t0 = time.time()
    m = self.mj_model
    actor = NablaActorTorch(actor_params_npz, device="cpu")
    n_real = len(ref_qpos)
    n_steps = n_real - 1
    ref_qpos64 = np.asarray(ref_qpos, dtype=np.float64)
    ref_qvel64 = np.asarray(ref_qvel, dtype=np.float64)

    ref_bp, ref_bq, ref_bl, ref_ba = compute_full_body_kinematics(
      m, ref_qpos64, ref_qvel64, body_names=TRACKED_14_BODIES
    )
    ref_bp64 = ref_bp.astype(np.float64)
    ref_bq64 = ref_bq.astype(np.float64)
    ref_bl64 = ref_bl.astype(np.float64)
    ref_ba64 = ref_ba.astype(np.float64)

    rq_wxyz = ref_qpos64[:, 3:7]
    goal_feat = np.concatenate([
      ref_qpos64[:, 7:] - self.q0[None, :],
      ref_qvel64[:, 6:] * _SCALE_JVEL,
      _gravity_b(rq_wxyz),
      ref_qvel64[:, 3:6] * _SCALE_AVEL,
      _rotate_inv(_yaw_quat(rq_wxyz), ref_qvel64[:, :3]),
      ref_qpos64[:, 2:3],
    ], axis=1).astype(np.float32)

    d = mujoco.MjData(m)
    d.qpos[:] = ref_qpos64[0]
    d.qvel[:] = ref_qvel64[0]
    mujoco.mj_forward(m, d)

    def _proprio(prev_a: np.ndarray) -> np.ndarray:
      return np.concatenate([
        d.qpos[7:] - self.q0,
        d.qvel[6:] * _SCALE_JVEL,
        d.qvel[3:6] * _SCALE_AVEL,
        _gravity_b(d.qpos[3:7]),
        prev_a,
      ]).astype(np.float32)

    def _anchor_rel(frame: int) -> np.ndarray:
      ra_p, ra_q = d.xpos[self.torso_id], d.xquat[self.torso_id]
      ma_p, ma_q = ref_bp64[frame, self.anchor_k], ref_bq64[frame, self.anchor_k]
      pos = _rotate_inv(ra_q, ma_p - ra_p)
      rot = _quat_to_mat(_quat_mul(_quat_inv(ra_q), ma_q))
      return np.concatenate([pos, rot[:, 0], rot[:, 1]]).astype(np.float32)

    prev_a = np.zeros(m.nu, dtype=np.float64)
    hist = np.tile(_proprio(prev_a)[None, :], (_HIST, 1))
    qpos_out = np.empty((n_real, m.nq), dtype=np.float32)
    qvel_out = np.empty((n_real, m.nv), dtype=np.float32)
    ctrl_out = np.empty((n_steps, m.nu), dtype=np.float32)
    torque_out = np.empty((n_steps, m.nu), dtype=np.float32)
    mpjpe_seq = np.zeros(n_steps, dtype=np.float32)
    qpos_out[0], qvel_out[0] = d.qpos.astype(np.float32), d.qvel.astype(np.float32)

    alive = True
    fall_t = -1
    fall_cause: str | None = None
    rew_sum = mpjpe_sum = mpjpe_max = root_sum = jmae_sum = ar_sq_sum = max_tau = 0.0
    vel6 = np.empty(6, dtype=np.float64)

    for t in range(n_steps):
      nf = min(t + 1, n_real - 1)
      idx_g = np.minimum(t + np.arange(_FUTURE), n_real - 1)
      goal = np.concatenate([goal_feat[idx_g].reshape(-1), _anchor_rel(t)[_GOAL_ANCHOR_SLICE]])
      act = actor(hist.reshape(-1), goal).astype(np.float64)
      ctrl = np.clip(ref_qpos64[nf, 7:] + self.action_scale * act, self.ctrl_lo, self.ctrl_hi)
      d.ctrl[self.joint_to_act] = ctrl
      for _ in range(4):
        mujoco.mj_step(m, d)
      mujoco.mj_comPos(m, d)
      mujoco.mj_comVel(m, d)

      qpos_out[t + 1] = d.qpos.astype(np.float32)
      qvel_out[t + 1] = d.qvel.astype(np.float32)
      ctrl_out[t] = ctrl.astype(np.float32)
      tau = np.asarray(d.qfrc_actuator[6 : 6 + m.nu], dtype=np.float64)
      torque_out[t] = tau.astype(np.float32)
      max_tau = max(max_tau, float(np.max(np.abs(tau) / np.maximum(self.effort, 1e-6))))

      pos, quat = d.xpos[self.body_ids_14], d.xquat[self.body_ids_14]
      lin = np.empty((len(self.body_ids_14), 3), dtype=np.float64)
      ang = np.empty((len(self.body_ids_14), 3), dtype=np.float64)
      for kb, bid in enumerate(self.body_ids_14):
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0)
        ang[kb], lin[kb] = vel6[:3], vel6[3:]
      rp, rq = ref_bp64[nf], ref_bq64[nf]
      ra_p, ra_q, ma_p, ma_q = pos[self.anchor_k], quat[self.anchor_k], rp[self.anchor_k], rq[self.anchor_k]
      d_pos = np.concatenate([ra_p[:2], ma_p[2:]])
      d_ori = _yaw_quat(_quat_mul(ra_q, _quat_inv(ma_q)))
      rel_p = d_pos[None, :] + _rotate(rp - ma_p[None, :], d_ori[None, :])
      rel_q = _quat_mul(d_ori[None, :], rq)

      b_err = np.linalg.norm(rel_p - pos, axis=-1)
      b_mean, b_max = float(b_err.mean()), float(b_err.max())
      r_err = float(np.linalg.norm(ma_p - ra_p))
      j_mae = float(np.mean(np.abs(d.qpos[7:] - ref_qpos64[nf, 7:])))
      mpjpe_seq[t] = float(b_mean * 1000.0)

      if t == 0:
        da = act + (ref_qpos64[nf, 7:] - self.q0) / self.action_scale
      else:
        da = act - prev_a + (ref_qpos64[nf, 7:] - ref_qpos64[t, 7:]) / self.action_scale

      err_d = dict(
        root_pos=float(np.sum((ma_p - ra_p) ** 2)),
        root_ori=float(_quat_angle_sq(ma_q, ra_q)),
        body_pos=float(np.mean(np.sum((rel_p - pos) ** 2, axis=-1))),
        body_ori=float(np.mean(_quat_angle_sq(rel_q, quat))),
        body_lin_vel=float(np.mean(np.sum((ref_bl64[nf] - lin) ** 2, axis=-1))),
        body_ang_vel=float(np.mean(np.sum((ref_ba64[nf] - ang) ** 2, axis=-1))),
      )
      s_rew = sum(_DT * _W[k_t] * math.exp(-v_t / (_STD[k_t] ** 2)) for k_t, v_t in err_d.items())
      s_rew += _DT * _W["action_rate"] * float(np.sum(da**2))
      q_j = d.qpos[7:]
      s_rew += _DT * _W["joint_limit"] * float(
        np.sum(np.maximum(q_j - self.soft_hi, 0.0) + np.maximum(self.soft_lo - q_j, 0.0))
      )

      dz = abs(float(d.qpos[2] - ref_qpos64[nf, 2]))
      dg = abs(float(_gravity_b(d.qpos[3:7])[2] - _gravity_b(ref_qpos64[nf, 3:7])[2]))
      bad = not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)))
      term = (dz > term_height) or (dg > term_tilt) or bad
      if alive:
        rew_sum += s_rew
        mpjpe_sum += b_mean
        mpjpe_max = max(mpjpe_max, b_max)
        root_sum += r_err
        jmae_sum += j_mae
        ar_sq_sum += float(np.mean(da**2))
        if term:
          alive = False
          fall_t = t + 1
          fall_cause = "nonfinite" if bad else ("both" if (dz > term_height and dg > term_tilt) else ("height" if dz > term_height else "tilt"))

      hist = np.concatenate([hist[1:], _proprio(act)[None, :]], axis=0)
      prev_a = act

    nal = float(fall_t if fall_t > 0 else n_steps)
    nal_s = max(nal, 1.0)
    failed = fall_t > 0
    if not failed or nabla_qpos is None or nabla_qvel is None:
      source = "mujoco4_closed_loop" if not failed else "mujoco4_partial"
      chosen_qpos, chosen_qvel, chosen_ctrl, chosen_tau = qpos_out, qvel_out, ctrl_out, torque_out
    else:
      source = "nabla_engine_fallback"
      chosen_qpos = nabla_qpos
      chosen_qvel = nabla_qvel
      chosen_ctrl = nabla_ctrl if nabla_ctrl is not None else ctrl_out
      chosen_tau = None

    opt_audit = audit_trajectory_kinematics(m, chosen_qpos, fps=50.0)
    verify_s = time.time() - t0
    return {
      "mujoco_version": mujoco.__version__,
      "export_source": source,
      "verify_s": round(verify_s, 3),
      "alive_frac": float(nal / float(n_steps)),
      "failed": failed,
      "fall_s": round(fall_t * _DT, 2) if failed else None,
      "fall_cause": fall_cause,
      "full_norm_return": float(rew_sum / (float(n_steps) * _MAX_REWARD)),
      "mpjpe_mm": float(mpjpe_sum / nal_s * 1000.0),
      "max_body_err_mm": float(mpjpe_max * 1000.0),
      "root_pos_err_mm": float(root_sum / nal_s * 1000.0),
      "joint_mae_rad": float(jmae_sum / nal_s),
      "action_rate_rms": float(math.sqrt(ar_sq_sum / nal_s)),
      "max_torque_utilization": round(max_tau, 4),
      "optimized_audit": opt_audit.to_dict(),
      "qpos": chosen_qpos,
      "qvel": chosen_qvel,
      "ctrl": chosen_ctrl,
      "torques": chosen_tau,
      "mpjpe_per_frame_mm": mpjpe_seq,
    }
