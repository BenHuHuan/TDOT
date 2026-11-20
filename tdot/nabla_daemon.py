"""Persistent preloaded NablaEngine + NablaMimic + SuperMimic-lite daemon for TDOT.

Stays resident in GPU memory while the TDOT web editor is open:
  1. Boots NablaEngine CUDA library + Menagerie G1 + STATE_SKIP ONCE at startup.
  2. Pre-compiles bucket 512 (clips up to 10.2s) in GPU RAM in the background while
     the user loads/views/edits their trajectory -> 0.0s cold start when clicking Optimize!
  3. Zero-recompile `dual_eval` parameterized by dynamic `n_real` inside each bucket
     so ANY clip length in the bucket executes with 0ms JIT overhead!
  4. Optional Warm-Start from previous optimized actor weights for rapid re-optimization
     after interactive curve edits (~3-6s instead of ~15s!).
  5. Clean JSON-line protocol on stdout (`PRELOAD_STATE`, `PROGRESS`, `RESULT`, `ERROR`)
     with non-blocking `CANCEL` support via dedicated stdin reader thread.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if "/usr/local/cuda/bin" not in os.environ.get("PATH", "") and Path(
  "/usr/local/cuda/bin/nvcc"
).exists():
  os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"

# Keep stdout 100% pure JSON-lines; redirect any library print() noise to stderr!
_PROTO_OUT = sys.stdout
sys.stdout = sys.stderr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from tdot.convergence import ConvergenceConfig, ConvergenceMonitor, EvalMetrics  # noqa: E402
from tdot.io_bridge import compute_full_body_kinematics  # noqa: E402
from tdot.progress import ProgressEstimator  # noqa: E402

ARCH_HIDDEN: dict[str, tuple[int, ...]] = {
  "lite": (512, 256, 128),
  "medium": (1024, 512, 256),
  "supermimic": (2048, 1024, 512, 256),
}
BUCKET_SIZES: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192)
_WRITE_LOCK = threading.Lock()


def emit(msg: dict[str, Any]) -> None:
  line = json.dumps(msg, ensure_ascii=False)
  with _WRITE_LOCK:
    _PROTO_OUT.write(line + "\n")
    _PROTO_OUT.flush()


def _choose_bucket(n_frames: int) -> int:
  for b in BUCKET_SIZES:
    if n_frames <= b:
      return b
  return int(((n_frames + 4095) // 4096) * 4096)


def _detect_vram_gb() -> float:
  try:
    stats = jax.local_devices()[0].memory_stats() or {}
    lim = stats.get("bytes_limit")
    if lim:
      return float(lim) / (1024.0**3)
  except Exception:
    pass
  return 12.0


def _tune_bucket_hp(bucket: int, vram_gb: float, arch: str) -> dict[str, Any]:
  if vram_gb >= 44.0:
    num_envs = 2048 if bucket >= 1024 else 1024
    replay_steps = 160
    horizon = 48 if bucket >= 1024 else 32
    at_batch = 16384
  elif vram_gb >= 20.0:
    num_envs = 1024 if bucket >= 1024 else 512
    replay_steps = 128
    horizon = 48 if bucket >= 1024 else 32
    at_batch = 8192
  else:
    num_envs = 512
    replay_steps = 96
    horizon = 32
    at_batch = 4096
  return {
    "arch": arch,
    "actor_hidden": ARCH_HIDDEN[arch],
    "actor_latent": 32,
    "critic_hidden": (256, 128),
    "num_envs": num_envs,
    "collect_envs": num_envs,
    "replay_steps": replay_steps,
    "horizon": horizon,
    "at_batch": at_batch,
    "utd": 0.4,
    "lr": 3e-4 if arch == "lite" else 2e-4,
    "action_target_lr": 1e-3 if arch == "lite" else 6e-4,
    "action_target_m": 8,
    "action_target_k": 1,
    "action_target_delta": 0.030,
    "action_target_delta_floor": 0.20,
    "lr_warmup_steps": 15,
    "lr_decay_steps": 500 if arch == "lite" else 1200,
    "lr_decay_to": 0.10,
    "actor_ema": 0.985,
  }


def _save_actor_params_npz(path: Path, actor_tree: dict[str, Any]) -> None:
  arrays: dict[str, np.ndarray] = {}
  enc = actor_tree["enc"]
  dec = actor_tree["dec"]
  arrays["n_enc"] = np.array([len(enc)], dtype=np.int32)
  arrays["n_dec"] = np.array([len(dec)], dtype=np.int32)
  for i, (w, b) in enumerate(enc):
    arrays[f"enc_w{i}"] = np.asarray(w, dtype=np.float32)
    arrays[f"enc_b{i}"] = np.asarray(b, dtype=np.float32)
  for i, (w, b) in enumerate(dec):
    arrays[f"dec_w{i}"] = np.asarray(w, dtype=np.float32)
    arrays[f"dec_b{i}"] = np.asarray(b, dtype=np.float32)
  if "anchor_dec" in actor_tree:
    arrays["anchor_dec"] = np.asarray(actor_tree["anchor_dec"], dtype=np.float32)
  np.savez(path, **arrays)


class _CompiledBucket:
  """All JIT-compiled JAX/NablaEngine functions & shapes for one `(bucket, arch)` pair."""

  def __init__(
    self,
    *,
    bucket: int,
    arch: str,
    mj_model: mujoco.MjModel,
    vram_gb: float,
    engine_pool: dict[int, Any],
    term_height: float = 0.30,
    term_tilt: float = 0.80,
    solver_iterations: int = 10,
  ) -> None:
    from nablaengine import jax_ffi
    from nablamimic import nets, sim
    from nablamimic.env import BM_BODIES, GOAL_ANCHOR, EnvConfig, TrackingEnv, W, _gravity_b
    from nablamimic.motion import MotionLibrary
    from nablamimic.nets import actor_apply, critic_apply, normalize
    from nablamimic.sim import BIN, select_reset, vmapped
    from nablamimic.train import make_rollout

    self.bucket = bucket
    self.arch = arch
    self.hp = _tune_bucket_hp(bucket, vram_gb, arch)
    self.n = int(self.hp["num_envs"])
    self.nc = int(self.hp["collect_envs"])
    self.horizon = int(self.hp["horizon"])
    self.replay_steps = int(self.hp["replay_steps"])
    self.at_batch = int(self.hp["at_batch"])
    self.gamma = 0.99
    self.lam = 0.95
    self.max_ep = max(500, bucket + 16)
    self.n_bins = max(1, (bucket - 1 + BIN - 1) // BIN)
    self.collect_noise = 0.05
    self.act_noise = 0.03
    self.term_height = term_height
    self.term_tilt = term_tilt

    # Dummy standing motion of shape `bucket` strictly for one-time JIT tracing.
    q_dummy = np.zeros((bucket, mj_model.nq), dtype=np.float32)
    q_dummy[:, 2] = 0.793
    q_dummy[:, 3] = 1.0
    v_dummy = np.zeros((bucket, mj_model.nv), dtype=np.float32)
    bp, bq, bl, ba = compute_full_body_kinematics(
      mj_model, q_dummy[:1], v_dummy[:1], body_names=BM_BODIES
    )
    dummy_lib = MotionLibrary(
      qpos=q_dummy,
      qvel=v_dummy,
      body_pos=np.repeat(bp, bucket, axis=0),
      last=np.full(bucket, bucket - 1, dtype=np.int32),
      names=["dummy"],
      starts=np.array([0], dtype=np.int32),
      fps=50.0,
      bodies=tuple(BM_BODIES),
      body_quat=np.repeat(bq, bucket, axis=0),
      body_lin_vel=np.repeat(bl, bucket, axis=0),
      body_ang_vel=np.repeat(ba, bucket, axis=0),
    )

    cfg = EnvConfig(
      solver_iterations=solver_iterations,
      dr=False,
      obs_noise=False,
      reset_noise=False,
      term_height=term_height,
      term_tilt=term_tilt,
      residual_action=True,
      action_rate_space="setpoint",
    )
    self.train_env = TrackingEnv(dummy_lib, cfg)
    self.eval_env = TrackingEnv(dummy_lib, cfg)
    self.g_max = float(self.train_env.max_reward / (1.0 - self.gamma))

    # Share NablaEngine handles across buckets!
    need_sizes = [s for s in sorted({self.n, self.nc, 2}) if s not in engine_pool]
    if need_sizes:
      new_eng = sim.make_engine(self.train_env, need_sizes)
      engine_pool.update(new_eng)
    self.train_env.warp = {self.n: engine_pool[self.n], self.nc: engine_pool[self.nc]}
    self.eval_env.warp = {2: engine_pool[2]}

    self.k_anc = GOAL_ANCHOR.stop - GOAL_ANCHOR.start
    self.actor_hidden = tuple(self.hp["actor_hidden"])
    self.critic_hidden = tuple(self.hp["critic_hidden"])
    self.actor_latent = int(self.hp["actor_latent"])

    def _sched(peak_lr: float):
      return optax.warmup_cosine_decay_schedule(
        init_value=peak_lr * 0.1,
        peak_value=peak_lr,
        warmup_steps=int(self.hp["lr_warmup_steps"]),
        decay_steps=int(self.hp["lr_decay_steps"]),
        end_value=peak_lr * float(self.hp["lr_decay_to"]),
      )

    self.opt_c = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adam(_sched(float(self.hp["lr"])), b1=0.9, b2=0.95),
    )
    self.opt_r = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adam(_sched(float(self.hp["action_target_lr"])), eps=1e-8),
    )

    n = self.n
    nc = self.nc
    horizon = self.horizon
    replay_steps = self.replay_steps
    psize = replay_steps * nc
    at_batch = self.at_batch
    n_bins = self.n_bins
    train_env = self.train_env
    eval_env = self.eval_env
    gamma, lam, g_max, max_ep = self.gamma, self.lam, self.g_max, self.max_ep
    collect_noise, act_noise = self.collect_noise, self.act_noise

    def bin_weights(stat, bcount):
      w = jnp.where(bcount > 0, jnp.maximum(1.0 - stat, 0.05), 0.0)
      return w / jnp.maximum(w.sum(), 1e-8)

    def frame_probs(bq, aux_):
      bins_j, valid_j, bcount = aux_["bins"], aux_["valid"], aux_["bin_count"]
      per_frame = lambda w: jnp.where(  # noqa: E731
        valid_j > 0, w[bins_j] / jnp.maximum(bcount[bins_j], 1.0), 0.0
      )
      uni = per_frame(
        jnp.where(bcount > 0, 1.0, 0.0) / jnp.maximum(jnp.sum(bcount > 0), 1.0)
      )
      hard = per_frame(bin_weights(bq[0], bcount))
      hist = per_frame(bq[1] / jnp.maximum(bq[1].sum(), 1e-8))
      return 0.45 * uni + 0.45 * hard + 0.10 * hist

    self.frame_probs_jit = jax.jit(frame_probs)

    def sample_resets(k, probs_, tables_, size=n):
      k1, k2 = jax.random.split(k)
      frames = jax.random.choice(k1, bucket, (size,), p=probs_)
      return vmapped(train_env, tables_)["reset"](jax.random.split(k2, size), frames)

    self.init_collector_jit = jax.jit(
      lambda k, p_, t_: sample_resets(k, p_, t_, nc)
    )

    # Probe STATE_SKIP once globally if not set yet.
    tables_d = train_env.tables()
    aux_d = self.make_aux(bucket)
    bq_d = jnp.stack([jnp.ones(n_bins), jnp.ones(n_bins) / n_bins])
    probs_d = self.frame_probs_jit(bq_d, aux_d)
    if sim.STATE_SKIP is None:
      e_b = train_env.bind(tables_d)

      def _probe(k):
        k1, k2, k3 = jax.random.split(k, 3)
        s_r = sample_resets(k1, probs_d, tables_d, 128)
        q = s_r.dx.qpos + 0.01 * jax.random.normal(k2, s_r.dx.qpos.shape)
        v = s_r.dx.qvel + 0.1 * jax.random.normal(k3, s_r.dx.qvel.shape)
        s_a = jax.vmap(e_b.finish_physics)(
          s_r, jnp.zeros((128, train_env.nu)), q, v, s_r.dx.qacc_warmstart
        )
        return s_r, s_a, jax.vmap(e_b.restore)(train_env.compact(s_r))

      s_r, s_a, s_c = jax.jit(_probe)(jax.random.PRNGKey(21))
      sim.STATE_SKIP = sim.skip_mask(s_r, s_a, s_c)
      del s_r, s_a, s_c

    rollout = make_rollout(
      train_env,
      horizon,
      gamma,
      max_ep,
      remat=False,
      act_clip=3.0,
      act_noise=act_noise,
      mask_after_done=False,
      term_penalty=(2.0, 0.08, 0.10),
      record_pre=True,
      record_obs=True,
      enc_hoist=True,
      remat_policy=jax_ffi.remat_policy(),
      v_bound=float(g_max),
      clip_end_pre=True,
    )

    @jax.jit(donate_argnums=(2, 3))
    def collect_step(actor_, cstate_, pool_, ptr_, probs_, k_, tables_):
      f = vmapped(train_env, tables_)
      k1, k2, k3 = jax.random.split(k_, 3)
      hist, goal = f["policy_obs"](cstate_, jax.random.split(k1, nc))
      a = actor_apply(actor_, hist, goal) + collect_noise * jax.random.normal(
        k2, (nc, train_env.nu)
      )
      s2 = f["step"](cstate_, a)
      term = f["terminated"](s2)
      done = term | f["clip_ended"](s2) | (s2.ep >= max_ep)
      s_next = select_reset(done, sample_resets(k3, probs_, tables_, nc), s2)
      pos = (ptr_[0] + jnp.arange(nc)) % psize
      row = train_env.compact(s_next)
      pool_ = jax.tree_util.tree_map(lambda p, v: p.at[pos].set(v), pool_, row)
      ptr_ = jnp.stack([(ptr_[0] + nc) % psize, jnp.minimum(ptr_[1] + nc, psize)])
      return s_next, pool_, ptr_, jnp.mean(term)

    k_win = int(self.hp["action_target_k"])
    m_steps = int(self.hp["action_target_m"])
    delta_floor = float(self.hp["action_target_delta_floor"])
    decay_steps = int(self.hp["lr_decay_steps"])
    opt_c, opt_r = self.opt_c, self.opt_r

    def _action_target_update(actor_, buf_, gd, outs, k_, delta0_):
      frac = jnp.minimum(buf_["it"].astype(jnp.float32) / float(decay_steps), 1.0)
      delta = delta0_ * (
        delta_floor + (1.0 - delta_floor) * 0.5 * (1.0 + jnp.cos(jnp.pi * frac))
      )
      g = jnp.where(jnp.isfinite(-gd), -gd, 0.0)
      nrm = jnp.sqrt(jnp.sum(g**2, axis=-1))
      live = nrm > 0
      thr = 3.0 * jnp.nanmedian(jnp.where(live, nrm, jnp.nan))
      thr = jnp.where(jnp.isfinite(thr), thr, jnp.inf)
      g = g * jnp.minimum(1.0, thr / (nrm + 1e-30))[..., None]
      s2 = train_env.action_scale**2
      ss = jnp.sum(jnp.where(live[..., None], g**2 * s2, 0.0))
      cnt = jnp.maximum(jnp.sum(live), 1).astype(g.dtype) * train_env.nu
      eta = delta / jnp.sqrt(ss / cnt + 1e-30)
      tgt = outs.pop("a_pi") + eta * g
      j = buf_["count"] % k_win
      buf_ = {
        **buf_,
        "hist": buf_["hist"].at[j].set(outs.pop("hist")),
        "goal": buf_["goal"].at[j].set(outs.pop("goal")),
        "tgt": buf_["tgt"].at[j].set(tgt),
        "w": buf_["w"].at[j].set(live.astype(g.dtype)),
        "count": buf_["count"] + 1,
        "it": buf_["it"] + 1,
      }
      nfill = jnp.minimum(buf_["count"], k_win)
      w_err = train_env.action_scale

      def reg(carry, kk):
        p, st = carry
        ka, kb, kc = jax.random.split(kk, 3)
        si = jax.random.randint(ka, (at_batch,), 0, nfill)
        ti = jax.random.randint(kb, (at_batch,), 0, horizon)
        ei = jax.random.randint(kc, (at_batch,), 0, n)
        hb = buf_["hist"][si, ti, ei].astype(jnp.float32)
        gb = buf_["goal"][si, ti, ei].astype(jnp.float32)
        yb = buf_["tgt"][si, ti, ei]
        wb = buf_["w"][si, ti, ei]

        def lf(q):
          err = (actor_apply(q, hb, gb) - yb) * w_err
          return jnp.sum(wb * jnp.mean(err**2, axis=-1)) / jnp.maximum(jnp.sum(wb), 1.0)

        lb, gr = jax.value_and_grad(lf)(p)
        u, st = opt_r.update(gr, st, p)
        return (optax.apply_updates(p, u), st), lb

      (actor_, st_r), ls = jax.lax.scan(
        reg, (actor_, buf_["opt"]), jax.random.split(k_, m_steps)
      )
      return actor_, {**buf_, "opt": st_r}, jnp.mean(ls)

    def _train_step_fn(
      actor_,
      critic_,
      st_c_,
      cnorm_,
      bin_q_,
      pool_,
      ptr_,
      probs_,
      k_,
      tables_,
      aux_,
      buf_,
      delta0_,
    ):
      k1, k2, k3, k4, k5, k6 = jax.random.split(k_, 6)
      reset_state = jax.lax.stop_gradient(sample_resets(k1, probs_, tables_, n))
      pick = jax.random.randint(k3, (n,), 0, ptr_[1])
      start = jax.vmap(train_env.bind(tables_).restore)(
        jax.tree_util.tree_map(lambda x: x[pick], pool_)
      )
      start = start._replace(key=jax.random.split(k4, n))
      (loss, (_, outs)), (gd, _) = jax.value_and_grad(
        rollout, argnums=(8, 7), has_aux=True
      )(
        actor_,
        critic_,
        cnorm_,
        start,
        reset_state,
        k2,
        tables_,
        jnp.zeros(2),
        jnp.zeros((horizon, n, train_env.nu)),
      )
      actor_, buf_, reg_loss = _action_target_update(
        actor_, buf_, gd, outs, k5, delta0_
      )
      tgt = nets.td_lambda(
        outs["r"], outs["term"], outs["done"], outs["v2"], gamma, lam
      )
      x = normalize(cnorm_, outs["cobs_pre"].reshape(-1, train_env.critic_dim))
      y = tgt.reshape(-1)
      m_tot = x.shape[0]
      mb = m_tot // 4

      def _c_loss(c, xb, yb):
        return jnp.mean((critic_apply(c, xb)[:, 0] - yb) ** 2)

      def _c_epoch(carry, ke):
        c, s = carry
        perm = jax.random.permutation(ke, m_tot)

        def _mini(cc, i):
          c_, s_ = cc
          idx = jax.lax.dynamic_slice(perm, (i * mb,), (mb,))
          lb, gc = jax.value_and_grad(_c_loss)(c_, x[idx], y[idx])
          u, s_ = opt_c.update(gc, s_, c_)
          return (optax.apply_updates(c_, u), s_), lb

        (c, s), ls = jax.lax.scan(_mini, (c, s), jnp.arange(4))
        return (c, s), jnp.mean(ls)

      (critic_, st_c_), closs_seq = jax.lax.scan(
        _c_epoch, (critic_, st_c_), jax.random.split(k6, 5)
      )
      xc = outs["cobs"].reshape(-1, train_env.critic_dim)
      mean_c, sq_c = xc.mean(0), (xc**2).mean(0)
      m0, v0 = cnorm_
      cnorm_ = (
        0.99 * m0 + 0.01 * mean_c,
        0.99 * v0 + 0.01 * jnp.maximum(sq_c - mean_c**2, 1e-6),
      )
      q_bin = tgt / g_max
      b_idx = aux_["bins"][outs["frame"].reshape(-1)]
      s_b = jnp.zeros(n_bins).at[b_idx].add(q_bin.reshape(-1))
      c_b = jnp.zeros(n_bins).at[b_idx].add(1.0)
      stat = jnp.where(
        c_b > 0, 0.9 * bin_q_[0] + 0.1 * s_b / jnp.maximum(c_b, 1.0), bin_q_[0]
      )
      past = 0.995 * bin_q_[1] + 0.005 * bin_weights(stat, aux_["bin_count"])
      bin_q_ = jnp.stack([stat, past])
      probs_ = frame_probs(bin_q_, aux_)
      summary = {
        "loss": loss,
        "reg_loss": reg_loss,
        "closs": jnp.mean(closs_seq),
        "rew_mean": jnp.mean(outs["r"]) / train_env.max_reward,
        "term_rate": jnp.mean(outs["term"]),
        "min_bin_q": jnp.min(jnp.where(aux_["bin_count"] > 0, stat, 1.0)),
      }
      return actor_, critic_, st_c_, cnorm_, bin_q_, probs_, buf_, summary

    self.collect_step = collect_step
    self.train_step = jax.jit(_train_step_fn, donate_argnums=(11,))
    self.ema_step = jax.jit(
      lambda e, a, d: jax.tree_util.tree_map(
        lambda x, y: (d * x + (1.0 - d) * y).astype(x.dtype)
        if jnp.issubdtype(x.dtype, jnp.floating)
        else y,
        e,
        a,
      )
    )

    # Zero-recompile dual full-clip eval over `bucket - 1` steps with dynamic `n_steps_real`!
    terms_order = tuple(W)
    n_scan = bucket - 1

    @jax.jit
    def dual_eval(actor_a, actor_b, tables_, n_steps_real):
      f = vmapped(eval_env, tables_)
      e = eval_env.bind(tables_)
      k_rst, k_step = jax.random.split(jax.random.PRNGKey(1))
      state0 = f["reset"](jax.random.split(k_rst, 2), jnp.zeros((2,), dtype=jnp.int32))
      q0_rec = state0.dx.qpos
      v0_rec = state0.dx.qvel

      def body(carry, xs):
        t, k = xs
        (
          state,
          alive,
          fall_t,
          cause,
          tsum,
          mpjpe_sum,
          mpjpe_max,
          root_sum,
          jmae_sum,
          ar_sq_sum,
        ) = carry
        hist, goal = f["policy_obs"](state, jax.random.split(k, 2))
        act = jnp.stack(
          [actor_apply(actor_a, hist[0], goal[0]), actor_apply(actor_b, hist[1], goal[1])],
          axis=0,
        )
        ctrl = jax.vmap(e.ctrl_of)(act, state.frame)
        s2 = f["step"](state, act)
        _, terms = f["reward"](s2, act, state.prev_a, prev=state)
        tv = jnp.stack([terms[nm] for nm in terms_order], axis=-1)

        def _err_one(dx, fr):
          pos, _, _, _, rel_p, _, _, _, ra_p, _, ma_p = e._track(dx, fr)
          b_err = jnp.linalg.norm(rel_p - pos, axis=-1)
          return jnp.mean(b_err), jnp.max(b_err), jnp.linalg.norm(ma_p - ra_p)

        b_mean, b_max, r_err = jax.vmap(_err_one)(s2.dx, s2.frame)
        rq = e.ref_qpos[s2.frame]
        q = s2.dx.qpos
        dz = jnp.abs(q[:, 2] - rq[:, 2])
        dg = jnp.abs(
          jax.vmap(_gravity_b)(q[:, 3:7])[:, 2] - jax.vmap(_gravity_b)(rq[:, 3:7])[:, 2]
        )
        dj_abs = jnp.mean(jnp.abs(q[:, 7:] - rq[:, 7:]), axis=-1)
        da = jax.vmap(e.action_change)(act, state.prev_a, state)
        ar_sq = jnp.mean(da**2, axis=-1)

        in_range = t < n_steps_real
        active = alive & in_range
        term = f["terminated"](s2) & in_range
        am = active.astype(jnp.float32)
        tsum = tsum + active[:, None] * tv
        mpjpe_sum = mpjpe_sum + am * b_mean
        mpjpe_max = jnp.maximum(mpjpe_max, am * b_max)
        root_sum = root_sum + am * r_err
        jmae_sum = jmae_sum + am * dj_abs
        ar_sq_sum = ar_sq_sum + am * ar_sq

        newfall = active & term
        fall_t = jnp.where(newfall, t + 1, fall_t)
        cause = jnp.where(
          newfall,
          (dz > eval_env.cfg.term_height).astype(jnp.int32) * 1
          + (dg > eval_env.cfg.term_tilt).astype(jnp.int32) * 2,
          cause,
        )
        alive = alive & ~term & (t + 1 < n_steps_real)
        return (
          s2,
          alive,
          fall_t,
          cause,
          tsum,
          mpjpe_sum,
          mpjpe_max,
          root_sum,
          jmae_sum,
          ar_sq_sum,
        ), (s2.dx.qpos, s2.dx.qvel, ctrl, b_mean * 1000.0)

      init = (
        state0,
        jnp.ones((2,), dtype=jnp.bool_),
        jnp.full((2,), -1, dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((2, len(terms_order)), dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.zeros((2,), dtype=jnp.float32),
      )
      (
        _,
        _,
        fall_t,
        cause,
        tsum,
        mpjpe_sum,
        mpjpe_max,
        root_sum,
        jmae_sum,
        ar_sq_sum,
      ), (q_seq, v_seq, c_seq, b_seq) = jax.lax.scan(
        body, init, (jnp.arange(n_scan), jax.random.split(k_step, n_scan))
      )
      q_full = jnp.concatenate([q0_rec[None, :, :], q_seq], axis=0).swapaxes(0, 1)
      v_full = jnp.concatenate([v0_rec[None, :, :], v_seq], axis=0).swapaxes(0, 1)
      return dict(
        fall_t=fall_t,
        cause=cause,
        tsum=tsum,
        mpjpe_sum=mpjpe_sum,
        mpjpe_max=mpjpe_max,
        root_sum=root_sum,
        jmae_sum=jmae_sum,
        ar_sq_sum=ar_sq_sum,
        qpos=q_full,
        qvel=v_full,
        ctrl=c_seq.swapaxes(0, 1),
        mpjpe_per_frame_mm=b_seq.swapaxes(0, 1),
      )

    self.dual_eval = dual_eval

    # Force immediate one-time JIT compilation of all 3 kernels right now during preload!
    key = jax.random.PRNGKey(0)
    actor0, critic0, st_c0, cnorm0, buf0 = self.fresh_params(key)
    cs0, pool0, ptr0 = self.fresh_collector(key, probs_d, tables_d)
    cs0, pool0, ptr0, _ = self.collect_step(
      actor0, cs0, pool0, ptr0, probs_d, key, tables_d
    )
    jax.block_until_ready(ptr0)
    eval0 = self.dual_eval(actor0, actor0, tables_d, jnp.int32(bucket - 1))
    jax.block_until_ready(eval0["fall_t"])
    _, _, _, _, _, _, _, s0 = self.train_step(
      actor0,
      critic0,
      st_c0,
      cnorm0,
      bq_d,
      pool0,
      ptr0,
      probs_d,
      key,
      tables_d,
      aux_d,
      buf0,
      jnp.float32(self.hp["action_target_delta"]),
    )
    jax.block_until_ready(s0["loss"])

  def make_aux(self, n_real: int) -> dict[str, Any]:
    from nablamimic.sim import BIN

    bins_np = np.minimum(np.arange(self.bucket, dtype=np.int32) // BIN, self.n_bins - 1)
    valid_np = (np.arange(self.bucket, dtype=np.int32) < (n_real - 1)).astype(np.float32)
    bin_count_np = np.bincount(bins_np, weights=valid_np, minlength=self.n_bins).astype(
      np.float32
    )
    return {
      "bins": jnp.asarray(bins_np),
      "valid": jnp.asarray(valid_np),
      "bin_count": jnp.asarray(bin_count_np),
    }

  def fresh_params(self, key: Any):
    from nablamimic.nets import actor_init, mlp_init

    ka, kc = jax.random.split(key)
    actor = actor_init(
      ka,
      self.train_env,
      hidden=self.actor_hidden,
      latent=self.actor_latent,
      anchor_to_decoder=self.k_anc,
    )
    critic = mlp_init(kc, [self.train_env.critic_dim, *self.critic_hidden, 1])
    st_c = self.opt_c.init(critic)
    cnorm = (
      jnp.zeros(self.train_env.critic_dim, jnp.float32),
      jnp.ones(self.train_env.critic_dim, jnp.float32),
    )
    k_win = int(self.hp["action_target_k"])
    buf: dict[str, Any] = {
      "hist": jnp.zeros((k_win, self.horizon, self.n, self.train_env.hist_dim), jnp.bfloat16),
      "goal": jnp.zeros((k_win, self.horizon, self.n, self.train_env.goal_dim), jnp.bfloat16),
      "tgt": jnp.zeros((k_win, self.horizon, self.n, self.train_env.nu), jnp.float32),
      "w": jnp.zeros((k_win, self.horizon, self.n), jnp.float32),
      "count": jnp.int32(0),
      "it": jnp.int32(0),
      "opt": self.opt_r.init(actor),
    }
    return actor, critic, st_c, cnorm, buf

  def fresh_collector(self, key: Any, probs: Any, tables: dict[str, Any]):
    k1, k2 = jax.random.split(key)
    cstate = self.init_collector_jit(k1, probs, tables)
    cstate = cstate._replace(
      ep=jax.random.randint(k2, (self.nc,), 0, self.max_ep).astype(cstate.ep.dtype)
    )
    pool = jax.tree_util.tree_map(
      lambda x: jnp.tile(x, (self.replay_steps,) + (1,) * (x.ndim - 1)),
      self.train_env.compact(cstate),
    )
    psize = self.replay_steps * self.nc
    ptr = jnp.array([self.nc % psize, self.nc], dtype=jnp.int32)
    return cstate, pool, ptr


class NablaDaemon:
  """Long-lived GPU daemon holding preloaded NablaEngine + compiled XLA buckets + warm weights."""

  def __init__(
    self,
    *,
    preload_buckets: list[int],
    arch: str = "lite",
    matmul_precision: str = "highest",
  ) -> None:
    from nablamimic import jax_cache, nets
    from nablamimic.env import EnvConfig, build_model

    t0 = time.time()
    emit({"type": "PRELOAD_STATE", "state": "booting", "detail": "Initializing JAX + NablaEngine..."})
    jax_cache.enable()
    nets.set_matmul_precision(matmul_precision)
    self.arch = arch
    self.matmul_precision = matmul_precision
    self.vram_gb = _detect_vram_gb()
    self.gpu_name = str(jax.devices()[0].device_kind)
    self.mj_model, _ = build_model(EnvConfig(solver_iterations=10, dr=False, obs_noise=False, reset_noise=False))
    self.engine_pool: dict[int, Any] = {}
    self.buckets: dict[tuple[int, str], _CompiledBucket] = {}
    self.warm_cache: dict[str, tuple[Any, Any, Any]] = {}  # arch -> (actor, critic, cnorm)

    for b in preload_buckets:
      tb = time.time()
      emit({
        "type": "PRELOAD_STATE",
        "state": "compiling",
        "detail": f"Pre-compiling GPU bucket {b} ({self.gpu_name}, {matmul_precision})...",
      })
      self.get_or_compile_bucket(b, arch)
      emit({
        "type": "PRELOAD_STATE",
        "state": "bucket_ready",
        "bucket": b,
        "elapsed_s": round(time.time() - tb, 2),
        "detail": f"Bucket {b} hot in GPU RAM ({time.time() - tb:.1f}s)",
      })

    emit({
      "type": "PRELOAD_STATE",
      "state": "ready",
      "gpu": self.gpu_name,
      "vram_gb": round(self.vram_gb, 2),
      "preloaded_buckets": [k[0] for k in self.buckets],
      "matmul_precision": self.matmul_precision,
      "boot_s": round(time.time() - t0, 2),
      "detail": f"Ready on {self.gpu_name} (buckets {[k[0] for k in self.buckets]} hot in RAM)",
    })

  def get_or_compile_bucket(self, bucket: int, arch: str) -> tuple[_CompiledBucket, float]:
    key = (bucket, arch)
    if key in self.buckets:
      return self.buckets[key], 0.0
    t0 = time.time()
    cb = _CompiledBucket(
      bucket=bucket,
      arch=arch,
      mj_model=self.mj_model,
      vram_gb=self.vram_gb,
      engine_pool=self.engine_pool,
    )
    self.buckets[key] = cb
    return cb, time.time() - t0

  def _build_tables(self, cb: _CompiledBucket, qpos: np.ndarray, qvel: np.ndarray, name: str):
    from nablamimic.env import BM_BODIES
    from nablamimic.motion import MotionLibrary

    n_real = len(qpos)
    bucket = cb.bucket
    bp, bq, bl, ba = compute_full_body_kinematics(self.mj_model, qpos, qvel, body_names=BM_BODIES)

    def _pad(arr: np.ndarray) -> np.ndarray:
      if len(arr) == bucket:
        return arr.astype(np.float32)
      tail = np.repeat(arr[-1:], bucket - len(arr), axis=0)
      return np.concatenate([arr, tail], axis=0).astype(np.float32)

    lib = MotionLibrary(
      qpos=_pad(qpos),
      qvel=_pad(qvel),
      body_pos=_pad(bp),
      last=np.full(bucket, n_real - 1, dtype=np.int32),
      names=[name],
      starts=np.array([0], dtype=np.int32),
      fps=50.0,
      bodies=tuple(BM_BODIES),
      body_quat=_pad(bq),
      body_lin_vel=_pad(bl),
      body_ang_vel=_pad(ba),
    )
    cb.train_env.set_library(lib)
    return cb.train_env.tables()

  def optimize(self, req: dict[str, Any], cancel_event: threading.Event) -> None:
    job_id = str(req["job_id"])
    t_job0 = time.time()
    try:
      inp = np.load(req["input_npz"])
      qpos = np.asarray(inp["qpos"], dtype=np.float32)
      qvel = np.asarray(inp["qvel"], dtype=np.float32)
      clip_name = str(req.get("clip_name", "motion"))
      arch = str(req.get("arch", self.arch))
      preset = str(req.get("preset", "standard"))
      warm_start = bool(req.get("warm_start", False))
      seed = int(req.get("seed", 0))
      out_dir = Path(req["out_dir"]).resolve()
      out_dir.mkdir(parents=True, exist_ok=True)

      n_real = len(qpos)
      bucket = _choose_bucket(n_real)
      conv_cfg = ConvergenceConfig.from_preset(
        preset,  # type: ignore[arg-type]
        max_seconds=req.get("max_seconds"),
        max_iters=req.get("max_iters"),
        target_return=req.get("target_return"),
        target_mpjpe_mm=req.get("target_mpjpe_mm"),
        fine_eval_every=8 if n_real >= 500 else None,
      )
      monitor = ConvergenceMonitor(cfg=conv_cfg)
      prog = ProgressEstimator(conv_cfg)

      emit({
        "type": "PROGRESS",
        "job_id": job_id,
        **prog.stage_event("SANITIZE", 4.0, time.time() - t_job0, f"Loaded {n_real} frames (bucket {bucket})").to_dict(),
      })

      if (bucket, arch) not in self.buckets:
        emit({
          "type": "PROGRESS",
          "job_id": job_id,
          **prog.stage_event("COMPILE", 7.0, time.time() - t_job0, f"Compiling new bucket {bucket} (cached for future runs)...").to_dict(),
        })
      cb, compile_s = self.get_or_compile_bucket(bucket, arch)

      t_tbl = time.time()
      tables = self._build_tables(cb, qpos, qvel, clip_name)
      aux = cb.make_aux(n_real)
      bin_q = jnp.stack([jnp.ones(cb.n_bins), jnp.ones(cb.n_bins) / cb.n_bins])
      probs = cb.frame_probs_jit(bin_q, aux)

      key = jax.random.PRNGKey(seed)
      key, kp, kc = jax.random.split(key, 3)
      actor, critic, st_c, cnorm, buf = cb.fresh_params(kp)
      used_warm = False
      delta0 = float(cb.hp["action_target_delta"])
      if warm_start and arch in self.warm_cache:
        w_actor, w_critic, w_cnorm = self.warm_cache[arch]
        actor = jax.tree_util.tree_map(jnp.asarray, w_actor)
        critic = jax.tree_util.tree_map(jnp.asarray, w_critic)
        cnorm = jax.tree_util.tree_map(jnp.asarray, w_cnorm)
        st_c = cb.opt_c.init(critic)
        buf["opt"] = cb.opt_r.init(actor)
        delta0 = 0.022
        used_warm = True

      cstate, pool, ptr = cb.fresh_collector(kc, probs, tables)
      actor_ema = actor
      n_steps_real_j = jnp.int32(n_real - 1)
      delta0_j = jnp.float32(delta0)
      table_load_s = time.time() - t_tbl

      emit({
        "type": "PROGRESS",
        "job_id": job_id,
        **prog.stage_event(
          "BALANCE",
          12.0,
          time.time() - t_job0,
          f"Bucket {bucket} hot (compile={compile_s:.1f}s, warm={used_warm}) — optimizing...",
        ).to_dict(),
      })

      t_opt0 = time.time()
      col_acc = 0.0
      total_eval_s = 0.0
      best_payload: dict[str, Any] | None = None
      best_actor_host: dict[str, Any] | None = None
      stop_reason = "MAX_ITERS"
      cause_map = {0: "nonfinite", 1: "height", 2: "tilt", 3: "both"}
      full_len = float(n_real - 1)

      for it in range(1, conv_cfg.max_iters + 1):
        if cancel_event.is_set():
          stop_reason = "CANCELLED"
          break

        key, k_col, k_tr = jax.random.split(key, 3)
        col_acc += 1.0 / float(cb.hp["utd"])
        n_col = int(col_acc)
        for j in range(n_col):
          cstate, pool, ptr, _ = cb.collect_step(
            actor, cstate, pool, ptr, probs, jax.random.fold_in(k_col, j), tables
          )
        col_acc -= n_col

        actor, critic, st_c, cnorm, bin_q, probs, buf, summ = cb.train_step(
          actor, critic, st_c, cnorm, bin_q, pool, ptr, probs, k_tr, tables, aux, buf, delta0_j
        )
        d_ema = min(float(cb.hp["actor_ema"]), (1.0 + it) / (10.0 + it))
        actor_ema = cb.ema_step(actor_ema, actor, jnp.float32(d_ema))
        jax.block_until_ready(summ["loss"])

        wall_s = time.time() - t_opt0
        loss_val = float(summ["loss"])
        if not np.isfinite(loss_val):
          stop_reason = "DIVERGED"
          break
        term_rate = float(summ["term_rate"])
        rew_mean = float(summ["rew_mean"])
        min_bin_q = float(summ["min_bin_q"])

        do_eval = monitor.should_eval(it, wall_s, train_term_rate=term_rate)
        if not do_eval and it % 5 == 0:
          snap = prog.on_train_beat(
            iteration=it,
            wall_s=wall_s,
            train_term_rate=term_rate,
            train_rew_mean=rew_mean,
            min_bin_q=min_bin_q,
          )
          emit({"type": "PROGRESS", "job_id": job_id, **snap.to_dict()})

        if do_eval:
          t_e0 = time.time()
          raw_eval = jax.device_get(cb.dual_eval(actor, actor_ema, tables, n_steps_real_j))
          eval_ms = (time.time() - t_e0) * 1000.0
          total_eval_s += eval_ms / 1000.0

          def _mk(idx: int, cand: str) -> EvalMetrics:
            fall_t = int(raw_eval["fall_t"][idx])
            failed = fall_t > 0
            nal = float(fall_t) if failed else full_len
            nal_s = max(nal, 1.0)
            tsum = float(np.asarray(raw_eval["tsum"][idx]).sum())
            return EvalMetrics(
              iteration=it,
              wall_s=round(wall_s, 3),
              candidate=cand,
              alive_frac=float(nal / full_len),
              failed=failed,
              fall_s=round(fall_t * 0.02, 2) if failed else None,
              fall_cause=cause_map.get(int(raw_eval["cause"][idx]), "unknown") if failed else None,
              full_norm_return=float(tsum / (full_len * cb.train_env.max_reward)),
              return_per_alive_step=float(tsum / (nal_s * cb.train_env.max_reward)),
              mpjpe_mm=float(raw_eval["mpjpe_sum"][idx] / nal_s * 1000.0),
              max_body_err_mm=float(raw_eval["mpjpe_max"][idx] * 1000.0),
              root_pos_err_mm=float(raw_eval["root_sum"][idx] / nal_s * 1000.0),
              joint_mae_rad=float(raw_eval["jmae_sum"][idx] / nal_s),
              action_rate_rms=float(np.sqrt(float(raw_eval["ar_sq_sum"][idx]) / nal_s)),
              min_bin_return=min_bin_q,
              eval_ms=round(eval_ms, 2),
            )

          m_raw = _mk(0, "actor")
          m_ema = _mk(1, "actor_ema")
          if m_ema.rank_key() > m_raw.rank_key():
            c_idx, chosen_m, chosen_tree = 1, m_ema, actor_ema
          else:
            c_idx, chosen_m, chosen_tree = 0, m_raw, actor

          should_stop, reason, is_new_best = monitor.update(chosen_m)
          if is_new_best:
            best_payload = {
              "qpos": np.asarray(raw_eval["qpos"][c_idx, :n_real], dtype=np.float32),
              "qvel": np.asarray(raw_eval["qvel"][c_idx, :n_real], dtype=np.float32),
              "ctrl": np.asarray(raw_eval["ctrl"][c_idx, : n_real - 1], dtype=np.float32),
              "mpjpe_per_frame_mm": np.asarray(
                raw_eval["mpjpe_per_frame_mm"][c_idx, : n_real - 1], dtype=np.float32
              ),
            }
            best_actor_host = jax.device_get(chosen_tree)

          snap = prog.on_eval(
            chosen_m,
            confirm_count=monitor.confirm_count,
            stale_feasible_evals=monitor.stale_feasible_evals,
            decision=reason,
          )
          emit({"type": "PROGRESS", "job_id": job_id, **snap.to_dict()})
          if should_stop:
            stop_reason = reason
            break

      opt_wall_s = time.time() - t_opt0

      if best_payload is None or best_actor_host is None:
        raw_eval = jax.device_get(cb.dual_eval(actor, actor_ema, tables, n_steps_real_j))
        best_payload = {
          "qpos": np.asarray(raw_eval["qpos"][0, :n_real], dtype=np.float32),
          "qvel": np.asarray(raw_eval["qvel"][0, :n_real], dtype=np.float32),
          "ctrl": np.asarray(raw_eval["ctrl"][0, : n_real - 1], dtype=np.float32),
          "mpjpe_per_frame_mm": np.asarray(
            raw_eval["mpjpe_per_frame_mm"][0, : n_real - 1], dtype=np.float32
          ),
        }
        best_actor_host = jax.device_get(actor)

      # Cache warm-start weights for instant re-optimization after user edits!
      if monitor.best_metrics is not None and not monitor.best_metrics.failed:
        self.warm_cache[arch] = (
          best_actor_host,
          jax.device_get(critic),
          jax.device_get(cnorm),
        )

      res_npz = out_dir / "nabla_result.npz"
      actor_npz = out_dir / "actor_params.npz"
      np.savez(
        res_npz,
        qpos=best_payload["qpos"],
        qvel=best_payload["qvel"],
        ctrl=best_payload["ctrl"],
        ref_qpos=qpos,
        ref_qvel=qvel,
        mpjpe_per_frame_mm=best_payload["mpjpe_per_frame_mm"],
      )
      _save_actor_params_npz(actor_npz, best_actor_host)

      emit({
        "type": "RESULT",
        "job_id": job_id,
        "stop_reason": stop_reason,
        "used_warm_start": used_warm,
        "bucket": bucket,
        "gpu": self.gpu_name,
        "matmul_precision": self.matmul_precision,
        "nabla_result_npz": str(res_npz),
        "actor_params_npz": str(actor_npz),
        "timing": {
          "compile_s": round(compile_s, 2),
          "table_load_s": round(table_load_s, 3),
          "optimization_s": round(opt_wall_s, 2),
          "eval_overhead_s": round(total_eval_s, 2),
          "daemon_total_s": round(time.time() - t_job0, 2),
        },
        "best_metrics": monitor.best_metrics.to_dict() if monitor.best_metrics else None,
        "eval_history": monitor.history,
      })
    except Exception as exc:
      import traceback

      emit({
        "type": "ERROR",
        "job_id": job_id,
        "error": str(exc),
        "traceback": traceback.format_exc(),
      })


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser(description="TDOT Persistent Preloaded NablaEngine Daemon")
  ap.add_argument("--preload-buckets", type=int, nargs="*", default=[512])
  ap.add_argument("--arch", default="lite", choices=["lite", "medium", "supermimic"])
  ap.add_argument("--matmul-precision", default="highest", choices=["highest", "nets_tf32"])
  args = ap.parse_args(argv)

  daemon = NablaDaemon(
    preload_buckets=list(args.preload_buckets),
    arch=args.arch,
    matmul_precision=args.matmul_precision,
  )

  cmd_q: queue.Queue[dict[str, Any] | None] = queue.Queue()
  active_cancel: dict[str, threading.Event] = {}
  cancel_lock = threading.Lock()

  def _stdin_reader() -> None:
    for raw_line in sys.stdin:
      line = raw_line.strip()
      if not line:
        continue
      try:
        msg = json.loads(line)
      except Exception:
        continue
      cmd = msg.get("cmd")
      if cmd == "CANCEL":
        jid = str(msg.get("job_id", ""))
        with cancel_lock:
          ev = active_cancel.get(jid)
          if ev is not None:
            ev.set()
      elif cmd == "PING":
        emit({"type": "PONG", "buckets": [k[0] for k in daemon.buckets]})
      elif cmd == "SHUTDOWN":
        cmd_q.put(None)
        return
      else:
        cmd_q.put(msg)
    cmd_q.put(None)

  threading.Thread(target=_stdin_reader, daemon=True).start()

  while True:
    req = cmd_q.get()
    if req is None:
      break
    cmd = req.get("cmd")
    if cmd == "OPTIMIZE":
      jid = str(req["job_id"])
      ev = threading.Event()
      with cancel_lock:
        active_cancel[jid] = ev
      try:
        daemon.optimize(req, ev)
      finally:
        with cancel_lock:
          active_cancel.pop(jid, None)
    elif cmd == "PRELOAD":
      b = int(req.get("bucket", 512))
      a = str(req.get("arch", daemon.arch))
      _, dt = daemon.get_or_compile_bucket(b, a)
      emit({"type": "PRELOAD_STATE", "state": "bucket_ready", "bucket": b, "elapsed_s": round(dt, 2)})


if __name__ == "__main__":
  main()
