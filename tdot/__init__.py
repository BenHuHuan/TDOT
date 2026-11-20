"""TDOT — Turbo Dynamical Optimization Tool for Unitree G1 Motion Editing.

Combines:
- Interactive 3D web motion editor (based on project-instinct/robot-motion-editor)
- Preloaded persistent NablaEngine + NablaMimic + SuperMimic-lite differentiable TO daemon
- Adaptive two-stage ConvergenceMonitor + Physics-Driven ProgressEstimator
- Preloaded strict-IEEE-FP32 MuJoCo 4.0 closed-loop verifier & unified MLAB4_Fast/Editor .npz exporter
"""

from tdot.convergence import (
  PRESET_CONFIGS,
  ConvergenceConfig,
  ConvergenceMonitor,
  EvalMetrics,
)
from tdot.progress import ProgressEstimator, ProgressSnapshot

__version__ = "0.1.0"

__all__ = [
  "ConvergenceConfig",
  "ConvergenceMonitor",
  "EvalMetrics",
  "PRESET_CONFIGS",
  "ProgressEstimator",
  "ProgressSnapshot",
  "__version__",
]
