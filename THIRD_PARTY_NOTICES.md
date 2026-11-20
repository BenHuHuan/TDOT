# Third-Party Notices & Attributions

TDOT builds upon and integrates the following open-source projects:

1. **robot-motion-editor** (`https://github.com/project-instinct/robot-motion-editor`)
   - Origin of the Three.js URDF 3D viewer, 2D interactive curve editor, and the Unitree G1 29-DoF URDF & STL visual assets under `static/unitree_g1/`.
2. **NablaMimic** (`https://github.com/sailab-ai/NablaMimic`) & **NablaEngine** (`https://github.com/OamIcs/NablaEngine`)
   - Differentiable physics / first-order analytic-gradient trajectory optimization backend (`NablaEngine` + `STATE_SKIP` + `SHAC/NablaMimic`).
3. **MLAB4_Fast** (`https://github.com/BenHuHuan/MLAB4_Fast`)
   - Strict IEEE FP32 MuJoCo 4.0 / `mujoco-warp` verification standards and unified `MotionCommand` `.npz` schema.
4. **MuJoCo Menagerie** (`https://github.com/google-deepmind/mujoco_menagerie`)
   - Unitree G1 MJCF physical model used for closed-loop dynamics verification.
5. **Three.js** & **urdf-loader** (loaded via CDN importmap in `templates/index.html`, MIT / Apache-2.0).
