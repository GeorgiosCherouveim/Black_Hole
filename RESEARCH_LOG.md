# Research Log - Black Hole Ray Tracing Thesis

## 2025-11-11
- **Environment Setup**: Created conda environment `thesis-bh` with Python 3.10
- **GPU**: Verified Taichi runs on RTX 3060 (CUDA backend)
- **Validation**: Generated first plot showing GR vs PN deflection angles
- **Result**: GPU kernel runs successfully, array operations working
- **Fix**: Taichi API changed; using `ti.lang.impl.current_cfg().arch` for backend detection
- **Next**: Implement Schwarzschild geodesic integrator (Phase 1, Week 1)