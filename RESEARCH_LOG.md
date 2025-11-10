# Research Log - Black Hole Ray Tracing Thesis

## 2025-11-11
- **Environment Setup**: Created conda environment `thesis-bh` with Python 3.10
- **GPU**: Verified Taichi runs on RTX 3060 (CUDA backend)
- **Validation**: Generated first plot showing GR vs PN deflection angles
- **Result**: GPU kernel runs successfully, array operations working
- **Fix**: Taichi API changed; using `ti.lang.impl.current_cfg().arch` for backend detection
- **Next**: Implement Schwarzschild geodesic integrator (Phase 1, Week 1)

## 2025-11-11 (continued)
- **GitHub**: Initialized repo, pushed first commit (hello_gpu.py, project structure)
- **Repo URL**: https://github.com/GeorgiosCherouveim/Black_Hole
- **Status**: Phase 0 complete. Environment ready for physics implementation.

## 2025-11-11 (Photon Deflection Validation)

**Task**: Validate GR photon deflection vs 2PN approximation (Phase 1, Week 1)

### Implementation
- **Exact Solution**: Schwarzschild null geodesics via elliptic integrals
  - Used `scipy.special.ellipk` for complete elliptic integral
  - Critical impact parameter: `b_crit = 3√3 M ≈ 5.196M`
  
- **2PN Approximation**: Weak-field expansion
  - Formula: `α = 4M/b + (15π/4)(M/b)²`

### Results at b = 10M
| Metric | Deflection Angle | Relative Error |
|--------|------------------|----------------|
| Exact (Elliptic) | **0.6399 rad (36.66°)** | — |
| 2PN Approximation | **0.5178 rad (29.67°)** | **19.08%** |

### Key Findings
- 2PN underestimates deflection in strong-field regime (b ≤ 20M)
- Error decreases rapidly for b &gt; 30M
- Validation script: `validation/deflection_test.py`
- Plot generated: `validation/deflection_test.png`

### Status
✅ Phase 1, Week 1 complete - Deflection validation successful
🔄 Next: Implement Schwarzschild geodesic integrator for full ray tracing