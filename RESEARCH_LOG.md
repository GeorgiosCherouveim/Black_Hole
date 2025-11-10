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

## 2025-11-11 (Geodesic Integrator Complete)

**Task**: Implement Schwarzschild geodesic RHS and RK4 integrator (Phase 1, Week 1)

### Implementation Details

**Kerr-Schild Cartesian Coordinates**:
- Avoids coordinate singularity at r=2M horizon
- Metric: g_μν = η_μν + (2M/r) l_μ l_ν where l_μ = (-1, x/r, y/r, z/r)
- State vector: [x, y, z, px, py, pz] with covariant momentum components

**Geodesic RHS (src/schwarzschild.py)**:
- Solves null constraint g^μν p_μ p_ν = 0 for photon energy p_0 at each step
- Quadratic: A p_0² + B p_0 + C = 0 with A = -1 + 2M/r, B = 4M(x·p)/r²
- Returns NaN for r &lt; 2.001M (horizon + safety buffer)
- Vectorized NumPy operations for efficiency

**RK4 Integrator (src/integrator.py)**:
- Adaptive step size: reduces dt by 0.5× on NaN detection
- Termination conditions: capture (r &lt; 2.01M), escape (r &gt; r_escape)
- Returns trajectory and status dictionary

### Test Results

**Test 1: Weak Field (b = 15M)**
- Status: ✅ **Escape** 
- Final radius: 144081.67M
- Steps: 265
- **Result**: As expected - minimal deflection

**Test 2: Strong Field (b = 5.2M)**
- Status: ✅ **Escape**
- Final radius: 81322748.498M
- **Result**: Photons escape even near critical impact parameter (b_crit ≈ 5.196M). This suggests the test impact parameter needs fine-tuning or the target vector precision should be improved for capture tests.

### Validation Files
- `validation/test_geodesics.py`: 3D momentum test (non-radial)
- `validation/test_integrator_basic.py`: Integration smoke tests

### Status
✅ **Phase 1, Week 1 Complete** - Core geodesic infrastructure operational
- Photon deflection validated against exact GR solution
- Geodesic equations implemented in regular coordinates
- Robust integrator with NaN handling

### Next Steps (Week 2)
- Fine-tune strong field test parameters for capture demonstration
- Implement full ray tracer with observer camera
- Add redshift calculation (frequency shift along geodesics)
- Begin accretion disk emission models