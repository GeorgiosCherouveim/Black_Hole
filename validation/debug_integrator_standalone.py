#!/usr/bin/env python3
"""
Standalone Diagnostic - Everything in one file, NO imports
"""

import numpy as np
from scipy.special import ellipk
from scipy.optimize import brentq

# === COPY schwarzschild_geodesic_rhs HERE ===
def schwarzschild_geodesic_rhs(state, M):
    x, y, z, px, py, pz = state
    r_vec = np.array([x, y, z], dtype=float)
    p_cov = np.array([px, py, pz], dtype=float)
    
    r = np.linalg.norm(r_vec)
    if r < 2.001 * M:
        return np.full(6, np.nan)
    
    r_inv = 1.0 / r
    r2_inv = r_inv**2
    r3_inv = r_inv**3
    
    xp = np.dot(r_vec, p_cov)
    
    # CRITICAL: Correct coefficients
    A = -1.0
    B = -4 * M * xp * r2_inv
    C = 1.0 + 2 * M * r_inv
    
    discriminant = B**2 - 4*A*C
    if discriminant < 0:
        return np.full(6, np.nan)
    
    sqrt_disc = np.sqrt(discriminant)
    p0_root1 = (-B + sqrt_disc) / (2*A)
    p0_root2 = (-B - sqrt_disc) / (2*A)
    p0 = p0_root1 if p0_root1 > 0 else p0_root2
    
    D = p0 * r2_inv + xp * r3_inv
    dx_dt = p_cov - 2 * M * r_vec * D
    
    E_term = p0**2 * r3_inv + 2 * p0 * xp * r2_inv**2 + xp**2 * r3_inv**2
    dp_dt = 2 * M * p_cov * D - M * r_vec * E_term
    
    return np.array([dx_dt[0], dx_dt[1], dx_dt[2], 
                     dp_dt[0], dp_dt[1], dp_dt[2]])

# === COPY integrate_geodesic HERE ===
def integrate_geodesic(initial_state, M, lambda_max=200.0, dt=0.1, r_escape=500.0):
    state = np.array(initial_state, dtype=float, copy=True)
    trajectory = []
    trajectory.append(np.array(state, copy=True))
    total_lambda = 0.0
    
    n_steps = int(lambda_max / dt)
    
    for step in range(n_steps):
        r = np.linalg.norm(state[:3])
        
        if r < 2.01 * M:
            return np.array(trajectory), {'reason': 'capture', 'final_r': r, 'steps': step}
        if r > r_escape * M:
            return np.array(trajectory), {'reason': 'escape', 'final_r': r, 'steps': step}
        
        # RK4 with constant dt
        k1 = schwarzschild_geodesic_rhs(state, M)
        k2 = schwarzschild_geodesic_rhs(state + 0.5*dt*k1, M)
        k3 = schwarzschild_geodesic_rhs(state + 0.5*dt*k2, M)
        k4 = schwarzschild_geodesic_rhs(state + dt*k3, M)
        
        state += dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        trajectory.append(np.array(state, copy=True))
        total_lambda += dt
    
    return np.array(trajectory), {'reason': 'max_steps', 'final_r': r, 'steps': n_steps}

# === COPY create_initial_state HERE ===
def create_initial_state(r0_vec, target_vec, M):
    p_spatial = target_vec / np.linalg.norm(target_vec)
    r = np.linalg.norm(r0_vec)
    xp = np.dot(r0_vec, p_spatial)
    
    A = -1.0
    B = -4 * M * xp / r**2
    C = 1.0 + 2 * M / r
    
    discriminant = B**2 - 4*A*C
    
    if discriminant < 0:
        raise ValueError(f"No real p_0 at r={r:.2f}M")
    
    sqrt_disc = np.sqrt(discriminant)
    p0 = (-B - sqrt_disc) / (2*A)  # Incoming photon
    
    return np.array([*r0_vec, *p_spatial]), p0

# === TEST CODE ===
def main():
    print("="*60)
    print("FIXED DIAGNOSTIC - Reference Solution")
    print("="*60)
    
    M = 1.0
    r0 = np.array([100.0, 0.0, 0.0])
    target = np.array([-100.0, 10.0, 0.0])
    state, p0 = create_initial_state(r0, target, M)
    
    # Highly resolved reference solution (RK4, dt=0.001)
    print("Computing reference solution (dt=0.001)...")
    ref_path, _ = integrate_geodesic(state, M, lambda_max=200.0, dt=0.001, r_escape=500.0)
    ref_momentum = ref_path[-1][3:6]
    
    # Test step sizes
    dt_values = [0.5, 0.25, 0.125, 0.0625]
    errors = []
    
    for dt in dt_values:
        path, status = integrate_geodesic(state, M, lambda_max=200.0, dt=dt, r_escape=500.0)
        final_momentum = path[-1][3:6]
        error = np.linalg.norm(final_momentum - ref_momentum)
        errors.append(error)
        print(f"dt={dt:6.3f}: Error = {error:.2e} | Steps = {len(path)}")
    
    # Check convergence rate
    print("\n" + "="*60)
    for i in range(len(errors)-1):
        rate = np.log2(errors[i] / errors[i+1])
        print(f"dt {dt_values[i]:.3f}→{dt_values[i+1]:.3f}: Convergence rate = {rate:.2f}")
        if 3.8 <= rate <= 4.2:
            print("  ✅ 4th-order convergence confirmed")
        elif errors[i+1] < 1e-12:
            print("  ✅ Hit machine precision")
        else:
            print("  ❌ Rate abnormal")
    
    print("="*60)

if __name__ == "__main__":
    main()