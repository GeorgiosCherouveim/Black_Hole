#!/usr/bin/env python3
"""
Schwarzschild Geodesic Integrator - CORRECT CONSTRAINT
======================================================
RK4 integration with fixed A coefficient.
"""

import numpy as np
from schwarzschild import schwarzschild_geodesic_rhs

def create_initial_state(r0_vec, target_vec, M):
    """Create initial state with CORRECT constraint."""
    p_spatial = target_vec / np.linalg.norm(target_vec)
    r0 = np.linalg.norm(r0_vec)
    xp = np.dot(r0_vec, p_spatial)
    p2 = np.dot(p_spatial, p_spatial)
    
    # ✅ CORRECTED: A = -(1 - 2M/r) not just -1
    A = -(1.0 - 2.0*M/r0)
    B = 4.0 * M * xp / r0**2
    C = p2 + 2.0*M/r0**3 * xp**2
    
    discriminant = B**2 - 4 * A * C
    
    if discriminant < 0:
        raise ValueError(f"No real solution for p₀ at r={r0:.2f}M (discriminant={discriminant:.2e})")
    
    sqrt_disc = np.sqrt(discriminant)
    p0 = (-B - sqrt_disc) / (2 * A)
    
    if p0 <= 0:
        raise ValueError(f"Non-positive p₀={p0:.3e}")
    
    return np.array([*r0_vec, *p_spatial]), p0


def integrate_geodesic(initial_state, M, lambda_max=5000.0, dt=0.5, r_escape=500.0):
    """Integrate with fixed step RK4."""
    state = np.array(initial_state, dtype=float, copy=True)
    trajectory = [np.array(state, copy=True)]
    
    n_steps = int(lambda_max / dt)
    
    for step in range(n_steps):
        r = np.linalg.norm(state[:3])
        
        if r < 2.01 * M:
            return np.array(trajectory), {
                'reason': 'capture',
                'final_r': r,
                'steps': step
            }
        if r > r_escape:
            return np.array(trajectory), {
                'reason': 'escape',
                'final_r': r,
                'steps': step
            }
        
        # RK4 step
        k1 = schwarzschild_geodesic_rhs(state, M)
        k2 = schwarzschild_geodesic_rhs(state + 0.5 * dt * k1, M)
        k3 = schwarzschild_geodesic_rhs(state + 0.5 * dt * k2, M)
        k4 = schwarzschild_geodesic_rhs(state + dt * k3, M)
        
        state += dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
        trajectory.append(np.array(state, copy=True))
    
    return np.array(trajectory), {
        'reason': 'max_steps',
        'final_r': np.linalg.norm(state[:3]),
        'steps': n_steps
    }