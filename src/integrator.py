#!/usr/bin/env python3
"""
Schwarzschild Geodesic Integrator
================================
RK4 integration for null geodesics in Schwarzschild spacetime.
"""

import numpy as np
from schwarzschild import schwarzschild_geodesic_rhs

def create_initial_state(r0_vec, target_vec, M):
    """
    Create initial state [x, y, z, px, py, pz] for photon ray.
    
    Parameters:
    -----------
    r0_vec : array (3,)
        Initial spatial position [x0, y0, z0]
    target_vec : array (3,)
        Vector from r0 to target point
    M : float
        Black hole mass
    
    Returns:
    --------
    state : array (6,)
        [x, y, z, px, py, pz] where p_i are covariant momentum components
    p0 : float
        Time component of momentum (energy)
    """
    # Normalize spatial momentum direction
    p_spatial = target_vec / np.linalg.norm(target_vec)
    r0 = np.linalg.norm(r0_vec)
    
    # Dot product r·p for Hamiltonian constraint
    xp = np.dot(r0_vec, p_spatial)
    
    # Solve -p₀² + pᵢpⁱ = 0 for null geodesic
    # pᵢpⁱ = (1 + 2M/r) p_spatial² - (4M/r²) xp p₀ + (2M/r³) xp²
    # For p_spatial² = 1 (normalized), this becomes quadratic in p₀
    
    A = -1.0
    B = -4 * M * xp / r0**2
    C = 1.0 + 2 * M / r0
    
    discriminant = B**2 - 4 * A * C
    
    if discriminant < 0:
        raise ValueError(f"No real solution for p₀ at r={r0:.2f}M (discriminant={discriminant:.2e})")
    
    sqrt_disc = np.sqrt(discriminant)
    # Choose incoming photon solution (p₀ > 0, but momentum points inward)
    p0_root1 = (-B + sqrt_disc) / (2 * A)
    p0_root2 = (-B - sqrt_disc) / (2 * A)
    p0 = max(p0_root1, p0_root2)  # Take the positive root
    
    return np.array([*r0_vec, *p_spatial]), p0

def integrate_geodesic(initial_state, M, lambda_max=5000.0, dt=0.5, r_escape=500.0):
    """
    Integrate null geodesic using RK4 with fixed step size.
    
    Parameters:
    -----------
    initial_state : array (6,)
        [x, y, z, px, py, pz]
    M : float
        Black hole mass
    lambda_max : float
        Maximum affine time to integrate
    dt : float
        Step size in affine time
    r_escape : float
        Termination radius (in units of M)
    
    Returns:
    --------
    trajectory : array (N, 6)
        State at each step
    status : dict
        {
            'reason': 'escape' | 'capture' | 'max_steps',
            'final_r': float,
            'steps': int
        }
    """
    state = np.array(initial_state, dtype=float, copy=True)
    trajectory = [np.array(state, copy=True)]
    
    n_steps = int(lambda_max / dt)
    
    for step in range(n_steps):
        r = np.linalg.norm(state[:3])
        
        # Termination conditions
        if r < 2.01 * M:  # Inside horizon (with buffer)
            return np.array(trajectory), {
                'reason': 'capture',
                'final_r': r,
                'steps': step
            }
        if r > r_escape * M:  # Escaped to infinity
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
    
    # Reached lambda_max
    return np.array(trajectory), {
        'reason': 'max_steps',
        'final_r': np.linalg.norm(state[:3]),
        'steps': n_steps
    }