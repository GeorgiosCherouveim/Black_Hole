#!/usr/bin/env python3
"""
RK4 Geodesic Integrator for Schwarzschild Ray Tracing
=====================================================

Robust integration with adaptive step size and NaN handling.
"""

import numpy as np
from schwarzschild import schwarzschild_geodesic_rhs

def integrate_geodesic(initial_state, M, lambda_max=100.0, dt=0.01, r_escape=50.0):
    """
    Integrate photon geodesic with NaN-safe RK4.
    
    Parameters
    ----------
    initial_state : ndarray (6,)
        Initial [x, y, z, px, py, pz]
    M : float
        Black hole mass
    lambda_max : float
        Maximum affine time
    dt : float
        Initial step size
    r_escape : float
        Escape radius in units of M
    
    Returns
    -------
    trajectory : ndarray (N, 6)
        Photon path
    status : dict
        Integration status
    """
    state = initial_state.astype(float).copy()
    trajectory = [state.copy()]
    total_lambda = 0.0
    
    status = {'reason': 'unknown', 'final_r': np.nan, 'steps': 0}
    
    max_steps = int(lambda_max / dt) * 10  # Allow for step size variations
    
    for step in range(max_steps):
        r = np.linalg.norm(state[:3])
        status['final_r'] = r
        status['steps'] = step
        
        # Termination conditions
        if r < 2.01 * M:
            status['reason'] = 'capture'
            break
        if r > r_escape * M:
            status['reason'] = 'escape'
            break
        
        # Compute derivatives
        k1 = schwarzschild_geodesic_rhs(state, M)
        
        # NaN check
        if np.any(np.isnan(k1)):
            # Reduce step size and retry
            dt *= 0.5
            if dt < 1e-6:
                status['reason'] = 'error'
                status['error'] = 'dt too small'
                break
            continue
        
        # RK4 step
        k2 = schwarzschild_geodesic_rhs(state + 0.5*dt*k1, M)
        k3 = schwarzschild_geodesic_rhs(state + 0.5*dt*k2, M)
        k4 = schwarzschild_geodesic_rhs(state + dt*k3, M)
        
        # Check for NaN in intermediate steps
        if any(np.any(np.isnan(k)) for k in [k2, k3, k4]):
            dt *= 0.5
            continue
        
        # Update state
        state += dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        
        trajectory.append(state.copy())
        total_lambda += dt
        
        # Adaptive step size: increase if safe
        if step % 10 == 0 and np.max(np.abs(k1)) < 0.1:
            dt = min(dt * 1.2, 0.5)  # Cap at 0.5
    
    return np.array(trajectory), status


def create_initial_state(r0_vec, target_vec, M):
    """Create physically consistent initial state."""
    p_spatial = target_vec / np.linalg.norm(target_vec)
    r = np.linalg.norm(r0_vec)
    xp = np.dot(r0_vec, p_spatial)
    
    # Quadratic: A p_0^2 + B p_0 + C = 0
    A = -1 + 2*M/r
    B = 4*M*xp / r**2
    C = 1 - 2*M*xp**2 / r**3  # |p|^2 = 1
    
    discriminant = B**2 - 4*A*C
    
    if discriminant < 0:
        raise ValueError(f"No real p_0 solution at r={r:.2f}M")
    
    sqrt_disc = np.sqrt(discriminant)
    p0_root1 = (-B + sqrt_disc) / (2*A)
    p0_root2 = (-B - sqrt_disc) / (2*A)
    
    # Choose incoming root (p_0 < 0)
    p0 = p0_root1 if p0_root1 < 0 else p0_root2
    
    return np.array([*r0_vec, *p_spatial]), p0


if __name__ == "__main__":
    # Simple smoke test
    import numpy as np
    
    M = 1.0
    
    # Create state
    r0 = np.array([20.0, 5.0, 0.0])
    target = np.array([-1.0, -0.25, 0.0])
    state0, p0 = create_initial_state(r0, target, M)
    
    print(f"Initial state: {state0}")
    print(f"p_0 = {p0:.6f}")
    
    # Integrate
    path, status = integrate_geodesic(state0, M, lambda_max=50.0, dt=0.1)
    
    print(f"\nIntegration status: {status}")
    print(f"Trajectory shape: {path.shape}")
    print(f"Final radius: {status['final_r']:.3f}M")