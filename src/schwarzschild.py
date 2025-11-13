#!/usr/bin/env python3
"""
Schwarzschild Geodesic RHS - CONSTRAINT-PRESERVING VERSION
==========================================================
Correct Hamilton equations that preserve null constraint.
"""

import numpy as np

def schwarzschild_geodesic_rhs(state, M):
    """
    Compute geodesic RHS using Hamilton's equations.
    
    Hamiltonian: H = -½(1-2M/r)p₀² + ½p² + 2M(x·p)p₀/r² + M(x·p)²/r³
    
    Constraint: H = 0 (null geodesic)
    
    Returns: d/dλ [x, y, z, px, py, pz]
    """
    x, y, z, px, py, pz = state
    r_vec = np.array([x, y, z], dtype=float)
    p_cov = np.array([px, py, pz], dtype=float)
    
    r = np.linalg.norm(r_vec)
    if r < 2.001 * M:
        return np.full(6, np.nan)
    
    r_inv = 1.0 / r
    r2_inv = r_inv * r_inv
    r3_inv = r2_inv * r_inv
    
    xp = np.dot(r_vec, p_cov)
    p2 = np.dot(p_cov, p_cov)
    
    # Solve for p₀ from constraint
    A = -(1.0 - 2.0*M*r_inv)
    B = 4.0 * M * xp * r2_inv
    C = p2 + 2.0*M*r3_inv * xp**2
    
    discriminant = B**2 - 4.0*A*C
    if discriminant < 0:
        return np.full(6, np.nan)
    
    sqrt_disc = np.sqrt(discriminant)
    p0 = (-B - sqrt_disc) / (2.0*A)  # Incoming photon (negative root)
    
    # ====================================================================
    # Position derivatives: dx^i/dλ = ∂H/∂p_i
    # ====================================================================
    dx_dt = (1.0 + 2.0*M*r_inv) * p_cov + 2.0*M*p0*r2_inv * r_vec
    
    # ====================================================================
    # Momentum derivatives: dp_i/dλ = -∂H/∂x^i
    # ====================================================================
    # This is the critical part that preserves the constraint!
    
    # Term 1: From ∂/∂x^i[(1-2M/r)p₀²]
    term1 = -M * p0**2 * r3_inv * r_vec
    
    # Term 2: From ∂/∂x^i[(x·p)p₀/r²]
    term2 = -2.0*M*p0*r2_inv * (p_cov - 2.0*xp*r2_inv * r_vec)
    
    # Term 3: From ∂/∂x^i[(x·p)²/r³]
    term3 = -M*xp**2*r3_inv * (2.0*r2_inv*p_cov - 3.0*xp*r2_inv*r2_inv*r_vec)
    
    dp_dt = term1 + term2 + term3
    
    return np.array([dx_dt[0], dx_dt[1], dx_dt[2], 
                     dp_dt[0], dp_dt[1], dp_dt[2]])


def create_initial_state(r0_vec, target_vec, M):
    """Create initial state satisfying null constraint."""
    p_spatial = target_vec / np.linalg.norm(target_vec)
    r = np.linalg.norm(r0_vec)
    xp = np.dot(r0_vec, p_spatial)
    p2 = np.dot(p_spatial, p_spatial)
    
    A = -(1.0 - 2.0*M/r)
    B = 4.0 * M * xp / r**2
    C = p2 + 2.0*M/r**3 * xp**2
    
    discriminant = B**2 - 4.0*A*C
    if discriminant < 0:
        raise ValueError(f"No real p_0: r={r:.2f}M, disc={discriminant:.2e}")
    
    sqrt_disc = np.sqrt(discriminant)
    p0 = (-B - sqrt_disc) / (2.0*A)
    
    if p0 <= 0:
        raise ValueError(f"Non-positive p₀={p0:.3e}")
    
    return np.array([*r0_vec, *p_spatial]), p0