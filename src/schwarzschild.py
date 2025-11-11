#!/usr/bin/env python3
"""
Schwarzschild Geodesic RHS - DEBUGGED
====================================

Fixed null constraint coefficients for Kerr-Schild coordinates.
"""

import numpy as np

def schwarzschild_geodesic_rhs(state, M):
    """
    Compute RHS with CORRECTED null constraint coefficients.
    """
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
    
    # CRITICAL FIX: Correct null constraint for Kerr-Schild
    # g^μν p_μ p_ν = -p_0^2 + (1+2M/r)p^2 + 4M p_0 (x·p)/r^2 = 0
    A = -1.0  # g^00
    B = -4 * M * xp * r2_inv  # g^0i (CORRECTED SIGN)
    C = 1.0 + 2 * M * r_inv   # g^ij (CORRECTED)
    
    # Solve for p_0
    discriminant = B**2 - 4*A*C
    
    if discriminant < 0:
        return np.full(6, np.nan)
    
    sqrt_disc = np.sqrt(discriminant)
    p0_root1 = (-B + sqrt_disc) / (2*A)
    p0_root2 = (-B - sqrt_disc) / (2*A)
    
    # Choose incoming photon (p_0 > 0 for these coordinates)
    p0 = p0_root1 if p0_root1 > 0 else p0_root2
    
    # Position derivatives
    D = p0 * r2_inv + xp * r3_inv
    dx_dt = p_cov - 2 * M * r_vec * D
    
    # Momentum derivatives
    E_term = p0**2 * r3_inv + 2 * p0 * xp * r2_inv**2 + xp**2 * r3_inv**2
    dp_dt = 2 * M * p_cov * D - M * r_vec * E_term
    
    return np.array([dx_dt[0], dx_dt[1], dx_dt[2], 
                     dp_dt[0], dp_dt[1], dp_dt[2]])

def create_initial_state(r0_vec, target_vec, M):
    """Create initial state with debug prints."""
    p_spatial = target_vec / np.linalg.norm(target_vec)
    r = np.linalg.norm(r0_vec)
    xp = np.dot(r0_vec, p_spatial)
    
    # CRITICAL: Must match coefficients in schwarzschild_geodesic_rhs
    A = -1.0
    B = -4 * M * xp / r**2
    C = 1.0 + 2 * M / r
    
    discriminant = B**2 - 4*A*C
    
    if discriminant < 0:
        raise ValueError(f"No real p_0: r={r:.2f}M, xp={xp:.3f}, disc={discriminant:.2e}")
    
    sqrt_disc = np.sqrt(discriminant)
    p0_root1 = (-B + sqrt_disc) / (2*A)
    p0_root2 = (-B - sqrt_disc) / (2*A)
    
    # Choose incoming (p_0 > 0)
    p0 = p0_root1 if p0_root1 > 0 else p0_root2
    
    if p0 <= 0:
        print(f"WARNING: p_0 = {p0:.3f} (non-positive)")
    
    return np.array([*r0_vec, *p_spatial]), p0

if __name__ == "__main__":
    # Simple test
    M = 1.0
    state, p0 = create_initial_state(np.array([50.0, 10.0, 0.0]), 
                                     np.array([-1.0, -0.2, 0.0]), M)
    print(f"Initial state: {state}")
    print(f"p_0 = {p0:.6f}")
    
    deriv = schwarzschild_geodesic_rhs(state, M)
    print(f"Derivatives: {deriv}")