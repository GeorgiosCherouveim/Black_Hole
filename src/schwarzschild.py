#!/usr/bin/env python3
"""
Schwarzschild Geodesic Right-Hand Side (RHS)
==========================================

Computes derivatives for photon geodesic integration in Kerr-Schild Cartesian coordinates.
These coordinates are regular at the horizon (r = 2M), enabling stable ray tracing.

Physics Background:
- Kerr-Schild metric: g_μν = η_μν + (2M/r) l_μ l_ν
- l_μ = (-1, x/r, y/r, z/r)
- Null constraint: g^μν p_μ p_ν = 0 determines p_0
- Geodesic equations from Hamiltonian formulation
"""

import numpy as np

def schwarzschild_geodesic_rhs(state, M, p0_sign=None):
    """
    Compute RHS of geodesic equations for a photon in Schwarzschild spacetime.
    
    Parameters
    ----------
    state : array_like (6,)
        State vector [x, y, z, px, py, pz]
        - (x, y, z): Cartesian position coordinates
        - (px, py, pz): Covariant momentum components p_i
    M : float
        Black hole mass in geometric units (G = c = 1)
    p0_sign : int, optional
        Sign of time component p_0: -1 for incoming photons (observer→BH),
        +1 for outgoing photons (BH→observer). Auto-detected if None.
    
    Returns
    -------
    derivs : ndarray (6,)
        Time derivatives [dx/dλ, dy/dλ, dz/dλ, dpx/dλ, dpy/dλ, dpz/dλ]
        Returns NaN if r < 2.001M (inside horizon + safety buffer)
    """
    # Unpack state vector
    x, y, z, px, py, pz = state
    
    # Vectorize for cleaner math
    r_vec = np.array([x, y, z], dtype=float)
    p_cov = np.array([px, py, pz], dtype=float)
    
    # Compute radius
    r = np.linalg.norm(r_vec)
    
    # Numerical stability: halt integration inside horizon
    if r < 2.001 * M:
        return np.full(6, np.nan)
    
    # Precompute inverse powers of r for efficiency
    r_inv = 1.0 / r
    r2_inv = r_inv ** 2
    r3_inv = r_inv ** 3
    r4_inv = r_inv ** 4
    r5_inv = r_inv ** 5
    
    # Dot product x^j p_j (appears frequently in equations)
    xp = np.dot(r_vec, p_cov)
    
    # Solve for p_0 (photon energy) using null constraint: g^μν p_μ p_ν = 0
    # For Kerr-Schild metric, this yields quadratic in p_0:
    # A p_0^2 + B p_0 + C = 0
    
    # CORRECTED COEFFICIENTS (sign error fixed):
    A = -1 + 2 * M * r_inv          # g^00 component
    B = 4 * M * xp * r2_inv          # g^0i components
    C = np.dot(p_cov, p_cov) - 2 * M * xp**2 * r3_inv  # g^ij components
    
    # Check for real solutions
    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return np.full(6, np.nan)
    
    # Compute both roots
    sqrt_disc = np.sqrt(discriminant)
    p0_root1 = (-B + sqrt_disc) / (2 * A)
    p0_root2 = (-B - sqrt_disc) / (2 * A)
    
    # Select appropriate root based on photon direction
    if p0_sign is None:
        # Auto-detect: incoming photons (x·p < 0) should have p_0 < 0
        # But also check which root gives physically consistent motion
        p0_candidates = [p0_root1, p0_root2]
        
        # Choose root that gives negative p_0 for incoming photons
        if xp < 0:  # Incoming
            valid_roots = [p for p in p0_candidates if p < 0]
        else:  # Outgoing
            valid_roots = [p for p in p0_candidates if p > 0]
            
        if len(valid_roots) == 0:
            return np.full(6, np.nan)
        
        p0 = valid_roots[0]
    else:
        p0 = p0_root1 if p0_sign * p0_root1 > 0 else p0_root2
    
    # Common term D = p_0 / r^2 + (x·p) / r^3
    D = p0 * r2_inv + xp * r3_inv
    
    # Position derivatives: dx^i/dλ = g^ij p_j + g^i0 p_0
    # Using inverse metric components in Kerr-Schild coordinates
    # g^ij = δ^ij - 2M x^i x^j / r^3
    # g^i0 = -2M x^i / r^2
    
    # Compute: dx^i/dλ = p_i - (2M x^i / r^2) p_0 - (2M x^i (x·p) / r^3)
    # This simplifies to: p_i - 2M x^i D
    dx_dt = p_cov - 2 * M * r_vec * D
    
    # Momentum derivatives: dp_i/dλ = -½ p_μ p_ν ∂_i g^μν
    # Explicit formula for Kerr-Schild metric:
    # dp_i/dλ = (2M / r^3) [ p_i (xp) + p_0 (xp + p_0 r) x_i/r 
    #                       + (xp)^2 x_i / r^2 - (p_0 + xp/r)^2 x_i ]
    # Simplified form:
    E_term = p0**2 * r3_inv + 2 * p0 * xp * r4_inv + xp**2 * r5_inv
    dp_dt = 2 * M * p_cov * D - M * r_vec * E_term
    
    return np.array([dx_dt[0], dx_dt[1], dx_dt[2], 
                     dp_dt[0], dp_dt[1], dp_dt[2]])


def create_initial_state(r0_vec, target_vec, M):
    """
    Create physically consistent initial state [x, y, z, px, py, pz].
    
    Parameters
    ----------
    r0_vec : ndarray (3,)
        Initial position vector from observer
    target_vec : ndarray (3,)
        Rough direction toward black hole
    M : float
        Black hole mass
        
    Returns
    -------
    state : ndarray (6,)
        Full state vector with covariant momentum components
    p0 : float
        Time component p_0 (photon energy)
    """
    # Normalize spatial momentum direction
    p_spatial = target_vec / np.linalg.norm(target_vec)
    
    # Solve for p_0 using null constraint
    r = np.linalg.norm(r0_vec)
    xp = np.dot(r0_vec, p_spatial)
    
    # Quadratic coefficients: A p_0^2 + B p_0 + C = 0
    A = -1 + 2*M/r
    B = 4*M*xp / r**2
    C = np.dot(p_spatial, p_spatial) - 2*M*xp**2 / r**3
    
    discriminant = B**2 - 4*A*C
    
    if discriminant < 0:
        raise ValueError(f"No real solution for p_0 at r={r:.2f}M, b={r*abs(p_spatial[1]):.2f}M")
    
    sqrt_disc = np.sqrt(discriminant)
    p0_root1 = (-B + sqrt_disc) / (2*A)
    p0_root2 = (-B - sqrt_disc) / (2*A)
    
    # Choose incoming photon root (p_0 < 0)
    p0 = p0_root1 if p0_root1 < 0 else p0_root2
    
    return np.array([*r0_vec, *p_spatial]), p0


def test_geodesic():
    """Test with normalized, physically consistent momentum."""
    M = 1.0
    
    # Photon at (10M, 5M, 0), aiming at impact parameter b ≈ 5.5M
    r_vec = np.array([10.0, 5.0, 0.0])
    
    # Normalize spatial momentum to unit magnitude
    p_dir = np.array([-1.0, -0.5, 0.0])  # Direction
    p_spatial = p_dir / np.linalg.norm(p_dir)
    
    # Solve for p_0 to satisfy null constraint
    r = np.linalg.norm(r_vec)
    xp = np.dot(r_vec, p_spatial)
    
    # Quadratic coefficients
    A = 1 + 2*M/r
    B = 4 * M * xp / r**2
    C = -1 + 2 * M * xp**2 / r**3  # p_spatial is unit
    
    # Choose incoming photon root
    sqrt_disc = np.sqrt(B**2 - 4*A*C)
    p0 = (-B - sqrt_disc) / (2*A)  # Negative for incoming
    
    # Full state: [x, y, z, px, py, pz]
    state = np.array([*r_vec, *p_spatial])
    
    derivs = schwarzschild_geodesic_rhs(state, M, p0_sign=-1)
    print("State:", state)
    print("Derivatives:", derivs)
    print(f"p_0 = {p0:.6f}")
    
    # Check no NaNs
    assert not np.any(np.isnan(derivs)), "Derivatives contain NaN!"
    
    # Verify photon is moving inward (dx/dt < 0 should be negative for x>0)
    if state[0] > 0:
        assert derivs[0] < 0, f"Photon not moving toward BH! dx/dt = {derivs[0]}"
    
    print("✓ Test passed: Geodesic RHS working correctly")
    return state, derivs, p0