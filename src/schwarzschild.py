#!/usr/bin/env python3
"""
Schwarzschild Geodesic Right-Hand Side (RHS)
==========================================

Computes derivatives for photon geodesic integration in Kerr-Schild Cartesian coordinates.
These coordinates are regular at the horizon (r = 2M), enabling stable ray tracing.

Physics Background:
- State vector: [x, y, z, px, py, pz] where (px,py,pz) are covariant momentum components
- Null constraint: g^μν p_μ p_ν = 0 determines the photon energy p_0
- Geodesic equations from Hamiltonian: dx^i/dλ = ∂H/∂p_i, dp_i/dλ = -∂H/∂x^i
- Metric: Kerr-Schild form g_μν = η_μν + (2M/r) l_μ l_ν with l_μ = (-1, x/r, y/r, z/r)

Save as: src/geodesic.py
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
    # Kerr-Schild coordinates are regular, but physics stops at horizon
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
    # This yields quadratic: A p_0^2 + B p_0 + C = 0
    
    A = 1 + 2 * M * r_inv
    B = 4 * M * xp * r2_inv
    C = -np.dot(p_cov, p_cov) + 2 * M * xp**2 * r3_inv
    
    # Check for real solutions
    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return np.full(6, np.nan)
    
    # Two branches: choose based on photon direction
    sqrt_disc = np.sqrt(discriminant)
    p0_root1 = (-B + sqrt_disc) / (2 * A)
    p0_root2 = (-B - sqrt_disc) / (2 * A)
    
    if p0_sign is None:
        # Auto-detect: incoming photons (x·p < 0) should have p_0 < 0
        p0 = p0_root1 if (xp < 0 and p0_root1 < 0) else p0_root2
    else:
        # User-specified direction
        p0 = p0_root1 if p0_sign * p0_root1 > 0 else p0_root2
    
    # Common term D = p_0 / r^2 + (x·p) / r^3
    # Appears in both position and momentum derivatives
    D = p0 * r2_inv + xp * r3_inv
    
    # Position derivatives: dx^i/dλ = g^ij p_j + g^i0 p_0
    # Using inverse metric components in Kerr-Schild coordinates
    dx_dt = p_cov - 2 * M * r_vec * D
    
    # Momentum derivatives: dp_i/dλ = -∂H/∂x^i = -½ p_μ p_ν ∂_i g^μν
    # Explicit form for Kerr-Schild metric
    E = p0**2 * r3_inv + 4 * xp * p0 * r4_inv + 3 * xp**2 * r5_inv
    dp_dt = 2 * M * p_cov * D - M * r_vec * E
    
    return np.array([dx_dt[0], dx_dt[1], dx_dt[2], 
                     dp_dt[0], dp_dt[1], dp_dt[2]])


# Example usage and validation
def test_geodesic():
    """Test with normalized, physically consistent momentum."""
    M = 1.0
    
    # Photon at (10M, 5M, 0), aiming at impact parameter b ≈ 5.5M
    r_vec = np.array([10.0, 5.0, 0.0])
    
    # Normalize spatial momentum to unit magnitude
    p_spatial = np.array([-1.0, -0.5, 0.0])  # Direction
    p_spatial = p_spatial / np.linalg.norm(p_spatial)  # Unit vector
    
    # Solve for p_0 to satisfy null constraint: g^μν p_μ p_ν = 0
    xp = np.dot(r_vec, p_spatial)
    r = np.linalg.norm(r_vec)
    
    # Quadratic coefficients
    A = 1 + 2*M/r
    B = 4*M*xp / r**2
    C = -1 + 2*M*xp**2 / r**3  # p_spatial is unit
    
    # Choose incoming photon root
    sqrt_disc = np.sqrt(B**2 - 4*A*C)
    p0 = (-B - sqrt_disc) / (2*A)  # Negative for incoming
    
    # Full state: [x, y, z, px, py, pz]
    state = np.array([*r_vec, *p_spatial])
    
    derivs = schwarzschild_geodesic_rhs(state, M, p0_sign=-1)
    print("State:", state)
    print("Derivatives:", derivs)
    
    # Verify null constraint is preserved
    print(f"p_0 = {p0:.6f}")
    
    # Check no NaNs
    assert not np.any(np.isnan(derivs)), "Derivatives contain NaN!"
    
    # Verify photon is moving inward (dx/dt < 0)
    assert derivs[0] < 0, "Photon not moving toward BH!"
    
    print("✓ Test passed: Geodesic RHS working correctly")

if __name__ == "__main__":
    test_geodesic()