#!/usr/bin/env python3
"""
Black Hole Ray Tracer - Photon Deflection Validation
====================================================

This script validates photon deflection calculations for a Schwarzschild black hole
by comparing the exact General Relativistic solution (using elliptic integrals)
with the 2nd-order Post-Newtonian approximation.

Physics Background:
- Schartzschild metric describes a static, spherical black hole of mass M
- Photons follow null geodesics with impact parameter b
- Critical impact parameter b_crit = 3√3 M: photons with b < b_crit are captured
- For b > b_crit, photons are deflected by angle α

Save as: validation/deflection_test.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
from scipy.optimize import brentq

# Set geometric units: G = c = 1
M = 1.0  # Black hole mass (scale for all lengths)

# Critical impact parameter for photon capture
b_crit = 3 * np.sqrt(3) * M  # ≈ 5.196M


def find_closest_approach(b, M):
    """
    Find distance of closest approach r0 for a photon with impact parameter b.
    
    Physics:
    At closest approach, radial velocity dr/dλ = 0, giving:
    b = r0 / sqrt(1 - 2M/r0)
    
    This leads to cubic equation: r0³ - b²r0 + 2Mb² = 0
    
    Parameters:
    -----------
    b : float
        Impact parameter (must be > b_crit for escape)
    M : float
        Black hole mass
        
    Returns:
    --------
    r0 : float
        Distance of closest approach (> 2M)
    """
    rs = 2 * M
    
    def f(r):
        return r**3 - b**2 * r + b**2 * rs
    
    # For b > b_crit, the physical root is > 3M
    # Lower bound: r_min = 3M (where f(r) < 0)
    r_min = 3 * M + 1e-6
    
    # Upper bound: r_max = 1.5 * b (where f(r) > 0 for all b > b_crit)
    r_max = b * 1.5
    
    return brentq(f, r_min, r_max)


def exact_deflection(b, M):
    """
    Calculate exact GR deflection angle using elliptic integrals.
    
    Physics:
    The null geodesic equation integrates to:
    α = 2 * sqrt(r0/(r0 - rs)) * K(k) - π
    
    where:
    - r0 is distance of closest approach
    - rs = 2M is Schwarzschild radius
    - K(k) is complete elliptic integral of the first kind
    - k² = rs/(r0 - rs) * (3r0 - rs)/(4r0)
    
    Parameters:
    -----------
    b : float
        Impact parameter
    M : float
        Black hole mass
        
    Returns:
    --------
    alpha : float
        Deflection angle in radians (inf if captured)
    """
    rs = 2 * M
    
    if b <= b_crit:
        return np.inf  # Photon captured by black hole
    
    # Find closest approach distance
    r0 = find_closest_approach(b, M)
    
    # Calculate elliptic integral parameters
    prefactor = np.sqrt(r0 / (r0 - rs))
    
    # Modulus squared (must be < 1 for real solution)
    k_squared = (rs / (r0 - rs)) * (3 * r0 - rs) / (4 * r0)
    
    # Numerical stability check
    if k_squared >= 1.0:
        k_squared = 0.999999
    
    # Complete elliptic integral of the first kind
    K = ellipk(k_squared)
    
    # Total angle swept from asymptote to asymptote
    delta_phi = 2 * prefactor * K
    
    # Deflection is deviation from straight-line path (π radians)
    alpha = delta_phi - np.pi
    
    return alpha


def deflection_2pn(b, M):
    """
    Calculate 2PN (Post-Newtonian) approximation for photon deflection.
    
    Physics:
    Weak-field expansion of the exact solution:
    α_2PN = 4M/b + (15π/4)(M/b)² + O((M/b)³)
    
    Valid for b >> M (weak gravitational field).
    
    Parameters:
    -----------
    b : float
        Impact parameter
    M : float
        Black hole mass
        
    Returns:
    --------
    alpha : float
        2PN deflection angle in radians
    """
    x = M / b
    return 4 * x + (15 * np.pi / 4) * x**2


def main():
    """Main validation routine."""
    
    # Create output directory if needed
    os.makedirs('validation', exist_ok=True)
    
    # Impact parameter range (avoid b < b_crit where photons are captured)
    b_min = max(4.0, b_crit + 0.1) * M  # Start just above critical value
    b_max = 20.0 * M
    num_points = 150
    
    b_values = np.linspace(b_min, b_max, num_points)
    
    print("Calculating photon deflection angles...")
    print(f"Impact parameter range: {b_min/M:.2f}M to {b_max/M:.2f}M")
    print(f"Critical impact parameter: b_crit = 3√3 M ≈ {b_crit/M:.3f}M\n")
    
    # Calculate deflection angles
    alpha_exact = np.array([exact_deflection(b, M) for b in b_values])
    alpha_pn2 = np.array([deflection_2pn(b, M) for b in b_values])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot deflection angles
    ax.plot(b_values / M, alpha_exact, 'b-', label='Exact (Elliptic Integrals)', 
            linewidth=2, zorder=3)
    ax.plot(b_values / M, alpha_pn2, 'r--', label='2PN Approximation', 
            linewidth=2, zorder=2)
    
    # Mark critical impact parameter
    ax.axvline(x=b_crit / M, color='k', linestyle=':', linewidth=1.5,
               label=f'Critical b = 3√3 M ≈ {b_crit/M:.2f}M',
               zorder=1)
    
    # Styling
    ax.set_xlabel('Impact Parameter (b/M)', fontsize=12)
    ax.set_ylabel('Deflection Angle α (radians)', fontsize=12)
    ax.set_title('Photon Deflection in Schwarzschild Metric', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('validation/deflection_test.png', dpi=150, bbox_inches='tight')
    print("Plot saved to: validation/deflection_test.png")
    
    # Calculate error at b = 10M
    b_test = 10.0 * M
    if b_test > b_crit:
        alpha_exact_10M = exact_deflection(b_test, M)
        alpha_pn2_10M = deflection_2pn(b_test, M)
        error_percent = abs(alpha_exact_10M - alpha_pn2_10M) / alpha_exact_10M * 100
        
        print(f"\n{'='*50}")
        print(f"Error Analysis at b = 10M:")
        print(f"{'='*50}")
        print(f"Exact deflection (elliptic):  {alpha_exact_10M:.6f} radians")
        print(f"2PN approximation:            {alpha_pn2_10M:.6f} radians")
        print(f"Absolute difference:          {abs(alpha_exact_10M - alpha_pn2_10M):.6f} radians")
        print(f"Relative error:               {error_percent:.2f}%")
        print(f"{'='*50}")
        
        # Also show in degrees for intuition
        deg_exact = np.degrees(alpha_exact_10M)
        deg_pn2 = np.degrees(alpha_pn2_10M)
        print(f"In degrees:")
        print(f"  Exact:  {deg_exact:.2f}°")
        print(f"  2PN:    {deg_pn2:.2f}°")
        print(f"  Diff:   {abs(deg_exact - deg_pn2):.2f}°")
        
    else:
        print(f"\nNote: b=10M is below critical impact parameter (b_crit={b_crit/M:.3f}M)")
        print("Photon would be captured by the black hole.")
    
    plt.show()


if __name__ == "__main__":
    main()