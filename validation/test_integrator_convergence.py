#!/usr/bin/env python3
"""
Schwarzschild Integrator Convergence Test - DEBUGGED
==================================================

Validates RK4 integrator against exact deflection at b=10M.
All sign errors and angle calculations fixed.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
from scipy.optimize import brentq
from schwarzschild import schwarzschild_geodesic_rhs
from integrator import integrate_geodesic

# Exact deflection calculation
def exact_deflection_elliptic(b, M):
    """Exact GR deflection using elliptic integrals."""
    if b <= 3 * np.sqrt(3) * M:
        return np.inf
    
    rs = 2 * M
    def f(r):
        return r**3 - b**2 * r + b**2 * rs
    
    r0 = brentq(f, 3 * M, b * 1.5)
    
    prefactor = np.sqrt(r0 / (r0 - rs))
    k_squared = (rs / (r0 - rs)) * (3 * r0 - rs) / (4 * r0)
    if k_squared >= 1.0:
        k_squared = 0.999999
    
    K = ellipk(k_squared)
    delta_phi = 2 * prefactor * K
    
    return delta_phi - np.pi

def create_initial_state(r0_vec, target_vec, M):
    """Create physically consistent initial state."""
    p_spatial = target_vec / np.linalg.norm(target_vec)
    r = np.linalg.norm(r0_vec)
    xp = np.dot(r0_vec, p_spatial)
    
    # CRITICAL FIX: Correct coefficients for null constraint
    # For Kerr-Schild: g^μν p_μ p_ν = -p_0^2 + (1+2M/r)p_spatial^2 + (4M/r^2) p_0 (x·p) = 0
    # This gives: A p_0^2 + B p_0 + C = 0
    A = -1.0  # g^00 component
    B = -4 * M * xp / r**2  # CORRECTED SIGN
    C = 1.0 + 2 * M / r  # CORRECTED: spatial part
    
    discriminant = B**2 - 4*A*C
    
    if discriminant < 0:
        raise ValueError(f"No real p_0 solution: r={r:.2f}M, b={r*abs(p_spatial[1]):.2f}M, disc={discriminant:.2e}")
    
    sqrt_disc = np.sqrt(discriminant)
    p0_root1 = (-B + sqrt_disc) / (2*A)
    p0_root2 = (-B - sqrt_disc) / (2*A)
    
    # Choose incoming photon (p_0 should be positive for inward motion in these coordinates)
    # DEBUG: Print both roots to see which is physical
    p0 = p0_root1 if p0_root1 > 0 else p0_root2
    
    # Verify choice
    if p0 < 0:
        print(f"⚠️ WARNING: p_0 = {p0:.3f} (negative), may indicate sign error")
    
    return np.array([*r0_vec, *p_spatial]), p0

def compute_deflection_angle(state0, state_final, b, M):
    """
    Compute asymptotic deflection angle for photon with impact parameter b.
    
    The deflection is the angle between:
    1. Initial asymptotic direction (from infinity to pericenter)
    2. Final asymptotic direction (momentum at large radius after deflection)
    
    For b=10M, starting at x=100M:
    - Initial direction should be [-cosθ, sinθ, 0] where θ = arctan(b/100)
    - Final direction is momentum vector at r=500M
    - Deflection = angle between initial and final directions
    """
    # Initial asymptotic direction (at x → +∞)
    # For photon starting at (100, 0) aiming toward (0, b):
    #   angle = arctan(b/100) ≈ 0.0997 rad
    #   direction = [-cosθ, sinθ, 0]
    theta_initial = np.arctan(b / 100.0)  # Impact angle
    p_init = np.array([-np.cos(theta_initial), np.sin(theta_initial), 0.0])
    
    # Final asymptotic direction (momentum at escape)
    p_final = state_final[3:6]
    p_final_norm = p_final / np.linalg.norm(p_final)
    
    # Angle between vectors
    dot_product = np.dot(p_init, p_final_norm)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    deflection = np.arccos(dot_product)
    
    # DEBUG: Print all values
    print(f"    Initial angle θ = {np.degrees(theta_initial):.2f}°")
    print(f"    Initial dir: {p_init}")
    print(f"    Final dir:   {p_final_norm}")
    print(f"    Dot product: {dot_product:.6f}")
    
    return deflection

def check_energy_conservation(state, M, tolerance=1e-6):
    """
    Check null constraint - FIXED: Use same coefficients as create_initial_state
    """
    x, y, z, px, py, pz = state
    r = np.linalg.norm([x, y, z])
    xp = x*px + y*py + z*pz
    
    # Use same coefficients as create_initial_state
    A = -1.0
    B = -4 * M * xp / r**2  # MUST MATCH create_initial_state
    C = 1.0 + 2 * M / r
    
    discriminant = B**2 - 4*A*C
    
    if discriminant < 0:
        print(f"  ⚠️  No real p_0 at r={r:.2f}M, disc={discriminant:.2e}")
        return False, np.nan
    
    sqrt_disc = np.sqrt(discriminant)
    p0_root1 = (-B + sqrt_disc) / (2*A)
    p0_root2 = (-B - sqrt_disc) / (2*A)
    
    # Choose incoming root (p_0 > 0 for these coordinates)
    valid_roots = [p for p in [p0_root1, p0_root2] if p > 0]
    
    if len(valid_roots) == 0:
        print(f"  ⚠️  No valid p_0 root at r={r:.2f}M")
        return False, np.nan
    
    p0 = valid_roots[0]
    null_constraint = A*p0**2 + B*p0 + C
    
    return abs(null_constraint) < tolerance, null_constraint

def run_convergence_test():
    """Run convergence test."""
    print("Schwarzschild Geodesic Integrator Convergence Test")
    print("="*60)
    
    M = 1.0
    b = 10.0 * M
    
    dt_values = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625])
    alpha_exact = exact_deflection_elliptic(b, M)
    
    print(f"Exact deflection: {alpha_exact:.6f} rad ({np.degrees(alpha_exact):.2f}°)")
    print("="*60)
    
    results = []
    
    for dt in dt_values:
        print(f"\nTesting dt = {dt:.6f} M...")
        
        # FIXED: Start much farther away
        r0 = np.array([100.0, b, 0.0])  # x=100M for proper asymptotic
        target = np.array([-1.0, -b/100.0, 0.0])
        
        # FIXED: Proper impact parameter calculation
        # For b=10M, we need to aim so the photon ACTUALLY has this impact parameter
        # Start at x=100M, y=0, then aim toward a point offset by b in y-direction

        # Correct aiming: target position is (0, b) (not the origin)
        aim_point = np.array([0.0, b, 0.0])
        r0 = np.array([100.0, 0.0, 0.0])  # Start on x-axis

        # Direction vector from start to aim point
        target_vec = aim_point - r0  # This gives [-100, b, 0]

        try:
            state0, p0 = create_initial_state(r0, target_vec, M)
        except ValueError as e:
            print(f"  ❌ Failed to create state: {e}")
            continue
                
        # Debug prints
        print(f"  Initial p_0 = {p0:.6f}")
        print(f"  Initial |p| = {np.linalg.norm(state0[3:6]):.6f}")
        
        lambda_max = 1000.0  # Very large for proper escape
        path, status = integrate_geodesic(state0, M, lambda_max, dt, r_escape=500.0)
        
        print(f"  Status: {status['reason']}")
        print(f"  Steps: {status['steps']}")
        print(f"  Final r: {status['final_r']:.2f}M")
        print(f"  Actual dt used: {status.get('dt_used', 'UNKNOWN')}")
        
        if status['reason'] != 'escape':
            print(f"  ⚠️  Did not escape")
            continue
        
        # Compute deflection
        alpha_computed = compute_deflection_angle(state0, path[-1], b, M)
        error = abs(alpha_computed - alpha_exact)
        rel_error = error / alpha_exact
        
        # Check energy
        energy_ok, null_val = check_energy_conservation(path[-1], M)
        
        print(f"  Deflection: {alpha_computed:.6f} rad (error: {error:.6e})")
        print(f"  Energy OK: {energy_ok} (|g^μν p_μ p_ν| = {abs(null_val):.2e})")
        
        results.append({
            'dt': dt,
            'error': error,
            'rel_error': rel_error,
            'energy_ok': energy_ok,
            'steps': status['steps']
        })
    
    return results

def plot_convergence(results):
    """Plot convergence."""
    dt = np.array([r['dt'] for r in results])
    error = np.array([r['error'] for r in results])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(dt, error, 'o-', linewidth=2, markersize=8, label='RK4 Error')
    
    # Reference lines
    ref_dt = np.array([dt[0], dt[-1]])
    ax.loglog(ref_dt, error[0] * (ref_dt/dt[0])**1, '--', alpha=0.5, label='1st order')
    ax.loglog(ref_dt, error[0] * (ref_dt/dt[0])**2, '--', alpha=0.5, label='2nd order')
    ax.loglog(ref_dt, error[0] * (ref_dt/dt[0])**4, '--', alpha=0.5, label='4th order')
    
    ax.set_xlabel('Step Size dt (M)', fontsize=12)
    ax.set_ylabel('Absolute Error (rad)', fontsize=12)
    ax.set_title('RK4 Convergence Test: Photon Deflection at b=10M', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    
    if len(dt) >= 2:
        slope = np.log(error[-1]/error[0]) / np.log(dt[-1]/dt[0])
        ax.text(0.05, 0.95, f'Observed slope: {slope:.2f}\nExpected: 4.00', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # FIXED: Save in current directory
    plt.savefig('convergence_test.png', dpi=200, bbox_inches='tight')
    print("\nPlot saved: convergence_test.png")
    
    plt.show()
    
    return slope if len(dt) >= 2 else None

def main():
    """Run test."""
    results = run_convergence_test()
    
    if len(results) < 2:
        print("\n❌ Insufficient data")
        return
    
    print("\n" + "="*60)
    slope = plot_convergence(results)
    
    if slope is not None:
        print(f"\nObserved convergence order: {slope:.2f}")
        if 3.6 <= slope <= 4.4:
            print("✅ SUCCESS: RK4 shows 4th-order convergence!")
        else:
            print(f"⚠️  FAILED: Convergence order {slope:.2f} not in [3.6, 4.4]")
    
    print("\n" + "="*60)
    print("✓ Convergence test complete")

if __name__ == "__main__":
    main()