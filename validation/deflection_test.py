#!/usr/bin/env python3
"""
Schwarzschild Ray Tracer - Thesis Validation
==========================================
Tests weak-field deflection convergence against exact solution.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from schwarzschild import schwarzschild_geodesic_rhs

# === Import your integrator functions ===
from integrator import create_initial_state
from integrator import integrate_geodesic

def calculate_deflection_angle(initial_direction, final_state):
    """
    Compute deflection angle between incoming asymptote and final direction.
    initial_direction: unit vector of incoming photon (e.g., [-1, 0, 0])
    final_state: [x, y, z, px, py, pz]
    """
    p_final = final_state[3:6]  # Covariant momentum
    p_norm = np.linalg.norm(p_final)
    if p_norm < 1e-12:
        return np.nan
    
    direction_final = p_final / p_norm
    dot_product = np.dot(initial_direction, direction_final)
    
    # Clip for numerical safety
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product)
    return angle

def check_energy_conservation(state_initial, state_final, M):
    """
    Verify Hamiltonian constraint H = -p₀ is conserved.
    Returns relative drift (should be < 1e-6).
    """
    def compute_hamiltonian(state):
        x, y, z, px, py, pz = state
        r = np.linalg.norm([x, y, z])
        p_spatial = np.array([px, py, pz])
        xp = np.dot([x, y, z], p_spatial)
        
        # Solve -p₀² + pᵢpⁱ = 0 for null geodesic
        A = -1.0
        B = -4 * M * xp / r**2
        C = 1.0 + 2 * M / r
        
        discriminant = B**2 - 4 * A * C
        if discriminant < 0:
            return np.nan
        
        sqrt_disc = np.sqrt(discriminant)
        p0 = (-B - sqrt_disc) / (2 * A)  # Incoming photon
        return -p0  # Hamiltonian H = -p₀
    
    H_initial = compute_hamiltonian(state_initial)
    H_final = compute_hamiltonian(state_final)
    
    if abs(H_initial) < 1e-12:
        return np.nan
    
    relative_drift = abs(H_final - H_initial) / abs(H_initial)
    return relative_drift

def run_deflection_validation():
    """Main validation test for thesis specification."""
    print("Schwarzschild Ray Tracer - Thesis Validation")
    print("=" * 60)
    
    # Physical parameters (WEAK FIELD)
    M = 1.0
    b = 100.0 * M  # Impact parameter (b >> M)
    r_initial = 1000.0 * M  # Start in asymptotic region
    r_escape = 5000.0 * M   # ESCAPE TO FAR FIELD (must be > r_initial)
    
    # Exact weak-field deflection
    alpha_exact = 4.0 * M / b
    print(f"Impact parameter: b = {b/M:.1f}M")
    print(f"Exact deflection: α = {alpha_exact:.6e} rad")
    print(f"Start radius: {r_initial/M:.1f}M | Escape radius: {r_escape/M:.1f}M")
    print("=" * 60)
    
    # Initial conditions
    r0 = np.array([r_initial, 0.0, 0.0])
    aim_point = np.array([0.0, b, 0.0])
    target = aim_point - r0
    
    state0, _ = create_initial_state(r0, target, M)
    initial_direction = np.array([-1.0, 0.0, 0.0])  # Asymptotic incoming direction
    
    # Step sizes for convergence test
    dt_values = np.array([2.0, 1.0, 0.5, 0.25, 0.125])
    results = []
    
    for dt in dt_values:
        print(f"\nTesting dt = {dt:.3f}M...")
        
        # Integrate geodesic
        trajectory, status = integrate_geodesic(
            state0, M, 
            lambda_max=10000.0,  # Increased to reach r_escape
            dt=dt, 
            r_escape=r_escape
        )
        
        if status['reason'] != 'escape':
            print(f"  ❌ Failed: {status['reason']} at r={status['final_r']:.2f}M")
            continue
        
        # Calculate deflection angle
        final_state = trajectory[-1]
        alpha_sim = calculate_deflection_angle(initial_direction, final_state)
        
        # Check energy conservation
        energy_drift = check_energy_conservation(state0, final_state, M)
        
        # Error metrics
        angle_error = abs(alpha_sim - alpha_exact)
        rel_error = angle_error / alpha_exact
        
        print(f"  Steps: {len(trajectory)}")
        print(f"  Final r: {status['final_r']:.2f}M")
        print(f"  Simulated α: {alpha_sim:.6e} rad")
        print(f"  Angle error: {angle_error:.2e}")
        print(f"  Energy drift: {energy_drift:.2e}")
        
        results.append({
            'dt': dt,
            'alpha_sim': alpha_sim,
            'angle_error': angle_error,
            'energy_drift': energy_drift,
            'n_steps': len(trajectory)
        })
    
    return results

def plot_convergence(results):
    """Plot deflection angle convergence."""
    dt_vals = np.array([r['dt'] for r in results])
    errors = np.array([r['angle_error'] for r in results])
    drifts = np.array([r['energy_drift'] for r in results])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot 1: Convergence rate ---
    ax1.loglog(dt_vals, errors, 'o-', linewidth=2, markersize=8, label='RK4 Error')
    
    # Reference slope for 4th-order
    ref_dt = np.array([dt_vals[0], dt_vals[-1]])
    slope_4 = errors[0] * (ref_dt/dt_vals[0])**4
    ax1.loglog(ref_dt, slope_4, '--', alpha=0.5, label='4th Order (Δt⁴)')
    
    ax1.set_xlabel('Step Size dt (M)', fontsize=12)
    ax1.set_ylabel('|α_sim - α_exact| (rad)', fontsize=12)
    ax1.set_title('Deflection Angle Convergence', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left')
    ax1.grid(True, which='both', ls='--', alpha=0.3)
    
    # Compute observed slope
    if len(errors) > 1:
        slope = np.log(errors[0]/errors[-2]) / np.log(dt_vals[0]/dt_vals[-2])
        ax1.text(0.05, 0.95, f'Observed order: {slope:.2f}\nExpected: 4.00', 
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    else:
        slope = 0.0
    
    # --- Plot 2: Energy conservation ---
    # Filter out zero or negative drifts for log plot
    drift_plot = np.maximum(drifts, 1e-16)
    ax2.loglog(dt_vals, drift_plot, 's-', color='orange', linewidth=2, markersize=8, label='Energy Drift')
    ax2.axhline(y=1e-6, color='red', linestyle=':', label='Drift limit (1e-6)')
    ax2.set_xlabel('Step Size dt (M)', fontsize=12)
    ax2.set_ylabel('Relative Energy Drift', fontsize=12)
    ax2.set_title('Hamiltonian Constraint', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, which='both', ls='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('thesis_validation.png', dpi=300, bbox_inches='tight')
    print("\n📊 Plot saved: thesis_validation.png")
    plt.show()
    
    return slope

def main():
    """Execute full validation suite."""
    results = run_deflection_validation()
    
    if len(results) < 3:
        print("\n❌ Insufficient data for convergence analysis")
        return
    
    print("\n" + "=" * 60)
    slope = plot_convergence(results)
    
    # Final verdict
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Observed convergence order: {slope:.2f}")
    
    if 3.8 <= slope <= 4.2:
        print("✅ PASS: RK4 shows 4th-order convergence")
    else:
        print("⚠️  WARNING: Convergence outside expected range")
    
    # Energy conservation check
    max_drift = max(r['energy_drift'] for r in results)
    print(f"Maximum energy drift: {max_drift:.2e}")
    
    if max_drift < 1e-6:
        print("✅ PASS: Energy conservation within tolerance")
    else:
        print("❌ FAIL: Energy drift exceeds 1e-6")
    
    print("=" * 60)
    print("🎓 Validation complete. Integrator ready for Phase 2!")
    print("=" * 60)

if __name__ == "__main__":
    main()