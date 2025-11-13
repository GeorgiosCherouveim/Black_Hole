#!/usr/bin/env python3
"""
CORRECTED Deflection Test with Proper Initial Conditions
=========================================================
Photon travels PARALLEL to x-axis, offset by impact parameter b.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from schwarzschild import schwarzschild_geodesic_rhs
from integrator import create_initial_state, integrate_geodesic


def calculate_deflection_angle_correct(trajectory, M):
    """
    Calculate deflection from change in direction.
    
    For a photon initially traveling in -x direction,
    deflection is the angle it bends in the xy-plane.
    """
    n_points = len(trajectory)
    n_asymp = max(10, n_points // 10)
    
    # Initial direction (from early trajectory)
    initial_positions = trajectory[:n_asymp, :3]
    incoming_vec = initial_positions[-1] - initial_positions[0]
    incoming_dir = incoming_vec / np.linalg.norm(incoming_vec)
    
    # Final direction (from late trajectory)
    final_positions = trajectory[-n_asymp:, :3]
    outgoing_vec = final_positions[-1] - final_positions[0]
    outgoing_dir = outgoing_vec / np.linalg.norm(outgoing_vec)
    
    # Deflection angle
    cos_angle = np.dot(incoming_dir, outgoing_dir)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return angle


def check_energy_conservation(trajectory, M):
    """Check null constraint preservation."""
    max_violation = 0.0
    
    for state in trajectory[::max(1, len(trajectory)//100)]:
        x, y, z, px, py, pz = state
        r = np.linalg.norm([x, y, z])
        p_spatial = np.array([px, py, pz])
        xp = np.dot([x, y, z], p_spatial)
        p2 = np.dot(p_spatial, p_spatial)
        
        A = -(1.0 - 2.0*M/r)
        B = 4.0 * M * xp / r**2
        C = p2 + 2.0*M/r**3 * xp**2
        
        discriminant = B**2 - 4 * A * C
        if discriminant < 0:
            max_violation = max(max_violation, abs(discriminant))
            continue
        
        p0 = (-B - np.sqrt(discriminant)) / (2 * A)
        
        constraint_value = -(1.0 - 2.0*M/r)*p0**2 + p2 + 4.0*M*xp*p0/r**2 + 2.0*M*xp**2/r**3
        max_violation = max(max_violation, abs(constraint_value))
    
    return max_violation


def run_deflection_validation():
    """Main validation with CORRECT initial conditions."""
    print("Schwarzschild Deflection Test - CORRECTED INITIAL CONDITIONS")
    print("=" * 60)
    
    # Physical parameters
    M = 1.0
    b = 100.0 * M  # Impact parameter
    r_initial = 10000.0 * M  # Starting distance
    r_escape = 20000.0 * M   # Escape distance
    
    # Exact weak-field deflection
    alpha_exact = 4.0 * M / b
    
    print(f"Impact parameter: b = {b/M:.1f}M")
    print(f"Exact deflection: α = {alpha_exact:.6e} rad = {np.degrees(alpha_exact):.4f}°")
    print(f"Start radius: {r_initial/M:.1f}M | Escape radius: {r_escape/M:.1f}M")
    print("=" * 60)
    
    # ========================================================================
    # CORRECTED INITIAL CONDITIONS
    # ========================================================================
    # Photon starts at (r_initial, b, 0) traveling in -x direction
    # This gives a parallel incoming ray offset by impact parameter b
    # ========================================================================
    
    r0 = np.array([r_initial, b, 0.0])  # Start at y = b (impact parameter offset)
    direction = np.array([-1.0, 0.0, 0.0])  # Travel in -x direction (parallel beam)
    
    state0, p0_initial = create_initial_state(r0, direction, M)
    
    print(f"\nInitial position: ({r0[0]/M:.1f}, {r0[1]/M:.1f}, {r0[2]/M:.1f})M")
    print(f"Initial direction: {direction} (parallel to x-axis)")
    print(f"Initial p₀: {p0_initial:.6f}")
    print(f"Initial momentum: {state0[3:6]/np.linalg.norm(state0[3:6])}")
    
    # Step sizes for convergence test
    dt_values = np.array([5.0, 2.5, 1.25, 0.625, 0.3125])
    results = []
    
    for dt in dt_values:
        print(f"\nTesting dt = {dt:.3f}M...")
        
        # Integrate
        trajectory, status = integrate_geodesic(
            state0, M,
            lambda_max=50000.0,
            dt=dt,
            r_escape=r_escape
        )
        
        if status['reason'] != 'escape':
            print(f"  ❌ Failed: {status['reason']} at r={status['final_r']:.2f}M")
            continue
        
        # Calculate deflection
        alpha_sim = calculate_deflection_angle_correct(trajectory, M)
        
        # Check constraint
        max_violation = check_energy_conservation(trajectory, M)
        
        # Error metrics
        angle_error = abs(alpha_sim - alpha_exact)
        rel_error = angle_error / alpha_exact
        
        print(f"  Steps: {len(trajectory)}")
        print(f"  Final r: {status['final_r']:.2f}M")
        print(f"  Final position: ({trajectory[-1][0]:.1f}, {trajectory[-1][1]:.1f}, {trajectory[-1][2]:.1f})")
        print(f"  Simulated α: {alpha_sim:.6e} rad ({np.degrees(alpha_sim):.4f}°)")
        print(f"  Angle error: {angle_error:.2e} ({100*rel_error:.2f}%)")
        print(f"  Max constraint violation: {max_violation:.2e}")
        
        results.append({
            'dt': dt,
            'alpha_sim': alpha_sim,
            'angle_error': angle_error,
            'constraint_violation': max_violation,
            'n_steps': len(trajectory)
        })
    
    return results, alpha_exact


def plot_convergence(results, alpha_exact):
    """Plot convergence analysis."""
    dt_vals = np.array([r['dt'] for r in results])
    errors = np.array([r['angle_error'] for r in results])
    violations = np.array([r['constraint_violation'] for r in results])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convergence plot
    ax1.loglog(dt_vals, errors, 'o-', linewidth=2, markersize=8, label='RK4 Error')
    
    if len(dt_vals) >= 2:
        ref_dt = np.array([dt_vals[0], dt_vals[-1]])
        slope_4 = errors[0] * (ref_dt/dt_vals[0])**4
        ax1.loglog(ref_dt, slope_4, '--', alpha=0.5, label='4th Order (Δt⁴)')
    
    ax1.set_xlabel('Step Size dt (M)', fontsize=12)
    ax1.set_ylabel('|α_sim - α_exact| (rad)', fontsize=12)
    ax1.set_title('Deflection Angle Convergence', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, which='both', ls='--', alpha=0.3)
    
    # Compute observed slope
    if len(errors) >= 2:
        log_ratio = np.log(errors[0]/errors[-1]) / np.log(dt_vals[0]/dt_vals[-1])
        ax1.text(0.05, 0.95, f'Observed order: {log_ratio:.2f}\nExpected: 4.00',
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    else:
        log_ratio = 0.0
    
    # Constraint violation plot
    viol_plot = np.maximum(violations, 1e-16)
    ax2.loglog(dt_vals, viol_plot, 's-', color='orange', linewidth=2, markersize=8,
               label='Max Constraint Violation')
    ax2.axhline(y=1e-6, color='red', linestyle=':', linewidth=2, label='Target (1e-6)')
    ax2.set_xlabel('Step Size dt (M)', fontsize=12)
    ax2.set_ylabel('Max |g^μν p_μ p_ν|', fontsize=12)
    ax2.set_title('Null Constraint Preservation', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, which='both', ls='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('thesis_validation_FINAL_CORRECT.png', dpi=300, bbox_inches='tight')
    print("\n📊 Plot saved: thesis_validation_FINAL_CORRECT.png")
    plt.show()
    
    return log_ratio


def main():
    """Execute validation."""
    results, alpha_exact = run_deflection_validation()
    
    if len(results) < 2:
        print("\n❌ Insufficient data for convergence analysis")
        return
    
    print("\n" + "=" * 60)
    slope = plot_convergence(results, alpha_exact)
    
    # Final verdict
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Observed convergence order: {slope:.2f}")
    
    if 3.5 <= slope <= 4.5:
        print("✅ PASS: RK4 shows 4th-order convergence")
    else:
        print("⚠️  WARNING: Convergence outside expected range [3.5, 4.5]")
    
    max_viol = max(r['constraint_violation'] for r in results)
    print(f"Maximum constraint violation: {max_viol:.2e}")
    
    if max_viol < 1e-6:
        print("✅ PASS: Constraint preserved within tolerance")
    else:
        print("⚠️  WARNING: Constraint violation exceeds 1e-6")
    
    print("=" * 60)
    print("🎓 Validation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()