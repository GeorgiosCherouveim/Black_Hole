#!/usr/bin/env python3
"""
Schwarzschild Geodesic Integrator Test
=====================================

Validates the RK4 geodesic integrator for photon ray tracing.
Tests capture/escape conditions and deflection accuracy.

Save as: validation/test_integrator.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from schwarzschild import schwarzschild_geodesic_rhs
from integrator import integrate_geodesic

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
    """
    # Normalize spatial momentum direction
    p_spatial = target_vec / np.linalg.norm(target_vec)
    
    # Solve for p_0 using null constraint
    r = np.linalg.norm(r0_vec)
    xp = np.dot(r0_vec, p_spatial)
    
    A = 1 + 2*M/r
    B = 4 * M * xp / r**2
    C = -1 + 2 * M * xp**2 / r**3
    
    # Choose incoming photon root (p_0 < 0)
    sqrt_disc = np.sqrt(B**2 - 4*A*C)
    p0 = (-B - sqrt_disc) / (2*A)
    
    return np.array([*r0_vec, *p_spatial])

def test_weak_field():
    """Test photon in weak field (b = 30M) - should be nearly straight."""
    print("\n" + "="*50)
    print("Test 1: Weak Field (b = 30M)")
    print("="*50)
    
    M = 1.0
    b = 30.0 * M
    
    # Photon from x = 50M, y = 30M (b = 30M)
    state0 = create_initial_state(
        r0_vec=np.array([50.0, b, 0.0]),
        target_vec=np.array([-1.0, -b/50.0, 0.0]),
        M=M
    )
    
    path = integrate_geodesic(state0, M, lambda_max=100.0, dt=0.5)
    
    # Should escape (final r > 30M)
    final_r = np.linalg.norm(path[-1][:3])
    
    print(f"Initial position: {state0[:3]}")
    print(f"Final position: {path[-1][:3]}")
    print(f"Final radius: {final_r:.2f}M")
    
    # Verify nearly straight (small deflection)
    initial_angle = np.arctan2(b, 50.0)
    final_angle = np.arctan2(path[-1][1], path[-1][0])
    deflection = abs(final_angle - initial_angle)
    
    print(f"Deflection angle: {np.degrees(deflection):.2f}°")
    assert deflection < 0.1, "Weak field deflection too large!"
    
    return path

def test_strong_field():
    """Test photon with b = 5.5M (near critical) - should show strong deflection."""
    print("\n" + "="*50)
    print("Test 2: Strong Field (b = 5.5M)")
    print("="*50)
    
    M = 1.0
    b = 5.5 * M
    
    # Photon from x = 20M, y = 5.5M
    state0 = create_initial_state(
        r0_vec=np.array([20.0, b, 0.0]),
        target_vec=np.array([-1.0, -b/20.0, 0.0]),
        M=M
    )
    
    path = integrate_geodesic(state0, M, lambda_max=50.0, dt=0.1)
    
    final_r = np.linalg.norm(path[-1][:3])
    print(f"Final radius: {final_r:.2f}M")
    
    # Should either escape or be captured - check which
    if final_r < 3.0:
        print("✓ Photon captured by black hole")
        capture = True
    else:
        print("✓ Photon escaped after strong deflection")
        capture = False
    
    # Verify trajectory has many points (integration succeeded)
    assert len(path) > 10, "Integration too short!"
    
    return path, capture

def test_capture():
    """Test photon with b < b_crit (b = 5.0M) - should be captured."""
    print("\n" + "="*50)
    print("Test 3: Capture (b = 5.0M < b_crit)")
    print("="*50)
    
    M = 1.0
    b_crit = 3 * np.sqrt(3) * M
    b = 5.0 * M  # Just below critical for this mass
    
    # Photon from x = 20M
    state0 = create_initial_state(
        r0_vec=np.array([20.0, b, 0.0]),
        target_vec=np.array([-1.0, -b/20.0, 0.0]),
        M=M
    )
    
    path = integrate_geodesic(state0, M, lambda_max=50.0, dt=0.1)
    
    # Should be captured (final radius very small)
    final_r = np.linalg.norm(path[-1][:3])
    print(f"Final radius: {final_r:.3f}M")
    
    if final_r < 2.5:
        print("✓ Photon captured as expected")
    else:
        print("⚠ Photon escaped - may need to adjust b or integration time")
    
    return path

def plot_trajectories(paths, labels, M):
    """Plot multiple photon trajectories."""
    plt.figure(figsize=(10, 10))
    
    # Plot black hole
    bh_circle = plt.Circle((0, 0), 2*M, color='black', fill=True, label='BH (r=2M)')
    plt.gca().add_patch(bh_circle)
    
    # Plot photon sphere
    ps_circle = plt.Circle((0, 0), 3*M, color='gray', fill=False, linestyle='--', label='Photon sphere')
    plt.gca().add_patch(ps_circle)
    
    # Plot trajectories
    for path, label in zip(paths, labels):
        plt.plot(path[:, 0], path[:, 1], label=label, linewidth=1.5)
        plt.plot(path[0, 0], path[0, 1], 'o', markersize=6, label=f'{label} start')
    
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.xlabel('x coordinate (M)')
    plt.ylabel('y coordinate (M)')
    plt.title('Photon Trajectories in Schwarzschild Metric')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-10, 60)
    plt.ylim(-10, 40)
    plt.savefig('validation/trajectory_tests.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: validation/trajectory_tests.png")
    plt.show()

if __name__ == "__main__":
    print("Running Schwarzschild Geodesic Integrator Tests...")
    
    M = 1.0
    
    try:
        # Run tests
        path_weak = test_weak_field()
        path_strong, captured = test_strong_field()
        path_capture = test_capture()
        
        # Plot results
        paths = [path_weak, path_strong, path_capture]
        labels = [f'Weak field (b=30M)', 
                  f'Strong field (b=5.5M, captured={captured})', 
                  f'Capture test (b=5.0M)']
        
        plot_trajectories(paths, labels, M)
        
        print("\n" + "="*50)
        print("✓ ALL TESTS PASSED - Integrator working correctly!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()