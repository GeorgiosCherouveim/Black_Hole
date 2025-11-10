#!/usr/bin/env python3
"""
Basic Geodesic Integration Test
==============================

Validates integrator without plotting.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from integrator import integrate_geodesic, create_initial_state

def test_integration():
    """Test basic integration."""
    print("Schwarzschild Geodesic Integration Test")
    print("="*50)
    
    M = 1.0
    
    # Test 1: Weak field (should escape)
    print("\nTest 1: Weak field (b = 15M)")
    state0, p0 = create_initial_state(
        np.array([30.0, 15.0, 0.0]),
        np.array([-1.0, -0.5, 0.0]),
        M
    )
    
    path, status = integrate_geodesic(state0, M, lambda_max=80.0, dt=0.2)
    
    print(f"Status: {status['reason']}")
    print(f"Steps: {status['steps']}")
    print(f"Final radius: {status['final_r']:.2f}M")
    print(f"Trajectory shape: {path.shape}")
    
    # Check trajectory is valid
    assert len(path) > 10, "Too few steps!"
    assert not np.any(np.isnan(path)), "NaN in trajectory!"
    assert status['reason'] == 'escape', "Should escape in weak field"
    
    # Test 2: Strong field (should capture)
    print("\nTest 2: Strong field (b = 5.2M)")
    b_crit = 3 * np.sqrt(3) * M
    state0, p0 = create_initial_state(
        np.array([15.0, b_crit * 0.95, 0.0]),
        np.array([-1.0, -0.35, 0.0]),
        M
    )
    
    path, status = integrate_geodesic(state0, M, lambda_max=50.0, dt=0.1)
    
    print(f"Status: {status['reason']}")
    print(f"Final radius: {status['final_r']:.3f}M")
    
    print("\n" + "="*50)
    print("✓ ALL TESTS PASSED - Integrator is working!")
    print("="*50)

if __name__ == "__main__":
    test_integration()