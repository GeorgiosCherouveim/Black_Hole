#!/usr/bin/env python3
"""
Diagnostic: Analyze trajectory geometry and deflection calculation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from schwarzschild import schwarzschild_geodesic_rhs
from integrator import create_initial_state, integrate_geodesic


def analyze_trajectory_geometry(trajectory, M, b):
    """Detailed analysis of trajectory geometry."""
    
    print("\n" + "="*60)
    print("TRAJECTORY GEOMETRY ANALYSIS")
    print("="*60)
    
    # Extract positions
    positions = trajectory[:, :3]
    
    # Key points
    r_start = positions[0]
    r_end = positions[-1]
    
    # Find closest approach point
    distances = np.linalg.norm(positions, axis=1)
    idx_closest = np.argmin(distances)
    r_closest = positions[idx_closest]
    closest_dist = distances[idx_closest]
    
    print(f"\nKey Points:")
    print(f"  Start:   {r_start}")
    print(f"  End:     {r_end}")
    print(f"  Closest: {r_closest} (r = {closest_dist:.2f}M)")
    print(f"  Impact parameter b = {b:.1f}M")
    
    # Asymptotic directions
    n_asymp = max(10, len(trajectory) // 10)
    
    incoming_vec = positions[n_asymp] - positions[0]
    incoming_dir = incoming_vec / np.linalg.norm(incoming_vec)
    
    outgoing_vec = positions[-1] - positions[-n_asymp]
    outgoing_dir = outgoing_vec / np.linalg.norm(outgoing_vec)
    
    print(f"\nAsymptotic Directions (using {n_asymp} points):")
    print(f"  Incoming: {incoming_dir}")
    print(f"  Outgoing: {outgoing_dir}")
    
    # Calculate deflection angle
    cos_angle = np.dot(incoming_dir, outgoing_dir)
    deflection = np.arccos(np.clip(cos_angle, -1, 1))
    
    print(f"\nDeflection Angle:")
    print(f"  Calculated: {deflection:.6f} rad = {np.degrees(deflection):.4f}°")
    print(f"  Expected:   {4*M/b:.6f} rad = {np.degrees(4*M/b):.4f}°")
    
    # Check if trajectory is in xy-plane
    z_coords = positions[:, 2]
    print(f"\nOut-of-plane motion:")
    print(f"  Z range: [{z_coords.min():.2e}, {z_coords.max():.2e}]")
    print(f"  Max |z|: {np.abs(z_coords).max():.2e}")
    
    # Y-coordinate analysis (should show the deflection)
    y_coords = positions[:, 1]
    print(f"\nY-coordinate (impact parameter direction):")
    print(f"  Start: {y_coords[0]:.2f}M")
    print(f"  Closest: {y_coords[idx_closest]:.2f}M")
    print(f"  End: {y_coords[-1]:.2f}M")
    print(f"  Change: {y_coords[-1] - y_coords[0]:.2f}M")
    
    # Expected deflection in y-direction
    # For small angle α, Δy ≈ α * x_travel
    x_travel = abs(r_end[0] - r_start[0])
    expected_y_change = (4*M/b) * x_travel
    print(f"\nExpected Y-change from deflection:")
    print(f"  X-distance traveled: {x_travel:.1f}M")
    print(f"  Expected Δy ≈ α*x: {expected_y_change:.2f}M")
    print(f"  Actual Δy: {y_coords[-1] - y_coords[0]:.2f}M")
    
    return {
        'incoming_dir': incoming_dir,
        'outgoing_dir': outgoing_dir,
        'deflection': deflection,
        'closest_dist': closest_dist,
        'positions': positions
    }


def plot_trajectory_3d(positions, M, b):
    """3D visualization of trajectory."""
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=1)
    ax1.scatter([0], [0], [0], c='r', s=100, marker='o', label='Black Hole')
    ax1.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 
                c='g', s=50, marker='o', label='Start')
    ax1.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], 
                c='orange', s=50, marker='o', label='End')
    
    ax1.set_xlabel('X (M)')
    ax1.set_ylabel('Y (M)')
    ax1.set_zlabel('Z (M)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # XY plane (main deflection plane)
    ax2 = fig.add_subplot(132)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1)
    ax2.scatter([0], [0], c='r', s=100, marker='o', label='Black Hole')
    ax2.scatter([positions[0, 0]], [positions[0, 1]], c='g', s=50, marker='o', label='Start')
    ax2.scatter([positions[-1, 0]], [positions[-1, 1]], c='orange', s=50, marker='o', label='End')
    
    # Draw impact parameter
    ax2.axhline(y=b, color='k', linestyle='--', alpha=0.3, label=f'b = {b:.0f}M')
    
    ax2.set_xlabel('X (M)')
    ax2.set_ylabel('Y (M)')
    ax2.set_title('XY Plane (Deflection Plane)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Zoomed view near closest approach
    ax3 = fig.add_subplot(133)
    
    # Find closest approach
    distances = np.linalg.norm(positions, axis=1)
    idx_closest = np.argmin(distances)
    
    # Select points near closest approach (middle 20% of trajectory)
    n_points = len(positions)
    start_idx = max(0, n_points//2 - n_points//10)
    end_idx = min(n_points, n_points//2 + n_points//10)
    
    zoom_positions = positions[start_idx:end_idx]
    
    ax3.plot(zoom_positions[:, 0], zoom_positions[:, 1], 'b-', linewidth=2)
    ax3.scatter([0], [0], c='r', s=200, marker='o', label='Black Hole')
    ax3.scatter([positions[idx_closest, 0]], [positions[idx_closest, 1]], 
                c='purple', s=100, marker='*', label='Closest')
    
    # Draw circle at closest approach distance
    closest_r = distances[idx_closest]
    circle = plt.Circle((0, 0), closest_r, fill=False, color='purple', linestyle='--', alpha=0.5)
    ax3.add_patch(circle)
    
    ax3.set_xlabel('X (M)')
    ax3.set_ylabel('Y (M)')
    ax3.set_title(f'Near Closest Approach (r = {closest_r:.1f}M)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    plt.tight_layout()
    plt.savefig('trajectory_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📊 Trajectory plot saved: trajectory_analysis.png")
    plt.show()


def test_straight_line_reference():
    """Compare against a straight-line trajectory (no gravity)."""
    
    print("\n" + "="*60)
    print("STRAIGHT LINE REFERENCE TEST")
    print("="*60)
    
    M = 1.0
    b = 100.0
    r_initial = 10000.0
    
    # Same initial setup
    r0 = np.array([r_initial, 0.0, 0.0])
    aim_point = np.array([0.0, b, 0.0])
    target = aim_point - r0
    
    # Create straight line trajectory
    direction = target / np.linalg.norm(target)
    
    # Generate points along straight line
    lambdas = np.linspace(0, 30000, 1000)
    straight_positions = np.array([r0 + lam * direction for lam in lambdas])
    
    # Calculate "deflection" using same method
    n_asymp = 100
    incoming_vec = straight_positions[n_asymp] - straight_positions[0]
    incoming_dir = incoming_vec / np.linalg.norm(incoming_vec)
    
    outgoing_vec = straight_positions[-1] - straight_positions[-n_asymp]
    outgoing_dir = outgoing_vec / np.linalg.norm(outgoing_vec)
    
    cos_angle = np.dot(incoming_dir, outgoing_dir)
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    
    print(f"\nStraight line 'deflection': {angle:.2e} rad")
    print(f"This should be ~0 for a correct measurement method")
    print(f"Expected GR deflection: {4*M/b:.2e} rad")
    
    if angle > 1e-10:
        print("\n⚠️  WARNING: Method gives non-zero angle for straight line!")
        print("This suggests the measurement method is picking up geometric effects")


def main():
    """Run complete diagnostic."""
    
    # Setup
    M = 1.0
    b = 100.0
    r_initial = 10000.0
    r_escape = 20000.0
    
    r0 = np.array([r_initial, 0.0, 0.0])
    aim_point = np.array([0.0, b, 0.0])
    target = aim_point - r0
    
    state0, p0_initial = create_initial_state(r0, target, M)
    
    print("Running single trajectory integration...")
    trajectory, status = integrate_geodesic(
        state0, M,
        lambda_max=50000.0,
        dt=1.0,
        r_escape=r_escape
    )
    
    print(f"Integration status: {status['reason']}")
    print(f"Number of points: {len(trajectory)}")
    
    # Analyze geometry
    analysis = analyze_trajectory_geometry(trajectory, M, b)
    
    # Plot
    plot_trajectory_3d(analysis['positions'], M, b)
    
    # Test straight line
    test_straight_line_reference()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()