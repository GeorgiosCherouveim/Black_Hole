import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from schwarzschild import schwarzschild_geodesic_rhs
from integrator import create_initial_state
import numpy as np

M = 1.0
r0 = np.array([10000.0, 0.0, 0.0])
target = np.array([-10000.0, 100.0, 0.0])

state0, p0_init = create_initial_state(r0, target, M)

print(f"Initial state: {state0}")
print(f"Initial p₀: {p0_init:.10f}")

# Check the RHS
rhs = schwarzschild_geodesic_rhs(state0, M)
print(f"\nRHS at initial state: {rhs}")

# Check constraint at initial state
r = np.linalg.norm(state0[:3])
p = state0[3:6]
xp = np.dot(state0[:3], p)
p2 = np.dot(p, p)

H = -(1.0 - 2*M/r)*p0_init**2 + p2 + 4*M*xp*p0_init/r**2 + 2*M*xp**2/r**3  # ✅ NEW
print(f"\nConstraint H (should be 0): {H:.6e}")

# Take one tiny step
dt = 0.1
state1 = state0 + dt * rhs

# Recompute constraint at new state
r1 = np.linalg.norm(state1[:3])
p1 = state1[3:6]
xp1 = np.dot(state1[:3], p1)
p2_1 = np.dot(p1, p1)

# Solve for new p₀
A = -(1.0 - 2.0*M/r1)  # ✅ NEW
B = 4.0*M*xp1/r1**2   # ✅ NEW (positive)
C = p2_1 + 2.0*M*xp1**2/r1**3  # ✅ NEW
disc = B**2 - 4*A*C
p0_new = (-B - np.sqrt(disc))/(2*A)

H1 = -p0_new**2 + (1.0 + 2*M/r1)*p2_1 + 4*M*xp1*p0_new/r1**2
print(f"\nAfter one Euler step (dt={dt}):")
print(f"  New p₀: {p0_new:.10f}")
print(f"  Constraint H: {H1:.6e}")
print(f"  Change in |p|: {np.linalg.norm(p1) - np.linalg.norm(p):.6e}")