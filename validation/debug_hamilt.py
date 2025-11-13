import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from integrator import create_initial_state, integrate_geodesic
import numpy as np

M = 1.0
b = 100.0 * M
r0 = np.array([1000.0, 0.0, 0.0])
target = np.array([-1000.0, b, 0.0])

# Check initial Hamiltonian
state0, p0 = create_initial_state(r0, target, M)
r = np.linalg.norm(state0[:3])
xp = np.dot(state0[:3], state0[3:6])
H_initial = -p0**2 + (1 + 2*M/r) - 4*M*xp/r**2 * p0 + 2*M*xp**2/r**3
print(f"Initial Hamiltonian H (should be 0): {H_initial:.2e}")

# Check after one RK4 step
from schwarzschild import schwarzschild_geodesic_rhs
k1 = schwarzschild_geodesic_rhs(state0, M)
k2 = schwarzschild_geodesic_rhs(state0 + 0.5*0.5*k1, M)
k3 = schwarzschild_geodesic_rhs(state0 + 0.5*0.5*k2, M)
k4 = schwarzschild_geodesic_rhs(state0 + 0.5*k3, M)
state1 = state0 + 0.5/6.0 * (k1 + 2*k2 + 2*k3 + k4)

# Recompute p0 at new state
r1 = np.linalg.norm(state1[:3])
xp1 = np.dot(state1[:3], state1[3:6])
A = -1.0; B = -4*M*xp1/r1**2; C = 1.0 + 2*M/r1
p0_new = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)
H_final = -p0_new**2 + (1 + 2*M/r1) - 4*M*xp1/r1**2 * p0_new + 2*M*xp1**2/r1**3
print(f"Hamiltonian after 1 step (should be 0): {H_final:.2e}")
print(f"Drift: {abs(H_final - H_initial):.2e}")