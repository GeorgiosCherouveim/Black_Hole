# Replace your test with this non-radial case:
import sys
sys.path.append('../src')
from schwarzschild import schwarzschild_geodesic_rhs
import numpy as np

# Photon with impact parameter b ≈ 5.5M (strong deflection)
# Position: x=10M, y=5M (not on symmetry axis)
# Momentum: pointing roughly toward origin (not purely radial)

r_vec = np.array([10.0, 5.0, 0.0])
p_dir = np.array([-1.0, -0.5, 0.0])  # Direction with y-component
p_spatial = p_dir / np.linalg.norm(p_dir)  # Normalize

# Solve for p_0 to satisfy null constraint
r = np.linalg.norm(r_vec)
xp = np.dot(r_vec, p_spatial)
A = 1 + 2/r
B = 4 * xp / r**2
C = -1 + 2 * xp**2 / r**3
sqrt_disc = np.sqrt(B**2 - 4*A*C)
p0 = (-B - sqrt_disc) / (2*A)  # Incoming photon

# Build full state [x, y, z, px, py, pz]
state = np.array([*r_vec, *p_spatial])
M = 1.0

derivatives = schwarzschild_geodesic_rhs(state, M, p0_sign=-1)
print("State:", state)
print("Derivatives:", derivatives)
print(f"p0 = {p0:.6f}")

# Check results
assert not np.any(np.isnan(derivatives)), "NaN detected!"
assert derivatives[1] != 0, "No y-motion! Check momentum normalization."
print("✓ Full 3D test passed")