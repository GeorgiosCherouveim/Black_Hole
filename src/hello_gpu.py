import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)  # or ti.cpu if GPU fails

@ti.kernel
def add_arrays(x: ti.types.ndarray(), y: ti.types.ndarray(), result: ti.types.ndarray()):
    for i in range(x.shape[0]):
        result[i] = x[i] + y[i]

# Test data
n = 1000000
x = np.ones(n, dtype=np.float32)
y = np.ones(n, dtype=np.float32) * 2
result = np.zeros(n, dtype=np.float32)

add_arrays(x, y, result)

print("First 5 results:", result[:5])
print("Backend:", ti.lang.impl.current_cfg().arch)