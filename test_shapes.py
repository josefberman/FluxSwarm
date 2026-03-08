import math as pymath
from phi.torch.flow import *
import phi
import numpy as np

# Create a mock environment
v = StaggeredGrid(0, boundary={'x': ZERO_GRADIENT, 'y': 0}, bounds=Box['x,y', 0:100, 0:4], x=10, y=10)

v_u, v_v = math.unstack(v.values, '~vector')
print(f"Original v_u shape: {v_u.shape}")

# Create mask
mask = math.ones(v_u.shape['y']) * 5.0
print(f"Mask shape: {mask.shape}")

# Option 1: Concat
mask_x1 = math.expand(mask, spatial(x=1))
v_tensor_u = math.concat([mask_x1, v_u[{'x': slice(1, None)}]], dim='x')
print(f"Concat v_tensor_u shape: {v_tensor_u.shape}")

# Option 2: math.where with spatial index
x_idx = math.range_tensor(v_u.shape['x'])
v_tensor_u_where = math.where(x_idx == 0, mask, v_u)
print(f"Where v_tensor_u shape: {v_tensor_u_where.shape}")
