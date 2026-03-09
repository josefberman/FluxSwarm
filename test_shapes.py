import numpy as np
import phi
phi.math.set_global_precision(32)
from phi.torch.flow import *

box = Box['x,y', 0:100, 0:4]
v = StaggeredGrid(0, boundary={'x': ZERO_GRADIENT, 'y': 0}, bounds=box, x=2000, y=80)
v_u, v_v = math.unstack(v.values, '~vector')
print("Staggered v_u shape:", v_u.shape)
print("Staggered v_v shape:", v_v.shape)

p = CenteredGrid(0, bounds=box, x=2000, y=80)
print("Centered p shape:", p.values.shape)
