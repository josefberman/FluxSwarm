import numpy as np
import torch
import warnings
from phi.torch.flow import *

box = Box['x,y', 0:100, 0:4]

x_list = [50, 60]
y_list = [2, 2]
r_list = [1.0, 1.0]

x_tensor = tensor(x_list, instance('members'))
y_tensor = tensor(y_list, instance('members'))
radii_tensor = tensor(r_list, instance('members'))

from phi.geom import Sphere
spheres = Sphere(x=x_tensor, y=y_tensor, radius=radii_tensor)

mask_grid = StaggeredGrid(spheres, boundary={'x': ZERO_GRADIENT, 'y': 0}, bounds=box, x=100, y=10)
print(mask_grid.shape)

mask_members = StaggeredGrid(spheres, boundary={'x': ZERO_GRADIENT, 'y': 0}, bounds=box, x=100, y=10).with_values(
    math.cast(spheres, StaggeredGrid)
)
print("Cast shape:", mask_members.shape)
