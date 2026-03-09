import numpy as np
from phi.torch.flow import *

box = Box['x,y', 0:100, 0:4]
x_list = [50, 60]
y_list = [2, 2]
r_list = [1.0, 1.0]

x_tensor = tensor(x_list, instance('members'))
y_tensor = tensor(y_list, instance('members'))
radii_tensor = tensor(r_list, instance('members'))

spheres = Sphere(x=x_tensor, y=y_tensor, radius=radii_tensor)

grid = StaggeredGrid(0, bounds=box, x=100, y=10)
mask = math.to_float(spheres.lies_inside(grid.elements.center))
print("lies_inside shape:", mask.shape)

