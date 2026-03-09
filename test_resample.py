import numpy as np
import torch
import warnings
from phi.torch.flow import *

box = Box['x,y', 0:100, 0:4]

forces_np = np.array([[10, 5], [-5, -10]])
x_list = [50, 60]
y_list = [2, 2]
r_list = [1.0, 1.0]

x_tensor = tensor(x_list, instance('members'))
y_tensor = tensor(y_list, instance('members'))
radii_tensor = tensor(r_list, instance('members'))

from phi.geom import Sphere
spheres = Sphere(x=x_tensor, y=y_tensor, radius=radii_tensor)
vals = tensor(forces_np, instance('members'), channel(vector='x,y'))

print(f"Spheres shape: {spheres.shape}")
print(f"Vals shape: {vals.shape}")

# Create a StaggeredGrid from the spheres
try:
    mask_grid = StaggeredGrid(spheres, boundary={'x': ZERO_GRADIENT, 'y': 0}, bounds=box, x=100, y=10)
    print("Mask grid created directly.")
    print("Mask grid shape:", mask_grid.values.shape)
    
    # We want to modulate each member's mask by its force and sum
    force_grid = mask_grid * vals
    print("Force grid shape before sum:", force_grid.values.shape)
    
    force_grid = math.sum(force_grid, 'members')
    print("Force grid shape after sum:", force_grid.values.shape)
    print("Success!")
except Exception as e:
    print(f"Method 1 failed: {e}")

