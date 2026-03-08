import os
from data_structures import Simulation, Inflow, Fluid
from phi.torch.flow import *
import phi

sim = Simulation(length_x=100, length_y=4, resolution=(10, 10), dt=0.05, total_time=2000)
box = Box['x,y', 0:sim.length_x, 0:sim.length_y]
v = CenteredGrid(1.0, bounds=box, x=10, y=10)

coords = tensor([[1, 2], [3, 4]], instance('pts'), channel(vector='x,y'))
try:
    print("Trying f.at()")
    res = v.at(coords)
    print("Success:", type(res))
except Exception as e:
    print("f.at failed:", e)

try:
    print("Trying f.at(PointCloud)")
    from phi.geom import PointCloud
    res = v.at(PointCloud(coords))
    print("Success:", type(res))
except Exception as e:
    print("f.at(PointCloud) failed:", e)

try:
    print("Trying field.sample")
    res = phi.field.sample(v, coords)
    print("Success:", type(res))
except Exception as e:
    print("field.sample failed:", e)
