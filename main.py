from phi.flow import *
from plotting import animate_save_simulation
from logs import create_run_name, create_folders_for_run, log_parameters
from data_structures import Simulation, Swarm, Inflow, Fluid
from simulation import run_simulation

# -------------- Parameter Definition -------------
# Simulation dimensions are μm and second
sim = Simulation(length_x=100, length_y=10, resolution=(1000, 1000), dt=0.1, total_time=10)
swarm = Swarm(num_x=5, num_y=5, member_radius=0.04)
inflow = Inflow(frequency=2, amplitude=1000, radius=sim.length_y / 2, center_y=sim.length_y / 2)
#inflow.center_x = inflow.radius + sim.dx
inflow.center_x = 0
fluid = Fluid(viscosity=0.0089)

# -------------- Container Generation --------------
box = Box['x,y', 0:sim.length_x, 0:sim.length_y]

# ---- initial v and p Vector Field Generation ----
boundary = {'x': ZERO_GRADIENT, 'y': 0}
velocity_field = StaggeredGrid(0, boundary=boundary, bounds=box, x=sim.resolution[0], y=sim.resolution[1])
inflow_sphere = Sphere(x=inflow.center_x, y=inflow.center_y, radius=inflow.radius)
inflow_field = CenteredGrid(0, boundary=boundary, bounds=box, x=sim.resolution[0], y=sim.resolution[1])

# ----------------- Calculation --------------------
folder_name = create_run_name()
create_folders_for_run(folder_name)
log_parameters(folder_name=folder_name, sim=sim, swarm=swarm, inflow=inflow, fluid=fluid)

run_simulation(velocity_field=velocity_field, pressure_field=None, inflow_field=inflow_field,
               inflow_sphere=inflow_sphere, inflow=inflow, sim=sim, swarm=swarm, folder_name=folder_name)

# ----------------- Animation --------------------
animate_save_simulation(sim=sim, folder_name=folder_name)
