from phi.flow import ZERO_GRADIENT, StaggeredGrid, Box
import numpy as np
from plotting import animate_save_simulation
from logs import create_run_name, create_folders_for_run, log_parameters
from data_structures import Simulation, Swarm, Inflow, Fluid
from simulation import run_simulation

# -------------- Parameter Definition -------------
# Simulation dimensions are length=mm and time=second, mass=ug
sim = Simulation(length_x=720, length_y=36, resolution=(3600, 180), dt=0.05, total_time=20)
swarm = Swarm(num_x=1, num_y=1, left_location=480, bottom_location=18, member_interval_x=1, member_interval_y=1,
              member_radius=1.8, member_density=5150)  # density in ug/mm^3
inflow = Inflow(frequency=2 * np.pi, amplitude=98, radius=sim.length_y / 2, center_y=sim.length_y / 2)
inflow.center_x = 0
fluid = Fluid(viscosity=1E3)  # viscosity of blood in ug/(mm*s)

# -------------- Container Generation --------------
box = Box['x,y', 0:sim.length_x, 0:sim.length_y]

# ---- initial v and p Vector Field Generation ----
boundary = {'x': ZERO_GRADIENT, 'y': 0}
velocity_field = StaggeredGrid(0, boundary=boundary, bounds=box, x=sim.resolution[0], y=sim.resolution[1])

# ----------------- Calculation --------------------
folder_name = create_run_name()
create_folders_for_run(folder_name)
log_parameters(folder_name=folder_name, sim=sim, swarm=swarm, inflow=inflow, fluid=fluid)

run_simulation(velocity_field=velocity_field, pressure_field=None, inflow=inflow, sim=sim,
               swarm=swarm, fluid_obj=fluid, folder_name=folder_name)

# ----------------- Animation --------------------
animate_save_simulation(sim=sim, swarm=swarm, folder_name=folder_name)
