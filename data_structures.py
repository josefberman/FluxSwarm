import phi.field
from phi.flow import *
from math import floor


class Swarm:
    def __init__(self, num_x: int = 0, num_y: int = 0, member_radius: float = 0):
        self.num_x = num_x
        self.num_y = num_y
        self.member_radius = member_radius

    def as_obstacle_list(self) -> list:
        swarm = []
        for i in np.linspace(2, 3, self.num_x):
            for j in np.linspace(0.1, 0.9, self.num_y):
                swarm.append(Obstacle(Sphere(x=i, y=j, radius=self.member_radius)))
        return swarm


class Inflow:
    def __init__(self, frequency: float = 0, amplitude: float = 0, radius: float = 0, center_x: float = 0,
                 center_y: float = 0):
        self.frequency = frequency
        self.amplitude = amplitude
        self.radius = radius
        self.center_x = center_x
        self.center_y = center_y


class Fluid:
    def __init__(self, viscosity: float):
        self.viscosity = viscosity


class Simulation:
    def __init__(self, length_x: float = 0, length_y: float = 0, resolution: tuple[int, int] = (0, 0), dt: float = 0,
                 total_time: float = 0):
        self.length_x = length_x
        self.length_y = length_y
        self.resolution = resolution
        self.dx = self.length_x / self.resolution[0]
        self.dy = self.length_y / self.resolution[1]
        self.dt = dt
        self.total_time = total_time
        self.time_steps = floor(self.total_time / self.dt)
