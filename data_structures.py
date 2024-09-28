from phi.flow import *
from math import floor


class Swarm:
    def __init__(self, num_x: int = 0, num_y: int = 0, left_location: float = 0, bottom_location: float = 0,
                 member_interval_x: float = 0, member_interval_y: float = 0, member_radius: float = 0):
        self.num_x = num_x
        self.num_y = num_y
        self.left_location = left_location
        self.bottom_location = bottom_location
        self.member_interval_x = member_interval_x
        self.member_interval_y = member_interval_y
        self.member_radius = member_radius

    def as_obstacle_list(self) -> list:
        swarm = []
        for i in np.linspace(self.left_location,
                             self.left_location + self.member_interval_x * (self.num_x - 1),
                             self.num_x):
            for j in np.linspace(self.bottom_location,
                                 self.bottom_location + self.member_interval_y * (self.num_y - 1),
                                 self.num_y):
                swarm.append(Obstacle(Sphere(x=i, y=j, radius=self.member_radius)))
        return swarm

    def as_coordinate_list(self) -> list:
        swarm = []
        for i in np.linspace(self.left_location,
                             self.left_location + self.member_interval_x * (self.num_x - 1),
                             self.num_x):
            for j in np.linspace(self.bottom_location,
                                 self.bottom_location + self.member_interval_y * (self.num_y - 1),
                                 self.num_y):
                swarm.append((i, j))
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
