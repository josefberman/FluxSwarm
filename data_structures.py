from phi.flow import *
from math import floor
from numpy.random import rand


class Member:
    def __init__(self, location=None, velocity=None, radius: float = 0, density: float = 1, max_force: float = 0):
        if location is None:
            location = {'x': 0, 'y': 0}
        self.location = location
        if velocity is None:
            velocity = {'x': 0, 'y': 0}
        self.velocity = velocity
        self.radius = radius
        self.density = density
        self.mass = self.density * np.pi * self.radius ** 2
        self.previous_locations = []
        self.previous_velocities = []
        self.previous_forces = []
        self.max_force = max_force

    def as_sphere(self):
        return Sphere(x=self.location['x'], y=self.location['y'], radius=self.radius)


class Swarm:
    def __init__(self, num_x: int = 0, num_y: int = 0, left_location: float = 0, bottom_location: float = 0,
                 member_interval_x: float = 0, member_interval_y: float = 0, member_radius: float = 0,
                 member_density: float = 1, member_max_force: float = 0):
        s = []
        for i in range(num_x):
            for j in range(num_y):
                # s.append(Member(
                #     location={'x': left_location + i * member_interval_x, 'y': bottom_location + j * member_interval_y,
                #               'theta': rand() * 2 * np.pi}, radius=member_radius, density=member_density, max_force=))
                s.append(Member(
                    location={'x': left_location + i * member_interval_x, 'y': bottom_location + j * member_interval_y,
                              'theta': 0}, radius=member_radius, density=member_density, max_force=member_max_force))
        self.members = s
        self.num_x = num_x
        self.num_y = num_y
        self.left_location = left_location
        self.bottom_location = bottom_location
        self.member_interval_x = member_interval_x
        self.member_interval_y = member_interval_y
        self.member_radius = member_radius
        self.member_max_force = member_max_force

    def as_obstacle_list(self) -> list:
        return [Obstacle(geometry=Sphere(x=m.location['x'], y=m.location['y'], radius=m.radius),
                         velocity=vec(x=m.velocity['x'], y=m.velocity['y'])) for m in self.members]

    def as_sphere_list(self) -> list:
        return [Sphere(x=m.location['x'], y=m.location['y'], radius=m.radius) for m in self.members]


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