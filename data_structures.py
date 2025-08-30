from phi.flow import *
from math import floor
import phi.field as field
import numpy as np


class Member:
    """
    Represents a physical member with properties such as location, velocity, radius, and density.

    This class models a physical member with the ability to store its location, velocity,
    radius, density, and related physical properties like mass. It also keeps track of
    historical data such as previous locations, velocities, and forces. The class provides
    a method to represent the member as a geometric sphere.

    :ivar location: The current location of the member as a dictionary with keys 'x' and 'y'.
    :type location: dict
    :ivar velocity: The current velocity of the member as a dictionary with keys 'x' and 'y'.
    :type velocity: dict
    :ivar radius: The radius of the member.
    :type radius: float
    :ivar density: The density of the member.
    :type density: float
    :ivar mass: The mass of the member, calculated based on its radius and density.
    :type mass: float
    :ivar previous_locations: A list of previous locations of the member.
    :type previous_locations: list of dict
    :ivar previous_velocities: A list of previous velocities of the member.
    :type previous_velocities: list of dict
    :ivar previous_forces: A list of previous forces applied on the member.
    :type previous_forces: list
    :ivar max_force: The maximum force that can be applied on the member.
    :type max_force: float
    """

    def __init__(self, location=None, velocity=None, radius: float = 0, density: float = 1, max_force: float = 0):
        if location is None:
            location = {'x': 0, 'y': 0}
        self.location = location
        if velocity is None:
            velocity = {'x': 0, 'y': 0}
        self.action = {'x': 0, 'y': 0}
        self.velocity = velocity
        self.radius = radius
        self.density = density
        self.mass = self.density * 4 / 3 * np.pi / radius ** 3
        self.previous_locations = [self.location]
        self.previous_velocities = [self.velocity]
        self.previous_actions = [self.action]
        self.max_force = max_force

    def as_sphere(self):
        return Sphere(x=self.location['x'], y=self.location['y'], radius=self.radius)


class Swarm:
    """
    Represents a swarm of members organized in a grid-like arrangement. This class is used to manage and
    initialize the members of the swarm with specific attributes such as location, interval, radius,
    density, and force.

    It provides utility methods to interpret the swarm members as obstacles or spheres.

    :ivar members: A list of `Member` objects that represent the entities of the swarm.
    :type members: list
    :ivar num_x: Number of members in the x-direction.
    :type num_x: int
    :ivar num_y: Number of members in the y-direction.
    :type num_y: int
    :ivar left_location: X-coordinate of the leftmost point of the swarm grid.
    :type left_location: float
    :ivar bottom_location: Y-coordinate of the bottommost point of the swarm grid.
    :type bottom_location: float
    :ivar member_interval_x: Distance between adjacent members in the x-direction.
    :type member_interval_x: float
    :ivar member_interval_y: Distance between adjacent members in the y-direction.
    :type member_interval_y: float
    :ivar member_radius: Radius of each member in the swarm.
    :type member_radius: float
    :ivar member_max_force: Maximum force applicable to a member in the swarm.
    :type member_max_force: float
    """

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
    """
    Represents an Inflow with properties that define its frequency, amplitude, radius,
    and center coordinates in a given space.

    This class models an inflow with parameters capable of defining oscillatory flows,
    spatial areas of influence, and their specific locations.

    :ivar frequency: Represents the frequency of the inflow.
    :type frequency: float
    :ivar amplitude: Represents the amplitude of the inflow.
    :type amplitude: float
    :ivar radius: Represents the radius within which the inflow has an effect.
    :type radius: float
    :ivar center_x: X-coordinate of the inflow's center.
    :type center_x: float
    :ivar center_y: Y-coordinate of the inflow's center.
    :type center_y: float
    """

    def __init__(self, frequency: float = 0, amplitude: float = 0, h_shift: float = 0, v_shift: float = 0):
        self.frequency = frequency
        self.amplitude = amplitude
        self.h_shift = h_shift
        self.v_shift = v_shift


class Fluid:
    """
    Represents a fluid with specific properties.

    The Fluid class is used to model characteristics of a fluid. It primarily
    focuses on the viscosity property, which describes the fluid's resistance
    to deformation. This class can serve as a basis for simulating fluid
    behavior in various engineering and scientific applications.

    :ivar viscosity: The viscosity of the fluid which determines its flow
        resistance.
    :type viscosity: float
    """

    def __init__(self, viscosity: float):
        self.viscosity = viscosity


class Simulation:
    """
    Represents a simulation environment with defined dimensions, resolution,
    and time parameters.

    The class initializes a simulation space characterized by its spatial
    dimensions, grid resolution, and temporal properties. It also computes
    derived attributes like grid spacings in both x and y directions and the
    total number of time steps. This class serves as a foundational setup
    for simulations requiring spatial and temporal discretization.

    :ivar length_x: The length of the simulation domain in the x-direction.
    :type length_x: float
    :ivar length_y: The length of the simulation domain in the y-direction.
    :type length_y: float
    :ivar resolution: A tuple representing the number of grid points in the
        x and y directions, respectively.
    :type resolution: tuple[int, int]
    :ivar dx: The grid spacing in the x-direction, derived from length_x
        and resolution.
    :type dx: float
    :ivar dy: The grid spacing in the y-direction, derived from length_y
        and resolution.
    :type dy: float
    :ivar dt: The time step for the simulation.
    :type dt: float
    :ivar total_time: The total simulation time.
    :type total_time: float
    :ivar time_steps: The number of discrete time steps in the simulation,
        derived from total_time and dt.
    :type time_steps: int
    """

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
