from numpy import pi, sin, arcsin, cos, arccos

TO_MMHG = 7.501E-6  # From mg/(mm*s^2) to mmHg


def trapezoidal_waveform(t: float, a: float = 1, tau: float = 1, h: float = 1, v: float = 0):
    """
    Computes the value of a trapezoidal waveform at time t
    :param t: time step (seconds)
    :param a: amplitude
    :param tau: wavelength (seconds)
    :param h: horizontal shift
    :param v: vertical shift
    :return: value of a trapezoidal waveform at time t
    """
    return a / pi * (arcsin(sin(pi / tau * t + h)) + arccos(cos(pi / tau * t + h))) - a / 2 + v
