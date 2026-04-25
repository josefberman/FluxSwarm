from numpy import pi, sin, arcsin, cos, arccos, exp

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


def beat_waveform(t: float, amplitude: float):
    """
    Computes the value of a beat waveform at time t
    :param t: time step (seconds)
    :param v_peak: peak velocity (mm/s)
    :param v_dia: diastolic velocity (seconds)  
    :param tau: wavelength (seconds)
    :param upstroke: upstroke duration (seconds)
    :param plateau: plateau duration (seconds)
    :param downstroke: downstroke duration (seconds)
    :return: value of a beat waveform at time t
    """
    # t_cycle = 1/tau
    # t = t % t_cycle
    # dv = v_peak-v_dia
    # upstroke = upstroke*t_cycle
    # plateau = plateau*t_cycle
    # downstroke = downstroke*t_cycle
    # if 0<=t<upstroke:
    #     return v_dia+dv*S(t/upstroke)
    # elif upstroke<=t<upstroke+plateau:
    #     return v_peak
    # elif upstroke+plateau<=t<upstroke+plateau+downstroke:
    #     return v_peak-dv*S((t-upstroke-plateau)/downstroke)
    # else:
    #     return v_dia
    tau = 1
    t_cycle = 1/tau
    t = t % t_cycle
    # if 0<=t<0.15:
    #     return -177.77 * amplitude * t * (t-0.15)  # parabola with positive peak at (0.075,100)
    # elif 0.15<=t<0.25:
    #     return 8.00 * amplitude * (t-0.15) * (t-0.25)  # parabola with negative peak at (0.2,-2)
    # else:
    #     return 0.32 * amplitude * (t-0.25) * (1-t)**6.5  # t-shifted beta distribution with positive peak at (0.35,2)
    return amplitude * exp(-((t-0.14)**2)/(2*0.04**2)) \
        - 0.1*amplitude * exp(-((t-0.32)**2)/(2*0.035**2)) \
        + 0.05*amplitude * exp(-((t-0.45)**2)/(2*0.05**2))


def S(x: float):
    """
    Computes the value of a cubic spline function at time t
    :param x: time step (seconds)
    :return: value of a cubic spline function at time t
    """
    if x<=0:
        return 0
    elif 0<x<1:
        return 6*x**5 - 15*x**4 + 10*x**3
    else:
        return 1