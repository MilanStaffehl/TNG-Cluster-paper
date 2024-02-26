"""
Compute complex quantities from simulation data.

This module provides functions to compute cell values from simulation
data, most prominently temperature: temperature is not directly given
in the simulation data and instead has to be calculated from internal
energy and electron abundance.
"""
import numpy as np
from numpy.typing import NDArray

from library.constants import HUBBLE, X_H, G, M_sol, k_B, kpc, m_p


@np.vectorize
def get_temperature_vectorized(
    internal_energy: float | NDArray,
    electron_abundance: float | NDArray,
    star_formation_rate: float | NDArray
) -> NDArray:
    """
    Return the temperature of the cell(s) given.

    Temperature is calculated according to the temperature formula from
    the `data access FAQ`_, using the quantities and values described
    therein. As described therein, the electron abundance for star forming
    gas is not physically accurate. Since star forming gas is generally
    cold, it is therefore assigned an artificial temperature of 10^3 K.

    The function is vectorized, meaning it also takes arrays as args.
    However, these arrays must be of shape (N,) and have the same length.

    .. _data access FAQ: https://www.tng-project.org/data/docs/faq/#gen6

    :param internal_energy: internal energy of the gas cell in units of
        km/s^2
    :param electron_abundance: number density of electrons in the gas as
        fraction of the hydrogen number density (n_e / n_H)
    :param star_formation_rate: the SFR of the gas cell in solar masses
        per year
    :return: temperature of the gas in Kelvin
    """
    if star_formation_rate > 0:
        return 1e3  # star forming gas electron abundance is not accurate
    # constants are in cgs
    molecular_weight = (4 * m_p / (1 + 3 * X_H + 4 * X_H * electron_abundance))
    temperature = (2 / 3 * internal_energy / k_B * 1e10 * molecular_weight)
    return temperature


def get_temperature(
    internal_energy: float | NDArray,
    electron_abundance: float | NDArray,
    star_formation_rate: float | NDArray
) -> NDArray:
    """
    Return the temperature of the cells given. Uses numpy array maths.

    Function does the same thing as :func:`get_temperature`, but it
    utilizes numpy functions in order to porcess entire arrays without
    the use of ``np.vectorize``.

    :param internal_energy: internal energy of the gas cell in units of
        km/s^2
    :param electron_abundance: number density of electrons in the gas as
        fraction of the hydrogen number density (n_e / n_H)
    :param star_formation_rate: the SFR of the gas cell in solar masses
        per year
    :return: temperature of the gas in Kelvin
    """
    # constants are in cgs
    molecular_weight = (4 * m_p / (1 + 3 * X_H + 4 * X_H * electron_abundance))
    temperature = (2 / 3 * internal_energy / k_B * 1e10 * molecular_weight)
    # star forming gas is assigned 10^3 Kelvin
    return np.where(star_formation_rate > 0, 1e3, temperature)


@np.vectorize
def get_virial_temperature(
    mass: float | NDArray, radius: float | NDArray
) -> NDArray:
    """
    Return the virial temperature of a halo with the given mass and radius.

    The function calculates the virial temperature of a galaxy cluster
    using the common relation 2U = K and relating the kinetic energy to
    temperature as K ~ T.

    Follows the formula for virial temperature in Barkana & Loeb (2001),
    using the virial radius from the simulation data instead of calculating
    it from the virial mass. The coefficients are combined assuming
    a value of 0.6 for the mean molecular weight.

    :param mass: mass of the halo in solar masses
    :param radius: radius of the halo in kpc
    :return: virial temperature of the halo in Kelvin
    """
    if radius == 0:
        return 0.0
    return 0.3 * G * mass * M_sol * m_p / (radius * kpc * k_B)


@np.vectorize
def get_virial_temperature_only_mass(mass: float | NDArray) -> NDArray:
    """
    Return the virial temperature of a halo of the given mass.

    This method is based on the virial temperature estimation in Nelson
    et al. (2013) and makes an approximation for the virial radius based
    on mass. It will produce estimates lower than that of the alternative
    method :meth:`get_virial_temperature` by usually ~0.3 orders of
    magnitude.

    This method is meant to be used in cases where the virial radius is
    not directly available.

    :param mass: mass of the halo in solar masses
    :return: virial temperature of the halo in K
    """
    return 4e5 * (mass / (1e11 * HUBBLE**(-1)))**(2 / 3) / 3


def get_radial_velocities(
    center: NDArray, positions: NDArray, velocities: NDArray
) -> NDArray:
    """
    Calcuate the radial velocities in direction of ``center``.

    The returned array is the radial velocities. Positive value denote
    velocities towards the center, negative velocities denote velocities
    away from the center.

    :param center: Position vector of the center of the sphere, shape
        (3, ).
    :param positions: Array of position vectors of the objects whose
        radial velocity to find. Shape (N, 3).
    :param velocities: Array of velocity vectors of the objects for
        which to find the radial velocity w.r.t. center. Shape (N, 3).
    :return: Array of shape (N, ) of velocity components of the given
        velocities in direction of the center. Positive values denote
        velocity towards center, negative values denote velocity away
        from center.
    """
    radial_vectors = center - positions
    norms = np.linalg.norm(radial_vectors, axis=1)
    unit_vectors = np.divide(radial_vectors, norms[:, np.newaxis])
    return np.sum(velocities * unit_vectors, axis=1)  # pair-wise dot product
