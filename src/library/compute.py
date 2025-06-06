"""
Compute complex quantities from simulation data.

This module provides functions to compute cell values from simulation
data, most prominently temperature: temperature is not directly given
in the simulation data and instead has to be calculated from internal
energy and electron abundance.
"""
import astropy.cosmology
import astropy.units
import numpy as np
from numpy.typing import NDArray

from library.constants import HUBBLE, X_H, G, M_sol, k_B, kpc, m_p


@np.vectorize
def get_temperature_vectorized(
    internal_energy: float | NDArray,
    electron_abundance: float | NDArray,
    star_formation_rate: float | NDArray
) -> float | NDArray:
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
    utilizes numpy functions in order to process entire arrays without
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
) -> float | NDArray:
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
    center: NDArray,
    halo_velocity: NDArray,
    positions: NDArray,
    velocities: NDArray,
) -> NDArray:
    """
    Calculate the radial velocities in direction of ``center``.

    The returned array is the radial velocities. Positive value denote
    velocities away from the center (outflowing), negative velocities
    denote velocities towards the center (infalling).

    :param center: Position vector of the center of the sphere, shape
        (3, ).
    :param halo_velocity: Velocity vector of the halo center, shape
        (3, 0). Must be in units of km/s. Set this to ``[0, 0, 0]`` if
        the velocities are already w.r.t. the halo center.
    :param positions: Array of position vectors of the objects whose
        radial velocity to find. Shape (N, 3).
    :param velocities: Array of velocity vectors of the objects for
        which to find the radial velocity. Shape (N, 3).
    :return: Array of shape (N, ) of velocity components of the given
        velocities in direction of the center. Positive values denote
        velocity away from the halo center, negative values denote
        velocity towards the halo center. In units of km/s.
    """
    relative_vel = velocities - halo_velocity
    radial_vectors = positions - center
    norms = np.linalg.norm(radial_vectors, axis=1)
    unit_vectors = np.divide(radial_vectors, norms[:, np.newaxis])
    return np.sum(relative_vel * unit_vectors, axis=1)  # pair-wise dot product


def get_virial_velocity(
    virial_mass: float | NDArray, virial_radius: float | NDArray
) -> float | NDArray:
    """
    Return the virial velocity of the halo with the given mass and radius.

    :param virial_mass: Mass of the halo in solar masses.
    :param virial_radius: Radius of the halo in kpc.
    :return: The virial velocity of the halo in km/s.
    """
    return np.sqrt(G * virial_mass * M_sol / (virial_radius * kpc)) / 100000


def get_distance_periodic_box(
    positions_a: NDArray, positions_b: NDArray, box_size: float
) -> NDArray | float:
    """
    Get the distance between points A and points B within a periodic box.

    The function calculates the distance between the points A and B,
    taking into account periodic boundaries of a box of edge size
    ``box_size``. It will automatically limit all distances in one
    direction to half the box size before finding the norm of the
    distance vector between points A and B.

    .. attention:: This function implicitly assumes that all coordinates
        are limited to the box domain, i.e. it assumes that no distance
        between two points can be larger than the box size itself. If
        you have coordinates that can lie outside the box volume, they
        must be normalized to the box volume first!

    :param positions_a: Either an array of shape (N, 3) or a vector of
        shape (3, ).
    :param positions_b: Either an array of shape (N, 3) or a vector of
        shape (3, ).
    :param box_size: The edge length of the cubic box. Distances are
        calculated assuming the box has periodic boundaries, i.e.
        crossing a face one will end up on the opposite face of the
        cube.
    :return: The distance between point(s) A and point(s) B taking into
        account periodic boundaries of a box of size ``box_size``.
    """
    if not np.issubdtype(positions_a.dtype, np.floating):
        positions_a = positions_a.astype(float)
    if not np.issubdtype(positions_b.dtype, np.floating):
        positions_b = positions_b.astype(float)

    half_box_size = 0.5 * box_size
    d = positions_a - positions_b
    # limit to box size
    d[d > half_box_size] -= box_size
    d[d < -half_box_size] += box_size

    if len(d.shape) > 1:
        # list of vectors
        return np.linalg.norm(d, axis=1)
    else:
        # single vector
        return np.linalg.norm(d)


def lookback_time_from_redshift(redshift: NDArray) -> NDArray:
    """
    Return the lookback time for a set of redshifts.

    Function computes the lookback time in Gyr for an array of redshifts.
    The lookback time is calculated using the Planck 2015 cosmology.
    Negative redshift values are ignored and returned as-is.

    :param redshift: Array of redshifts.
    :return: Array of corresponding lookback time in units of Gyr.
        Negative redshifts lead to negative lookback times; these have
        no meaning and should be ignored.
    """
    planck15 = astropy.cosmology.Planck15
    lookback_time = redshift.copy()
    z_pos = np.nonzero(redshift >= 0)[0]
    if z_pos.size != 0:
        t = planck15.lookback_time(redshift[z_pos])
        lookback_time[z_pos] = t.value  # keep negative values as-is
    return lookback_time


def redshift_from_lookback_time(lookback_time: NDArray) -> NDArray:
    """
    Return redshift belonging to a set of lookback times in Gyr.

    Function computes the redshift belonging to the given lookback times
    in Gyr. The redshifts are calculated using the Planck 2015 cosmology.
    Negative lookback times are ignored and returned as-is; lookback times
    that exceed the age of the universe of the Planck 2015 cosmology are
    mapped to ``np.inf``.

    :param lookback_time: An array of lookback times in units of Gyr.
    :return: An array of corresponding redshifts assuming Planck 2015.
        Negative lookback times lead to negative redshifts, lookback
        times greater than the age of the universe lead to ``np.inf``.
        Neither have any meaning and both should be ignored.
    """
    planck15 = astropy.cosmology.Planck15
    redshift = lookback_time.copy()
    universe_age = planck15.age(0).value
    # remove values that are older than the universe
    redshift[lookback_time > universe_age] = np.inf
    # find valid entries
    t_valid = np.nonzero(
        (lookback_time > 0) & (lookback_time <= universe_age)
    )[0]
    if t_valid.size != 0:
        lookback_time_quant = astropy.units.Quantity(
            lookback_time[t_valid], unit="Gyr"
        )
        z = astropy.cosmology.z_at_value(
            planck15.lookback_time, lookback_time_quant
        )
        redshift[t_valid] = z.value
    return redshift
