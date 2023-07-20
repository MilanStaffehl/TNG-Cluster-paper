"""
Compute complex quantities from simulation data.

This module provides functions to compute cell values from simulation 
data, most prominently temperature: temperature is not directly given 
in the simulation data and instead has to be calculated from internal 
energy and electron abundance. 
"""
import numpy as np
from astropy.constants import k_B, m_p

from constants import X_H


@np.vectorize
def get_temperature(internal_energy, electron_abundance):
    """
    Return the temperature of the cell(s) given.

    Temperature is calculated according to the temperature formula from
    the `data access FAQ`_, using the quantities and values described
    therein.

    The function is vectorized, meaning it also takes arrays as args.
    However, these arrays must be of shape (N,) and have the same length.

    :param internal_energy: internal energy of the gas cell in units of
        km/s^2
    :param electron_abundance: number density of electrons in the gas as
        fraction of the hydrogen number density (n_e / n_H)
    :return: temperature of the gas in Kelvin
    """
    # constants are in cgs
    molecular_weight = 4 * m_p.cgs.value / (1 + 3 * X_H + 4 * X_H * electron_abundance)
    temperature = 2 / 3 * internal_energy / k_B.cgs.value * 1e10 * molecular_weight
    return temperature
