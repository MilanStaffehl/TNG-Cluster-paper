"""
Module contains reusable constants.
"""
import copy

import astropy.constants

# simulation-specific
HUBBLE = 0.6774
X_H = 0.76

# physical constants (copied to minimize lookup time)
G = copy.copy(astropy.constants.G.cgs.value)
k_B = copy.copy(astropy.constants.k_B.cgs.value)
kpc = copy.copy(astropy.constants.kpc.cgs.value)
m_p = copy.copy(astropy.constants.m_p.cgs.value)
M_sol = copy.copy(astropy.constants.M_sun.cgs.value)
