"""
Module contains reusable constants.
"""
import copy

import astropy.constants

# simulation-specific
HUBBLE = 0.6774
X_H = 0.76

# physical constants (copied to avoid repeated reference to astropy)
k_B = copy.copy(astropy.constants.k_B.cgs.value)
m_p = copy.copy(astropy.constants.m_p.cgs.value)
