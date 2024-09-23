"""
Utilities for unit conversion.
"""
from typing import ClassVar, TypeVar

import numpy as np
from numpy.typing import NDArray

from library import constants

N = TypeVar("N", bound=NDArray | float)


class UnsupportedUnitError(Exception):
    """Raise when a unit is converted that is not supported"""

    def __init__(self, field: str, *args: object) -> None:
        super().__init__(*args)
        self.field = field

    def __str__(self) -> str:
        return (
            f"The field {self.field} currently has no supported unit "
            f"conversion available."
        )


class UnitConverter:
    """
    Provides utilities to convert units of simulation data fields.
    """

    fields: ClassVar[dict[list[str]]] = {
        "massLike":
            [
                "GroupMass",
                "GroupMassType",
                "Group_M_Crit200",
                "Group_M_Crit500",
                "Group_M_Mean200",
                "Group_M_TopHat200",
                "Masses",
                "BH_Mass",
                "BH_CumMassGrowth_QM",
                "BH_CumMassGrowth_RM",
                "SubhaloMass",
                "SubhaloMassInRadType",
            ],  # noqa: E123
        "distanceLike":
            [
                "Group_R_Crit200",
                "Group_R_Crit500",
                "Group_R_Mean200",
                "Group_R_TopHat200",
                "GroupPos",
                "Coordinates",
                "SubhaloPos",
            ],  # noqa: E123
        "velocityLike": ["Velocities"],
        "groupVelocityLike": ["GroupVel"],
        "subhaloVelocityLike": ["SubhaloVel"],
        "densityLike": ["Density"],
        "sfrLike": ["GroupSFR", "StarFormationRate", "SubhaloSFR"],
        "massFlowLike": ["GroupBHMdot", "BH_Mdot", "BH_MdotEddington"],
        "energyLike": ["BH_CumEgyInjection_QM", "BH_CumEgyInjection_RM"],
        "metallicityLike":
            [
                "GroupGasMetallicity",
                "GFM_Metallicity",
                "SubhaloGasMetallicity"
            ],  # noqa: E123
        "unitless":
            [
                "count",
                "IDs",
                "GroupLen",
                "GroupFirstSub",
                "GroupNsubs",
                "SubhaloLen",
                "BH_Progs",
                "ParticleIDs",
                "ParentID",
                "TracerID",
            ],  # noqa: E123
    }

    @classmethod
    def supported_fields(cls):
        """
        Return a list of all supported fields.
        """
        supported = []
        for field_list in cls.fields.values():
            supported += field_list
        return supported

    @classmethod
    def convert(cls, quantity: N, field: str, snap_num: int = 99) -> N:
        """
        Automatically convert the quantity into physical units.

        :param quantity: Quantity in code units.
        :param field: Name of the field of the quantity.
        :param snap_num: Number of the snapshot in which unit is located.
        :raises UnsupportedUnitError: When the field has no supported
            unit conversion available.
        :return: Quantity in physical units.
        """
        if field in cls.fields["massLike"]:
            return cls.convert_masslike(quantity)
        elif field in cls.fields["distanceLike"]:
            return cls.convert_distancelike(quantity)
        elif field in cls.fields["velocityLike"]:
            return cls.convert_velocitylike(quantity, snap_num)
        elif field in cls.fields["groupVelocityLike"]:
            return cls.convert_groupvelocitylike(quantity, snap_num)
        elif field in cls.fields["subhaloVelocityLike"]:
            return quantity  # already in km/s
        elif field in cls.fields["densityLike"]:
            return cls.convert_densitylike(quantity)
        elif field in cls.fields["massFlowLike"]:
            return cls.convert_massflowlike(quantity)
        elif field in cls.fields["sfrLike"]:
            return quantity  # Already in M_sol /yr
        elif field in cls.fields["energyLike"]:
            return cls.convert_energylike(quantity)
        elif field in cls.fields["metallicityLike"]:
            return cls.convert_metallicitylike(quantity)
        elif field in cls.fields["unitless"]:
            return quantity
        else:
            raise UnsupportedUnitError(field)

    @staticmethod
    def convert_masslike(quantity: N) -> N:
        """
        Return the mass-like quantity in solar masses.

        :param quantity: Mass-like quantity in code units.
        :return: Mass-like quantity in units of solar masses.
        """
        return quantity * 1e10 / constants.HUBBLE

    @staticmethod
    def convert_distancelike(quantity: N) -> N:
        """
        Return the distance-like quantity in kiloparsec (comoving).

        :param quantity: Distance-like quantity in code units.
        :return: Distance-like quantity in ckpc.
        """
        return quantity / constants.HUBBLE

    @staticmethod
    def convert_velocitylike(quantity: N, snap_num: int) -> N:
        """
        Return the velocity-like quantity in km/s.

        :param quantity: Velocity-like quantity in km/s * sqrt(a).
        :param snap_num: The snapshot number.
        :return: Velocity-like quantity in km/s.
        """
        a = 1 / (1 + constants.REDSHIFTS[snap_num])
        return quantity * np.sqrt(a)

    @staticmethod
    def convert_groupvelocitylike(quantity: N, snap_num: int | N) -> N:
        """
        Return the group-velocity-like quantity in km/s.

        :param quantity: Velocity-like quantity in km/s/sqrt(a).
        :param snap_num: The snapshot number.
        :return: Group-velocity-like in km/s.
        """
        a = 1 / (1 + constants.REDSHIFTS[snap_num])
        return quantity / a

    @staticmethod
    def convert_densitylike(quantity: N) -> N:
        """
        Return the density like quantity in solar masses per ckpc cubed.

        :param quantity: Density-like quantity in code units.
        :return: Desnity-like quantity in M_sol per ckpc cubed.
        """
        return quantity * 1e10 * constants.HUBBLE**2

    @staticmethod
    def convert_massflowlike(quantity: N) -> N:
        """
        Return the mass-flow-like quantity in solar masses per Gyr.

        :param quantity: Mass-flow-like quantity in code units.
        :return: Mass-flow-like quantity in solar masses per Gyr.
        """
        return quantity * 1e10 / 0.978

    @staticmethod
    def convert_energylike(quantity: N) -> N:
        """
        Return the energy-like quantity in solar masses * ckpc^2 per Gyr^2.

        Note that this conversion only works for quantities that are
        already in code units representing solar masses, ckpc and Gyr.
        There are other energy fields that require a different conversion.

        :param quantity: Energy-like quantity in code units.
        :return: Energy-like quantity in solar masses times comoving
            kiloparsec squared per gigayear squared.
        """
        return quantity * 1e10 / (constants.HUBBLE * 0.978**2)

    @staticmethod
    def convert_metallicitylike(quantity: N) -> N:
        """
        Retrun the metallicity-like quantity in solar metallicities (Z_sol).

        :param quantity: Metallicity-like (unit-less; M_Z / M_tot)
        :return: Metallicity in solar units (Z)
        """
        return quantity / 0.0127
