from dataclasses import dataclass, asdict
from typing import Literal, Optional, List
import json

from dolfinx import fem
from petsc4py.PETSc import ScalarType
import ufl

SedfType = Literal["Neo-Hookean", "Mooney-Rivlin", "HGO"]

# parameters that must be set (not None) for each model
REQUIRED = {
    "Neo-Hookean":   ("C_10", "D"),
    "Mooney-Rivlin": ("C_10", "C_01", "D"),
    "HGO":           ("C_10", "C_01", "D", "k1", "k2", "a4", "a6", "kappa"),
}


@dataclass
class HyperelasticMaterial:
    sedf_type: SedfType = "Mooney-Rivlin" # default value

    # isotropic matrix (shared by both models)
    C_10: Optional[float] = None      # [stress]
    C_01: Optional[float] = None      # [stress], Mooney-Rivlin only
    D: Optional[float] = None         # psi_vol = (J - 1)^2 / D  [1/stress]

    # HGO fibre family
    k1: Optional[float] = None        # fibre stiffness [stress]
    k2: Optional[float] = None        # fibre exponential coefficient [-]
    kappa: Optional[float] = None        # fibre dispersion
    a4: Optional[List[float]] = None        # orientation vector [1,0,0]
    a6: Optional[List[float]] = None        # orientation vector [1,0,0]

    def __post_init__(self):
        valid = SedfType.__args__
        if self.sedf_type not in valid:
            raise ValueError(f"sedf_type must be one of {valid}, got {self.sedf_type!r}")
        missing = [name for name in REQUIRED[self.sedf_type] if getattr(self, name) is None]
        if missing:
            raise ValueError(f"{self.sedf_type} requires: {missing}")
        if self.D <= 0:
            raise ValueError("D must be > 0 (D -> 0 is the incompressible limit)")

        # fibres
        if self.k1 is not None and self.k1 < 0:
            raise ValueError("k1 must be >= 0")
        if self.k2 is not None and self.k2 <= 0:
            raise ValueError("k2 must be > 0 (psi_fibre contains k1 / (2 k2))")
        if self.kappa is not None and self.kappa < 0:
            raise ValueError("kappa must be >= 0")
        for name in ("a4", "a6"):
            a = getattr(self, name)
            if a is None:                       # not used by this law
                continue
            if len(a) != 3 or not any(a):
                raise ValueError(f"{name} must be a non-zero vector [a_u, a_v, a_theta], got {a}")
            setattr(self, name, [float(c) for c in a])      # e.g. [1, 0, 0] -> [1.0, 0.0, 0.0]
            
    @property
    def mu(self):
        """Initial shear modulus of the matrix."""
        return 2 * (self.C_10 + (self.C_01 or 0.0))

    @property
    def bulk_modulus(self):
        """Initial bulk modulus: 2 / D."""
        return 2 / self.D

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

        
    def strain_energy_density_function(self, mech):
        """
        called by build_weak_form. returns psi : the strain energy density function. 
        """
        if self.sedf_type=='Neo-Hookean':
            C_dev = mech.J**(-2/3)*ufl.dot(mech.F.T, mech.F)
            I1_dev = ufl.tr(C_dev)

            C10 = fem.Constant(mech.domain, ScalarType(self.C_10))
            D = fem.Constant(mech.domain, ScalarType(self.D))

            psi = C10 *(I1_dev - 3) + 1/D *(mech.J - 1)**2

        elif self.sedf_type=='Mooney-Rivlin':
            C_dev = mech.J**(-2/3)*ufl.dot(mech.F.T, mech.F)
            I1_dev = ufl.tr(C_dev)
            I2_dev = 1/2* (ufl.tr(C_dev) ** 2 - ufl.tr(ufl.dot(C_dev, C_dev)))

            C01 = fem.Constant(mech.domain, ScalarType(self.C_01))
            C10 = fem.Constant(mech.domain, ScalarType(self.C_10))
            D = fem.Constant(mech.domain, ScalarType(self.D))

            psi = C10 *(I1_dev - 3) + C01 *(I2_dev-3) + 1/D *(mech.J - 1)**2

        elif self.sedf_type=='HGO':
            # Isotropic contribution
            C_dev = mech.J**(-2/3)*ufl.dot(mech.F.T, mech.F)
            I1_dev = ufl.tr(C_dev)
            I2_dev = 1/2* (ufl.tr(C_dev) ** 2 - ufl.tr(ufl.dot(C_dev, C_dev)))

            C01 = fem.Constant(mech.domain, ScalarType(self.C_01))
            C10 = fem.Constant(mech.domain, ScalarType(self.C_10))
            D = fem.Constant(mech.domain, ScalarType(self.D))

            psi = C10 *(I1_dev - 3) + C01 *(I2_dev-3) + 1/(2*D) * (mech.J**2 - 1 - 2*ufl.ln(mech.J))#+ 1/D *(mech.J - 1)**2

            k1 = fem.Constant(mech.domain, ScalarType(self.k1))
            k2 = fem.Constant(mech.domain, ScalarType(self.k2))
            kappa = fem.Constant(mech.domain, ScalarType(self.kappa))

            C = ufl.dot(mech.F.T, mech.F)
            I4 = ufl.inner(ufl.outer(mech.a4, mech.a4), C)
            I6 = ufl.inner(ufl.outer(mech.a6, mech.a6), C)

            E4 = kappa*(I1_dev-3) + (1-3*kappa)*(I4 -1)
            E6 = kappa*(I1_dev-3) + (1-3*kappa)*(I6 -1)

            def fibre_energy(E):
                return ufl.conditional(ufl.gt(E, 0.0), ufl.exp(k2 * E**2) - 1, 0.0)

            psi += k1/(2*k2) * (fibre_energy(E4) + fibre_energy(E6))
        return(psi)
