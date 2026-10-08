from dataclasses import dataclass, asdict, fields
from typing import Literal, Optional, List
import json

from dolfinx import fem
from petsc4py.PETSc import ScalarType
import ufl

# File that contains the parameter classes for the elastic and poroelastic materials

# Solid part
SedfType = Literal["Neo-Hookean", "Mooney-Rivlin", "HGO"]
REQUIRED_SOLID = {
    "Neo-Hookean":   ("C_10",), 
    "Mooney-Rivlin": ("C_10", "C_01"),
    "HGO":           ("C_10", "C_01", "k1", "k2", "a4", "a6", "kappa"),
}

PorosityType = Literal["Isotropic"]
REQUIRED_POROUS = {
    "Isotropic":     ("phi_s", "k")
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
        missing = [name for name in REQUIRED_SOLID[self.sedf_type] if getattr(self, name) is None]
        if missing:
            raise ValueError(f"{self.sedf_type} requires: {missing}")
        
        if self.D is not None and self.D <= 0:
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
        
        self._psi = None
        self._psi_iso = None
            
    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @property
    def psi(self):
        if self._psi is None:
            raise RuntimeError("Call strain_energy_density_function(mech) first")
        return self._psi

    @property
    def psi_iso(self):
        if self.sedf_type == "HGO":
            raise NotImplementedError(
                "psi_iso is not defined for HGO: the fibre invariants I4, I6 are "
                "computed from the full C, so the fibre term is not purely deviatoric."
            )
        if self._psi_iso is None:
            raise RuntimeError("Call strain_energy_density_function(mech) first")
        return self._psi_iso

    def strain_energy_density_function(self, mech):
        """Sets self.psi (total) and self.psi_iso (isochoric part, not for HGO)."""
        C_dev = mech.J**(-2/3) * ufl.dot(mech.F.T, mech.F)
        I1_dev = ufl.tr(C_dev)
        I2_dev = 0.5 * (ufl.tr(C_dev)**2 - ufl.tr(ufl.dot(C_dev, C_dev)))
        const = lambda x: fem.Constant(mech.domain, ScalarType(x))

        C10 = const(self.C_10)
        psi_iso = C10 * (I1_dev - 3)
        if self.sedf_type in ("Mooney-Rivlin", "HGO"):
            psi_iso += const(self.C_01) * (I2_dev - 3)

        if self.sedf_type == "HGO":
            k1, k2, kappa = const(self.k1), const(self.k2), const(self.kappa)
            C = ufl.dot(mech.F.T, mech.F)
            I4 = ufl.inner(ufl.outer(mech.a4, mech.a4), C)
            I6 = ufl.inner(ufl.outer(mech.a6, mech.a6), C)
            E4 = kappa * (I1_dev - 3) + (1 - 3 * kappa) * (I4 - 1)
            E6 = kappa * (I1_dev - 3) + (1 - 3 * kappa) * (I6 - 1)
            fibre = lambda E: ufl.conditional(ufl.gt(E, 0.0), ufl.exp(k2 * E**2) - 1, 0.0)
            psi_iso += k1 / (2 * k2) * (fibre(E4) + fibre(E6))
            self._psi_iso = None
        else:
            self._psi_iso = psi_iso

        # volumetric part, requires D
        if self.D is None:
            self._psi = None            # psi unavailable without D
        else:
            D = const(self.D)
            if self.sedf_type == "HGO":
                psi_vol = 1/(2*D) * (mech.J**2 - 1 - 2*ufl.ln(mech.J))
            else:
                psi_vol = 1/D * (mech.J - 1)**2
            self._psi = psi_iso + psi_vol


@dataclass
class PorousMaterial:
    porosity_type: PorosityType = "Isotropic"
    k: Optional[float] = None
    phi_s: Optional[float] = None

    def __post_init__(self):
        valid = PorosityType.__args__
        if self.porosity_type not in valid:
            raise ValueError(f"porosity_type must be one of {valid}, got {self.porosity_type!r}")
        missing = [n for n in REQUIRED_POROUS[self.porosity_type] if getattr(self, n) is None]
        if missing:
            raise ValueError(f"{self.porosity_type} requires: {missing}")
        if self.k < 0:
            raise ValueError("k must be >= 0")
        if not (0 < self.phi_s <= 1):
            raise ValueError("phi_s must be in (0, 1]")

    def set_permeability(self, mech):
        """K = J F^-1 k F^-T"""
        I = ufl.Identity(3)
        const = lambda x: fem.Constant(mech.domain, ScalarType(x))
        self.phi_s_cst = const(self.phi_s)
        self.k_cst = const(self.k)
        self.K_pullback = mech.J * ufl.dot(ufl.dot(mech.F_inv, self.k_cst * I), mech.F_inv.T)

    

@dataclass
class PoroelasticMaterial:
    solid: HyperelasticMaterial
    porous: PorousMaterial

    @classmethod
    def from_dict(cls, d):
        solid_keys = {f.name for f in fields(HyperelasticMaterial)}
        porous_keys = {f.name for f in fields(PorousMaterial)}
        unknown = set(d) - solid_keys - porous_keys
        if unknown:
            raise ValueError(f"Unknown parameters: {sorted(unknown)}")
        solid = HyperelasticMaterial(**{k: v for k, v in d.items() if k in solid_keys})
        porous = PorousMaterial(**{k: v for k, v in d.items() if k in porous_keys})
        return cls(solid, porous)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump({**asdict(self.solid), **asdict(self.porous)}, f, indent=2)

    # delegation (explicit)
    def set_material_laws(self, mech):
        # set all the needed parameters : solid part
        self.solid.strain_energy_density_function(mech)
        # porous part : permeability
        self.porous.set_permeability(mech)

    def strain_energy_density_function(self, mech):
        return self.solid.strain_energy_density_function(mech)

    def pullback_permeability_tensor(self, mech):
        return self.porous.pullback_permeability_tensor(mech)

    @property
    def psi_iso(self):
        return self.solid.psi_iso

    @property
    def K_pullback(self):
        return self.porous.K_pullback

    @property
    def phi_s(self):
        return self.porous.phi_s_cst
