from dataclasses import dataclass, asdict
from typing import Literal, Optional
import json

SedfType = Literal["Neo-Hookean", "Mooney-Rivlin", "HGO"]

# parameters that must be set (not None) for each model
REQUIRED = {
    "Neo-Hookean":   ("C_10", "D"),
    "Mooney-Rivlin": ("C_10", "C_01", "D"),
    "HGO":           ("C_10", "D", "k1", "k2", "kappa_disp"),
}


@dataclass
class MechParams:
    sedf_type: SedfType = "Mooney-Rivlin" # default value

    # isotropic matrix (shared by both models)
    C_10: Optional[float] = None      # [stress]
    C_01: Optional[float] = None      # [stress], Mooney-Rivlin only
    D: Optional[float] = None         # psi_vol = (J - 1)^2 / D  [1/stress]

    # HGO fibre family
    k1: Optional[float] = None        # fibre stiffness [stress]
    k2: Optional[float] = None        # fibre exponential coefficient [-]
    kappa_disp: Optional[float] = None  # fibre dispersion, 0 (aligned) to 1/3 (isotropic)
    fiber_angle: float = 0.0          # fibre angle in the reference square [deg], 0 = along u

    def __post_init__(self):
        valid = SedfType.__args__
        if self.sedf_type not in valid:
            raise ValueError(f"sedf_type must be one of {valid}, got {self.sedf_type!r}")
        missing = [name for name in REQUIRED[self.sedf_type] if getattr(self, name) is None]
        if missing:
            raise ValueError(f"{self.sedf_type} requires: {missing}")
        if self.D <= 0:
            raise ValueError("D must be > 0 (D -> 0 is the incompressible limit)")
        if self.kappa_disp is not None and not 0 <= self.kappa_disp <= 1 / 3:
            raise ValueError("kappa_disp must be in [0, 1/3]")

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