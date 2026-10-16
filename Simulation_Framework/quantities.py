# All the output quantities of the frameworks, defined in one place.
#
# Symbolic only: nothing is compiled here, the OutputManager compiles what it is given.
# The expressions only need the attributes u, F, J and PK1 of the framework. Two features
# of the framework select the extra entries (no test on the class, so there is no import
# of the frameworks and no change here when a new class is added):
#   - framework.output_meta["axisymmetric"]: True adds hoop_stretch (needs framework.r)
#   - hasattr(framework, "p"): the pore pressure unknown, adds fluid_pressure
# For the poroelastic frameworks PK1 already contains the pressure term, so cauchy_stress
# is the total Cauchy stress.
#
# Usage in a framework class:
#     from ..quantities import build_quantities
#
#     @cached_property
#     def quantities(self):
#         return build_quantities(self)

import ufl


def build_quantities(framework):
    """Dictionary of UFL expressions of the standard output quantities. Call after
    build_weak_form (needs framework.u, .F, .J, .PK1)."""
    I = ufl.Identity(3)
    F, J, PK1 = framework.F, framework.J, framework.PK1
    sigma = (1 / J) * ufl.dot(PK1, F.T)
    dev = sigma - ufl.tr(sigma) / 3 * I
    q = {
        "displacement": framework.u,
        "PK1_stress": PK1,
        "cauchy_stress": sigma,
        "green_lagrange": 0.5 * (ufl.dot(F.T, F) - I),
        "J": J,
        "von_mises": ufl.sqrt(3 / 2 * ufl.inner(dev, dev)),
        "hydrostatic_pressure": -ufl.tr(sigma) / 3,
    }
    if framework.output_meta["axisymmetric"]:
        q["hoop_stretch"] = 1 + framework.u[0] / framework.r
    if hasattr(framework, "p"):
        q["fluid_pressure"] = framework.p
    return q
