# Output management for the axisymmetric hyperelastic framework (dolfinx 0.12).
#
# Two kinds of outputs, each picked from a registry by name:
#   FIELDS  : written to an XDMF file (one time step per call to write)
#   SCALARS : one column each in a CSV file (one row per call to write)
#
# Usage:
#   out = OutputManager(mech, "results/cornea_IOP",
#                       fields=["displacement", "cauchy_stress"],
#                       scalars=["apex_uz_anterior", "central_thickness"])
#   out.write(t)      # after each solve
#   out.close()
# or as a context manager: `with OutputManager(...) as out: ...`

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, geometry, io
from petsc4py.PETSc import ScalarType

# physical tag of the limbus in cornea.msh (see cornea/mesh_io.py)
LIMBUS = 3


# --------------------------------------------------------------------------- #
# Kinematics and stresses, built from the framework's UFL objects
# --------------------------------------------------------------------------- #
def _PK1(mech):
    """First Piola-Kirchhoff stress, 3x3 (r, z, theta). mech.F is a ufl.variable."""
    return ufl.diff(mech.psi, mech.F)


def _cauchy(mech):
    """Cauchy stress sigma = J^-1 P F^T, 3x3 (r, z, theta)."""
    return (1 / mech.J) * ufl.dot(_PK1(mech), mech.F.T)


def _von_mises(mech):
    s = _cauchy(mech)
    dev = s - ufl.tr(s) / 3 * ufl.Identity(3)
    return ufl.sqrt(3 / 2 * ufl.inner(dev, dev))


def _inplane(A, w):
    """(r, z) components of A . (w_r, w_z, 0)."""
    return ufl.as_vector([A[0, 0] * w[0] + A[0, 1] * w[1],
                          A[1, 0] * w[0] + A[1, 1] * w[1]])


# --------------------------------------------------------------------------- #
# Field outputs (XDMF)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldOutput:
    description: str
    build: Callable            # mech -> UFL expression
    space: tuple               # ("Lagrange", 1) nodal, or ("DG", 0) one value per cell


FIELDS = {
    "displacement": FieldOutput(
        "displacement (u_r, u_z), interpolated from P2 to P1",
        lambda m: m.u, ("Lagrange", 1)),
    "cauchy_stress": FieldOutput(
        "Cauchy stress, 3x3 in (r, z, theta)",
        _cauchy, ("DG", 0)),
    "PK1_stress": FieldOutput(
        "first Piola-Kirchhoff stress, 3x3 in (r, z, theta)",
        _PK1, ("DG", 0)),
    "green_lagrange": FieldOutput(
        "Green-Lagrange strain E = (F^T F - I) / 2, 3x3",
        lambda m: 0.5 * (ufl.dot(m.F.T, m.F) - ufl.Identity(3)), ("DG", 0)),
    "J": FieldOutput(
        "volume ratio J = det F",
        lambda m: m.J, ("DG", 0)),
    "hoop_stretch": FieldOutput(
        "hoop stretch 1 + u_r / r",
        lambda m: 1 + m.u[0] / m.r, ("DG", 0)),
    "psi": FieldOutput(
        "strain energy density",
        lambda m: m.psi, ("DG", 0)),
    "von_mises": FieldOutput(
        "von Mises equivalent Cauchy stress",
        _von_mises, ("DG", 0)),
    "hydrostatic_pressure": FieldOutput(
        "hydrostatic pressure -tr(sigma) / 3",
        lambda m: -ufl.tr(_cauchy(m)) / 3, ("DG", 0)),
}


# --------------------------------------------------------------------------- #
# Scalar outputs (CSV)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScalarOutput:
    description: str
    build: Callable            # mech -> zero-argument function returning a float


def _integral(mech, integrand):
    """Compile once; return a function giving the global value (all ranks)."""
    form = fem.form(integrand)
    comm = mech.domain.comm
    return lambda: comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)


def _axis_apex(mech, which):
    """Reference coordinates of the anterior (highest) or posterior (lowest) point on the axis."""
    x = mech.domain.geometry.x
    on_axis = np.isclose(x[:, 0], 0.0, atol=1e-10)
    z = x[on_axis, 1]
    comm = mech.domain.comm
    if which == "anterior":
        return comm.allreduce(z.max() if z.size else -np.inf, op=MPI.MAX)
    return comm.allreduce(z.min() if z.size else np.inf, op=MPI.MIN)

def _point_probe(mech, point):
    """Return a function evaluating mech.u at a fixed reference point (all ranks)."""
    domain = mech.domain
    pts = np.array([[point[0], point[1], 0.0]])
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    cells = geometry.compute_colliding_cells(domain, candidates, pts).links(0)
    comm = domain.comm

    def probe():
        if len(cells):
            local = np.asarray(mech.u.eval(pts, cells[:1])).reshape(len(pts), -1)[0]  # [u_r, u_z]
        else:
            local = None
        values = [v for v in comm.allgather(local) if v is not None]
        return values[0]

    return probe


def _apex_uz(which):
    def build(mech):
        probe = _point_probe(mech, (0.0, _axis_apex(mech, which)))
        return lambda: probe()[1]
    return build


def _central_thickness(mech):
    z_a, z_p = _axis_apex(mech, "anterior"), _axis_apex(mech, "posterior")
    ant, post = _point_probe(mech, (0.0, z_a)), _point_probe(mech, (0.0, z_p))
    return lambda: (z_a + ant()[1]) - (z_p + post()[1])


def _limbus_reaction(component):
    def build(mech):
        N = ufl.FacetNormal(mech.domain)
        traction = _inplane(_PK1(mech), N)          # force per reference area on the cornea
        return _integral(mech, 2 * np.pi * traction[component] * mech.r * mech.ds(LIMBUS))
    return build


SCALARS = {
    "apex_uz_anterior": ScalarOutput(
        "axial displacement of the anterior apex", _apex_uz("anterior")),
    "apex_uz_posterior": ScalarOutput(
        "axial displacement of the posterior apex", _apex_uz("posterior")),
    "central_thickness": ScalarOutput(
        "deformed central thickness", _central_thickness),
    "limbus_reaction_r": ScalarOutput(
        "radial force of the limbus support on the cornea (full 3D ring)", _limbus_reaction(0)),
    "limbus_reaction_z": ScalarOutput(
        "axial force of the limbus support on the cornea (full 3D ring)", _limbus_reaction(1)),
    "volume": ScalarOutput(
        "deformed volume of the cornea (full 3D)",
        lambda m: _integral(m, 2 * np.pi * m.J * m.r * m.dx)),
    "strain_energy": ScalarOutput(
        "total strain energy (full 3D)",
        lambda m: _integral(m, 2 * np.pi * m.psi * m.r * m.dx)),
}


def available_outputs():
    """Print the names and descriptions of all field and scalar outputs."""
    print("Field outputs (XDMF):")
    for name, f in FIELDS.items():
        print(f"  {name:22s} {f.description}")
    print("Scalar outputs (CSV):")
    for name, s in SCALARS.items():
        print(f"  {name:22s} {s.description}")


def _check(names, registry, kind):
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"unknown {kind} output(s) {unknown}; available: {list(registry)}")


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #
class OutputManager:
    """
    Writes the selected fields to <basename>.xdmf and scalars to <basename>_scalars.csv.
    Create it after mech.build_BCs (scalars on the limbus use mech.ds and the facet tags).
    """

    def __init__(self, mech, basename, fields=(), scalars=()):
        _check(fields, FIELDS, "field") # verify that the specified fields are all defined in the registry of available fields
        _check(scalars, SCALARS, "scalar")
        if not hasattr(mech, "facet_tag"): # check that the BC are defined before
            raise RuntimeError("call mech.build_BCs(...) before creating the OutputManager")

        self.mech = mech
        self.comm = mech.domain.comm
        base = Path(basename)
        if self.comm.rank == 0:
            base.parent.mkdir(parents=True, exist_ok=True) # check and create folder for output
        self.comm.barrier()

        # ---- fields: one output Function and one compiled Expression each ----
        self.fields = []
        self.xdmf = None
        if fields: # manage all the fields
            spaces = {}
            for name in fields:
                spec = FIELDS[name]         # create a spec
                expr = spec.build(mech)     # build the fem form
                shape = expr.ufl_shape      
                key = spec.space if shape == () else (*spec.space, shape)
                if key not in spaces:                 # share spaces between fields
                    spaces[key] = fem.functionspace(mech.domain, key)
                V = spaces[key]
                f = fem.Function(V, name=name)
                self.fields.append((f, fem.Expression(expr, V.element.interpolation_points))) # Build the Expression and associate with a function that interpolates it

            # create and manage the XDMF file
            self.xdmf = io.XDMFFile(self.comm, str(base.with_suffix(".xdmf")), "w")
            self.xdmf.write_mesh(mech.domain)
            fdim = mech.domain.topology.dim - 1
            mech.domain.topology.create_connectivity(fdim, mech.domain.topology.dim)
            mech.facet_tag.name = "facet_tags"
            self.xdmf.write_meshtags(mech.facet_tag, mech.domain.geometry)

        # ---- scalars: one compiled evaluator each, CSV written on rank 0 ----
        self.scalar_names = list(scalars)
        self.scalars = [SCALARS[n].build(mech) for n in scalars]
        self.history = {name: [] for name in ["t", *self.scalar_names]}
        self._csv = None
        if scalars and self.comm.rank == 0:
            self._csv_file = open(f"{base}_scalars.csv", "w", newline="")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(["t", *self.scalar_names])

    def write(self, t):
        """Write all selected outputs at time (or load) t. Must be called on all ranks."""
        for f, expr in self.fields:
            f.interpolate(expr)
            self.xdmf.write_function(f, t)

        values = [float(evaluate()) for evaluate in self.scalars]
        for name, value in zip(["t", *self.scalar_names], [t, *values]):
            self.history[name].append(value)
        if self._csv is not None:
            self._csv.writerow([t, *values])
            self._csv_file.flush()             # rows are on disk even if a later step fails

    def get_history(self):
        """Scalar outputs written so far, as numpy arrays (same on all ranks)."""
        return {name: np.array(v) for name, v in self.history.items()}

    def close(self):
        if self.xdmf is not None:
            self.xdmf.close()
            self.xdmf = None
        if self._csv is not None:
            self._csv_file.close()
            self._csv = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
