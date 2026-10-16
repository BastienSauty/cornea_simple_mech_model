# author : B. Sauty ; 23 Sept 2026 (updated 28 Sept 2026: adopt the mesh's local
# reference system e_1/e_2/e_3 # and move mesh reading out to mesh_io, shared by every framework)
# This file aims at building a framework class to model mechanical problems using fenicsx. Simple cases
# Different classes are built in order to run several types of simulation. 
# Axisymmetric or full 3D geometry. Single material
#
# The physics is written once, in Hyperelastic_framework. What differs between the axisymmetric
# and the 3D case (gradients, weight of dx and ds, normal product, integral factor) is provided
# by a geometry object, see geometry.py. The two public classes at the end only choose it.

from .geometry import Axisymmetric, Cartesian3D, push_local_frame
from .materials import HyperelasticMaterial
from .mesh_io import read_mesh
from .solver import make_problem, solve_and_report
from .quantities import build_quantities
from .postprocessing_tools.log_utils import log

from functools import cached_property

from mpi4py import MPI
 
from dolfinx import fem
import ufl
from petsc4py.PETSc import ScalarType
 

class Hyperelastic_framework:
    """
    This class builds a simple hyperelastic framework. Several hyperelastic law can be implemented.
    The geometry (axisymmetric or 3D) is the class attribute `geometry`, set by the subclasses.
    """
    geometry = None

    def __init__(self, mesh_file, mech_params_json, comm=MPI.COMM_WORLD):
        """
        mesh_file        : gmsh 4.1 .msh file, as written by meshing_tools (the (r, z) section
                           for the axisymmetric case) : self.domain, self.facet_tag,
                           self.cell_tags and, if the file stores them as
                           $ElementData "e_1"/"e_2", the local reference system
                           self.e_1, self.e_2, self.e_3 (used for the
                           fibres) -- see mesh_io.read_mesh.
        mech_params_json : material parameters, see materials.HyperelasticMaterial.
        """
        self.mesh_file = mesh_file
        mesh_data = read_mesh(mesh_file, comm)
        self.domain = mesh_data.domain
        self.facet_tag = mesh_data.facet_tag
        self.cell_tags = mesh_data.cell_tags
        self.e_1 = mesh_data.e_1
        self.e_2 = mesh_data.e_2
        self.e_3 = mesh_data.e_3

        log(f"[setup] Mesh read from {mesh_file}"
              + (" (with local reference system)" if mesh_data.has_lrs else ""), comm=self.domain.comm)

        self.V_u = fem.functionspace(self.domain, ("Lagrange", 2, (self.domain.geometry.dim,))) # disp function space

        self.u = fem.Function(self.V_u)   # displacement unknown — real Function, holds DOF values
        self.v = ufl.TestFunction(self.V_u) # test function - shape function in the FEM

        # Store the mechanical parameters in the volume using a DataClass. See parameter_class
        self.mech_params = HyperelasticMaterial.from_json(mech_params_json)
        log(f'[setup] Framework initialized; SEDF type : {self.mech_params.sedf_type}', comm=self.domain.comm)

        if self.mech_params.sedf_type == "HGO":     # a4, a6 exist only for HGO
            self._fibre_orientation_field()

    def _fibre_orientation_field(self):
        """
        Build self.a4, self.a6: fibre directions (unit UFL vectors in r, z, theta, or x, y, z),
        given in the mesh's local reference system (LRS) by mech_params.a4, a6 =
        (a_1, a_2, a_3), and expressed in physical coordinates by a plain orthonormal
        change of basis (see push_local_frame).
        """
        if not hasattr(self, "e_1") or self.e_1 is None:
            raise RuntimeError(f"fibre directions need the mesh's local reference system, but "
                               f"{self.mesh_file} has no $ElementData 'e_1' and 'e_2'")
        a4_local = fem.Constant(self.domain, ScalarType(self.mech_params.a4))
        a6_local = fem.Constant(self.domain, ScalarType(self.mech_params.a6))
 
        self.a4 = push_local_frame(self.e_1, self.e_2, self.e_3, a4_local)
        self.a6 = push_local_frame(self.e_1, self.e_2, self.e_3, a6_local)
        log(f"[setup] Fibre fields built: a4 = {self.mech_params.a4}, "
              f"a6 = {self.mech_params.a6} (local frame)", comm=self.domain.comm)

    def build_weak_form(self):
        """
        Construct the weak form of the mechanical problem. 
        """
        self.metadata = {"quadrature_degree": 4}
        self.dx = ufl.Measure("dx", domain=self.domain, metadata=self.metadata)
        geo = self.geometry

        # Identity tensor
        I = ufl.variable(ufl.Identity(3))

        # kinematic quantities
        self.x = ufl.SpatialCoordinate(self.domain)
        if geo.axisymmetric:
            self.r = self.x[0]                      # radial coordinate (axisymmetric only)
        self.weight = geo.weight(self.x)            # multiplies dx and ds: r (axisymmetric) or 1

        # Deformation gradient
        self.F = ufl.variable(I + geo.grad_vector(self.u, self.x))
        self.F_inv = ufl.inv(self.F)
        self.J = ufl.det(self.F)

        # Strain energy density function
        self.mech_params.strain_energy_density_function(self)
        self.PK1 = ufl.diff(self.mech_params.psi, self.F) # PK1 stress

        # Residuals
        self.R_u = ufl.inner(geo.grad_vector(self.v, self.x), self.PK1) * self.weight * self.dx


    @property
    def output_meta(self):
        """Metadata for the OutputManager: labels of the components, axisymmetric or not."""
        return {"axes": self.geometry.axes, "axisymmetric": self.geometry.axisymmetric}

    @cached_property
    def quantities(self):
        """
        Standard output quantities as UFL expressions. Symbolic only:
        nothing is compiled here, the OutputManager compiles what it is given.
        See quantities.py for the list.
        """
        return build_quantities(self)

    def volume_integral(self, expr):
        """UFL form of the integral of expr over the full 3D body (2 pi r weight if axisymmetric)."""
        return self.geometry.integral_factor * expr * self.weight * self.dx

    def surface_integral(self, expr, marker):
        """UFL form of the integral of expr over the full 3D surface of tag `marker`. Needs build_BCs."""
        if not hasattr(self, "ds"):
            raise RuntimeError("call build_BCs before surface_integral (self.ds is defined there)")
        return self.geometry.integral_factor * expr * self.weight * self.ds(marker)

    def build_BCs(self, boundary_conditions, slip_penalty=1e3):
        """
        facet_tag           : dolfinx MeshTags of the boundary facets (from cornea.msh).
        boundary_conditions : list of [type, marker, values]:
            "Dirichlet"         values = ("clamped" or Constant, axis), axis 0 = r, 1 = z
                                (axisymmetric) or 0 = x, 1 = y, 2 = z (3D)
            "Slip"              values = None: slide along the boundary (u . N = 0, N the
                                reference facet normal), or a constant vector m (u . m = 0).
                                Enforced by a penalty, stiffness slip_penalty * mu / h.
            "Neumann_follower"  values = p (Constant): follower pressure normal to the
                                deformed surface, p > 0 pushes into the body
            "Neumann"           values = dead-load traction vector (Constant)
            "Robin"             values = (k, u_ref)
 
        Sets
            self.bcs     : list of fem.DirichletBC, to pass to NonlinearProblem(bcs=...)
            self.bc_form : UFL form to add to the residual
        """
        self.fdim = self.domain.topology.dim - 1
        # subdomain_data is required for ds(marker) to integrate over the tagged facets
        self.ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tag,
                              metadata=self.metadata)
        geo, w = self.geometry, self.weight
 
        N = ufl.FacetNormal(self.domain)       # outward normal, reference configuration
        h = ufl.CellDiameter(self.domain)
 
        self.bcs, self.bc_form = [], 0
        for bc_type, marker, values in boundary_conditions:
            ds = self.ds(marker)
 
            if bc_type == "Dirichlet":
                value, axis = values
                facets = self.facet_tag.find(marker)
                dofs = fem.locate_dofs_topological(self.V_u.sub(axis), self.fdim, facets)
                if isinstance(value, str) and value == "clamped":
                    value = ScalarType(0.0)
                self.bcs.append(fem.dirichletbc(value, dofs, self.V_u.sub(axis)))
 
            elif bc_type == "Neumann_follower":
                # Nanson: n da = J F^-T N dA ; traction t = -p n  ->  residual term + p J F^-T N . v
                self.bc_form += values * self.J * ufl.dot(geo.normal_product(self.F_inv.T, N), self.v) * w * ds
 
            elif bc_type == "Slip":
                # u . m = 0 by penalty. On a straight boundary (the limbus) m = N is constant,
                # so the nodes stay exactly on the same line.
                m = N if values is None else ufl.as_vector(values)
                k = slip_penalty * self.mech_params.mu / h
                self.bc_form += k * ufl.dot(self.u, m) * ufl.dot(self.v, m) * w * ds

            elif bc_type == "Neumann":
                self.bc_form += - ufl.dot(self.v, ufl.as_vector(values)) * w * ds
 
            elif bc_type == "Robin":
                k, u_ref = values
                self.bc_form += k * ufl.inner(self.u - u_ref, self.v) * w * ds
 
            else:
                raise TypeError(f"Unknown boundary condition: {bc_type}")

        log('[setup] Boundary conditions defined', comm=self.domain.comm)


    def build_solver(self, **petsc_overrides):
        """
        build the solver based on the residuals definition. Options: see solver.py,
        petsc_overrides replace the default ones (e.g. snes_atol=1e-12).
        """
        # Assemble Residuals
        self.residuals = self.R_u + self.bc_form

        self.problem = make_problem(self.residuals, self.u, self.bcs, "hyperelasticity",
                                    **petsc_overrides)
        log('[setup] Solver defined', comm=self.domain.comm)


    def solve_one_step(self, n, p):
        """
        Solve one load step (called after changing the BCs) and print a summary.
        n : step number, p : load value (for the printout)
        """
        solve_and_report(self.problem, n, p, self.domain.comm)


class Hyperelastic_axisymmetric_framework(Hyperelastic_framework):
    """Hyperelastic framework, axisymmetric case: (r, z) section, components (r, z, theta)."""
    geometry = Axisymmetric()


class Hyperelastic_3D_framework(Hyperelastic_framework):
    """Hyperelastic framework, full 3D case: components (x, y, z)."""
    geometry = Cartesian3D()
