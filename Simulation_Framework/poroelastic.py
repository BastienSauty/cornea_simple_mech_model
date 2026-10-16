# author : B. Sauty ; 5 Oct 2026 
# based on the hyperelastic framework, and previous development in the sandbox. 
#
# This file manages the framework for the simulation of a poroelastic material with either an axisymmetric case or a full 3D case
# Uses the same IO as hyperelastic framework
#
# The physics is written once, in Poroelastic_framework. What differs between the axisymmetric
# and the 3D case (gradients, weight of dx and ds, normal product, integral factor) is provided
# by a geometry object, see geometry.py. The two public classes at the end only choose it.

from .geometry import Axisymmetric, Cartesian3D
from .materials import PoroelasticMaterial
from .mesh_io import read_mesh
from .solver import make_problem, solve_and_report
from .quantities import build_quantities
from .postprocessing_tools.log_utils import log

from functools import cached_property

from mpi4py import MPI
 
from dolfinx import fem
import ufl
from petsc4py.PETSc import ScalarType


class Poroelastic_framework:
    """
    This class builds a simple poroelastic framework. Several hyperelastic law can be implemented.
    The geometry (axisymmetric or 3D) is the class attribute `geometry`, set by the subclasses.
    """
    geometry = None

    def __init__(self, mesh_file, mech_params_json, comm=MPI.COMM_WORLD):
        """
        mesh_file        : gmsh 4.1 .msh file, as written by Meshing_tools (the (r, z) section
                           for the axisymmetric case) : self.domain, self.facet_tag,
                           self.cell_tags and, if the file stores them as
                           $ElementData "e_1"/"e_2", the local reference system
                           self.e_1, self.e_2, self.e_3 (used for the
                           fibres) -- see mesh_io.read_mesh.
        mech_params_json : material parameters, see materials.PoroelasticMaterial.
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
        self.V_p = fem.functionspace(self.domain, ("Lagrange", 1)) 
        self.V_up   = ufl.MixedFunctionSpace(self.V_u, self.V_p)

        self.u = fem.Function(self.V_u)   # displacement unknown — real Function, holds DOF values
        self.p = fem.Function(self.V_p)   # pressure unknown

        self.v, self.q = ufl.TestFunctions(self.V_up)  # test function - shape function in the FEM
        self.u_n = fem.Function(self.V_u) # store results at step n for time discretization in J_dot

        # Store the mechanical parameters in the volume using a DataClass. See parameter_class
        self.mech_params = PoroelasticMaterial.from_json(mech_params_json)
        log(f'[setup] Framework initialized; SEDF type : {self.mech_params.solid.sedf_type}; Porosity type : {self.mech_params.porous.porosity_type}', comm=self.domain.comm)


    def build_weak_form(self, dt):
        """
        Construct the weak form of the mechanical problem.
        dt : fem.Constant, it is a simulation level parameters controled as a simulation variable
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

        # Conservation of momentum : static equilibrium 
        # Displacement residuals - Strain energy density function
        self.mech_params.set_material_laws(self)

        PK1_iso = ufl.diff(self.mech_params.psi_iso, self.F)
        # PK1_solid = ufl.diff(self.mech_params.psi, self.F)
        self.PK1 = self.mech_params.phi_s * PK1_iso - self.J * self.p * self.F_inv.T
        self.R_u = ufl.inner(geo.grad_vector(self.v, self.x), self.PK1) * self.weight * self.dx

        # Conservation of mass : porous equilibrium
        # Backward euler derivatives
        F_n = ufl.variable(I + geo.grad_vector(self.u_n, self.x))
        J_n = ufl.det(F_n)
        J_dot = (self.J - J_n) / dt   # scalar now: (J - J_n) is scalar, dt is scalar

        # self.mech_params.pullback_permeability_tensor(self)
        K_pullback = self.mech_params.K_pullback
        self.R_p = ( J_dot * self.q + ufl.inner(ufl.dot(geo.grad_scalar(self.p, self.x), K_pullback), geo.grad_scalar(self.q, self.x))) * self.weight * self.dx 


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
        boundary_conditions : list of [type, marker, values]

        Displacement field u
            "Dirichlet_disp"           values = ("clamped" or scalar/Constant, axis), axis 0 = r, 1 = z
                                    (axisymmetric) or 0 = x, 1 = y, 2 = z (3D)
            "Neumann_disp"             values = dead-load traction vector (reference area)
            "Neumann_follower_disp"    values = pressure that is applied on the normal of the surface
            "Slip_disp"                values = None (u . N = 0) or a constant vector m (u . m = 0),
                                    penalty stiffness slip_penalty * mu / h
            "Robin_disp"               values = (k, u_ref)

        Pressure field p
            "Dirichlet_pressure"       values = scalar, Constant or callable x -> value
            "Neumann_pressure"         values = outward fluid flux q_n = w . n per reference area
            "Neumann_follower_pressure" values = outward fluid flux q_n per current (deformed) area

        Sets
            self.bcs     : list of fem.DirichletBC
            self.bc_form : UFL form to add to the residual
        """
        self.fdim = self.domain.topology.dim - 1
        self.ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tag,
                            metadata=self.metadata)
        geo, w = self.geometry, self.weight

        N = ufl.FacetNormal(self.domain)       # outward normal, reference configuration
        h = ufl.CellDiameter(self.domain)

        self.bcs, self.bc_form = [], 0
        for bc_type, marker, values in boundary_conditions:
            ds = self.ds(marker)

            # ---------------- displacement ----------------
            if bc_type == "Dirichlet_disp":
                value, axis = values
                V = self.V_u.sub(axis)
                facets = self.facet_tag.find(marker)
                dofs = fem.locate_dofs_topological(V, self.fdim, facets)
                if isinstance(value, str) and value == "clamped":
                    value = ScalarType(0.0)
                self.bcs.append(fem.dirichletbc(value, dofs, V))

            elif bc_type == "Neumann_disp":
                self.bc_form += - ufl.dot(self.v, ufl.as_vector(values)) * w * ds

            elif bc_type == "Neumann_follower_disp":
                # Nanson: n da = J F^-T N dA ; traction t = -p n  ->  residual term + p J F^-T N . v
                self.bc_form += values * self.J * ufl.dot(geo.normal_product(self.F_inv.T, N), self.v) * w * ds

            elif bc_type == "Slip_disp":
                # u . m = 0 by penalty. On a straight boundary (the limbus) m = N is constant,
                # so the nodes stay exactly on the same line.
                m = N if values is None else ufl.as_vector(values)
                k = slip_penalty * self.mech_params.mu / h
                self.bc_form += k * ufl.dot(self.u, m) * ufl.dot(self.v, m) * w * ds

            elif bc_type == "Robin_disp":
                k, u_ref = values
                self.bc_form += k * ufl.inner(self.u - u_ref, self.v) * w * ds

            # ---------------- fluid pressure ----------------
            elif bc_type == "Dirichlet_pressure":
                facets = self.facet_tag.find(marker)
                dofs = fem.locate_dofs_topological(self.V_p, self.fdim, facets)
                if isinstance(values, str) and values == "drained":
                    values = ScalarType(0.0)
                self.bcs.append(fem.dirichletbc(values, dofs, self.V_p))

            elif bc_type == "Neumann_pressure":
                # flux per reference area: + q_n q dA
                self.bc_form += values * self.q * w * ds

            elif bc_type == "Neumann_follower_pressure":
                # flux per current area: da = J |F^-T N| dA (Nanson)
                n_ref = geo.normal_product(self.F_inv.T, N)
                area_ratio = self.J * ufl.sqrt(ufl.dot(n_ref, n_ref))
                self.bc_form += values * area_ratio * self.q * w * ds

            else:
                raise TypeError(f"Unknown boundary condition: {bc_type}")

        log('[setup] Boundary conditions defined', comm=self.domain.comm)


    def build_solver(self, **petsc_overrides):
        """
        build the solver based on the residuals definition. Options: see solver.py,
        petsc_overrides replace the default ones (e.g. snes_atol=1e-12).
        """
        # Assemble Residuals
        self.residuals = self.R_u + self.R_p + self.bc_form
        res_block = ufl.extract_blocks(self.residuals)

        self.problem = make_problem(res_block, [self.u, self.p], self.bcs, "poroelasticity",
                                    **petsc_overrides)
        log('[setup] Solver defined', comm=self.domain.comm)


    def solve_one_step(self, n, p):
        """
        Solve one load step (called after changing the BCs) and print a summary.
        n : step number, p : step value (for the printout)
        """
        solve_and_report(self.problem, n, p, self.domain.comm)

        # update history variable for time discretization
        self.u_n.interpolate(self.u)


class Poroelastic_axisymmetric_framework(Poroelastic_framework):
    """Poroelastic framework, axisymmetric case: (r, z) section, components (r, z, theta)."""
    geometry = Axisymmetric()


class Poroelastic_3D_framework(Poroelastic_framework):
    """Poroelastic framework, full 3D case: components (x, y, z)."""
    geometry = Cartesian3D()
