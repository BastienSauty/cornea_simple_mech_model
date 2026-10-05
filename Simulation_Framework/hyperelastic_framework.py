# author : B. Sauty ; 23 Sept 2026 (updated 28 Sept 2026: adopt the mesh's local
# reference system e_1/e_2/e_3 # and move mesh reading out to mesh_io, shared by every framework)
# This file aims at building a framework class to model mechanical problems using fenicsx. Simple cases
# Different classes are built in order to run several types of simulation. 
# The geometry is always assumed to be axisymetrical. Single material

from .parameters_class import HyperelasticMaterial
from .mesh_io import read_mesh
from .Postprocessing_tools.log_utils import log

from functools import cached_property

from mpi4py import MPI
 
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx import fem, mesh, io
import ufl
from petsc4py.PETSc import ScalarType



def _grad_axi(u, r):
    """
    Function to compute the axisymmetric gradient in the coordinate system (r, z, theta)
    with z the axis of symmetry and r = x[0] the radial coordinate
    """
    grad2D = ufl.grad(u)
    grad_axi = ufl.as_tensor([[grad2D[0,0], grad2D[0,1], 0],
                                [grad2D[1,0], grad2D[1,1], 0],
                                [0, 0, u[0]/r]])
    return(grad_axi)

def _inplane(A, w):
    """
    axisymmetric product of A * w for the Neumann follower pull back operation.
    (r, z) components of A . (w_r, w_z, 0). Works for a 2x2 tensor or a 3x3
    axisymmetric one (block-diagonal: the hoop component does not mix in)."""
    return ufl.as_vector([A[0, 0] * w[0] + A[0, 1] * w[1],
                          A[1, 0] * w[0] + A[1, 1] * w[1]])

def _push_local_frame(e_1, e_2, e_3, A):
    """
    A = (a_1, e_2, a_3): a direction given by its components in the
    mesh's local reference system (e_1, e_2, e_3) -> the same
    direction in physical (r, z, theta)/(x,y,z) space, as a unit UFL vector.

    (e_1, e_2, e_3) is an orthonormal triad by construction directly within
    the mesh. In the case of axisymmetry, the e_3 is built as the out of plane direction
    So this is a plain change of basis 
    A does not need to be a unit vector itself only the final result is 
    normalised.
    """
    a = A[0] * e_1 + A[1] * e_2 + A[2] * e_3
    return a / ufl.sqrt(ufl.dot(a, a))
 

class Hyperelastic_axisymmetric_framework:
    """
    This class builds a simple hyperelastic framework. Several hyperelastic law can be implemented.
    """
    def __init__(self, mesh_file, mech_params_json, comm=MPI.COMM_WORLD):
        """
        mesh_file        : gmsh 4.1 .msh file of the (r, z) section, as written by
                           meshing_tools : self.domain, self.facet_tag,
                           self.cell_tags and, if the file stores them as
                           $ElementData "e_1"/"e_2", the local reference system
                           self.e_1, self.e_2, self.e83 (used for the
                           fibres) -- see mesh_io.read_mesh.
        mech_params_json : material parameters, see parameters_class.HyperelasticMaterial.
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


    def _fibre_orientation_field(self):
        """
        Build self.a4, self.a6: fibre directions (unit UFL vectors in r, z, theta),
        given in the mesh's local reference system (LRS) by mech_params.a4, a6 =
        (a_1, a_2, a_3), and expressed in physical (r, z, theta)
        coordinates by a plain orthonormal change of basis (see _push_local_frame).
        """
        if not hasattr(self, "e_1") or self.e_1 is None:
            raise RuntimeError(f"fibre directions need the mesh's local reference system, but "
                               f"{self.mesh_file} has no $ElementData 'e_1' and 'e_2'")
        a4_local = fem.Constant(self.domain, ScalarType(self.mech_params.a4))
        a6_local = fem.Constant(self.domain, ScalarType(self.mech_params.a6))
 
        self.a4 = _push_local_frame(self.e_1, self.e_2, self.e_3, a4_local)
        self.a6 = _push_local_frame(self.e_1, self.e_2, self.e_3, a6_local)
        log(f"[setup] Fibre fields built: a4 = {self.mech_params.a4}, "
              f"a6 = {self.mech_params.a6} (local frame)", comm=self.domain.comm)

    def build_weak_form(self):
        """
        Construct the weak form of the mechanical problem. 
        """
        self.metadata = {"quadrature_degree": 4}
        self.dx = ufl.Measure("dx", domain=self.domain, metadata=self.metadata)

        # Identity tensor
        I = ufl.variable(ufl.Identity(3))

        # kinematic quantities
        self.x = ufl.SpatialCoordinate(self.domain)
        self.r = self.x[0]

        # Deformation gradient
        self.F = ufl.variable(I + _grad_axi(self.u, self.r))
        self.F_inv = ufl.inv(self.F)
        self.J = ufl.det(self.F)

        # Strain energy density function
        self.psi = self.mech_params.strain_energy_density_function(self)
        PK1 = ufl.diff(self.psi, self.F) # PK1 stress

        # Residuals
        self.R_u = ufl.inner(_grad_axi(self.v, self.r), PK1) * self.r * self.dx


    @property
    def output_meta(self):
        """Metadata for the OutputManager: labels of the components (r, z, theta), axisymmetric."""
        return {"axes": ["r", "z", "t"], "axisymmetric": True}

    @cached_property
    def quantities(self):
        """
        Standard output quantities as UFL expressions in (r, z, theta). Symbolic only:
        nothing is compiled here, the OutputManager compiles what it is given.
        """
        I = ufl.Identity(3)
        PK1 = ufl.diff(self.psi, self.F)
        sigma = (1 / self.J) * ufl.dot(PK1, self.F.T)
        dev = sigma - ufl.tr(sigma) / 3 * I
        return {
            "displacement": self.u,
            "PK1_stress": PK1,
            "cauchy_stress": sigma,
            "green_lagrange": 0.5 * (ufl.dot(self.F.T, self.F) - I),
            "J": self.J,
            "psi": self.psi,
            "von_mises": ufl.sqrt(3 / 2 * ufl.inner(dev, dev)),
            "hydrostatic_pressure": -ufl.tr(sigma) / 3,
            "hoop_stretch": 1 + self.u[0] / self.r,
        }

    def volume_integral(self, expr):
        """UFL form of the integral of expr over the full 3D body (2 pi r weight)."""
        return 2 * ufl.pi * expr * self.r * self.dx

    def surface_integral(self, expr, marker):
        """UFL form of the integral of expr over the full 3D surface of tag `marker`. Needs build_BCs."""
        if not hasattr(self, "ds"):
            raise RuntimeError("call build_BCs before surface_integral (self.ds is defined there)")
        return 2 * ufl.pi * expr * self.r * self.ds(marker)

    def build_BCs(self, boundary_conditions, slip_penalty=1e3):
        """
        facet_tag           : dolfinx MeshTags of the boundary facets (from cornea.msh).
        boundary_conditions : list of [type, marker, values]:
            "Dirichlet"         values = ("clamped" or Constant, axis), axis 0 = r, 1 = z
            "Pressure"          values = p (Constant): follower pressure normal to the
                                deformed surface, p > 0 pushes into the body
            "Slip"              values = None: slide along the boundary (u . N = 0, N the
                                reference facet normal), or a constant vector m (u . m = 0).
                                Enforced by a penalty, stiffness slip_penalty * mu / h.
            "Neumann_follower"  values = reference traction vector (Constant)
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
 
            elif bc_type == "Pressure":
                # Nanson: n da = J F^-T N dA ; traction t = -p n  ->  residual term + p J F^-T N . v
                p = values
                self.bc_form += p * self.J * ufl.dot(_inplane(self.F_inv.T, N), self.v) * self.r * ds
 
            elif bc_type == "Slip":
                # u . m = 0 by penalty. On a straight boundary (the limbus) m = N is constant,
                # so the nodes stay exactly on the same line.
                m = N if values is None else values
                k = slip_penalty * self.mech_params.mu / h
                self.bc_form += k * ufl.dot(self.u, m) * ufl.dot(self.v, m) * self.r * ds
 
            elif bc_type == "Neumann_follower":
                self.bc_form += - self.J * ufl.dot(_inplane(self.F_inv.T, self.v), values) * self.r * ds
 
            elif bc_type == "Neumann":
                self.bc_form += - ufl.dot(self.v, values) * self.r * ds
 
            elif bc_type == "Robin":
                k, u_ref = values
                self.bc_form += k * ufl.inner(self.u - u_ref, self.v) * self.r * ds
 
            else:
                raise TypeError(f"Unknown boundary condition: {bc_type}")

        log(f'[setup] Boundary conditions defined', comm=self.domain.comm)


    def build_solver(self):
        """
        build the solver based on the residuals definition.
        """
        # Assemble Residuals
        # self.residuals = ufl.derivative(self.R_u, self.u, self.v) + self.bc_form
        self.residuals = self.R_u + self.bc_form

        # log.set_log_level(log.LogLevel.INFO)

        petsc_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "none",
            "snes_monitor": None,
            "snes_atol": 1e-8,
            "snes_rtol": 1e-8,
            "snes_stol": 1e-8,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        self.problem = NonlinearProblem(
            self.residuals,
            self.u,
            bcs=self.bcs,
            petsc_options=petsc_options,
            petsc_options_prefix="hyperelasticity",
        )
        log(f'[setup] Solver defined', comm=self.domain.comm)


    def solve_one_step(self, n, p):
        """
        Solve one load step (called after changing the BCs) and print a summary.
        n : step number, p : load value (for the printout)
        """
        self.problem.solve()
        snes = self.problem.solver
        reason = snes.getConvergedReason()
        num_its = snes.getIterationNumber()
        res_norm = snes.getFunctionNorm()

        log(f"Step {n:3d} | load {p:.3e} | Newton iterations {num_its:2d} | "
            f"residual {res_norm:.3e} | reason {reason}", comm=self.domain.comm)

        if reason <= 0:
            print(snes.getConvergedReason(), snes.getIterationNumber())
            print(snes.getConvergenceHistory())
            raise RuntimeError(f"no convergence at step {n}, p = {p} (SNES reason {reason})")




class Hyperelastic_3D_framework:
    """
    This class builds a simple hyperelastic framework. Several hyperelastic law can be implemented.
    """
    def __init__(self, mesh_file, mech_params_json, comm=MPI.COMM_WORLD):
        """
        mesh_file        : gmsh 2.2 .msh file of the (r, z) section, as written by
                           cornea_meshing_package: self.domain, self.facet_tag,
                           self.cell_tags and, if the file stores them as
                           $ElementData "e_1"/"e_2", the local reference system
                           self.e_1, self.e_2, self.e_3 (used for the
                           fibres) -- see mesh_io.read_mesh.
        mech_params_json : material parameters, see parameters_class.HyperelasticMaterial.
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

    def _fibre_orientation_field(self):
        """
        Build self.a4, self.a6: fibre directions (unit UFL vectors in x,y,z),
        given in the mesh's local reference system (LRS) by mech_params.a4, a6 =
        (a_1, a_2, a_3), and expressed in physical (x,y,z)3
        coordinates by a plain orthonormal change of basis (see _push_local_frame).
        """
        if not hasattr(self, "e_1") or self.e_1 is None:
            raise RuntimeError(f"fibre directions need the mesh's local reference system, but "
                               f"{self.mesh_file} has no $ElementData 'e_1' and 'e_2'")
        a4_local = fem.Constant(self.domain, ScalarType(self.mech_params.a4))
        a6_local = fem.Constant(self.domain, ScalarType(self.mech_params.a6))
 
        self.a4 = _push_local_frame(self.e_1, self.e_2, self.e_3, a4_local)
        self.a6 = _push_local_frame(self.e_1, self.e_2, self.e_3, a6_local)
        log(f"[setup] Fibre fields built: a4 = {self.mech_params.a4}, "
              f"a6 = {self.mech_params.a6} (local frame)", comm=self.domain.comm)

    def build_weak_form(self):
        """
        Construct the weak form of the mechanical problem. 
        """
        self.metadata = {"quadrature_degree": 4}
        self.dx = ufl.Measure("dx", domain=self.domain, metadata=self.metadata)

        # Identity tensor
        I = ufl.variable(ufl.Identity(3))

        # kinematic quantities
        self.x = ufl.SpatialCoordinate(self.domain)

        # Deformation gradient
        self.F = ufl.variable(I + ufl.grad(self.u))
        self.F_inv = ufl.inv(self.F)
        self.J = ufl.det(self.F)

        # Strain energy density function
        self.psi = self.mech_params.strain_energy_density_function(self)
        PK1 = ufl.diff(self.psi, self.F) # PK1 stress

        # Residuals
        self.R_u = ufl.inner(ufl.grad(self.v), PK1) * self.dx


    @property
    def output_meta(self):
        """Metadata for the OutputManager: labels of the components, not axisymmetric."""
        return {"axes": ["x", "y", "z"], "axisymmetric": False}

    @cached_property
    def quantities(self):
        """
        Standard output quantities as UFL expressions in (x, y, z). Symbolic only:
        nothing is compiled here, the OutputManager compiles what it is given.
        """
        I = ufl.Identity(3)
        PK1 = ufl.diff(self.psi, self.F)
        sigma = (1 / self.J) * ufl.dot(PK1, self.F.T)
        dev = sigma - ufl.tr(sigma) / 3 * I
        return {
            "displacement": self.u,
            "PK1_stress": PK1,
            "cauchy_stress": sigma,
            "green_lagrange": 0.5 * (ufl.dot(self.F.T, self.F) - I),
            "J": self.J,
            "psi": self.psi,
            "von_mises": ufl.sqrt(3 / 2 * ufl.inner(dev, dev)),
            "hydrostatic_pressure": -ufl.tr(sigma) / 3,
        }

    def volume_integral(self, expr):
        """UFL form of the integral of expr over the body."""
        return expr * self.dx

    def surface_integral(self, expr, marker):
        """UFL form of the integral of expr over the surface of tag `marker`. Needs build_BCs."""
        if not hasattr(self, "ds"):
            raise RuntimeError("call build_BCs before surface_integral (self.ds is defined there)")
        return expr * self.ds(marker)

    def build_BCs(self, boundary_conditions, slip_penalty=1e3):
        """
        facet_tag           : dolfinx MeshTags of the boundary facets (from cornea.msh).
        boundary_conditions : list of [type, marker, values]:
            "Dirichlet"         values = ("clamped" or Constant, axis), axis 0 = r, 1 = z
            "Pressure"          values = p (Constant): follower pressure normal to the
                                deformed surface, p > 0 pushes into the body
            "Slip"              values = None: slide along the boundary (u . N = 0, N the
                                reference facet normal), or a constant vector m (u . m = 0).
                                Enforced by a penalty, stiffness slip_penalty * mu / h.
            "Neumann_follower"  values = reference traction vector (Constant)
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
 
            elif bc_type == "Pressure":
                # Nanson: n da = J F^-T N dA ; traction t = -p n  ->  residual term + p J F^-T N . v
                p = values
                self.bc_form += p * self.J * ufl.dot(ufl.dot(self.F_inv.T, N), self.v) * ds
 
            elif bc_type == "Slip":
                # u . m = 0 by penalty. On a straight boundary (the limbus) m = N is constant,
                # so the nodes stay exactly on the same line.
                m = N if values is None else values
                k = slip_penalty * self.mech_params.mu / h
                self.bc_form += k * ufl.dot(self.u, m) * ufl.dot(self.v, m) * ds
 
            elif bc_type == "Neumann_follower":
                self.bc_form += - self.J * ufl.dot(ufl.dot(self.F_inv.T, self.v), ufl.as_vector(values)) * ds
 
            elif bc_type == "Neumann":
                self.bc_form += - ufl.dot(self.v, ufl.as_vector(values)) * ds
 
            elif bc_type == "Robin":
                k, u_ref = values
                self.bc_form += k * ufl.inner(self.u - u_ref, self.v) * ds
 
            else:
                raise TypeError(f"Unknown boundary condition: {bc_type}")

        log(f'[setup] Boundary conditions defined', comm=self.domain.comm)


    def build_solver(self):
        """
        build the solver based on the residuals definition.
        """
        # Assemble Residuals
        # self.residuals = ufl.derivative(self.R_u, self.u, self.v) + self.bc_form
        self.residuals = self.R_u + self.bc_form

        # log.set_log_level(log.LogLevel.INFO)

        petsc_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "none",
            "snes_monitor": None,
            "snes_atol": 1e-8,
            "snes_rtol": 1e-8,
            "snes_stol": 1e-8,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        self.problem = NonlinearProblem(
            self.residuals,
            self.u,
            bcs=self.bcs,
            petsc_options=petsc_options,
            petsc_options_prefix="hyperelasticity",
        )
        log(f'[setup] Solver defined', comm=self.domain.comm)


    def solve_one_step(self, n, p):
        """
        Solve one load step (called after changing the BCs) and print a summary.
        n : step number, p : load value (for the printout)
        """
        self.problem.solve()
        snes = self.problem.solver
        reason = snes.getConvergedReason()
        num_its = snes.getIterationNumber()
        res_norm = snes.getFunctionNorm()

        log(f"Step {n:3d} | load {p:.3e} | Newton iterations {num_its:2d} | "
            f"residual {res_norm:.3e} | reason {reason}", comm=self.domain.comm)

        if reason <= 0:

            print(snes.getConvergedReason(), snes.getIterationNumber())
            print(snes.getConvergenceHistory())
            raise RuntimeError(f"no convergence at step {n}, p = {p} (SNES reason {reason})")