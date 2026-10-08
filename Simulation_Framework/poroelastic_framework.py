# author : B. Sauty ; 5 Oct 2026 
# based on the hyperelastic framework, and previous development in the sandbox. 
#
# This file manages the framework for the simulation of a poroelastic material with either an axisymmetric case or a full 3D case
# Uses the same IO as hyperelastic framework

from .parameters_class import PoroelasticMaterial
from .mesh_io import read_mesh
from .Postprocessing_tools.log_utils import log

from .hyperelastic_framework import _grad_vector_axi, _inplane

from functools import cached_property

from mpi4py import MPI
 
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx import fem, mesh, io
import ufl
from petsc4py.PETSc import ScalarType



def _grad_scalar_axi(u, r):
    """
    Function to compute the axisymmetric gradient of a scalar in the coordinate system (r, z, theta)
    with z the axis of symmetry and r = x[0] the radial coordinate
    u : scalar
    grad_axi : 3*1 vector
    """
    grad2D = ufl.grad(u)
    grad_axi = ufl.as_vector([grad2D[0], grad2D[1], 0])
    return(grad_axi)


class Poroelastic_axisymmetric_framework:
    """
    This class builds a simple poroelastic framework. Several hyperelastic law can be implemented.
    """
    def __init__(self, mesh_file, mech_params_json, comm=MPI.COMM_WORLD):
        """
        mesh_file        : gmsh 4.1 .msh file of the (r, z) section, as written by
                           Meshing_tools : self.domain, self.facet_tag,
                           self.cell_tags and, if the file stores them as
                           $ElementData "e_1"/"e_2", the local reference system
                           self.e_1, self.e_2, self.e83 (used for the
                           fibres) -- see mesh_io.read_mesh.
        mech_params_json : material parameters, see parameters_class.PoroelasticMaterial.
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

        # Identity tensor
        I = ufl.variable(ufl.Identity(3))

        # kinematic quantities
        self.x = ufl.SpatialCoordinate(self.domain)
        self.r = self.x[0]

        # Deformation gradient
        self.F = ufl.variable(I + _grad_vector_axi(self.u, self.r))
        self.F_inv = ufl.inv(self.F)
        self.J = ufl.det(self.F)

        # Conservation of momentum : static equilibrium 
        # Displacement residuals - Strain energy density function
        self.mech_params.set_material_laws(self)

        # self.mech_params.strain_energy_density_function(self)
        PK1_iso = ufl.diff(self.mech_params.psi_iso, self.F)

        self.PK1 = self.mech_params.phi_s * PK1_iso - self.J * self.p * self.F_inv.T
        self.R_u = ufl.inner(_grad_vector_axi(self.v, self.r), self.PK1) * self.r * self.dx

        # Conservation of mass : porous equilibrium
        # Backward euler derivatives
        F_n = ufl.variable(I + _grad_vector_axi(self.u_n, self.r))
        J_n = ufl.det(F_n)
        J_dot = (self.J - J_n) / dt   # scalar now: (J - J_n) is scalar, dt is scalar

        # self.mech_params.pullback_permeability_tensor(self)
        K_pullback = self.mech_params.K_pullback
        self.R_p = ( J_dot * self.q + ufl.inner(ufl.dot(_grad_scalar_axi(self.p, self.r), K_pullback), _grad_scalar_axi(self.q, self.r))) * self.dx 


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
        sigma = (1 / self.J) * ufl.dot(self.PK1, self.F.T)
        dev = sigma - ufl.tr(sigma) / 3 * I
        return {
            "displacement": self.u,
            "PK1_stress": self.PK1,
            "cauchy_stress": sigma,
            "green_lagrange": 0.5 * (ufl.dot(self.F.T, self.F) - I),
            "J": self.J,
            "von_mises": ufl.sqrt(3 / 2 * ufl.inner(dev, dev)),
            "hydrostatic_pressure": -ufl.tr(sigma) / 3,
            "hoop_stretch": 1 + self.u[0] / self.r,
            "fluid_pressure": self.p
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
        boundary_conditions : list of [type, marker, values]

        Displacement field u
            "Dirichlet_disp"           values = ("clamped" or scalar/Constant, axis), axis 0 = r, 1 = z
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
                self.bc_form += - ufl.dot(self.v, ufl.as_vector(values)) * self.r * ds

            elif bc_type == "Neumann_follower_disp":
                # Nanson: n da = J F^-T N dA ; traction t = -p n  ->  residual term + p J F^-T N . v
                self.bc_form += values * self.J * ufl.dot(_inplane(self.F_inv.T, N), self.v) * self.r * ds
            elif bc_type == "Slip_disp":
                m = N if values is None else values
                k = slip_penalty * self.mech_params.mu / h
                self.bc_form += k * ufl.dot(self.u, m) * ufl.dot(self.v, m) * self.r * ds

            elif bc_type == "Robin_disp":
                k, u_ref = values
                self.bc_form += k * ufl.inner(self.u - u_ref, self.v) * self.r * ds

            # ---------------- fluid pressure ----------------
            elif bc_type == "Dirichlet_pressure":
                facets = self.facet_tag.find(marker)
                dofs = fem.locate_dofs_topological(self.V_p, self.fdim, facets)
                if isinstance(values, str) and values == "drained":
                    values = ScalarType(0.0)
                self.bcs.append(fem.dirichletbc(values, dofs, self.V_p))

            elif bc_type == "Neumann_pressure":
                # flux per reference area: + q_n q dA
                self.bc_form += values * self.q * self.r * ds

            elif bc_type == "Neumann_follower_pressure":
                # flux per current area: da = J |F^-T N| dA (Nanson)
                area_ratio = self.J * ufl.sqrt(ufl.dot(_inplane(self.F_inv.T, N), _inplane(self.F_inv.T, N)))
                self.bc_form += values * area_ratio * self.q * self.r * ds

            else:
                raise TypeError(f"Unknown boundary condition: {bc_type}")

        log('[setup] Boundary conditions defined', comm=self.domain.comm)


    def build_solver(self):
        """
        build the solver based on the residuals definition.
        """
        # Assemble Residuals
        self.residuals = self.R_u + self.R_p + self.bc_form
        res_block = ufl.extract_blocks(self.residuals)

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
            res_block,
            [self.u, self.p],
            bcs=self.bcs,
            petsc_options=petsc_options,
            petsc_options_prefix="poroelasticity",
        )
        log(f'[setup] Solver defined', comm=self.domain.comm)



    def solve_one_step(self, n, p):
        """
        Solve one load step (called after changing the BCs) and print a summary.
        n : step number, p : step value (for the printout)
        """
        self.problem.solve()
        snes = self.problem.solver
        reason = snes.getConvergedReason()
        num_its = snes.getIterationNumber()
        res_norm = snes.getFunctionNorm()

        log(f"Step {n:3d} | load {p:.3e} | Newton iterations {num_its:2d} | "
            f"residual {res_norm:.3e} | reason {reason}", comm=self.domain.comm)

        # update history variable for time discretization
        self.u_n.interpolate(self.u)

        if reason <= 0:
            print(snes.getConvergedReason(), snes.getIterationNumber())
            print(snes.getConvergenceHistory())
            raise RuntimeError(f"no convergence at step {n}, p = {p} (SNES reason {reason})")



class Poroelastic_3D_framework:
    """
    This class builds a simple poroelastic framework. Several hyperelastic law can be implemented.
    """
    def __init__(self, mesh_file, mech_params_json, comm=MPI.COMM_WORLD):
        """
        mesh_file        : gmsh 4.1 .msh file of the (r, z) section, as written by
                           Meshing_tools : self.domain, self.facet_tag,
                           self.cell_tags and, if the file stores them as
                           $ElementData "e_1"/"e_2", the local reference system
                           self.e_1, self.e_2, self.e83 (used for the
                           fibres) -- see mesh_io.read_mesh.
        mech_params_json : material parameters, see parameters_class.PoroelasticMaterial.
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

        # Identity tensor
        I = ufl.variable(ufl.Identity(3))

        # Deformation gradient
        self.F = ufl.variable(I + ufl.grad(self.u))
        self.F_inv = ufl.inv(self.F)
        self.J = ufl.det(self.F)

        # Conservation of momentum : static equilibrium 
        # Displacement residuals - Strain energy density function
        self.mech_params.set_material_laws(self)

        # self.mech_params.strain_energy_density_function(self)
        PK1_iso = ufl.diff(self.mech_params.psi_iso, self.F)

        self.PK1 = self.mech_params.phi_s * PK1_iso - self.J * self.p * self.F_inv.T
        self.R_u = ufl.inner(ufl.grad(self.v), self.PK1) *self.dx

        # Conservation of mass : porous equilibrium
        # Backward euler derivatives
        F_n = ufl.variable(I + ufl.grad(self.u_n))
        J_n = ufl.det(F_n)
        J_dot = (self.J - J_n) / dt   # scalar now: (J - J_n) is scalar, dt is scalar

        # self.mech_params.pullback_permeability_tensor(self)
        K_pullback = self.mech_params.K_pullback
        self.R_p = ( J_dot * self.q + ufl.inner(ufl.dot(ufl.grad(self.p), K_pullback), ufl.grad(self.q))) * self.dx 


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
        sigma = (1 / self.J) * ufl.dot(self.PK1, self.F.T)
        dev = sigma - ufl.tr(sigma) / 3 * I
        return {
            "displacement": self.u,
            "PK1_stress": self.PK1,
            "cauchy_stress": sigma,
            "green_lagrange": 0.5 * (ufl.dot(self.F.T, self.F) - I),
            "J": self.J,
            "von_mises": ufl.sqrt(3 / 2 * ufl.inner(dev, dev)),
            "hydrostatic_pressure": -ufl.tr(sigma) / 3,
            "hoop_stretch": 1 + self.u[0] / self.r,
            "fluid_pressure": self.p
        }

    def volume_integral(self, expr):
        """UFL form of the integral of expr over the full 3D body (2 pi r weight)."""
        return 2 * ufl.pi * expr * self.dx

    def surface_integral(self, expr, marker):
        """UFL form of the integral of expr over the full 3D surface of tag `marker`. Needs build_BCs."""
        if not hasattr(self, "ds"):
            raise RuntimeError("call build_BCs before surface_integral (self.ds is defined there)")
        return 2 * ufl.pi * expr * self.ds(marker)

    def build_BCs(self, boundary_conditions, slip_penalty=1e3):
        """
        boundary_conditions : list of [type, marker, values]

        Displacement field u
            "Dirichlet_disp"           values = ("clamped" or scalar/Constant, axis), axis 0 = r, 1 = z
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
                self.bc_form += - ufl.dot(self.v, ufl.as_vector(values)) * ds

            elif bc_type == "Neumann_follower_disp":
                # Nanson: n da = J F^-T N dA ; traction t = -p n  ->  residual term + p J F^-T N . v
                self.bc_form += values * self.J * ufl.dot(ufl.dot(self.F_inv.T, N), self.v) *  ds

            elif bc_type == "Robin_disp":
                k, u_ref = values
                self.bc_form += k * ufl.inner(self.u - u_ref, self.v) * ds

            # ---------------- fluid pressure ----------------
            elif bc_type == "Dirichlet_pressure":
                facets = self.facet_tag.find(marker)
                dofs = fem.locate_dofs_topological(self.V_p, self.fdim, facets)
                if isinstance(values, str) and values == "drained":
                    values = ScalarType(0.0)
                self.bcs.append(fem.dirichletbc(values, dofs, self.V_p))

            elif bc_type == "Neumann_pressure":
                # flux per reference area: + q_n q dA
                self.bc_form += values * self.q * ds

            elif bc_type == "Neumann_follower_pressure":
                # flux per current area: da = J |F^-T N| dA (Nanson)
                area_ratio = self.J * ufl.sqrt(ufl.dot(ufl.dot(self.F_inv.T, N), ufl.dot(self.F_inv.T, N)))
                self.bc_form += values * area_ratio * self.q * ds

            else:
                raise TypeError(f"Unknown boundary condition: {bc_type}")

        log('[setup] Boundary conditions defined', comm=self.domain.comm)


    def build_solver(self):
        """
        build the solver based on the residuals definition.
        """
        # Assemble Residuals
        self.residuals = self.R_u + self.R_p + self.bc_form
        res_block = ufl.extract_blocks(self.residuals)

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
            res_block,
            [self.u, self.p],
            bcs=self.bcs,
            petsc_options=petsc_options,
            petsc_options_prefix="poroelasticity",
        )
        log(f'[setup] Solver defined', comm=self.domain.comm)



    def solve_one_step(self, n, p):
        """
        Solve one load step (called after changing the BCs) and print a summary.
        n : step number, p : step value (for the printout)
        """
        self.problem.solve()
        snes = self.problem.solver
        reason = snes.getConvergedReason()
        num_its = snes.getIterationNumber()
        res_norm = snes.getFunctionNorm()

        log(f"Step {n:3d} | load {p:.3e} | Newton iterations {num_its:2d} | "
            f"residual {res_norm:.3e} | reason {reason}", comm=self.domain.comm)

        # update history variable for time discretization
        self.u_n.interpolate(self.u)

        if reason <= 0:
            print(snes.getConvergedReason(), snes.getIterationNumber())
            print(snes.getConvergenceHistory())
            raise RuntimeError(f"no convergence at step {n}, p = {p} (SNES reason {reason})")