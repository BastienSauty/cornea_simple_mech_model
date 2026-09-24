# author : B. Sauty ; 23 Sept 2026
# This file aims at building a framework class to model mechanical problems using fenicsx. Simple cases
# Different classes are built in order to run several types of simulation. 
# The geometry is always assumed to be axisymetrical. Single material

from .parameters_class import MechParams


import gmsh
from mpi4py import MPI
import numpy as np
 
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx import fem, mesh, io
from dolfinx.io import gmsh as gmshio
import ufl
from petsc4py.PETSc import ScalarType
from dolfinx import log


def _grad_axi(u, r):
    """
    Function to compute the axisymmetric gradient in the coordinate system (r, z, theta)
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
  
def _push_axisymmetric(F_map, A):
    """
    Direction A = (a_u, a_v, a_theta) given in the reference square -> unit UFL vector
    (a_r, a_z, a_theta) on the cornea.
    The in-plane part gives the new in-plane direction, F_map . (a_u, a_v), with
    F_map = d(r, z)/d(u, v), but keeps the length it has in A; the theta part is
    untouched. So the split between in-plane and hoop components set in the square is
    preserved exactly (e.g. A = [1, 0, 1] stays at 45 deg from the hoop direction).
    """
    A = np.asarray(A, dtype=float)
    n_plane = np.linalg.norm(A[:2])
    if n_plane == 0:                                   # pure hoop direction
        return ufl.as_vector([0.0, 0.0, 1.0])
    a_rz = ufl.dot(F_map, ufl.as_vector(A[:2]))
    a_rz = n_plane * a_rz / ufl.sqrt(ufl.dot(a_rz, a_rz))
    a = ufl.as_vector([a_rz[0], a_rz[1], A[2]])
    return a / ufl.sqrt(ufl.dot(a, a))
 


class Hyperelastic_framework:
    """
    This class builds a simple hyperelastic framework. Several hyperelastic law can be implemented.
    """
    def __init__(self, mesh_file, mech_params_json, comm=MPI.COMM_WORLD):
        """
        mesh_file        : gmsh 2.2 .msh file of the (r, z) section. Read once, here:
                           self.domain, self.facet_tag, self.cell_tags and, if the
                           file stores them as $NodeData, the square coordinates
                           self.square_coords = [u, v] (P1 fields, used for the fibres).
        mech_params_json : material parameters, see parameters_class.MechParams.
        """
        self.mesh_file = mesh_file
        self._read_mesh(mesh_file, comm)
 
        self.V_u = fem.functionspace(self.domain, ("Lagrange", 2, (self.domain.geometry.dim,))) # disp function space

        self.u = fem.Function(self.V_u)   # displacement unknown — real Function, holds DOF values
        self.v = ufl.TestFunction(self.V_u) # test function - shape function in the FEM

        # Store the mechanical parameters in the volume using a DataClass. See parameter_class
        self.mech_params = self.mech_params = MechParams.from_json(mech_params_json)

        # Build the weak form directly
        self.build_weak_form()
        print(f'[setup] Framework initialized; SEDF type : {self.mech_params.sedf_type}')


    def _read_mesh(self, mesh_file, comm, rank=0):
        """
        Open the file once with gmsh (on `rank`) and extract the mesh, the physical
        tags and the $NodeData u, v, which correspond to the unmapped coordinates. gmsh turns each $NodeData block into a view.
        """
        node_data = None
        if comm.rank == rank:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.merge(mesh_file)
            mesh_data = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=2)
            node_data = {}
            for tag in gmsh.view.getTags():
                name = gmsh.option.getString(f"View[{gmsh.view.getIndex(tag)}].Name")
                _, nodes, values, _, _ = gmsh.view.getModelData(tag, 0)
                data = np.full(int(max(nodes)), np.nan)
                data[np.asarray(nodes, dtype=int) - 1] = np.asarray(values).ravel()
                node_data[name] = data          # indexed by node tag - 1
            gmsh.finalize()
        else:
            mesh_data = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=2)
        node_data = comm.bcast(node_data, root=rank)
 
        self.domain = mesh_data.mesh
        self.facet_tag = mesh_data.facet_tags
        self.cell_tags = mesh_data.cell_tags
 
        # square coordinates as P1 fields. For P1 on a linear mesh the dofs of a cell
        # are its geometry nodes in the same local order, and input_global_indices
        # gives the original number (gmsh tag - 1) of each local geometry node.
        self.square_coords = None
        if "u" in node_data and "v" in node_data:
            if self.domain.geometry.cmap.degree != 1:
                raise RuntimeError("u, v node data can only be loaded on a linear mesh")
            V = fem.functionspace(self.domain, ("Lagrange", 1))
            original = self.domain.geometry.input_global_indices
            self.square_coords = []
            for name in ("u", "v"):
                f = fem.Function(V, name=name)
                f.x.array[V.dofmap.list] = node_data[name][original][self.domain.geometry.dofmap]
                f.x.scatter_forward()
                self.square_coords.append(f)
        print(f"[setup] Mesh read from {mesh_file}"
              + (" (with square coordinates u, v)" if self.square_coords else ""))
 

    def strain_energy_density_function(self):
        """
        called by build_weak_form. returns psi : the strain energy density function. 
        """
        if self.mech_params.sedf_type=='Neo-Hookean':
            C_dev = self.J**(-2/3)*ufl.dot(self.F.T, self.F)
            I1_dev = ufl.tr(C_dev)

            C10 = fem.Constant(self.domain, ScalarType(self.mech_params.C_10))
            D = fem.Constant(self.domain, ScalarType(self.mech_params.D))

            psi = C10 *(I1_dev - 3) + 1/D *(self.J - 1)**2

        elif self.mech_params.sedf_type=='Mooney-Rivlin':
            C_dev = self.J**(-2/3)*ufl.dot(self.F.T, self.F)
            I1_dev = ufl.tr(C_dev)
            I2_dev = 1/2* (ufl.tr(C_dev) ** 2 - ufl.tr(ufl.dot(C_dev, C_dev)))

            C01 = fem.Constant(self.domain, ScalarType(self.mech_params.C_01))
            C10 = fem.Constant(self.domain, ScalarType(self.mech_params.C_10))
            D = fem.Constant(self.domain, ScalarType(self.mech_params.D))

            psi = C10 *(I1_dev - 3) + C01 *(I2_dev-3) + 1/D *(self.J - 1)**2

        elif self.mech_params.sedf_type=='HGO':
            # Isotropic contribution
            C_dev = self.J**(-2/3)*ufl.dot(self.F.T, self.F)
            I1_dev = ufl.tr(C_dev)
            I2_dev = 1/2* (ufl.tr(C_dev) ** 2 - ufl.tr(ufl.dot(C_dev, C_dev)))

            C01 = fem.Constant(self.domain, ScalarType(self.mech_params.C_01))
            C10 = fem.Constant(self.domain, ScalarType(self.mech_params.C_10))
            D = fem.Constant(self.domain, ScalarType(self.mech_params.D))

            psi = C10 *(I1_dev - 3) + C01 *(I2_dev-3) + 1/D *(self.J - 1)**2

            # Anisotropic contribution. The fibre directions are fields, one unit vector
            # per cell, transported from the reference square (see _fibre_orientation_field)
            if not hasattr(self, "a4"):
                self._fibre_orientation_field()

            k1 = fem.Constant(self.domain, ScalarType(self.mech_params.k1))
            k2 = fem.Constant(self.domain, ScalarType(self.mech_params.k2))

            C = ufl.dot(self.F.T, self.F)
            I4 = ufl.inner(ufl.outer(self.a4, self.a4), C)
            I6 = ufl.inner(ufl.outer(self.a6, self.a6), C)

            psi += k1/(2*k2) * (ufl.exp(k2 * (I4 - 1)**2) + ufl.exp(k2 * (I6 - 1)**2) - 2 )
        return(psi)

    def _fibre_orientation_field(self):
        """
        Build self.a4, self.a6: fibre directions (unit UFL vectors in r, z, theta),
        given in the reference square by mech_params.a4, a6 and pushed onto the cornea.
 
        self.square_coords = [u, v] are the square coordinates of the nodes, read from
        the mesh file. grad(u, v) = d(u, v)/d(r, z) is the Jacobian of the inverse map,
        so F_map = d(r, z)/d(u, v) is its inverse. Everything stays in UFL: the
        directions are evaluated exactly at the quadrature points.
        """
        if self.square_coords is None:
            raise RuntimeError(f"fibre directions need the square coordinates u, v, but "
                               f"{self.mesh_file} has no $NodeData 'u' and 'v'")
 
        G = ufl.grad(ufl.as_vector(self.square_coords))   # d(u, v)/d(r, z)
        F_map = ufl.inv(G)                                 # d(r, z)/d(u, v)
 
        self.a4 = _push_axisymmetric(F_map, self.mech_params.a4)
        self.a6 = _push_axisymmetric(F_map, self.mech_params.a6)
        print(f"[setup] Fibre fields built: a4 = {self.mech_params.a4}, "
              f"a6 = {self.mech_params.a6} (square)")


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
        self.psi = self.strain_energy_density_function()
        PK1 = ufl.diff(self.psi, self.F) # PK1 stress

        # Residuals
        self.R_u = ufl.inner(_grad_axi(self.v, self.r), PK1) * self.r * self.dx


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

        print(f'[setup] Boundary conditions defined')


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
        print(f'[setup] Solver defined')


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

        if self.domain.comm.rank == 0:
            print(f"Step {n:3d} | load {p:.3e} | Newton iterations {num_its:2d} | "
                  f"residual {res_norm:.3e} | reason {reason}")

        if reason <= 0:
            raise RuntimeError(f"no convergence at step {n}, p = {p} (SNES reason {reason})")