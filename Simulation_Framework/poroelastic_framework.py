# author : B. Sauty ; 5 Oct 2026 
# based on the hyperelastic framework, and previous development in the sandbox. 
#
# This file manages the framework for the simulation of a poroelastic material with either an axisymmetric case or a full 3D case
# Uses the same IO as hyperelastic framework

from .parameters_class import MechParams
from .mesh_io import read_mesh

from functools import cached_property

from mpi4py import MPI
 
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx import fem, mesh, io
import ufl
from petsc4py.PETSc import ScalarType

from Simulation_Framework import log