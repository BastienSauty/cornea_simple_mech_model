# author : B. Sauty ; 23 Sept 2026
# Main file to run a simulation for the cornea under pressure.
# This file is specific to this run. It should be adapted for each 

import numpy as np

from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
from dolfinx import fem
from petsc4py.PETSc import ScalarType

from Simulation_Framework import Hyperelastic_3D_framework, OutputManager, available_outputs


def run_simulation(name, mesh_file, mech_params_json, fields, scalars):

    # Build the hyperelastic framework. 
    mech = Hyperelastic_axisymmetric_framework(mesh_file, mech_params_json)
 
    # Boundary conditions
    # intraocular pressure, ramped during the simulation
    # (15 mmHg = 2.0e-3 MPa if lengths are in mm and stresses in MPa)
    p_iop = fem.Constant(mech.domain, ScalarType(0.0))
    p_max = 20* 0.000133322 # 18 mmHg to MPa
    n_steps = 20

    # physical tags written in cornea.msh by cornea/mesh_io.py
    ANTERIOR, POSTERIOR, LIMBUS, CENTRAL_LINE = 1, 2, 3, 4
    X0, X1, Y0, Y1, Z0, Z1 = 1, 2, 3, 4, 5, 6

    boundary_conditions = [
        ["Dirichlet", X0, ("clamped", 0)],   
        ["Dirichlet", Y0, ("clamped", 1)],   
        ["Dirichlet", Z0, ("clamped", 1)], 
        ["Neumann",  Z1,    p_iop],           # Normal pressure applied on the upper border
    ]
    mech.build_BCs(boundary_conditions)
    mech.build_solver()

    # Outputs: results/<name>.xdmf and results/<name>_scalars.csv, "time" = pressure
    with OutputManager(mech, f"results/{name}", fields=fields, scalars=scalars) as out:
        out.write(0.0)                                 # reference state
        for n, p in enumerate(np.linspace(0, p_max, n_steps)[1:], start=1):
            p_iop.value = p
            mech.solve_one_step(n, p)
            out.write(p)

    return out.get_history()


if __name__ == '__main__':
    name = "Testing_tools/cubetest"
    mesh_file = "Testing_tools/cube_hexa.msh"
    mech_params_json = "Testing_tools/mech_params_cube.json"


    # available_outputs()    # prints every output name with its description

    fields = ["displacement", "cauchy_stress", "von_mises", "J", "green_lagrange", "PK1_stress"]
    scalars = ["apex_uz_anterior", "central_thickness", "limbus_reaction_z", "volume"]

    history = run_simulation(name, mesh_file, mech_params_json, fields, scalars)

    if MPI.COMM_WORLD.rank == 0:
        import matplotlib.pyplot as plt
        plt.plot(history["apex_uz_anterior"] * 1e3, history["t"] / 0.000133322, "o-")
        plt.ylabel("IOP [mmHg]")
        plt.xlabel("anterior apex displacement [µm]")
        plt.savefig(f"results/{name}_apex.png", dpi=150)