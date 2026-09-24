# author : B. Sauty ; 23 Sept 2026
# Main file to run a simulation for the cornea under pressure.

import numpy as np

from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
from dolfinx import fem
from petsc4py.PETSc import ScalarType

from Simulation_Framework import Hyperelastic_framework, OutputManager, available_outputs


def run_simulation(name, mesh_file, mech_params_json, fields, scalars):

    # read_from_msh returns a MeshData object (needs the gmsh Python module)
    mesh_data = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0, gdim=2)
    domain, facet_tags = mesh_data.mesh, mesh_data.facet_tags

    # Build the hyperelastic framework
    mech = Hyperelastic_framework(domain, mech_params_json)

    # Boundary conditions
    # intraocular pressure, ramped during the simulation
    # (15 mmHg = 2.0e-3 MPa if lengths are in mm and stresses in MPa)
    p_iop = fem.Constant(mech.domain, ScalarType(0.0))

    # physical tags written in cornea.msh by cornea/mesh_io.py
    ANTERIOR, POSTERIOR, LIMBUS, CENTRAL_LINE = 1, 2, 3, 4

    boundary_conditions = [
        ["Dirichlet", CENTRAL_LINE, ("clamped", 0)],   # rolling on the axis: u_r = 0, u_z free
        # ["Slip",      LIMBUS,       None],           # limbus slides along its own line
        ["Dirichlet", LIMBUS,       ("clamped", 0)],
        ["Dirichlet", LIMBUS,       ("clamped", 1)],
        ["Pressure",  POSTERIOR,    p_iop],            # follower pressure normal to the posterior face
    ]                                                  # anterior face: free (zero traction)
    mech.build_BCs(facet_tags, boundary_conditions)
    mech.build_solver()

    # Outputs: results/<name>.xdmf and results/<name>_scalars.csv, "time" = pressure
    with OutputManager(mech, f"results/{name}", fields=fields, scalars=scalars) as out:
        out.write(0.0)                                 # reference state
        for n, p in enumerate(np.linspace(0, 2.0e-3, 20)[1:], start=1):
            p_iop.value = p
            mech.solve_one_step(n, p)
            out.write(p)

    return out.get_history()


if __name__ == '__main__':
    name = "cornea_IOP"
    mesh_file = "cornea.msh"
    mech_params_json = "mech_params.json"

    # available_outputs()    # prints every output name with its description

    fields = ["displacement", "cauchy_stress", "von_mises", "J", "green_lagrange", "PK1_stress"]
    scalars = ["apex_uz_anterior", "central_thickness", "limbus_reaction_z", "volume"]

    history = run_simulation(name, mesh_file, mech_params_json, fields, scalars)

    if MPI.COMM_WORLD.rank == 0:
        import matplotlib.pyplot as plt
        plt.plot(history["t"] * 1e3, history["apex_uz_anterior"] * 1e3, "o-")
        plt.xlabel("IOP [kPa]")
        plt.ylabel("anterior apex displacement [µm]")
        plt.savefig(f"results/{name}_apex.png", dpi=150)