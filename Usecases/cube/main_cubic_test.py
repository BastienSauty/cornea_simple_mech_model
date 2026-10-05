# author : B. Sauty ; 23 Sept 2026
# Main file to run a simulation of a cube under traction.
# This file is specific to this run: the quantities of interest are defined here.

import numpy as np
from pathlib import Path

from mpi4py import MPI
from dolfinx import fem
from petsc4py.PETSc import ScalarType

from Simulation_Framework import Hyperelastic_3D_framework, OutputManager, average

MMHG = 0.000133322      # 1 mmHg in MPa

# physical tags of the faces in cube_hexa.msh
X0, X1, Y0, Y1, Z0, Z1 = 1, 2, 3, 4, 5, 6


def cube_outputs(mech):
    """Quantities of interest of this case. Returns (fields, scalars) for the OutputManager."""
    q = mech.quantities       # standard quantities of the framework; print(q.keys()) lists them

    fields = {"displacement": q["displacement"],
              "cauchy_stress": q["cauchy_stress"],
              "J": q["J"]}
    comm = mech.domain.comm
    one = fem.Constant(mech.domain, ScalarType(1.0))
    scalars = {
        "volume": mech.volume_integral(q["J"]),
        "strain_energy": mech.volume_integral(q["psi"]),
        # mean axial displacement of the upper face Z1
        "uz_Z1": average(mech.surface_integral(q["displacement"][2], Z1),
                         mech.surface_integral(one, Z1), comm),
        # mean zz Cauchy stress over the cube
        "sigma_zz_mean": average(mech.volume_integral(q["cauchy_stress"][2, 2]),
                                 mech.volume_integral(one), comm),
    }
    return fields, scalars


def run_simulation(name, mesh_file, mech_params_json, outputs):
    """outputs : function mech -> (fields, scalars), see cube_outputs."""

    mech = Hyperelastic_3D_framework(mesh_file, mech_params_json)

    # Set the fiber orientation -> preprocess
    if mech.mech_params.sedf_type=='HGO':
        mech._fibre_orientation_field()
    # Set the weak form    
    mech.build_weak_form()

    # Boundary conditions: pressure ramped during the simulation
    p_iop = fem.Constant(mech.domain, ScalarType(0.0))
    p_max = 500 * MMHG
    n_steps = 20

    boundary_conditions = [
        ["Dirichlet", X0, ("clamped", 0)],
        ["Dirichlet", Y0, ("clamped", 1)],
        ["Dirichlet", Z0, ("clamped", 2)],
        # ["Dirichlet", Z1, (p_iop, 2)],
        
        ["Neumann_follower",  Z1,  [0, 0, p_iop]],           # load applied on the upper face
    ]
    mech.build_BCs(boundary_conditions)
    mech.build_solver()

    # Outputs: results/<name>.xdmf and results/<name>_scalars.csv, "time" = pressure
    fields, scalars = outputs(mech)                 # after build_BCs: surface integrals need mech.ds
    with OutputManager(f"results/{name}", mech.domain, fields, scalars,
                       meshtags={"facet_tag": mech.facet_tag},
                       meta=mech.output_meta) as out:
        out.write(0.0)                              # reference state
        for n, p in enumerate(np.linspace(0, p_max, n_steps)[1:], start=1):
            p_iop.value = p
            mech.solve_one_step(n, p)
            out.write(p)

    return out.get_history()


if __name__ == '__main__':
    name = "Usecases/cube/cubetest"
    mesh_file = "Usecases/cube/cube_hexa.msh"
    mech_params_json = "Usecases/cube/mech_params_cube.json"

    history = run_simulation(name, mesh_file, mech_params_json, cube_outputs)

    if MPI.COMM_WORLD.rank == 0:
        import matplotlib.pyplot as plt
        plt.plot(history["t"] / MMHG, history["volume"], "o-")
        plt.xlabel("load [mmHg]")
        plt.ylabel("deformed volume")
        plt.savefig(f"results/{name}_volume.png", dpi=150)