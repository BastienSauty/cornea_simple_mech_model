# author : B. Sauty ; 23 Sept 2026
# Main file to run a simulation for the cornea under pressure.
# This file is specific to this run: the quantities of interest are defined here.

import numpy as np

from mpi4py import MPI
from dolfinx import fem
from petsc4py.PETSc import ScalarType
import ufl

from Simulation_Framework import Hyperelastic_axisymmetric_framework, OutputManager, point_probe

MMHG = 0.000133322      # 1 mmHg in MPa

# physical tags written in cornea.msh by cornea/mesh_io.py
ANTERIOR, POSTERIOR, LIMBUS, CENTRAL_LINE = 1, 2, 3, 4


def axis_apex(mech, which):
    """z coordinate (reference configuration) of the anterior (highest) or posterior (lowest) point on the axis."""
    x = mech.domain.geometry.x
    z = x[np.isclose(x[:, 0], 0.0, atol=1e-10), 1]
    comm = mech.domain.comm
    if which == "anterior":
        return comm.allreduce(z.max() if z.size else -np.inf, op=MPI.MAX)
    return comm.allreduce(z.min() if z.size else np.inf, op=MPI.MIN)


def cornea_outputs(mech):
    """Quantities of interest of this case. Returns (fields, scalars) for the OutputManager."""
    q = mech.quantities       # standard quantities of the framework; print(q.keys()) lists them

    fields = {name: q[name] for name in
              ["displacement", "cauchy_stress", "von_mises", "J", "green_lagrange", "PK1_stress"]}

    # axial displacement of the apexes and deformed central thickness, probed on the axis
    z_ant, z_post = axis_apex(mech, "anterior"), axis_apex(mech, "posterior")
    u_ant = point_probe(mech.domain, mech.u, (0.0, z_ant))
    u_post = point_probe(mech.domain, mech.u, (0.0, z_post))

    # axial force of the limbus support on the cornea (full 3D ring): PK1 . N integrated on the limbus
    N = ufl.FacetNormal(mech.domain)
    P = q["PK1_stress"]
    traction_z = P[1, 0] * N[0] + P[1, 1] * N[1]

    scalars = {
        "apex_uz_anterior": lambda: u_ant()[1],
        "central_thickness": lambda: (z_ant + u_ant()[1]) - (z_post + u_post()[1]),
        "limbus_reaction_z": mech.surface_integral(traction_z, LIMBUS),
        "volume": mech.volume_integral(q["J"]),
    }
    return fields, scalars


def run_simulation(name, mesh_file, mech_params_json, outputs):
    """outputs : function mech -> (fields, scalars), see cornea_outputs."""

    # Build the hyperelastic framework. The mesh file is read once, inside: mesh,
    # facet tags and local reference system -> mech.domain, ...
    mech = Hyperelastic_axisymmetric_framework(mesh_file, mech_params_json)

    # Boundary conditions
    # intraocular pressure, ramped during the simulation
    # (15 mmHg = 2.0e-3 MPa if lengths are in mm and stresses in MPa)
    p_iop = fem.Constant(mech.domain, ScalarType(0.0))
    p_max = 20 * MMHG
    n_steps = 20

    boundary_conditions = [
        ["Dirichlet", CENTRAL_LINE, ("clamped", 0)],   # rolling on the axis: u_r = 0, u_z free
        # ["Slip",      LIMBUS,       None],           # limbus slides along its own line
        ["Dirichlet", LIMBUS,       ("clamped", 0)],
        ["Dirichlet", LIMBUS,       ("clamped", 1)],
        ["Pressure",  POSTERIOR,    p_iop],            # follower pressure normal to the posterior face
    ]                                                  # anterior face: free (zero traction)
    mech.build_BCs(boundary_conditions)
    mech.build_solver()

    # Outputs: results/<name>.xdmf and results/<name>_scalars.csv, "time" = pressure
    fields, scalars = outputs(mech)                    # after build_BCs: surface integrals need mech.ds
    with OutputManager(f"results/{name}", mech.domain, fields, scalars,
                       meshtags={"facet_tag": mech.facet_tag},
                       meta=mech.output_meta) as out:
        out.write(0.0)                                 # reference state
        for n, p in enumerate(np.linspace(0, p_max, n_steps)[1:], start=1):
            p_iop.value = p
            mech.solve_one_step(n, p)
            out.write(p)

    return out.get_history()


if __name__ == '__main__':
    name = "Usecases/cornea_axi/cornea_IOP_Giammarini"
    mesh_file = "Usecases/cornea_axi/cornea.msh"
    mech_params_json = "Usecases/cornea_axi/mech_params_Giammarini2026.json"

    history = run_simulation(name, mesh_file, mech_params_json, cornea_outputs)

    if MPI.COMM_WORLD.rank == 0:
        import matplotlib.pyplot as plt
        plt.plot(history["apex_uz_anterior"] * 1e3, history["t"] / MMHG, "o-")
        plt.ylabel("IOP [mmHg]")
        plt.xlabel("anterior apex displacement [µm]")
        plt.savefig(f"results/{name}_apex.png", dpi=150)