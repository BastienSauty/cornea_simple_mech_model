# author : B. Sauty ; 3 Oct 2026
# Main file to run a simulation for the cornea under pressure.
# 3D mesh generated using Merlini's specifications. See Meshing_tools, make_3D_cornea
# This file is specific to this run: the quantities of interest are defined here.

import numpy as np

from mpi4py import MPI
from dolfinx import fem
from petsc4py.PETSc import ScalarType
import ufl

from Simulation_Framework import Hyperelastic_3D_framework, OutputManager, point_probe

MMHG = 0.000133322      # 1 mmHg in MPa

# physical tags written in cornea_3D_hex.msh by Meshing_tools toolbox
ANTERIOR, POSTERIOR, LIMBUS = 1, 2, 3


def axis_apex(mech, which):
    """z coordinate (reference configuration) of the anterior (highest) or posterior (lowest)
    node on the axis x = y = 0. Needs mesh nodes on the axis (the structured hexahedral mesh
    has them; the tetrahedral one does not)."""
    x = mech.domain.geometry.x
    on_axis = np.abs(x[:, :2]).max(axis=1) < 1e-8
    z = x[on_axis, 2]
    comm = mech.domain.comm
    if which == "anterior":
        apex = comm.allreduce(z.max() if z.size else -np.inf, op=MPI.MAX)
    else:
        apex = comm.allreduce(z.min() if z.size else np.inf, op=MPI.MIN)
    if not np.isfinite(apex):
        raise ValueError("no mesh node on the axis x = y = 0: cannot probe the apex")
    return apex

def cornea_outputs(mech):
    """Quantities of interest of this case. Returns (fields, scalars) for the OutputManager."""
    q = mech.quantities       # standard quantities of the framework; print(q.keys()) lists them

    fields = {name: q[name] for name in
              ["displacement", "cauchy_stress", "von_mises", "J", "green_lagrange", "PK1_stress"]}

    # axial displacement of the apexes and deformed central thickness, probed on the axis
    z_ant, z_post = axis_apex(mech, "anterior"), axis_apex(mech, "posterior")
    u_ant = point_probe(mech.domain, mech.u, (0.0, 0.0, z_ant))
    u_post = point_probe(mech.domain, mech.u, (0.0, 0.0, z_post))

    # axial force of the limbus support on the cornea (full 3D ring): PK1 . N integrated on the limbus
    N = ufl.FacetNormal(mech.domain)
    P = q["PK1_stress"]
    # traction_z = ufl.dot(N, ufl.dot(P, N)) # P[1, 0] * N[0] + P[1, 1] * N[1]
    traction_z = ufl.dot(P, N)[2]

    scalars = {
        "apex_uz_anterior": lambda: u_ant()[2],
        "central_thickness": lambda: (z_ant + u_ant()[2]) - (z_post + u_post()[2]),
        "limbus_reaction_z": mech.surface_integral(traction_z, LIMBUS),
        "posterior_reaction_z": mech.surface_integral(traction_z, POSTERIOR),
        "anterior_reaction_z": mech.surface_integral(traction_z, ANTERIOR),
        "volume": mech.volume_integral(q["J"]),
    }
    return fields, scalars


def run_simulation(name, mesh_file, mech_params_json, outputs):
    """outputs : function mech -> (fields, scalars), see cornea_outputs."""

    # Build the hyperelastic framework. The mesh file is read once, inside: mesh,
    # facet tags and local reference system -> mech.domain, ...
    mech = Hyperelastic_3D_framework(mesh_file, mech_params_json)

    # Set the fiber orientation -> preprocess
    if mech.mech_params.sedf_type=='HGO':
        mech._fibre_orientation_field()
    # Set the weak form    
    mech.build_weak_form()

    # Boundary conditions
    # intraocular pressure, ramped during the simulation
    # (15 mmHg = 2.0e-3 MPa if lengths are in mm and stresses in MPa)
    p_iop = fem.Constant(mech.domain, ScalarType(0.0))
    p_max = 20 * MMHG
    n_steps = 20

    boundary_conditions = [
        ["Dirichlet", LIMBUS,       ("clamped", 0)],
        ["Dirichlet", LIMBUS,       ("clamped", 1)],
        ["Dirichlet", LIMBUS,       ("clamped", 2)],
        ["Neumann_follower",  POSTERIOR,    p_iop],            # follower pressure normal to the posterior face
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
    name = "Usecases/cornea_3D/cornea_3D_IOP_fine"
    mesh_file = "Usecases/cornea_3D/cornea_3D_hex_fine.msh"
    mech_params_json = "Usecases/cornea_3D/mech_params_Pandolfi2006.json"

    history = run_simulation(name, mesh_file, mech_params_json, cornea_outputs)

    if MPI.COMM_WORLD.rank == 0:
        import matplotlib.pyplot as plt
        plt.plot(history["apex_uz_anterior"] * 1e3, history["t"] / MMHG, "o-")
        plt.ylabel("IOP [mmHg]")
        plt.xlabel("anterior apex displacement [µm]")
        plt.savefig(f"results/{name}_apex.png", dpi=150)

    # run in parallel with : OMP_NUM_THREADS=1 mpirun -n 7 python Usecases/cornea_3D/main_cornea_3D.py
