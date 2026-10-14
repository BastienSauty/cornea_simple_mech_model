# Verification of the 3D framework for the HGO law (Mooney-Rivlin matrix + 2 fibre families).
#
# Cube under axial (z) load: symmetry planes X0, Y0, Z0, follower load on Z1, X1 and Y1 free.
# The state is homogeneous, F = diag(lam_x, lam_y, lam_z), with sigma_xx = sigma_yy = 0.
# Two fibre configurations, each in tension and in compression (check of the compression
# switch: a fibre carries load only where I4bar > 1):
#     x_y : a4 = x, a6 = y   (both transverse to the load)
#     x_z : a4 = x, a6 = z   (one transverse, one axial)
# The mesh is given from outside (cube_hexa.msh).

from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx import fem
from petsc4py.PETSc import ScalarType

from Simulation_Framework import Hyperelastic_3D_framework, OutputManager, average
import hgo_homogeneous as hgo

# physical tags of the faces in cube_hexa.msh
X0, X1, Y0, Y1, Z0, Z1 = 1, 2, 3, 4, 5, 6
AXIAL = 2                 # principal axes: 0 = x, 1 = y, 2 = z
N_STEPS = 20

FIBRE_CASES = {
    "x_y": {"a4": [1, 0, 0], "a6": [0, 1, 0], "fibres": (0, 1)},
    "x_z": {"a4": [1, 0, 0], "a6": [0, 0, 1], "fibres": (0, 2)},
}


# ----------------------------------------------------------------------------------------
# 1. outputs of the simulation that will be compared
# ----------------------------------------------------------------------------------------
def cube_outputs(mech):
    """Quantities of interest of the cube under axial loading.
    Returns (fields, scalars) for the OutputManager.

    Reference averages (J, Green-Lagrange strains, PK1 on Z1) are divided by the reference
    volume / area. The Cauchy stress average is int(sigma_zz J dV) / int(J dV).
    """
    q = mech.quantities
    fields = {name: q[name] for name in ["displacement", "cauchy_stress", "J", "green_lagrange"]}
    sigma, E, J, P = q["cauchy_stress"], q["green_lagrange"], q["J"], q["PK1_stress"]

    comm = mech.domain.comm
    one = fem.Constant(mech.domain, ScalarType(1.0))
    V0 = mech.volume_integral(one)
    A_Z1 = mech.surface_integral(one, Z1)

    scalars = {
        "J_mean": average(mech.volume_integral(J), V0, comm),
        "E_xx_mean": average(mech.volume_integral(E[0, 0]), V0, comm),
        "E_yy_mean": average(mech.volume_integral(E[1, 1]), V0, comm),
        "E_zz_mean": average(mech.volume_integral(E[2, 2]), V0, comm),
        "sigma_zz_mean": average(mech.volume_integral(sigma[2, 2] * J),
                                 mech.volume_integral(J), comm),
        # mean nominal axial stress on Z1 = axial reaction / reference area
        "P_zz_Z1": average(mech.surface_integral(P[2, 2], Z1), A_Z1, comm),
    }
    return fields, scalars


# ----------------------------------------------------------------------------------------
# 2. run simulation function
# ----------------------------------------------------------------------------------------
def run_simulation(name, mesh_file, mech_params_json, outputs, p_max):
    """outputs : function mech -> (fields, scalars), see cube_outputs.
    p_max : signed final load on Z1 (positive = compression), ramped in N_STEPS steps."""
    mech = Hyperelastic_3D_framework(mesh_file, mech_params_json)

    if mech.mech_params.sedf_type == 'HGO':
        mech._fibre_orientation_field()
    mech.build_weak_form()

    p_load = fem.Constant(mech.domain, ScalarType(0.0))
    boundary_conditions = [
        ["Dirichlet", X0, ("clamped", 0)],
        ["Dirichlet", Y0, ("clamped", 1)],
        ["Dirichlet", Z0, ("clamped", 2)],
        ["Neumann_follower", Z1, p_load],              # X1, Y1: free
    ]
    mech.build_BCs(boundary_conditions)
    mech.build_solver()

    fields, scalars = outputs(mech)                    # after build_BCs: surface integrals need mech.ds
    with OutputManager(f"results/{name}", mech.domain, fields, scalars,
                       meshtags={"facet_tag": mech.facet_tag},
                       meta=mech.output_meta) as out:
        out.write(0.0)                                 # reference state
        for n, p in enumerate(np.linspace(0, p_max, N_STEPS + 1)[1:], start=1):
            p_load.value = p
            mech.solve_one_step(n, p)
            out.write(p)

    return out.get_history()


# ----------------------------------------------------------------------------------------
# 3. Comparison with the analytical solution
# ----------------------------------------------------------------------------------------
def compare(results):
    """Add to each result the simulated and analytical curves and the errors.

    The analytical solution is evaluated at the axial stretch of the simulation,
    lam_z = sqrt(1 + 2 E_zz). Errors are relative norms over the load steps (J: on J - 1).
    """
    for r in results:
        h = r["history"]
        lam_z = np.sqrt(1.0 + 2.0 * h["E_zz_mean"])
        a = hgo.hgo_uniaxial(lam_z, r["material"], r["fibres"], AXIAL)
        an = {"lam_z": lam_z, "lat0": a["lam"][:, 0], "lat1": a["lam"][:, 1], "J": a["J"],
              "sigma_zz": a["sigma_ax"], "P_zz": a["P_ax"]}
        sim = {"lam_z": lam_z, "lat0": np.sqrt(1.0 + 2.0 * h["E_xx_mean"]),
               "lat1": np.sqrt(1.0 + 2.0 * h["E_yy_mean"]), "J": h["J_mean"],
               "sigma_zz": h["sigma_zz_mean"], "P_zz": h["P_zz_Z1"]}
        rel = hgo.relative_error
        r["sim"], r["an"] = sim, an
        r["active"] = hgo.fibre_activity(a["I4b"], r["fibres"])
        r["errors"] = {"lam_x": rel(sim["lat0"], an["lat0"]),
                       "lam_y": rel(sim["lat1"], an["lat1"]),
                       "J": rel(sim["J"], an["J"], offset=1.0),
                       "sigma_zz": rel(sim["sigma_zz"], an["sigma_zz"]),
                       "P_zz": rel(sim["P_zz"], an["P_zz"])}


if __name__ == "__main__":
    mesh_file = "Usecases/cube/cube_hexa.msh"
    base_json = "Usecases/cornea_axi/mech_params_Pandolfi2006.json"
    folder = "results/Testing_tools/Check_against_analytical/Check_hyper_3D_HGO_axial"

    cases = hgo.make_cases(base_json, FIBRE_CASES, axial=AXIAL)
    results = hgo.run_cases(cases, run_simulation, cube_outputs, mesh_file, folder)
    if MPI.COMM_WORLD.rank == 0:
        compare(results)
        hgo.report(results)
        hgo.plot(results, Path(folder) / "cube_HGO_verification.png", ("lam_x", "lam_y"))