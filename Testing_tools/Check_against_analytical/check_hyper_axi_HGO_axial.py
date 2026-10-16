# Verification of the axisymmetric framework for the HGO law (Mooney-Rivlin matrix + 2 fibre
# families). Filled cylinder under axial load, free lateral surface, bottom u_z = 0.
#
# Principal axes: 0 = radial, 1 = axial, 2 = hoop (local frame e_1, e_2, e_3). Two fibre
# configurations, each in tension and in compression (check of the compression switch: a
# fibre carries load only where I4bar > 1):
#     radial_hoop  : a4 = [1 0 0], a6 = [0 0 1]   (both transverse to the load)
#     radial_axial : a4 = [1 0 0], a6 = [0 1 0]   (one transverse, one axial)
# The solution is homogeneous (lam_r = lam_theta) only if sigma_r = 0 implies sigma_theta = 0.
# This fails for radial_axial in compression, where the radial fibre is stretched but there is
# no hoop fibre: that case has no homogeneous solution and is skipped (see hgo_homogeneous).

from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem
from petsc4py.PETSc import ScalarType

from Simulation_Framework import Hyperelastic_axisymmetric_framework, OutputManager
import hgo_homogeneous as hgo

# mesh lines
CENTRAL_LINE, LATERAL, BOTTOM, TOP = 4, 3, 2, 1
AXIAL = 1                 # principal axes: 0 = radial, 1 = axial, 2 = hoop
N_STEPS = 20

FIBRE_CASES = {
    "radial_hoop": {"a4": [1, 0, 0], "a6": [0, 0, 1], "fibres": (0, 2)},
    "radial_axial": {"a4": [1, 0, 0], "a6": [0, 1, 0], "fibres": (0, 1)},
}


# ----------------------------------------------------------------------------------------
# 1. outputs of the simulation that will be compared
# ----------------------------------------------------------------------------------------
def cylinder_outputs(mech):
    """Quantities of interest of the filled cylinder under axial loading.
    Returns (fields, scalars) for the OutputManager.

    Reference averages are forms divided by the reference volume V0 (assembled once).
    The Cauchy stress average is int(sigma_zz J dV) / int(J dV): both integrals are
    written, the ratio is taken in the post-processing.
    """
    q = mech.quantities
    fields = {name: q[name] for name in
              ["displacement", "cauchy_stress", "von_mises", "J", "green_lagrange"]}
    sigma, E, J = q["cauchy_stress"], q["green_lagrange"], q["J"]

    one = fem.Constant(mech.domain, 1.0)
    V0 = mech.domain.comm.allreduce(
        fem.assemble_scalar(fem.form(mech.volume_integral(one))))

    # axial reaction on the top surface: PK1 . N integrated on TOP
    N = ufl.FacetNormal(mech.domain)
    P = q["PK1_stress"]
    traction_z = P[1, 0] * N[0] + P[1, 1] * N[1]

    scalars = {
        "J_mean": mech.volume_integral(J / V0),
        "radial_strain_mean": mech.volume_integral(E[0, 0] / V0),
        "axial_strain_mean": mech.volume_integral(E[1, 1] / V0),
        "hoop_strain_mean": mech.volume_integral(E[2, 2] / V0),
        "volume": mech.volume_integral(J),                          # deformed volume
        "axial_stress_integral": mech.volume_integral(sigma[1, 1] * J),
        "top_reaction_z": mech.surface_integral(traction_z, TOP),
    }
    return fields, scalars


# ----------------------------------------------------------------------------------------
# 2. run simulation function
# ----------------------------------------------------------------------------------------
def run_simulation(name, mesh_file, mech_params_json, outputs, p_max):
    """outputs : function mech -> (fields, scalars), see cylinder_outputs.
    p_max : signed final load on TOP (positive = compression), ramped in N_STEPS steps."""
    mech = Hyperelastic_axisymmetric_framework(mesh_file, mech_params_json)

    mech.build_weak_form()

    p_load = fem.Constant(mech.domain, ScalarType(0.0))
    boundary_conditions = [
        ["Dirichlet", CENTRAL_LINE, ("clamped", 0)],   # symmetry axis: u_r = 0, u_z free
        ["Dirichlet", BOTTOM,       ("clamped", 1)],   # bottom: u_z = 0, u_r free
        ["Neumann_follower", TOP,   p_load],           # axial load on the top surface
    ]                                                  # lateral surface: free
    mech.build_BCs(boundary_conditions)
    mech.build_solver()

    fields, scalars = outputs(mech)                    # after build_BCs: surface integrals need mech.ds
    with OutputManager(f"{name}", mech.domain, fields, scalars,
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
def compare(results, rect):
    """Add to each result the simulated and analytical curves and the errors.

    The analytical solution is evaluated at the axial stretch of the simulation,
    lam_z = sqrt(1 + 2 E_zz). Errors are relative norms over the load steps (J: on J - 1).
    """
    area = np.pi * (rect.re**2 - rect.ri**2)
    for r in results:
        h = r["history"]
        lam_z = np.sqrt(1.0 + 2.0 * h["axial_strain_mean"])
        a = hgo.hgo_uniaxial(lam_z, r["material"], r["fibres"], AXIAL, tie=True)
        an = {"lam_z": lam_z, "lat0": a["lam"][:, 0], "lat1": a["lam"][:, 2], "J": a["J"],
              "sigma_zz": a["sigma_ax"], "P_zz": a["P_ax"]}
        sim = {"lam_z": lam_z, "lat0": np.sqrt(1.0 + 2.0 * h["radial_strain_mean"]),
               "lat1": np.sqrt(1.0 + 2.0 * h["hoop_strain_mean"]), "J": h["J_mean"],
               "sigma_zz": h["axial_stress_integral"] / h["volume"],
               "P_zz": h["top_reaction_z"] / area}
        rel = hgo.relative_error
        r["sim"], r["an"] = sim, an
        r["active"] = hgo.fibre_activity(a["I4b"], r["fibres"])
        r["errors"] = {"lam_r": rel(sim["lat0"], an["lat0"]),
                       "lam_theta": rel(sim["lat1"], an["lat1"]),
                       "J": rel(sim["J"], an["J"], offset=1.0),
                       "sigma_zz": rel(sim["sigma_zz"], an["sigma_zz"]),
                       "P_zz": rel(sim["P_zz"], an["P_zz"])}


if __name__ == "__main__":
    from Meshing_tools.make_axisym_rectangle import RectangleParams, make_rectangle

    base_json = "Usecases/cornea_axi/mech_params_Pandolfi2006.json"
    folder = Path("results/Testing_tools/Check_against_analytical/Check_hyper_axi_HGO_axial")
    rect = RectangleParams(ri=0.0, re=1.0, h=1.0, n_r=4, n_z=4)

    mesh_file = folder / "rectangle.msh"
    if MPI.COMM_WORLD.rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
        make_rectangle(str(mesh_file), rect)
    MPI.COMM_WORLD.barrier()

    cases = hgo.make_cases(base_json, FIBRE_CASES, axial=AXIAL, tie=True)
    results = hgo.run_cases(cases, run_simulation, cylinder_outputs, mesh_file, folder)
    if MPI.COMM_WORLD.rank == 0:
        compare(results, rect)
        hgo.report(results)
        hgo.plot(results, folder / "cylinder_HGO_verification.png", ("lam_r", "lam_theta"))