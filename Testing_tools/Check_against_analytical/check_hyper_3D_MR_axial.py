# Verification of the 3D framework for hyperelastic laws.
#
# Cube under axial load: symmetry planes X0, Y0, Z0 (u_n = 0), follower pressure on Z1,
# lateral faces X1 and Y1 free. The state is homogeneous, F = diag(lam_x, lam_y, lam_z) with
# lam_x = lam_y = lam_r such that sigma_xx = sigma_yy = 0: scalar analytical solution for
# the compressible Mooney-Rivlin law
#     W = C10 (I1bar - 3) + C01 (I2bar - 3) + (J - 1)^2 / D
# Two loadings are run: compression and tension. The mesh is given from outside
# (cube_hexa.msh), only a few cases are run.

import json
from itertools import product
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx import fem
from petsc4py.PETSc import ScalarType
from scipy.optimize import brentq

from Simulation_Framework import Hyperelastic_3D_framework, OutputManager, average

# physical tags of the faces in cube_hexa.msh
X0, X1, Y0, Y1, Z0, Z1 = 1, 2, 3, 4, 5, 6

P_MAX = 0.5          # maximum load magnitude on Z1 (same units as C10)
N_STEPS = 10
# Signed load on Z1: Neumann_follower is assumed to be a pressure (positive = compression).
# compare() checks the sign of lam_z - 1 and warns if this convention is reversed.
LOADS = {"compression": +P_MAX, "tension": -P_MAX}


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
def run_simulation(name, mesh_file, mech_params_json, outputs, p_max=P_MAX):
    """outputs : function mech -> (fields, scalars), see cube_outputs.
    p_max : signed final load on Z1 (ramped linearly from 0 in N_STEPS steps)."""
    mech = Hyperelastic_3D_framework(mesh_file, mech_params_json)

    if mech.mech_params.sedf_type == 'HGO':
        mech._fibre_orientation_field()
    mech.build_weak_form()

    # load on Z1, ramped during the simulation
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
# 3. Analytical solution
# ----------------------------------------------------------------------------------------
def principal_cauchy(lam_r, lam_z, C10, C01, D):
    """Principal Cauchy stresses (xx, zz, yy) for F = diag(lam_r, lam_z, lam_r)."""
    lam = np.array([lam_r, lam_z, lam_r])
    J = np.prod(lam)
    bb = lam**2 * J ** (-2.0 / 3.0)                  # eigenvalues of bbar = J^(-2/3) F F^T
    I1 = bb.sum()
    A = (C10 + C01 * I1) * bb - C01 * bb**2
    dev = A - A.mean()                               # deviatoric part (trace/3 removed)
    return 2.0 / J * dev + 2.0 / D * (J - 1.0)


def analytical_uniaxial(lam_z, C10, C01, D, bracket=(0.05, 10.0)):
    """Homogeneous solution as a function of the axial stretch.

    Returns lam_r (= lam_x = lam_y), J, sigma_zz (Cauchy) and P_zz (first Piola-Kirchhoff).
    lam_r is the root of sigma_rr(lam_r) = 0 in `bracket`.
    """
    lam_z = np.atleast_1d(np.asarray(lam_z, float))
    lam_r = np.empty_like(lam_z)
    sigma_zz = np.empty_like(lam_z)
    for i, lz in enumerate(lam_z):
        lam_r[i] = brentq(lambda lr: principal_cauchy(lr, lz, C10, C01, D)[0], *bracket)
        sigma_zz[i] = principal_cauchy(lam_r[i], lz, C10, C01, D)[1]
    J = lam_r**2 * lam_z
    return {"lam_z": lam_z, "lam_r": lam_r, "J": J,
            "sigma_zz": sigma_zz, "P_zz": J * sigma_zz / lam_z}   # P = J sigma F^-T


# ----------------------------------------------------------------------------------------
# 4. Make and run several simulations
# ----------------------------------------------------------------------------------------
def make_cases(C10=1.0, ratio_D=((-0.175, 0.4),), loads=LOADS):
    """Cases = material parameter sets x loadings (the mesh is heavy: keep the lists short).

    Material: C01 = ratio * C10, with its D (default: baseline Mooney-Rivlin only).
    Other sets to add if needed: (0.0, 0.4) neo-Hookean-like, (0.5, 0.04) nearly incompressible.
    """
    return [{"label": label, "p_max": p_max,
             "material": {"C_10": C10, "C_01": r * C10, "D": D}}
            for (r, D), (label, p_max) in product(ratio_D, loads.items())]


def run_cases(cases, run_simulation, outputs, mesh_file, folder):
    """Run one simulation per case on the given mesh, in a fixed folder.

    Layout:  <folder>/case_<k>.json            (inputs)
             <folder>/simulations/case_<k>*    (simulation results)
    run_simulation(name, mesh_file, mech_params_json, outputs, p_max) -> history
    Returns a list of {"label", "params", "history"} (same on all ranks).
    """
    comm = MPI.COMM_WORLD
    folder = Path(folder)
    sim_dir = folder / "simulations"
    if comm.rank == 0:
        sim_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    results = []
    for k, case in enumerate(cases):
        json_file = folder / f"case_{k}.json"
        if comm.rank == 0:
            with open(json_file, "w") as f:
                json.dump({"sedf_type": "Mooney-Rivlin", **case["material"]}, f, indent=2)
        comm.barrier()
        history = run_simulation(str(sim_dir / f"case_{k}_{case['label']}"), str(mesh_file),
                                 str(json_file), outputs, p_max=case["p_max"])
        results.append({"label": case["label"], "params": case["material"], "history": history})
    return results


# ----------------------------------------------------------------------------------------
# 5. Comparison and plots
# ----------------------------------------------------------------------------------------
def compare(results):
    """Add to each result the simulated and analytical curves and the errors.

    The analytical solution is evaluated at the axial stretch of the simulation,
    lam_z = sqrt(1 + 2 E_zz). Errors are relative norms over the load steps (J: on J - 1).
    """
    for r in results:
        h, p = r["history"], r["params"]
        lam_z = np.sqrt(1.0 + 2.0 * h["E_zz_mean"])
        an = analytical_uniaxial(lam_z, p["C_10"], p["C_01"], p["D"])
        sim = {"lam_z": lam_z,
               "lam_x": np.sqrt(1.0 + 2.0 * h["E_xx_mean"]),
               "lam_y": np.sqrt(1.0 + 2.0 * h["E_yy_mean"]),
               "J": h["J_mean"],
               "sigma_zz": h["sigma_zz_mean"],
               "P_zz": h["P_zz_Z1"]}
        rel = lambda s, a: np.linalg.norm(s - a) / np.linalg.norm(a)
        r["sim"], r["an"] = sim, an
        expected = {"compression": -1.0, "tension": +1.0}.get(r["label"])
        if expected is not None and np.sign(lam_z[-1] - 1.0) != expected:
            print(f"WARNING: case '{r['label']}' ends at lam_z = {lam_z[-1]:.3f}: the sign "
                  "convention of Neumann_follower is reversed, swap the signs in LOADS.")
        r["errors"] = {"lam_x": rel(sim["lam_x"], an["lam_r"]),
                       "lam_y": rel(sim["lam_y"], an["lam_r"]),
                       "J": rel(sim["J"] - 1.0, an["J"] - 1.0),
                       "sigma_zz": rel(sim["sigma_zz"], an["sigma_zz"]),
                       "P_zz": rel(sim["P_zz"], an["P_zz"])}


def report(results):
    keys = list(results[0]["errors"])
    print(f"{'load':>12} {'C_10':>6} {'C_01':>8} {'D':>7} | " + " ".join(f"{k:>10}" for k in keys))
    for r in results:
        p = r["params"]
        print(f"{r['label']:>12} {p['C_10']:6.3g} {p['C_01']:8.3g} {p['D']:7.3g} | "
              + " ".join(f"{r['errors'][k]:10.2e}" for k in keys))


def plot(results, filename):
    import matplotlib.pyplot as plt
    panels = [("P_zz", "P_zz (PK1)"), ("sigma_zz", "sigma_zz (Cauchy)"), ("J", "J")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    for ax, (key, label) in zip(axes.flat, panels):
        for c, r in zip(colors, results):
            p = r["params"]
            ax.plot(r["an"]["lam_z"], r["an"][key], "-", color=c)
            ax.plot(r["sim"]["lam_z"], r["sim"][key], "o", color=c, ms=3,
                    label=f"{r['label']}, C01={p['C_01']:.3g}, D={p['D']:.3g}")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    ax = axes[1, 1]                                    # lateral stretches, x and y
    for c, r in zip(colors, results):
        ax.plot(r["an"]["lam_z"], r["an"]["lam_r"], "-", color=c)
        ax.plot(r["sim"]["lam_z"], r["sim"]["lam_x"], "o", color=c, ms=3)
        ax.plot(r["sim"]["lam_z"], r["sim"]["lam_y"], "x", color=c, ms=4)
    ax.set_ylabel("lam_x (o), lam_y (x)")
    ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("lam_z")
    axes[0, 0].legend(fontsize=6, title="line: analytical, markers: simulation")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)


if __name__ == "__main__":
    mesh_file = "Usecases/cube/cube_hexa.msh"
    folder = "Testing_tools/Check_against_analytical/Check_hyper_3D_MR_axial"

    results = run_cases(make_cases(), run_simulation, cube_outputs, mesh_file, folder)
    if MPI.COMM_WORLD.rank == 0:
        compare(results)
        report(results)
        plot(results, Path(folder) / "cube_axial_verification.png")