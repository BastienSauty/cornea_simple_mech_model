# Verification of the axisymmetric poroelastic framework.
#
# Terzaghi consolidation: a laterally confined cylinder (1D column along z) with
# incompressible constituents and constant permeability, loaded at once by a constant
# compressive stress p0 on its drained top face. Compared with the linear analytical solution
# (see the document "Terzaghi consolidation: equations for the poroelastic check"):
#     p(0,t)/p0 = sum_m 2(-1)^m / M_m  exp(-M_m^2 T)           (bottom, impervious)
#     U(T)      = 1 - sum_m 2 / M_m^2  exp(-M_m^2 T)           (degree of consolidation)
#     with T = c_v t / H^2,  c_v = k M,  M = K + 4 mu / 3,  M_m = (2m + 1) pi / 2
#     and, for the Mooney-Rivlin law linearised about the reference state,
#     mu = 2 (C_10 + C_01), K = 2 / D.

import numpy as np

from mpi4py import MPI
from dolfinx import fem
from petsc4py.PETSc import ScalarType
from itertools import product
import json

from pathlib import Path

from Simulation_Framework import Poroelastic_axisymmetric_framework, OutputManager, average

# mesh lines
CENTRAL_LINE, LATERAL, BOTTOM, TOP = 4, 3, 2, 1

HEIGHT = 1.0          # column height H
STRAIN = 5e-3         # p0 / M: small, so that the response stays linear
T_MIN, T_END = 1e-3, 2.0     # first time step and end time, in units of H^2 / c_v
N_STEPS = 100         # geometric time steps between T_MIN and T_END


# ----------------------------------------------------------------------------------------
# 1. outputs of the simulation that will be compared
# ----------------------------------------------------------------------------------------
def column_outputs(mech):
    """Quantities of interest of the consolidating column.
    Returns (fields, scalars) for the OutputManager.

    p_bottom: mean fluid pressure on BOTTOM (uniform in the 1D solution).
    uz_top: mean axial displacement of TOP.
    """
    q = mech.quantities
    fields = {name: q[name] for name in
              ["displacement", "cauchy_stress", "J", "fluid_pressure"]}

    comm = mech.domain.comm
    one = fem.Constant(mech.domain, ScalarType(1.0))
    scalars = {
        "p_bottom": average(mech.surface_integral(q["fluid_pressure"], BOTTOM),
                            mech.surface_integral(one, BOTTOM), comm),
        "uz_top": average(mech.surface_integral(q["displacement"][1], TOP),
                          mech.surface_integral(one, TOP), comm),
    }
    return fields, scalars


# ----------------------------------------------------------------------------------------
# 2. run simulation function
# ----------------------------------------------------------------------------------------
def terzaghi_parameters(params):
    """Linearised constrained modulus M, consolidation coefficient c_v = k M, and the load
    p0 = STRAIN * M for a material parameter dictionary (Mooney-Rivlin, k = permeability)."""
    mu = 2.0 * (params["C_10"] + params["C_01"])
    K = 2.0 / params["D"]
    M = K + 4.0 * mu / 3.0
    return {"M": M, "c_v": params["k"] * M, "p0": STRAIN * M}


def run_simulation(name, mesh_file, mech_params_json, outputs):
    """outputs : function mech -> (fields, scalars), see column_outputs."""

    # load and time steps from the material parameters: geometric steps in T = c_v t / H^2
    with open(mech_params_json) as f:
        tp = terzaghi_parameters(json.load(f))
    p0 = tp["p0"]
    t_c = HEIGHT**2 / tp["c_v"]
    t_list = t_c * np.concatenate([[0.0], np.geomspace(T_MIN, T_END, N_STEPS)])

    mech = Poroelastic_axisymmetric_framework(mesh_file, mech_params_json)

    dt = fem.Constant(mech.domain, ScalarType(0))
    mech.build_weak_form(dt)

    # constant load applied at once on the top face, drained top
    p_load = fem.Constant(mech.domain, ScalarType(p0))

    boundary_conditions = [
        ["Dirichlet_disp", CENTRAL_LINE, ("clamped", 0)],   # confined: u_r = 0 on the axis
        ["Dirichlet_disp", LATERAL,      ("clamped", 0)],   # and on the lateral surface
        ["Dirichlet_disp", BOTTOM,       ("clamped", 1)],   # bottom: u_z = 0
        ["Neumann_follower_disp", TOP,   p_load],           # compressive load on the top
        ["Dirichlet_pressure", TOP, "drained"],             # drained top, other faces impervious
    ]
    mech.build_BCs(boundary_conditions)
    mech.build_solver()

    fields, scalars = outputs(mech)                    # after build_BCs: surface integrals need mech.ds
    with OutputManager(f"{name}", mech.domain, fields, scalars,
                       meshtags={"facet_tag": mech.facet_tag},
                       meta=mech.output_meta) as out:
        out.write(0.0)                                 # reference state
        for n, t in enumerate(t_list[1:], start=1):
            dt.value = t - t_list[n - 1]
            mech.solve_one_step(n, t)
            out.write(t)

    return out.get_history()


# ----------------------------------------------------------------------------------------
# 3. Analytical solution
# ----------------------------------------------------------------------------------------
def terzaghi_solution(T, z_over_H=0.0, n_terms=500):
    """Linear Terzaghi solution as a function of T = c_v t / H^2 (T > 0, T >= 1e-3 for 500 terms).

    Returns p(z,t)/p0 at z/H = z_over_H (0 = bottom, impervious; 1 = top, drained) and the
    degree of consolidation U(T) = -u_z(H,t) M / (p0 H).
    """
    T = np.atleast_1d(np.asarray(T, float))[:, None]
    m = np.arange(n_terms)
    Mm = (2 * m + 1) * np.pi / 2.0
    decay = np.exp(-Mm**2 * T)
    p = (2.0 * (-1.0)**m / Mm * np.cos(Mm * z_over_H) * decay).sum(axis=1)
    U = 1.0 - (2.0 / Mm**2 * decay).sum(axis=1)
    return {"T": T[:, 0], "p": p, "U": U}


# ----------------------------------------------------------------------------------------
# 4. Make and run several simulations
# ----------------------------------------------------------------------------------------
BASE = {"sedf_type": "Mooney-Rivlin", "C_10": 0.5, "C_01": -0.175, "D": 0.4,
        "porosity_type": "Isotropic", "phi_s": 0.2, "k": 1.5e-3}


def make_cases(base=BASE, variations=({}, {"k": 1.5e-2}, {"D": 0.1})):
    """Parameter sets: the baseline, a 10 times larger permeability, a stiffer skeleton.
    In T = c_v t / H^2 all cases must fall on the same analytical curves."""
    return [{**base, **v} for v in variations]


def run_cases(cases, run_simulation, outputs, rect, folder):
    """Run one simulation per case in a fixed folder.

    Layout:  <folder>/rectangle.msh, <folder>/case_<k>.json   (inputs)
             <folder>/simulations/case_<k>*                   (simulation results)
    run_simulation(name, mesh_file, mech_params_json, outputs) -> history
    outputs: column_outputs. Returns a list of {"params", "history"} (same on all ranks).
    """
    comm = MPI.COMM_WORLD
    folder = Path(folder)
    sim_dir = folder / "simulations"
    if comm.rank == 0:
        sim_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    mesh_file = folder / "rectangle.msh"
    if comm.rank == 0:
        make_rectangle(str(mesh_file), rect)
    comm.barrier()

    results = []
    for k, params in enumerate(cases):
        json_file = folder / f"case_{k}.json"
        if comm.rank == 0:
            with open(json_file, "w") as f:
                json.dump(params, f, indent=2)
        comm.barrier()
        history = run_simulation(str(sim_dir / f"case_{k}"), str(mesh_file), str(json_file),
                                 outputs)
        results.append({"params": params, "history": history})
    return results


# ----------------------------------------------------------------------------------------
# 5. Comparison and plots
# ----------------------------------------------------------------------------------------
def compare(results):
    """Add to each result the simulated and analytical curves and the errors.

    Both are functions of T = c_v t / H^2 (the reference state t = 0 is left out).
    Errors: relative L2 norm over the time steps, and maximum absolute error
    (in units of p0 for the pressure, of the final settlement for U).
    """
    for r in results:
        h, p = r["history"], r["params"]
        tp = terzaghi_parameters(p)
        T = h["t"][1:] * tp["c_v"] / HEIGHT**2
        an = terzaghi_solution(T)
        sim = {"T": T,
               "p": h["p_bottom"][1:] / tp["p0"],
               "U": -h["uz_top"][1:] * tp["M"] / (tp["p0"] * HEIGHT)}
        r["sim"], r["an"] = sim, an
        r["errors"] = {}
        for k in ["p", "U"]:
            r["errors"][f"{k}_rel"] = np.linalg.norm(sim[k] - an[k]) / np.linalg.norm(an[k])
            r["errors"][f"{k}_max"] = np.max(np.abs(sim[k] - an[k]))


def report(results):
    keys = list(results[0]["errors"])
    print(f"{'C_10':>6} {'C_01':>8} {'D':>7} {'k':>9} | " + " ".join(f"{k:>10}" for k in keys))
    for r in results:
        p = r["params"]
        print(f"{p['C_10']:6.3g} {p['C_01']:8.3g} {p['D']:7.3g} {p['k']:9.3g} | "
              + " ".join(f"{r['errors'][k]:10.2e}" for k in keys))


def plot(results, filename):
    import matplotlib.pyplot as plt
    panels = [("p", "p(0,t) / p0"), ("U", "U = -u_z(H,t) M / (p0 H)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    T = np.geomspace(T_MIN, T_END, 300)
    an = terzaghi_solution(T)
    for ax, (key, label) in zip(axes, panels):
        ax.plot(T, an[key], "k-", lw=1, label="analytical")
        for c, r in zip(colors, results):
            p = r["params"]
            ax.plot(r["sim"]["T"], r["sim"][key], "o", color=c, ms=3,
                    label=f"k={p['k']:.3g}, D={p['D']:.3g}")
        ax.set_xscale("log")
        ax.set_xlabel("T = c_v t / H^2")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)


if __name__ == "__main__":
    from Meshing_tools.make_axisym_rectangle import RectangleParams, make_rectangle

    folder = "results/Testing_tools/Check_against_analytical/Check_poro_axi_MR_terzaghi"
    rect = RectangleParams(ri=0.0, re=1.0, h=HEIGHT, n_r=2, n_z=40)
    results = run_cases(make_cases(), run_simulation, column_outputs, rect, folder)
    if MPI.COMM_WORLD.rank == 0:
        compare(results)
        report(results)
        plot(results, Path(folder) / "terzaghi_verification.png")
