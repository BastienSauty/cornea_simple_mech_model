# Verification of the 3D poroelastic framework.
#
# Terzaghi consolidation on a cube: a laterally confined column (1D along z) with
# incompressible constituents and constant permeability, loaded at once by a constant
# compressive stress p0 on its drained top face Z1. Compared with the linear analytical solution
# (see the document "Terzaghi consolidation: equations for the poroelastic check"):
#     p(0,t)/p0 = sum_m 2(-1)^m / M_m  exp(-M_m^2 T)           (bottom, impervious)
#     U(T)      = 1 - sum_m 2 / M_m^2  exp(-M_m^2 T)           (degree of consolidation)
#     with T = c_v t / H^2,  c_v = k M,  M_m = (2m + 1) pi / 2.
# In the framework the skeleton stress is phi_s dpsi_iso/dF - J p F^-T, with psi_iso the
# isochoric part of the Mooney-Rivlin energy only (Giammarini 2026): the parameter D is not
# used, and the drained constrained modulus is
#     M = (4/3) phi_s mu,  mu = 2 (C_10 + C_01).
# Cases: the baseline, a 10 times larger permeability, and a different phi_s (M changes).
# In T all cases must fall on the same analytical curves. The mesh is given from outside
# (cube_hexa.msh, side 1).

import numpy as np

from mpi4py import MPI
from dolfinx import fem
from petsc4py.PETSc import ScalarType
import json

from pathlib import Path

from Simulation_Framework import Poroelastic_3D_framework, OutputManager, average, point_probe

# physical tags of the faces in cube_hexa.msh
X0, X1, Y0, Y1, Z0, Z1 = 1, 2, 3, 4, 5, 6

HEIGHT = 1.0          # column height H (side of the cube)
N_ELEM_Z = 5          # elements along z in cube_hexa.msh
STRAIN = 5e-3         # p0 / M: small, so that the response stays linear
# The diffusion length sqrt(T) H must span at least 2 elements at the first step: T_MIN = (2 / n_z)^2.
# Below it the boundary layer is not resolved (and, with a refined mesh, T_MIN can be lowered).
T_MIN, T_END = (2.0 / N_ELEM_Z)**2, 3.0     # first time step and end time, in units of H^2 / c_v
N_STEPS = 100         # geometric time steps between T_MIN and T_END
N_PROFILE = 13        # points of the pressure profile along the vertical line, z/H from 0 to 1
T_PROFILES = (0.15, 0.3, 0.5, 0.8, 1.2, 2.0)   # times shown in the profile plot


# ----------------------------------------------------------------------------------------
# 1. outputs of the simulation that will be compared
# ----------------------------------------------------------------------------------------
def column_outputs(mech):
    """Quantities of interest of the consolidating column.
    Returns (fields, scalars) for the OutputManager.

    p_bottom: mean fluid pressure on Z0 (uniform in the 1D solution).
    uz_top: mean axial displacement of Z1.
    p_z_<k>: fluid pressure on the vertical line through the centre of the cube, at
             z = H k / (N_PROFILE - 1).
    """
    q = mech.quantities
    fields = {name: q[name] for name in
              ["displacement", "cauchy_stress", "J", "fluid_pressure"]}

    comm = mech.domain.comm
    one = fem.Constant(mech.domain, ScalarType(1.0))
    scalars = {
        "p_bottom": average(mech.surface_integral(q["fluid_pressure"], Z0),
                            mech.surface_integral(one, Z0), comm),
        "uz_top": average(mech.surface_integral(q["displacement"][2], Z1),
                          mech.surface_integral(one, Z1), comm),
    }
    for k, z in enumerate(np.linspace(0.0, HEIGHT, N_PROFILE)):
        probe = point_probe(mech.domain, mech.p, (0.5, 0.5, z))
        scalars[f"p_z_{k}"] = lambda probe=probe: float(np.ravel(probe())[0])
    return fields, scalars


# ----------------------------------------------------------------------------------------
# 2. run simulation function
# ----------------------------------------------------------------------------------------
def terzaghi_parameters(params):
    """Drained constrained modulus M = (4/3) phi_s mu, consolidation coefficient c_v = k M, and
    the load p0 = STRAIN * M for a material parameter dictionary (Mooney-Rivlin, k = permeability).
    D does not enter: the framework has no volumetric skeleton energy."""
    mu = 2.0 * (params["C_10"] + params["C_01"])
    M = 4.0 * params["phi_s"] * mu / 3.0
    return {"M": M, "c_v": params["k"] * M, "p0": STRAIN * M}


def run_simulation(name, mesh_file, mech_params_json, outputs):
    """outputs : function mech -> (fields, scalars), see column_outputs."""

    # load and time steps from the material parameters: geometric steps in T = c_v t / H^2
    with open(mech_params_json) as f:
        tp = terzaghi_parameters(json.load(f))
    p0 = tp["p0"]
    t_c = HEIGHT**2 / tp["c_v"]
    t_list = t_c * np.concatenate([[0.0], np.geomspace(T_MIN, T_END, N_STEPS)])

    mech = Poroelastic_3D_framework(mesh_file, mech_params_json)

    dt = fem.Constant(mech.domain, ScalarType(0))
    mech.build_weak_form(dt)

    # constant load applied at once on the top face, drained top
    p_load = fem.Constant(mech.domain, ScalarType(p0))

    boundary_conditions = [
        ["Dirichlet_disp", X0, ("clamped", 0)],         # confined: u_n = 0 on the four lateral
        ["Dirichlet_disp", X1, ("clamped", 0)],         # faces (no flow there) ...
        ["Dirichlet_disp", Y0, ("clamped", 1)],
        ["Dirichlet_disp", Y1, ("clamped", 1)],
        ["Dirichlet_disp", Z0, ("clamped", 2)],         # ... and bottom: u_z = 0, no flow
        ["Neumann_follower_disp", Z1, p_load],          # compressive load on the top
        ["Dirichlet_pressure", Z1, "drained"],          # drained top, other faces impervious
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


def make_cases(base=BASE, variations=({}, {"k": 1.5e-2}, {"phi_s": 0.4})):
    """Parameter sets: the baseline, a 10 times larger permeability, a different solid fraction
    (M is multiplied by 2). In T = c_v t / H^2 all cases must fall on the same analytical curves.
    The independence from D is checked in the axisymmetric test."""
    return [{**base, **v} for v in variations]


def run_cases(cases, run_simulation, outputs, mesh_file, folder):
    """Run one simulation per case on the given mesh, in a fixed folder.

    Layout:  <folder>/case_<k>.json            (inputs)
             <folder>/simulations/case_<k>*    (simulation results)
    run_simulation(name, mesh_file, mech_params_json, outputs) -> history
    outputs: column_outputs. Returns a list of {"params", "history"} (same on all ranks).
    """
    comm = MPI.COMM_WORLD
    folder = Path(folder)
    sim_dir = folder / "simulations"
    if comm.rank == 0:
        sim_dir.mkdir(parents=True, exist_ok=True)
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
    (in units of p0 for the pressure, of the final settlement for U). profile_max is the
    maximum error of p(z,t)/p0 over all probed heights and time steps.
    """
    for r in results:
        h, p = r["history"], r["params"]
        tp = terzaghi_parameters(p)
        T = h["t"][1:] * tp["c_v"] / HEIGHT**2
        an = terzaghi_solution(T)
        sim = {"T": T,
               "p": h["p_bottom"][1:] / tp["p0"],
               "U": -h["uz_top"][1:] * tp["M"] / (tp["p0"] * HEIGHT)}
        # pressure profile along the axis, rows = time steps, columns = z / H
        sim["profile"] = np.stack([h[f"p_z_{k}"][1:] for k in range(N_PROFILE)], axis=1) / tp["p0"]
        an["profile"] = np.stack([terzaghi_solution(T, z)["p"]
                                  for z in np.linspace(0.0, 1.0, N_PROFILE)], axis=1)
        r["sim"], r["an"] = sim, an
        r["errors"] = {}
        for k in ["p", "U"]:
            r["errors"][f"{k}_rel"] = np.linalg.norm(sim[k] - an[k]) / np.linalg.norm(an[k])
            r["errors"][f"{k}_max"] = np.max(np.abs(sim[k] - an[k]))
        r["errors"]["profile_max"] = np.max(np.abs(sim["profile"] - an["profile"]))


def report(results):
    keys = list(results[0]["errors"])
    print(f"{'C_10':>6} {'C_01':>8} {'D':>7} {'phi_s':>6} {'k':>9} | "
          + " ".join(f"{k:>10}" for k in keys))
    for r in results:
        p = r["params"]
        print(f"{p['C_10']:6.3g} {p['C_01']:8.3g} {p['D']:7.3g} {p['phi_s']:6.3g} "
              f"{p['k']:9.3g} | "
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
                    label=f"k={p['k']:.3g}, phi_s={p['phi_s']:.3g}, D={p['D']:.3g}")
        ax.set_xscale("log")
        ax.set_xlabel("T = c_v t / H^2")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)


def plot_profiles(results, filename, T_values=T_PROFILES):
    """Pressure profile p(z,t)/p0 along the height, one panel per case, one colour per time.
    Lines: analytical solution; markers: simulation. For each requested T, the closest time
    step of the simulation is used (both curves are evaluated at its T)."""
    import matplotlib.pyplot as plt
    n = len(results)
    ncols = min(n, 2)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    zs = np.linspace(0.0, 1.0, N_PROFILE)
    zf = np.linspace(0.0, 1.0, 200)
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(T_values)))
    for ax, r in zip(axes.flat, results):
        p = r["params"]
        T_sim = r["sim"]["T"]
        for c, T_target in zip(colors, T_values):
            i = np.argmin(np.abs(np.log(T_sim / T_target)))
            ax.plot([terzaghi_solution(T_sim[i], z)["p"][0] for z in zf], zf, "-", color=c, lw=1)
            ax.plot(r["sim"]["profile"][i], zs, "o", color=c, ms=3, label=f"T = {T_sim[i]:.2g}")
        ax.set_title(f"k={p['k']:.3g}, phi_s={p['phi_s']:.3g}, D={p['D']:.3g}", fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("p / p0")
    for ax in axes[:, 0]:
        ax.set_ylabel("z / H  (1 = drained top)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8,
               title="line: analytical, markers: simulation")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(filename, dpi=150)


if __name__ == "__main__":
    mesh_file = "Usecases/cube/cube_hexa.msh"
    folder = "results/Testing_tools/Check_against_analytical/Check_poro_3D_MR_terzaghi"

    results = run_cases(make_cases(), run_simulation, column_outputs, mesh_file, folder)
    if MPI.COMM_WORLD.rank == 0:
        compare(results)
        report(results)
        plot(results, Path(folder) / "terzaghi_verification.png")
        plot_profiles(results, Path(folder) / "terzaghi_profiles.png")
