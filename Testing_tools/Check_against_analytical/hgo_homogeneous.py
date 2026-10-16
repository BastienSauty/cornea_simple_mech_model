# Analytical homogeneous solution of a body under uniaxial load with free lateral faces,
# for the compressible Mooney-Rivlin matrix + two fibre families (HGO) aligned with the
# principal axes of the deformation. Shared by the axisymmetric and the 3D tests.
#
#   W = C10 (I1b - 3) + C01 (I2b - 3) + (J - 1)^2 / D
#       + sum_{i in fibres} k1 / (2 k2) (exp(k2 <I*_i - 1>^2) - 1)
#   I*_i = kappa I1b + (1 - 3 kappa) I4b_i,   I4b_i = J^(-2/3) lam_i^2,   <x> = max(x, 0)
#
# Principal axes are numbered 0, 1, 2 (3D: x, y, z; axisymmetric: r, z, theta).
# With F = diag(lam_0, lam_1, lam_2), the nominal principal stresses are P_i = dW / dlam_i
# and the Cauchy stresses sigma_i = P_i lam_i / J.

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import sympy as sp
from mpi4py import MPI
from scipy.optimize import brentq, root

# target axial stretches of the two loadings. The fibre law is very stiff (k2 = 750), keep
# the range small: at lam = 1.05 the axial fibre stress is already of the order of 1 MPa.
TARGETS = {"compression": 0.97, "tension": 1.03}


class NoHomogeneousSolution(Exception):
    """The free-lateral uniaxial state is not homogeneous for this fibre configuration."""


@lru_cache(maxsize=None)
def build_laws(fibres):
    """P_i = dW/dlam_i as numpy functions (lam_0, lam_1, lam_2, C10, C01, D, k1, k2, kappa)."""
    lam = sp.symbols("l0 l1 l2", positive=True)
    C10, C01, kappa = sp.symbols("C10 C01 kappa", real=True)
    D, k1, k2 = sp.symbols("D k1 k2", positive=True)
    J = lam[0] * lam[1] * lam[2]
    I1 = sum(x**2 for x in lam)
    I2 = (lam[0] * lam[1])**2 + (lam[1] * lam[2])**2 + (lam[2] * lam[0])**2
    I1b = J**sp.Rational(-2, 3) * I1
    W = C10 * (I1b - 3) + C01 * (J**sp.Rational(-4, 3) * I2 - 3) + (J - 1)**2 / D
    for i in fibres:
        I4b = J**sp.Rational(-2, 3) * lam[i]**2
        x = sp.Max(kappa * I1b + (1 - 3 * kappa) * I4b - 1, 0)
        W += k1 / (2 * k2) * (sp.exp(k2 * x**2) - 1)
    args = (*lam, C10, C01, D, k1, k2, kappa)
    return [sp.lambdify(args, sp.diff(W, x), "numpy") for x in lam]


def hgo_uniaxial(lam_ax, mat, fibres, axial, tie=False):
    """Homogeneous solution as a function of the stretch along the loading axis.

    mat    : dict with C_10, C_01, D, k1, k2, kappa
    fibres : indices of the principal axes carrying a fibre family, e.g. (0, 1)
    axial  : index of the loading axis
    tie    : True if the two lateral stretches are forced equal (axisymmetric solid cylinder,
             lam_r = lam_theta). Then sigma_lat1 = sigma_lat2 = 0 cannot both hold if the
             fibres are not symmetric in the lateral plane: NoHomogeneousSolution is raised.

    Returns lam (n, 3) principal stretches, J, I4b (n, 3) isochoric invariants of the three
    axes (a fibre carries load where I4b > 1), sigma_ax (Cauchy) and P_ax (nominal).
    """
    P = build_laws(tuple(fibres))
    m = (mat["C_10"], mat["C_01"], mat["D"], mat["k1"], mat["k2"], mat["kappa"])
    lat = [i for i in range(3) if i != axial]
    lam_ax = np.atleast_1d(np.asarray(lam_ax, float))
    lam = np.empty((len(lam_ax), 3))
    P_ax = np.empty(len(lam_ax))

    def stretches(la, x):
        s = [0.0, 0.0, 0.0]
        s[axial], s[lat[0]], s[lat[1]] = la, x[0], x[1]
        return s

    for n, la in enumerate(lam_ax):
        g = la**-0.5                                         # incompressible estimate
        if tie:
            x = brentq(lambda t: P[lat[0]](*stretches(la, [t, t]), *m),
                       0.8 * g, 1.25 * g, xtol=1e-14)
            x = [x, x]
        else:
            sol = root(lambda t: [P[lat[0]](*stretches(la, t), *m),
                                  P[lat[1]](*stretches(la, t), *m)], [g, g], tol=1e-12)
            if not (sol.success or np.abs(sol.fun).max() < 1e-10):
                raise RuntimeError(f"lateral stretches not found for lam_ax = {la}: {sol.message}")
            x = sol.x
        s = stretches(la, x)
        P_ax[n] = P[axial](*s, *m)
        if tie and abs(P[lat[1]](*s, *m)) > 1e-6 * (abs(P_ax[n]) + 1e-12):
            raise NoHomogeneousSolution(
                f"sigma_{lat[0]} = 0 does not imply sigma_{lat[1]} = 0 at lam_ax = {la:.4f}: "
                "no homogeneous solution")
        lam[n] = s

    J = lam.prod(axis=1)
    return {"lam": lam, "J": J, "I4b": J[:, None]**(-2.0 / 3.0) * lam**2,
            "sigma_ax": P_ax * lam[:, axial] / J, "P_ax": P_ax}


def follower_load(sigma_ax):
    """Signed follower load giving the Cauchy axial stress sigma_ax on the loaded face,
    assuming Neumann_follower is a pressure (positive = compression): p = -sigma_ax."""
    return -float(sigma_ax)


# ----------------------------------------------------------------------------------------
# Cases, runs, report and plot (common to the axisymmetric and 3D tests)
# ----------------------------------------------------------------------------------------
def make_cases(base_json, fibre_cases, axial, tie=False, targets=TARGETS):
    """One case per (fibre case, loading). The load amplitude is computed from the analytical
    solution so that the simulation ends at the target stretch: p_max = -sigma_ax(target).

    fibre_cases: {name: {"a4": [...], "a6": [...], "fibres": (i, j)}}, where fibres are the
    principal axes of a4 and a6. Cases without homogeneous solution are skipped.
    """
    with open(base_json) as f:
        base = json.load(f)
    cases = []
    for name, fc in fibre_cases.items():
        for load, target in targets.items():
            try:
                an = hgo_uniaxial([target], base, fc["fibres"], axial, tie)
            except NoHomogeneousSolution as e:
                print(f"case '{name}', {load}: skipped, {e}")
                continue
            cases.append({"fibre_case": name, "load": load, "target": target,
                          "fibres": tuple(fc["fibres"]),
                          "p_max": follower_load(an["sigma_ax"][0]),
                          "material": {**base, "a4": fc["a4"], "a6": fc["a6"]}})
    return cases


def run_cases(cases, run_simulation, outputs, mesh_file, folder):
    """Run one simulation per case on the given mesh, in a fixed folder.

    Layout:  <folder>/case_<k>_<fibre case>_<load>.json   (inputs)
             <folder>/simulations/case_<k>_<fibre case>_<load>*   (simulation results)
    run_simulation(name, mesh_file, mech_params_json, outputs, p_max) -> history
    Returns the cases with their "history" (same on all ranks).
    """
    comm = MPI.COMM_WORLD
    folder = Path(folder)
    sim_dir = folder / "simulations"
    if comm.rank == 0:
        sim_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    results = []
    for k, case in enumerate(cases):
        tag = f"case_{k}_{case['fibre_case']}_{case['load']}"
        json_file = folder / f"{tag}.json"
        if comm.rank == 0:
            with open(json_file, "w") as f:
                json.dump(case["material"], f, indent=2)
        comm.barrier()
        history = run_simulation(str(sim_dir / tag), str(mesh_file), str(json_file), outputs,
                                 p_max=case["p_max"])
        results.append({**case, "history": history})
    return results


def relative_error(sim, ref, offset=0.0):
    """Relative norm over the load steps of (sim - ref), normalised by ref - offset."""
    return np.linalg.norm(sim - ref) / max(np.linalg.norm(ref - offset), 1e-12)


def fibre_activity(I4b, fibres):
    """State of the two fibre families at the last step: 'a4:on a6:off' (I4b > 1 = on)."""
    return " ".join(f"{n}:{'on' if I4b[-1, i] > 1.0 else 'off'}"
                    for n, i in zip(("a4", "a6"), fibres))


def report(results):
    keys = list(results[0]["errors"])
    print(f"{'fibres':>14} {'load':>12} | " + " ".join(f"{k:>10}" for k in keys)
          + " | fibres at last step (analytical)")
    for r in results:
        print(f"{r['fibre_case']:>14} {r['load']:>12} | "
              + " ".join(f"{r['errors'][k]:10.2e}" for k in keys) + f" | {r['active']}")


def plot(results, filename, lateral_labels):
    """One row per fibre case, tension and compression on the same axes (lam_z around 1).
    r["sim"] and r["an"] hold lam_z, lat0, lat1, J, sigma_zz, P_zz."""
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(r["fibre_case"] for r in results))
    colors = {"compression": "tab:blue", "tension": "tab:red"}
    fig, axes = plt.subplots(len(names), 4, figsize=(15, 3.8 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        for r in (r for r in results if r["fibre_case"] == name):
            c, an, sim = colors.get(r["load"], "k"), r["an"], r["sim"]
            for ax, key in zip(axes[row, :3], ["P_zz", "sigma_zz", "J"]):
                ax.plot(an["lam_z"], an[key], "-", color=c)
                ax.plot(sim["lam_z"], sim[key], "o", color=c, ms=3, label=r["load"])
                ax.set_ylabel(key)
            ax = axes[row, 3]
            for key, ls, mk, lab in (("lat0", "-", "o", lateral_labels[0]),
                                     ("lat1", "--", "x", lateral_labels[1])):
                ax.plot(an["lam_z"], an[key], ls, color=c)
                ax.plot(sim["lam_z"], sim[key], mk, color=c, ms=3)
            ax.set_ylabel(f"{lateral_labels[0]} (o), {lateral_labels[1]} (x)")
        axes[row, 0].set_title(f"fibres {name}", loc="left", fontsize=10)
        for ax in axes[row]:
            ax.set_xlabel("lam_z")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, title="line: analytical, markers: simulation")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
