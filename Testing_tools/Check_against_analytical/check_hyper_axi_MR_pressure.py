# Verification of the axisymmetric framework for the different kinds of hyperelastic laws
# 
# Top and bottom faces slide (u_z = 0, u_r free), the lateral face is free, a follower
# pressure is applied on the inner face (r = ri, tag CENTRAL_LINE). The state is lam_z = 1 and
# depends on r only: it is the finite-strain compressible Lame problem, solved here by
# shooting on the equilibrium ODE. Compressible Mooney-Rivlin:
#     W = C10 (I1bar - 3) + C01 (I2bar - 3) + (J - 1)^2 / D
 
import json
from functools import partial
from itertools import product
from pathlib import Path
 
import numpy as np
import sympy as sp
import ufl
from mpi4py import MPI
from dolfinx import fem
from petsc4py.PETSc import ScalarType
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
 
from Simulation_Framework import Hyperelastic_axisymmetric_framework, OutputManager, point_probe

# mesh lines
CENTRAL_LINE, LATERAL, BOTTOM, TOP = 4, 3, 2, 1


P_MAX = 0.5          # maximum internal pressure (same units as C10)
N_STEPS = 20
N_PROBE = 5          # radial positions where u_r is probed, from ri to re

# ----------------------------------------------------------------------------------------
# 1. outputs of the simulation that will be compared 
# ----------------------------------------------------------------------------------------
def probe_radii(rect, n=N_PROBE):
    return np.linspace(rect.ri, rect.re, n)
 
 
def hollow_cylinder_outputs(mech, rect):
    """Quantities of interest. Returns (fields, scalars) for the OutputManager.
 
    J_mean: volume average of J (reference volume). top_reaction_z: axial force on TOP
    (PK1 . N). ur_k: radial displacement at the radii probe_radii(rect), at mid-height.
    Use partial(hollow_cylinder_outputs, rect=rect) to get a function of mech only.
    """
    q = mech.quantities
    fields = {name: q[name] for name in
              ["displacement", "cauchy_stress", "von_mises", "J", "green_lagrange"]}
    J = q["J"]
 
    # reference volume, constant: assembled once
    one = fem.Constant(mech.domain, 1.0)
    V0 = mech.domain.comm.allreduce(
        fem.assemble_scalar(fem.form(mech.volume_integral(one))))
 
    # axial reaction on the top surface: PK1 . N integrated on TOP
    N = ufl.FacetNormal(mech.domain)
    P = q["PK1_stress"]
    traction_z = P[1, 0] * N[0] + P[1, 1] * N[1]
 
    scalars = {
        "J_mean": mech.volume_integral(J / V0),
        "top_reaction_z": mech.surface_integral(traction_z, TOP),
    }
    for k, R in enumerate(probe_radii(rect)):
        probe = point_probe(mech.domain, mech.u, (R, 0.5 * rect.h))
        scalars[f"ur_{k}"] = lambda probe=probe: float(probe()[0])
    return fields, scalars

# ----------------------------------------------------------------------------------------
# 2. run simulation function
# ----------------------------------------------------------------------------------------
def run_simulation(name, mesh_file, mech_params_json, outputs):
    """outputs : function mech -> (fields, scalars), see hollow_cylinder_outputs."""
    mech = Hyperelastic_axisymmetric_framework(mesh_file, mech_params_json)
 
    if mech.mech_params.sedf_type == 'HGO':
        mech._fibre_orientation_field()
    mech.build_weak_form()
 
    # internal pressure, ramped during the simulation
    p_iop = fem.Constant(mech.domain, ScalarType(0.0))
 
    boundary_conditions = [
        ["Dirichlet", BOTTOM, ("clamped", 1)],         # sliding: u_z = 0, u_r free
        ["Dirichlet", TOP,    ("clamped", 1)],
        ["Neumann_follower", CENTRAL_LINE, p_iop],     # follower pressure on the inner face
    ]                                                  # lateral face: free (zero traction)
    mech.build_BCs(boundary_conditions)
    mech.build_solver()
 
    fields, scalars = outputs(mech)                    # after build_BCs: surface integrals need mech.ds
    with OutputManager(f"{name}", mech.domain, fields, scalars,
                       meshtags={"facet_tag": mech.facet_tag},
                       meta=mech.output_meta) as out:
        out.write(0.0)                                 # reference state
        for n, p in enumerate(np.linspace(0, P_MAX, N_STEPS)[1:], start=1):
            p_iop.value = p
            mech.solve_one_step(n, p)
            out.write(p)
 
    return out.get_history()


# ----------------------------------------------------------------------------------------
# 3. Analytical solution
# ----------------------------------------------------------------------------------------
def _build_laws():
    """Nominal principal stresses P_i = dW/dlam_i (lam_z set to 1) and the derivatives of P_r
    needed by the equilibrium equation, as numpy functions of (lam_r, lam_theta, C10, C01, D)."""
    lr, lt, lz = sp.symbols("lr lt lz", positive=True)
    C10, C01 = sp.symbols("C10 C01", real=True)
    D = sp.Symbol("D", positive=True)
    J = lr * lt * lz
    I1 = lr**2 + lt**2 + lz**2
    I2 = (lr * lt)**2 + (lt * lz)**2 + (lz * lr)**2
    W = (C10 * (J**sp.Rational(-2, 3) * I1 - 3) + C01 * (J**sp.Rational(-4, 3) * I2 - 3)
         + (J - 1)**2 / D)
    Pr, Pt, Pz = (sp.diff(W, x) for x in (lr, lt, lz))
    f = lambda e: sp.lambdify((lr, lt, C10, C01, D), e.subs(lz, 1), "numpy")
    return {"Pr": f(Pr), "Pt": f(Pt), "Pz": f(Pz),
            "Wrr": f(sp.diff(Pr, lr)), "Wrt": f(sp.diff(Pr, lt))}
 
 
_LAWS = _build_laws()
 
 
def _integrate(s0, a, ri, re, mat, dense=False):
    """Integrate the equilibrium ODE from R = ri to re, with r(ri) = a, r'(ri) = s0.
 
    State y = (r, r', F_z), lam_r = r', lam_theta = r / R. Equilibrium in the reference
    configuration: dP_r/dR + (P_r - P_theta) / R = 0, with P_r = P_r(r', r/R).
    F_z accumulates the axial force int P_z 2 pi R dR.
    """
    L = _LAWS
 
    def rhs(R, y):
        r, s, _ = y
        lt = r / R
        Pr, Pt = L["Pr"](s, lt, *mat), L["Pt"](s, lt, *mat)
        rpp = ((Pt - Pr) / R - L["Wrt"](s, lt, *mat) * (s / R - r / R**2)) / L["Wrr"](s, lt, *mat)
        return [s, rpp, L["Pz"](s, lt, *mat) * 2.0 * np.pi * R]
 
    return solve_ivp(rhs, (ri, re), [a, s0, 0.0], rtol=1e-10, atol=1e-12, dense_output=dense)
 
 
def analytical_pressurised_cylinder(lam_a, ri, re, C10, C01, D, R_probe):
    """Solution as a function of the inner stretch lam_a = r(ri) / ri.
 
    For each lam_a, r'(ri) is found such that the outer face is traction free, P_r(re) = 0.
    Returns p (internal pressure, sigma_rr(ri) = -p), J_mean (volume average, equal to
    (b^2 - a^2) / (re^2 - ri^2)), F_z (axial force on the top face) and u_r at R_probe.
    """
    lam_a = np.atleast_1d(np.asarray(lam_a, float))
    R_probe = np.asarray(R_probe, float)
    mat = (C10, C01, D)
    out = {"lam_a": lam_a, "p": np.empty(len(lam_a)), "J_mean": np.empty(len(lam_a)),
           "F_z": np.empty(len(lam_a)), "u_r": np.empty((len(lam_a), len(R_probe)))}
 
    def outer_traction(s0, a):
        sol = _integrate(s0, a, ri, re, mat)
        return _LAWS["Pr"](sol.y[1, -1], sol.y[0, -1] / re, *mat)
 
    for i, la in enumerate(lam_a):
        a = ri * la
        guess = 1.0 / la                                   # incompressible estimate of r'(ri)
        s0 = root_scalar(outer_traction, args=(a,), x0=guess, x1=1.02 * guess,
                         method="secant", xtol=1e-12).root
        sol = _integrate(s0, a, ri, re, mat, dense=True)
        b = sol.y[0, -1]
        out["p"][i] = -_LAWS["Pr"](s0, la, *mat) / la      # P_r(ri) = lam_theta sigma_rr = -p la
        out["J_mean"][i] = (b**2 - a**2) / (re**2 - ri**2)
        out["F_z"][i] = sol.y[2, -1]
        out["u_r"][i] = sol.sol(R_probe)[0] - R_probe
    return out
 
 
# ----------------------------------------------------------------------------------------
# 4. Make and run several simulations
# ----------------------------------------------------------------------------------------
def make_cases(C10=1.0, ratios=(0.0, -0.175, 0.5), Ds=(4.0, 0.4, 0.04)):
    """Material parameter sets: C01 = ratio * C10, for each D."""
    return [{"C_10": C10, "C_01": r * C10, "D": D} for r, D in product(ratios, Ds)]
 
 
def run_cases(cases, run_simulation, outputs, rect, folder):
    """Run one simulation per case in a fixed folder.
 
    Layout:  <folder>/rectangle.msh, <folder>/case_<k>.json   (inputs)
             <folder>/simulations/case_<k>*                   (simulation results)
    run_simulation(name, mesh_file, mech_params_json, outputs) -> history
    outputs: function mech -> (fields, scalars). Returns a list of {"params", "history"}.
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
                json.dump({"sedf_type": "Mooney-Rivlin", **params}, f, indent=2)
        comm.barrier()
        history = run_simulation(str(sim_dir / f"case_{k}"), str(mesh_file), str(json_file),
                                 outputs)
        results.append({"params": params, "history": history})
    return results
 
 
# ----------------------------------------------------------------------------------------
# 5. Comparison and plots
# ----------------------------------------------------------------------------------------
def compare(results, rect):
    """Add to each result the simulated and analytical curves and the errors.
 
    The analytical solution is evaluated at the inner stretch lam_a of the simulation,
    lam_a = 1 + u_r(ri) / ri. Errors are relative norms over the load steps (J: on J - 1,
    u_r: on the probes other than ri, which defines lam_a).
    """
    Rp = probe_radii(rect)
    for r in results:
        h, m = r["history"], r["params"]
        lam_a = 1.0 + h["ur_0"] / rect.ri
        an = analytical_pressurised_cylinder(lam_a, rect.ri, rect.re,
                                             m["C_10"], m["C_01"], m["D"], Rp)
        sim = {"lam_a": lam_a, "p": h["t"], "J_mean": h["J_mean"], "F_z": h["top_reaction_z"],
               "u_r": np.stack([h[f"ur_{k}"] for k in range(len(Rp))], axis=1)}
        rel = lambda s, a: np.linalg.norm(s - a) / np.linalg.norm(a)
        r["sim"], r["an"], r["R_probe"] = sim, an, Rp
        r["errors"] = {"p": rel(sim["p"], an["p"]),
                       "J_mean": rel(sim["J_mean"] - 1.0, an["J_mean"] - 1.0),
                       "F_z": rel(sim["F_z"], an["F_z"]),
                       "u_r": rel(sim["u_r"][:, 1:], an["u_r"][:, 1:])}
 
 
def report(results):
    keys = list(results[0]["errors"])
    print(f"{'C_10':>6} {'C_01':>8} {'D':>7} | " + " ".join(f"{k:>10}" for k in keys))
    for r in results:
        p = r["params"]
        print(f"{p['C_10']:6.3g} {p['C_01']:8.3g} {p['D']:7.3g} | "
              + " ".join(f"{r['errors'][k]:10.2e}" for k in keys))
 
 
def plot(results, filename):
    import matplotlib.pyplot as plt
    panels = [("p", "internal pressure p"), ("J_mean", "J_mean"), ("F_z", "axial force F_z")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    for ax, (key, label) in zip(axes.flat, panels):
        for c, r in zip(colors, results):
            m = r["params"]
            ax.plot(r["an"]["lam_a"], r["an"][key], "-", color=c)
            ax.plot(r["sim"]["lam_a"], r["sim"][key], "o", color=c, ms=3,
                    label=f"C01={m['C_01']:.3g}, D={m['D']:.3g}")
        ax.set_xlabel("inner stretch lam_a")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    ax = axes[1, 1]                                    # radial profile at the last load step
    for c, r in zip(colors, results):
        ax.plot(r["R_probe"], r["an"]["u_r"][-1], "-", color=c)
        ax.plot(r["R_probe"], r["sim"]["u_r"][-1], "o", color=c, ms=4)
    ax.set_xlabel("R")
    ax.set_ylabel("u_r, last step")
    ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6, title="line: analytical, markers: simulation")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
 
 
if __name__ == "__main__":
    from Meshing_tools.make_axisym_rectangle import RectangleParams, make_rectangle
 
    folder = "results/Testing_tools/Check_against_analytical/Check_hyper_axi_MR_pressure"
    rect = RectangleParams(ri=0.5, re=1.0, h=1.0, n_r=20, n_z=2)
    outputs = partial(hollow_cylinder_outputs, rect=rect)
    results = run_cases(make_cases(), run_simulation, outputs, rect, folder)
    if MPI.COMM_WORLD.rank == 0:
        compare(results, rect)
        report(results)
        plot(results, Path(folder) / "cylinder_pressure_verification.png")