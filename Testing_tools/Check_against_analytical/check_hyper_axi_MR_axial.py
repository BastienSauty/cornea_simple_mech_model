# Verification of the axisymmetric framework for the different kinds of hyperelastic laws
# 
# Test on a cylinder under axial load, compare with the analytical response.

import numpy as np

from mpi4py import MPI
from dolfinx import fem
from petsc4py.PETSc import ScalarType
import ufl
from itertools import product
import shutil
import tempfile
import json

from scipy.optimize import brentq
from pathlib import Path

from Simulation_Framework import Hyperelastic_axisymmetric_framework, OutputManager, point_probe

# mesh lines
CENTRAL_LINE, LATERAL, BOTTOM, TOP = 4, 3, 2, 1


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
def run_simulation(name, mesh_file, mech_params_json, outputs):
    """outputs : function mech -> (fields, scalars), see cornea_outputs."""

    # Build the hyperelastic framework. The mesh file is read once, inside: mesh,
    # facet tags and local reference system -> mech.domain, ...
    mech = Hyperelastic_axisymmetric_framework(mesh_file, mech_params_json)

    # Set the fiber orientation -> preprocess
    if mech.mech_params.sedf_type=='HGO':
        mech._fibre_orientation_field()
    # Set the weak form    
    mech.build_weak_form()

    # Boundary conditions
    # intraocular pressure, ramped during the simulation
    # (15 mmHg = 2.0e-3 MPa if lengths are in mm and stresses in MPa)
    p_iop = fem.Constant(mech.domain, ScalarType(0.0))
    p_max = 1
    n_steps = 20

    boundary_conditions = [
        ["Dirichlet", CENTRAL_LINE, ("clamped", 0)],   # rolling on the axis: u_r = 0, u_z free
        ["Dirichlet", BOTTOM,       ("clamped", 1)],
        ["Neumann_follower",  TOP,    p_iop],            # follower pressure normal to the posterior face
    ]                                                  # anterior face: free (zero traction)
    mech.build_BCs(boundary_conditions)
    mech.build_solver()

    # Outputs: results/<name>.xdmf and results/<name>_scalars.csv, "time" = pressure
    fields, scalars = outputs(mech)                    # after build_BCs: surface integrals need mech.ds
    with OutputManager(f"{name}", mech.domain, fields, scalars,
                       meshtags={"facet_tag": mech.facet_tag},
                       meta=mech.output_meta) as out:
        out.write(0.0)                                 # reference state
        for n, p in enumerate(np.linspace(0, p_max, n_steps)[1:], start=1):
            p_iop.value = p
            mech.solve_one_step(n, p)
            out.write(p)

    return out.get_history()


# ----------------------------------------------------------------------------------------
# 3. Analytical solution
# ----------------------------------------------------------------------------------------
def principal_cauchy(lam_r, lam_z, C10, C01, D):
    """Principal Cauchy stresses (rr, zz, theta-theta) for F = diag(lam_r, lam_z, lam_r)."""
    lam = np.array([lam_r, lam_z, lam_r])
    J = np.prod(lam)
    bb = lam**2 * J ** (-2.0 / 3.0)                  # eigenvalues of bbar = J^(-2/3) F F^T
    I1 = bb.sum()
    A = (C10 + C01 * I1) * bb - C01 * bb**2
    dev = A - A.mean()                               # deviatoric part (trace/3 removed)
    return 2.0 / J * dev + 2.0 / D * (J - 1.0)
 
 
def analytical_cylinder(lam_z, C10, C01, D, bracket=(0.05, 10.0)):
    """Homogeneous solution as a function of the axial stretch.
 
    Returns lam_r (= lam_theta), J, sigma_zz (Cauchy) and P_zz (first Piola-Kirchhoff).
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
def make_cases(C10=1.0, ratios=(0.0, -0.175, 0.5), Ds=(4.0, 0.4, 0.04)):
    """Material parameter sets: C01 = ratio * C10, for each D."""
    return [{"C_10": C10, "C_01": r * C10, "D": D} for r, D in product(ratios, Ds)]

def run_cases(cases, run_simulation, outputs, rect, folder):
    """Run one simulation per case in a fixed folder.
 
    Layout:  <folder>/rectangle.msh, <folder>/case_<k>.json   (inputs)
             <folder>/simulations/case_<k>*                   (simulation results)
    run_simulation(name, mesh_file, mech_params_json, outputs) -> history
    outputs: cylinder_outputs. Returns a list of {"params", "history"} (same on all ranks).
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
 
    Error = max |sim - analytical| / max |analytical| over the load steps.
    """
    area = np.pi * (rect.re**2 - rect.ri**2)
    for r in results:
        h, p = r["history"], r["params"]
        lam_z = np.sqrt(1.0 + 2.0 * h["axial_strain_mean"])
        an = analytical_cylinder(lam_z, p["C_10"], p["C_01"], p["D"])
        sim = {"lam_z": lam_z,
               "lam_r": np.sqrt(1.0 + 2.0 * h["radial_strain_mean"]),
               "lam_theta": np.sqrt(1.0 + 2.0 * h["hoop_strain_mean"]),
               "J": h["J_mean"],
               "sigma_zz": h["axial_stress_integral"] / h["volume"],
               "P_zz": h["top_reaction_z"] / area}
        ref = {**an, "lam_theta": an["lam_r"]}
        r["sim"], r["an"] = sim, an
        r["errors"] = {k: np.linalg.norm(sim[k] - ref[k]) / np.linalg.norm(ref[k])
                       for k in ["lam_r", "lam_theta", "J", "sigma_zz", "P_zz"]}
 
 
def report(results):
    keys = list(results[0]["errors"])
    print(f"{'C_10':>6} {'C_01':>8} {'D':>7} | " + " ".join(f"{k:>10}" for k in keys))
    for r in results:
        p = r["params"]
        print(f"{p['C_10']:6.3g} {p['C_01']:8.3g} {p['D']:7.3g} | "
              + " ".join(f"{r['errors'][k]:10.2e}" for k in keys))
 
 
def plot(results, filename):
    import matplotlib.pyplot as plt
    panels = [("P_zz", "P_zz (PK1)"), ("sigma_zz", "sigma_zz (Cauchy)"),
              ("J", "J"), ("lam_r", "lam_r")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    for ax, (key, label) in zip(axes.flat, panels):
        for c, r in zip(colors, results):
            p = r["params"]
            ax.plot(r["an"]["lam_z"], r["an"][key], "-", color=c)
            ax.plot(r["sim"]["lam_z"], r["sim"][key], "o", color=c, ms=3,
                    label=f"C01={p['C_01']:.3g}, D={p['D']:.3g}")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("lam_z")
    axes[0, 0].legend(fontsize=6, title="line: analytical, markers: simulation")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)


if __name__ == "__main__":
    from Meshing_tools.make_axisym_rectangle import RectangleParams, make_rectangle
    
    folder = "results/Testing_tools/Check_against_analytical/Check_hyper_axi_MR_axial"
    rect = RectangleParams(ri=0.0, re=1.0, h=1.0, n_r=4, n_z=4)
    results = run_cases(make_cases(), run_simulation, cylinder_outputs, rect, folder)
    if MPI.COMM_WORLD.rank == 0:
        compare(results, rect)
        report(results)
        plot(results, Path(folder) / "cylinder_axial_verification.png")