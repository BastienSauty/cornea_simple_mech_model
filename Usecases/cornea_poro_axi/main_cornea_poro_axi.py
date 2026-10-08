# author : B. Sauty ; 7 Oct 2026
# Main file to run a simulation for the cornea under pressure.
# Using the axisymmetric poroelastic framework

# Reproduce the conditions of Giammarini 2026

import numpy as np

from mpi4py import MPI
from dolfinx import fem
from petsc4py.PETSc import ScalarType
import ufl

from Simulation_Framework import Poroelastic_axisymmetric_framework, OutputManager, point_probe

MMHG = 0.000133322      # 1 mmHg in MPa

# physical tags written in cornea_3D_hex.msh by Meshing_tools toolbox
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
              ["displacement", "cauchy_stress", "von_mises", "J", "green_lagrange", "PK1_stress", "fluid_pressure"]}

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

    # Build the poroelastic framework. The mesh file is read once, inside: mesh,
    # facet tags and local reference system -> mech.domain, ...
    mech = Poroelastic_axisymmetric_framework(mesh_file, mech_params_json)

    dt = fem.Constant(mech.domain, ScalarType(0))
    # Set the weak form    
    mech.build_weak_form(dt)

    # Boundary conditions
    # intraocular pressure, ramped during the simulation
    # (15 mmHg = 2.0e-3 MPa if lengths are in mm and stresses in MPa)
    
    p_max = 18 * MMHG
    p_iop = fem.Constant(mech.domain, ScalarType(p_max))
    n_steps = 100
    t_max = 1000
    t_list = np.linspace(0, t_max, n_steps+1)

    zeta = fem.Constant(mech.domain, ScalarType(1.5e-5))
    r0 = fem.Constant(mech.domain, ScalarType(2))
    w = fem.Constant(mech.domain, ScalarType(0.1))
    chi = 0.5 * (1 - ufl.tanh((mech.r - r0) / w))   # w controls the transition width
    zeta_expr = zeta * chi

    boundary_conditions = [
        ["Dirichlet_disp", CENTRAL_LINE, ("clamped", 0)],   # rolling on the axis: u_r = 0, u_z free
        ["Dirichlet_disp", LIMBUS,       ("clamped", 0)],   # Limbus clamped
        ["Dirichlet_disp", LIMBUS,       ("clamped", 1)],
        # ["Neumann_disp",   POSTERIOR,    -p_iop*ufl.FacetNormal(mech.domain)],            # pressure normal to the posterior face
        ["Neumann_follower_disp",   POSTERIOR,    p_iop],            # pressure normal to the posterior face

        # ["Dirichlet_pressure", POSTERIOR, "drained"],  # drained condition, equivalent to 0.0
        ["Dirichlet_pressure", LIMBUS, "drained"],
        ["Neumann_follower_pressure",   POSTERIOR,    zeta_expr]
    ]
    mech.build_BCs(boundary_conditions)
    mech.build_solver()

    # Outputs: results/<name>.xdmf and results/<name>_scalars.csv, "time" = pressure
    fields, scalars = outputs(mech)                    # after build_BCs: surface integrals need mech.ds
    with OutputManager(f"results/{name}", mech.domain, fields, scalars,
                       meshtags={"facet_tag": mech.facet_tag},
                       meta=mech.output_meta) as out:
        out.write(0.0)                                 # reference state
        for n, t in enumerate(t_list[1:], start=1):
            dt.value = t - t_list[n-1]
            mech.solve_one_step(n, t)
            out.write(t)

    return out.get_history()


if __name__ == '__main__':
    folder_name = "Usecases/cornea_poro_axi"
    name = f"{folder_name}/cornea_IOP_Poro_pathological"
    mesh_file = f"{folder_name}/cornea.msh"
    mech_params_json = f"{folder_name}/mech_params_Giammarini2026.json"

    history = run_simulation(name, mesh_file, mech_params_json, cornea_outputs)

    if MPI.COMM_WORLD.rank == 0:
        import matplotlib.pyplot as plt
        plt.plot(history["apex_uz_anterior"][1:] * 1e3, history["t"][1:])
        plt.ylabel("Time [s]")
        plt.xlabel("anterior apex displacement [µm]")
        plt.savefig(f"results/{name}_apex.png", dpi=150)