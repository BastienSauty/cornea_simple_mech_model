# Example plotting script for the cornea IOP run. Runs without dolfinx:
#   python plot_cornea.py
import matplotlib.pyplot as plt
from Simulation_Framework import PlotManager, compare_scalars

pm = PlotManager("results/cornea_IOP")
pm.info()

# ---- scalar histories ----
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
pm.plot_scalars(["apex_uz_anterior"], ax=axes[0]) #, "apex_uz_posterior"], ax=axes[0])
pm.plot_scalars("central_thickness", ax=axes[1])
pm.plot_scalars("limbus_reaction_z", x="apex_uz_anterior", ax=axes[2])   # force-displacement
fig.tight_layout()
fig.savefig("results/cornea_IOP_scalars.png", dpi=200)

# ---- fields at the last step, on the deformed full section ----
fig, axes = plt.subplots(2, 1, figsize=(10, 7))
pm.plot_field("von_mises", deformed=True, mirror=True, ax=axes[0])
pm.plot_field("cauchy_stress", component="tt", deformed=True, mirror=True,
              symmetric=True, ax=axes[1])                                # hoop stress
fig.tight_layout()
fig.savefig("results/cornea_IOP_fields.png", dpi=200)


fig, axes = plt.subplots(2, 1, figsize=(10, 7))
pm.plot_field("PK1_stress", component="zz", deformed=False, mirror=False,
              symmetric=True, ax=axes[0])
pm.plot_field("green_lagrange", component="zz", deformed=False, mirror=False,
              symmetric=True, ax=axes[1])
fig.tight_layout()
fig.savefig("results/cornea_IOP_zz.png", dpi=200)

# ---- profile along the central axis r = 0 (posterior -> anterior apex) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
pm.plot_profile("green_lagrange", component="zz", ax=axes[0])            # last step
pm.plot_profile("green_lagrange", component="zz", t="all", ax=axes[1])   # every load step
fig.tight_layout()
fig.savefig("results/cornea_IOP_Err_axis.png", dpi=200)

# ---- animation over the load steps ----
pm.animate_field("von_mises", "results/cornea_IOP_von_mises.gif",
                 deformed=True, mirror=True, fps=20)

# ---- parameter study: overlay several runs ----
# runs = [PlotManager(f"results/cornea_mu{mu}", label=f"mu = {mu}") for mu in (0.05, 0.1, 0.2)]
# compare_scalars(runs, "apex_uz_anterior")

plt.show()