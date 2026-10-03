# Plotting script for the cornea IOP run. Runs without dolfinx:
#   python plot_cornea.py
import matplotlib.pyplot as plt
from Simulation_Framework import PlotManager, compare_scalars

basename = "Usecases/cornea_axi/cornea_IOP_Giammarini"
out = f"results/{basename}"

# labels of the case-specific scalars (standard quantities are labelled by the library)
labels = {
    "apex_uz_anterior": r"anterior apex $u_z$",
    "apex_uz_posterior": r"posterior apex $u_z$",
    "central_thickness": "central thickness",
    "limbus_reaction_z": r"limbus reaction $R_z$",
}

pm = PlotManager(out, labels=labels)
pm.info()

# ---- composed figures: pass `ax`, save the figure yourself ----
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
pm.plot_scalars("apex_uz_anterior", ax=axes[0])
pm.plot_scalars("central_thickness", ax=axes[1])
pm.plot_scalars("limbus_reaction_z", x="apex_uz_anterior", ax=axes[2])   # force-displacement
fig.tight_layout()
fig.savefig(f"{out}_scalars.png", dpi=200)

fig, axes = plt.subplots(2, 1, figsize=(10, 7))
pm.plot_field("von_mises", deformed=True, mirror=True, ax=axes[0])
pm.plot_field("cauchy_stress", component="tt", deformed=True, mirror=True,
              symmetric=True, ax=axes[1])                                 # hoop stress
fig.tight_layout()
fig.savefig(f"{out}_fields.png", dpi=200)

# ---- stand-alone figures: save= creates, saves and closes the figure ----
pm.plot_field("displacement", component="z", deformed=True, save=f"{out}_disp.png")
pm.plot_field("cauchy_stress", component="rr", symmetric=True, save=f"{out}_cauchyrr.png")
pm.plot_field("cauchy_stress", component="tt", symmetric=True, save=f"{out}_cauchytt.png")
pm.plot_field("PK1_stress", component="zz", symmetric=True, save=f"{out}_PK1zz.png")
pm.plot_field("green_lagrange", component="zz", symmetric=True, save=f"{out}_Ezz.png")

# ---- profile along the central axis (posterior -> anterior apex) ----
pm.plot_profile("green_lagrange", component="zz", save=f"{out}_Ezz_axis_last.png")
pm.plot_profile("green_lagrange", component="zz", t="all", save=f"{out}_Ezz_axis_all.png")

# ---- animation over the load steps ----
pm.animate_field("von_mises", f"{out}_von_mises.gif", deformed=True, mirror=True, fps=20)

# ---- parameter study: overlay several runs ----
# runs = [PlotManager(f"results/cornea_mu{mu}", label=f"mu = {mu}", labels=labels)
#         for mu in (0.05, 0.1, 0.2)]
# compare_scalars(runs, "apex_uz_anterior")
# plt.show()