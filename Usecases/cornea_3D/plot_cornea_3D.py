# Plotting script for the cornea IOP run. Runs without dolfinx:
#   python plot_cornea.py
import matplotlib.pyplot as plt
from Simulation_Framework import PlotManager, compare_scalars

basename = "Usecases/cornea_3D/cornea_3D_IOP_fine"
out = f"results/{basename}"

# labels of the case-specific scalars (standard quantities are labelled by the library)
labels = {
    "apex_uz_anterior": r"anterior apex $u_z$",
    "apex_uz_posterior": r"posterior apex $u_z$",
    "central_thickness": "central thickness",
    "limbus_reaction_z": r"limbus reaction $R_z$",
    "posterior_reaction_z": r"posterior reaction $R_z$",
    "anterior_reaction_z": r"anterior reaction $R_z$",
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


fig, axes = plt.subplots(1, 3, figsize=(14, 4))
pm.plot_scalars("posterior_reaction_z", ax=axes[0])
pm.plot_scalars("anterior_reaction_z", ax=axes[1])
pm.plot_scalars("limbus_reaction_z", ax=axes[2])   # force-displacement
fig.tight_layout()
fig.savefig(f"{out}_reaction_scalars.png", dpi=200)
