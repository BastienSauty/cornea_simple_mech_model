# Shared nonlinear solver set-up of the hyperelastic and poroelastic frameworks.
#
# One place for the PETSc/SNES options, the creation of the NonlinearProblem and the
# one-line report of a solved step, so that a tolerance is changed once for all the frameworks
# (or for one simulation, with overrides) and the convergence check is identical everywhere.
#
# Usage in a framework class:
#     from ..solver import make_problem, solve_and_report
#
#     def build_solver(self, **petsc_overrides):
#         self.problem = make_problem(residual, unknown, self.bcs, "hyperelasticity",
#                                     **petsc_overrides)
#
#     def solve_one_step(self, n, p):
#         solve_and_report(self.problem, n, p, self.domain.comm)

from dolfinx.fem.petsc import NonlinearProblem

from .postprocessing_tools.log_utils import log

# Newton without line search, direct solver. The absolute tolerance is small on purpose: with
# stresses in MPa and small loads, the residuals can be of order 1e-8 or below, and a larger
# snes_atol makes SNES accept the initial guess (0 iterations) and freeze the solution.
DEFAULT_PETSC_OPTIONS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_monitor": None,
    "snes_atol": 1e-14,
    "snes_rtol": 1e-10,
    "snes_stol": 1e-8,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def petsc_options(**overrides):
    """Default Newton + LU options, with optional overrides, e.g. petsc_options(snes_atol=1e-12).
    An override of an option that is not in the defaults is added to them."""
    options = dict(DEFAULT_PETSC_OPTIONS)
    options.update(overrides)
    return options


def make_problem(residual, unknown, bcs, prefix, **overrides):
    """NonlinearProblem for `residual` with the shared options.

    residual : UFL form, or list of blocks (ufl.extract_blocks) for a mixed problem
    unknown  : fem.Function, or [u, p] for a mixed problem
    prefix   : PETSc options prefix, e.g. "hyperelasticity", "poroelasticity"
    """
    return NonlinearProblem(residual, unknown, bcs=bcs,
                            petsc_options=petsc_options(**overrides),
                            petsc_options_prefix=prefix)


def solve_and_report(problem, n, value, comm):
    """Solve one step, log one summary line, raise if SNES did not converge.

    n     : step number
    value : load (or time) of the step, for the printout
    Returns {"reason", "iterations", "residual"}.
    """
    problem.solve()
    snes = problem.solver
    reason = snes.getConvergedReason()
    iterations = snes.getIterationNumber()
    residual = snes.getFunctionNorm()

    log(f"Step {n:3d} | load {value:.3e} | Newton iterations {iterations:2d} | "
        f"residual {residual:.3e} | reason {reason}", comm=comm)

    if reason <= 0:
        print(reason, iterations)
        print(snes.getConvergenceHistory())
        raise RuntimeError(f"no convergence at step {n}, value = {value} (SNES reason {reason})")
    return {"reason": reason, "iterations": iterations, "residual": residual}
