"""Printing for MPI runs: one line per message, in order with the PETSc output."""
from mpi4py import MPI
from petsc4py import PETSc


def log(*args, comm=MPI.COMM_WORLD, **kwargs):
    """Print on rank 0 only. Goes through PETSc's stdout, so it stays in order with
    the SNES monitor lines."""
    PETSc.Sys.Print(*args, comm=comm, **kwargs)


def log_ranks(msg, comm=MPI.COMM_WORLD):
    """One line per rank, gathered and printed in rank order (for diagnostics)."""
    msgs = comm.gather(msg, root=0)
    if comm.rank == 0:
        for r, m in enumerate(msgs):
            PETSc.Sys.Print(f"  [rank {r}] {m}")