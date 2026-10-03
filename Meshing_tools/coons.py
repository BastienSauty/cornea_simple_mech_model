"""2D Coons patch: map from the unit square onto a region bounded by four curves."""
import numpy as np


def segment(a, b):
    """Straight curve t -> a + t (b - a), t in [0, 1]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return lambda t: a + np.asarray(t, float)[..., None] * (b - a)


def coons_map(bottom, top, left, right):
    """phi: (..., 2) -> (..., 2). Curves map [0,1] to points (..., 2):
    bottom(u) at v=0, top(u) at v=1, left(v) at u=0, right(v) at u=1.
    They must agree at the four corners."""
    P00, P10, P01, P11 = bottom(0.0), bottom(1.0), top(0.0), top(1.0)

    def phi(X):
        X = np.asarray(X, float)
        u, v = X[..., 0], X[..., 1]
        U, V = u[..., None], v[..., None]
        return ((1 - V) * bottom(u) + V * top(u) + (1 - U) * left(v) + U * right(v)
                - ((1 - U) * (1 - V) * P00 + U * (1 - V) * P10
                   + (1 - U) * V * P01 + U * V * P11))
    return phi