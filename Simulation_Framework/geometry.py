# Geometry of the frameworks: the few operators that differ between an axisymmetric (r, z)
# section and a full 3D body. The physics (hyperelastic, poroelastic) is written once in its
# framework class and calls these operators; the class decides which geometry it uses.
#
#                       Axisymmetric                       Cartesian3D
#   grad_vector(u, x)   3x3 gradient, hoop u_r / r         ufl.grad(u)
#   grad_scalar(p, x)   (dp/dr, dp/dz, 0)                  ufl.grad(p)
#   normal_product      (r, z) block of A . N              A . N
#   weight(x)           r  (multiplies dx and ds)          1
#   integral_factor     2 pi (full revolved body)          1
#   axes, axisymmetric  (r, z, t), True                    (x, y, z), False
#
# x is the ufl.SpatialCoordinate of the domain. Components of vectors are (r, z) in
# axisymmetric (2D mesh, z the axis of symmetry) and (x, y, z) in 3D.

import ufl


def _grad_vector_axi(u, r):
    """
    Function to compute the axisymmetric gradient in the coordinate system (r, z, theta)
    with z the axis of symmetry and r = x[0] the radial coordinate
    """
    grad2D = ufl.grad(u)
    grad_axi = ufl.as_tensor([[grad2D[0,0], grad2D[0,1], 0],
                                [grad2D[1,0], grad2D[1,1], 0],
                                [0, 0, u[0]/r]])
    return(grad_axi)


def _grad_scalar_axi(u, r):
    """
    Function to compute the axisymmetric gradient of a scalar in the coordinate system (r, z, theta)
    with z the axis of symmetry and r = x[0] the radial coordinate
    u : scalar
    grad_axi : 3*1 vector
    """
    grad2D = ufl.grad(u)
    grad_axi = ufl.as_vector([grad2D[0], grad2D[1], 0])
    return(grad_axi)


def _inplane(A, w):
    """
    axisymmetric product of A * w for the Neumann follower pull back operation.
    (r, z) components of A . (w_r, w_z, 0). Works for a 2x2 tensor or a 3x3
    axisymmetric one (block-diagonal: the hoop component does not mix in)."""
    return ufl.as_vector([A[0, 0] * w[0] + A[0, 1] * w[1],
                          A[1, 0] * w[0] + A[1, 1] * w[1]])


def push_local_frame(e_1, e_2, e_3, A):
    """
    A = (a_1, e_2, a_3): a direction given by its components in the
    mesh's local reference system (e_1, e_2, e_3) -> the same
    direction in physical (r, z, theta)/(x,y,z) space, as a unit UFL vector.

    (e_1, e_2, e_3) is an orthonormal triad by construction directly within
    the mesh. In the case of axisymmetry, the e_3 is built as the out of plane direction
    So this is a plain change of basis 
    A does not need to be a unit vector itself only the final result is 
    normalised.
    """
    a = A[0] * e_1 + A[1] * e_2 + A[2] * e_3
    return a / ufl.sqrt(ufl.dot(a, a))

class Axisymmetric:
    """(r, z) section of a body of revolution, components (r, z, theta)."""
    axisymmetric = True
    axes = ["r", "z", "t"]
    integral_factor = 2 * ufl.pi

    def weight(self, x):
        return x[0]

    def grad_vector(self, u, x):
        return _grad_vector_axi(u, x[0])

    def grad_scalar(self, p, x):
        return _grad_scalar_axi(p, x[0])

    def normal_product(self, A, N):
        return _inplane(A, N)


class Cartesian3D:
    """Full 3D body, components (x, y, z)."""
    axisymmetric = False
    axes = ["x", "y", "z"]
    integral_factor = 1

    def weight(self, x):
        return ufl.as_ufl(1.0)

    def grad_vector(self, u, x):
        return ufl.grad(u)

    def grad_scalar(self, p, x):
        return ufl.grad(p)

    def normal_product(self, A, N):
        return ufl.dot(A, N)
