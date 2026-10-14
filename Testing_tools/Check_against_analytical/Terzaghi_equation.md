---
title: "Terzaghi consolidation: equations for the poroelastic check"
date: "14 October 2026"
---

# Problem and assumptions

A saturated column of height $H$, loaded at once by a constant compressive stress $p_0$ on its top face, consolidates as fluid leaves through the top only. The axisymmetric cylinder ($r_i = 0$, height $H$) is laterally confined, so the deformation is uniaxial along $z$, with $z = 0$ at the bottom and $z = H$ at the top.

- The constituents are incompressible: Biot coefficient equal to 1, no storage term.
- The permeability $k$ is constant (Darcy law, $w = -k\,\partial p/\partial z$).
- The strain is small: the load is small compared with the constrained modulus $M$, so the response is linear.
- Bottom: $u_z = 0$ and no flow. Lateral surface: $u_r = 0$ and no flow. Top: drained ($p = 0$) and loaded by $p_0$.

# Governing equations

The pore pressure obeys a diffusion equation with coefficient $c_v = kM$. It follows from four relations, with the effective stress $\sigma'$ taken positive in tension.

Momentum balance (no body force) in one dimension. The total stress is uniform and equal to the applied one:

$$
\frac{\partial}{\partial z}\left(\sigma'_{zz} - p\right) = 0
\quad\Rightarrow\quad
\sigma'_{zz} - p = -p_0 .
$$

Skeleton law, with the constrained (oedometric) modulus $M$:

$$
\sigma'_{zz} = M\,\varepsilon_{zz}, \qquad \varepsilon_{zz} = \frac{\partial u_z}{\partial z} .
$$

Darcy law and mass balance for incompressible constituents (the volume change equals minus the divergence of the flux):

$$
w = -k\,\frac{\partial p}{\partial z}, \qquad
\frac{\partial \varepsilon_{zz}}{\partial t} = -\frac{\partial w}{\partial z}
= k\,\frac{\partial^2 p}{\partial z^2} .
$$

Since $M\,\varepsilon_{zz} = p - p_0$, one has $\partial\varepsilon_{zz}/\partial t = (1/M)\,\partial p/\partial t$, and the consolidation equation is

$$
\frac{\partial p}{\partial t} = c_v\,\frac{\partial^2 p}{\partial z^2},
\qquad c_v = k\,M .
$$

# Conditions and solution

The load is carried entirely by the fluid at $t = 0^+$, then transferred to the skeleton as the pressure dissipates through the drained top:

$$
p(z, 0^+) = p_0, \qquad p(H, t) = 0, \qquad \frac{\partial p}{\partial z}(0, t) = 0 .
$$

With the dimensionless time $T = c_v t / H^2$ and $M_m = (2m + 1)\pi/2$, the series solution for the pressure is

$$
\frac{p(z,t)}{p_0}
= \sum_{m=0}^{\infty} \frac{2(-1)^m}{M_m}\,
\cos\!\left(\frac{M_m z}{H}\right) e^{-M_m^2 T} .
$$

The bottom is fixed, so the top displacement is the integral of $\varepsilon_{zz} = (p - p_0)/M$ over the height. It gives the degree of consolidation $U(T)$, which goes from $0$ at $T = 0$ to $1$ for $T \to \infty$:

$$
u_z(H, t) = -\frac{p_0 H}{M}\,U(T),
\qquad
U(T) = 1 - \sum_{m=0}^{\infty} \frac{2}{M_m^2}\,e^{-M_m^2 T} .
$$

The final settlement is $-p_0 H / M$. The series converge fast for $T$ above about $0.01$ and slowly at very small $T$, where several hundred terms are needed (500 are used in the check).

# Parameters and compared quantities

The constrained modulus comes from the linearisation of the Mooney-Rivlin law about the reference state,

$$
W = C_{10}\,(\bar I_1 - 3) + C_{01}\,(\bar I_2 - 3) + \frac{(J - 1)^2}{D}
$$

(convention to be confirmed against the framework), which gives

$$
\mu = 2\,(C_{10} + C_{01}), \qquad K = \frac{2}{D}, \qquad
M = K + \frac{4}{3}\,\mu .
$$

The load must keep the strain $p_0/M$ small (below about 1 percent), so that the nonlinear solution stays close to the linear one. The time scale is $t_c = H^2/c_v$, with $k$ in the units of the Darcy law of the framework (length to the fourth power per force and time).

| Quantity | Simulation | Analytical |
|:--------------|:-------------------------------------------|:-------------------------------------------|
| $p(0,t)/p_0$ | fluid pressure averaged over the bottom face | series for $z = 0$ |
| $U(T)$ | $-u_z(H,t)\,M/(p_0 H)$, with $u_z$ averaged over the top face | $1 - \sum_m \frac{2}{M_m^2} e^{-M_m^2 T}$ |

Both quantities are compared as functions of $T$, so the curves do not depend on $k$, $M$ or $H$.
