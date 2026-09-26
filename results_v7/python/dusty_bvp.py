"""
Python port of Dhia_Problem3_Inclined_New_Formula.m
Dusty hybrid nanofluid (Al2O3-Cu/water) boundary layer over an inclined
stretching/shrinking sheet with velocity & thermal slip.

State vector (same ordering as the MATLAB code):
 y1=f, y2=f', y3=f'', y4=F, y5=F', y6=t, y7=t', y8=tp
"""
import numpy as np
from scipy.integrate import solve_bvp

RHO_F, K_F, C_F = 997.1, 0.6130, 4179.0   # water
RHO_1, K_1, C_1 = 3970.0, 40.0, 765.0     # Al2O3
RHO_2, K_2, C_2 = 8933.0, 400.0, 385.0    # Cu
B_1, B_2, B_F = 0.85, 1.67, 21.0


def coeffs(phi1, phi2):
    A1 = 1.0 / (((1 - phi1) ** 2.5) * ((1 - phi2) ** 2.5))
    A2 = (1 - phi2) * ((1 - phi1) + phi1 * (RHO_1 / RHO_F)) + phi2 * (RHO_2 / RHO_F)
    A3 = (1 - phi2) * ((1 - phi1) + phi1 * ((RHO_1 * C_1) / (RHO_F * C_F))) \
        + phi2 * ((RHO_2 * C_2) / (RHO_F * C_F))
    A4 = (K_1 + 2 * K_F - 2 * phi1 * (K_F - K_1)) / (K_1 + 2 * K_F + phi1 * (K_F - K_1))
    A5 = ((K_2 + 2 * A4 * K_F - 2 * phi2 * (A4 * K_F - K_2))
          / (K_2 + 2 * A4 * K_F + phi2 * (A4 * K_F - K_2))) * A4
    A6 = ((1 - phi2) * ((1 - phi1) * (RHO_F * B_F) + phi1 * (RHO_1 * B_1) + phi2 * (RHO_2 * B_2))) \
        / ((1 - phi2) * ((1 - phi1) * RHO_F + phi1 * RHO_1 + phi2 * RHO_2))
    A7 = B_F
    A8 = A6 / A7
    return A1, A2, A3, A5, A8


DEFAULTS = dict(phi1=0.01, phi2=0.01, Pr=6.2, lam=-1.0, L=1.0, Bv=0.5, BT=0.5,
                m=1.0, Ec=2.0, eps=1.0, s=2.0, sigma1=0.2, sigma2=0.2,
                Gr=0.001, X=0.0)


def make_funcs(p):
    A1, A2, A3, A5, A8 = coeffs(p['phi1'], p['phi2'])
    Pr, lam, L, Bv, BT = p['Pr'], p['lam'], p['L'], p['Bv'], p['BT']
    m, Ec, eps, s = p['m'], p['Ec'], p['eps'], p['s']
    s1, s2, Gr, X = p['sigma1'], p['sigma2'], p['Gr'], p['X']

    def ode(x, y):
        y1, y2, y3, y4, y5, y6, y7, y8 = y
        return np.vstack([
            y2,
            y3,
            (A2 / A1) * (-y1 * y3 + y2 ** 2 - ((L * Bv) / A2) * (y5 - y2)
                         - (Gr * A8 * y6) * np.cos(X)),
            y5,
            (1.0 / y4) * (y5 ** 2 - Bv * (y2 - y5)),
            y7,
            ((Pr * A3) / A5) * (-y1 * y7 + 2 * y2 * y6 - ((L * BT) / (m * A3)) * (y8 - y6)
                                - ((L * Bv * Ec) / (m * A3)) * ((y5 - y2) ** 2)
                                - (A1 / A3) * (Ec * y3 ** 2)),
            (1.0 / y4) * (2 * y5 * y8 - eps * BT * (y6 - y8)),
        ])

    def bc(ya, yb):
        return np.array([
            ya[0] - s,
            ya[1] - lam - s1 * ya[2],
            ya[5] - 1 - s2 * ya[6],
            yb[1],
            yb[4],
            yb[3] - yb[0],
            yb[5],
            yb[7],
        ])

    return ode, bc


def make_funcs_pinned(p, fpp_target, pin_idx=2):
    """lambda is an UNKNOWN parameter; extra BC pins ya[pin_idx]=fpp_target.
    Used to walk around the fold (pseudo-arclength substitute)."""
    A1, A2, A3, A5, A8 = coeffs(p['phi1'], p['phi2'])
    Pr, L, Bv, BT = p['Pr'], p['L'], p['Bv'], p['BT']
    m, Ec, eps, s = p['m'], p['Ec'], p['eps'], p['s']
    s1, s2, Gr, X = p['sigma1'], p['sigma2'], p['Gr'], p['X']

    def ode(x, y, pp):
        y1, y2, y3, y4, y5, y6, y7, y8 = y
        return np.vstack([
            y2,
            y3,
            (A2 / A1) * (-y1 * y3 + y2 ** 2 - ((L * Bv) / A2) * (y5 - y2)
                         - (Gr * A8 * y6) * np.cos(X)),
            y5,
            (1.0 / y4) * (y5 ** 2 - Bv * (y2 - y5)),
            y7,
            ((Pr * A3) / A5) * (-y1 * y7 + 2 * y2 * y6 - ((L * BT) / (m * A3)) * (y8 - y6)
                                - ((L * Bv * Ec) / (m * A3)) * ((y5 - y2) ** 2)
                                - (A1 / A3) * (Ec * y3 ** 2)),
            (1.0 / y4) * (2 * y5 * y8 - eps * BT * (y6 - y8)),
        ])

    def bc(ya, yb, pp):
        lam = pp[0]
        return np.array([
            ya[0] - s,
            ya[1] - lam - s1 * ya[2],
            ya[5] - 1 - s2 * ya[6],
            ya[pin_idx] - fpp_target,
            yb[1],
            yb[4],
            yb[3] - yb[0],
            yb[5],
            yb[7],
        ])

    return ode, bc


def solve_pinned(p, eta_max, guess, fpp_target, lam_guess, tol=1e-8,
                 max_nodes=100000, pin_idx=2):
    ode, bc = make_funcs_pinned(p, fpp_target, pin_idx)
    x, Y = guess
    if not np.isclose(x[-1], eta_max) or len(x) > 1500:
        xn = np.linspace(0, eta_max, min(len(x), 801))
        Y = np.vstack([np.interp(xn, x, Y[i]) for i in range(8)])
        x = xn
    return solve_bvp(ode, bc, x, Y, p=np.array([lam_guess]), tol=tol,
                     max_nodes=max_nodes, verbose=0)


def guess1(x):
    """OdeInit1 variant #6 from the MATLAB file."""
    e = np.exp(-x)
    return np.vstack([1.112 + e * e, e, e, 0.15 + e, e, e, e, 0.15 + e])


def guess2(x):
    """OdeInit2 variant #7 from the MATLAB file."""
    e = np.exp(-x)
    return np.vstack([3.912 / e, e, e, e, e, e, e, e])


def solve(p, eta_max, guess, nodes=None, tol=1e-8, max_nodes=200000):
    """guess: callable(x)->(8,n) array or tuple (x, Y) for continuation."""
    ode, bc = make_funcs(p)
    if callable(guess):
        x = np.linspace(0, eta_max, nodes or max(int(eta_max) * 15, 61))
        Y = guess(x)
    else:
        x, Y = guess
        if not np.isclose(x[-1], eta_max) or len(x) > 1500:
            xn = np.linspace(0, eta_max, min(len(x), 801))
            Y = np.vstack([np.interp(xn, x, Y[i]) for i in range(8)])
            x = xn
    sol = solve_bvp(ode, bc, x, Y, tol=tol, max_nodes=max_nodes, verbose=0)
    return sol


def outputs(sol):
    """f''(0), F'(0), -t'(0), tp(0)"""
    y0 = sol.y[:, 0]
    return y0[2], y0[4], -y0[6], y0[7]


def is_physical(sol, eta_max):
    """Converged and decayed: f', F', t, tp -> 0 at eta_max."""
    if sol.status != 0:
        return False
    yb = sol.y[:, -1]
    return (abs(yb[1]) < 1e-5 and abs(yb[5]) < 1e-5
            and np.all(np.isfinite(sol.y)) and np.max(np.abs(sol.y[1])) < 50)
