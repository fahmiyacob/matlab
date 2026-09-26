"""
PROMPT 5 core computation.  Trace BOTH branches and carry the SECOND (lower)
solution all the way up to lambda = 0, for alpha = 0, pi/4, pi/2.

Strategy:
  1. First (upper) branch: natural lambda-continuation from lambda=1 down to the
     fold, storing full profiles.
  2. Fold crossing: pin f''(0) and step a little past the fold to land on the
     second branch (lambda as unknown parameter).
  3. Second (lower) branch: natural lambda-continuation with lambda INCREASING
     from the fold up to 0 (robust away from the fold).

PROMPT 4 parameters (sigma2 = 0), Gr = 0.1 primary.
"""
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import dusty_bvp as db

P4 = dict(db.DEFAULTS)
P4.update(phi1=0.1, phi2=0.1, Bv=0.5, BT=0.5, s=1.0, Ec=1.0,
          sigma1=0.2, sigma2=0.0, Gr=0.1, m=1.0, eps=1.0, L=1.0, Pr=6.2)
EM = 8.0


def params(X, lam):
    p = dict(P4); p['X'] = X; p['lam'] = lam
    return p


def wall(sol):
    y0 = sol.y[:, 0]
    return np.array([sol.y[2, 0], sol.y[4, 0], -sol.y[6, 0], sol.y[7, 0]])
    # f''(0), F'(0), -t'(0), tp(0)


def ok_sol(sol):
    return (sol.status == 0 and np.all(np.isfinite(sol.y))
            and abs(sol.y[1, -1]) < 1e-4 and abs(sol.y[4, -1]) < 1e-4)


TOL = 1e-7
MAXN = 15000


def trace_first(X, dlam=0.01):
    lam = 1.0
    sol = db.solve(params(X, lam), EM, db.guess1, tol=TOL, max_nodes=MAXN)
    assert ok_sol(sol), "first seed failed"
    lams, W, sols = [lam], [wall(sol)], [sol]
    seed = (sol.x, sol.y); step = dlam; fails = 0
    while step > 1e-4:
        lam2 = lam - step
        sol = db.solve(params(X, lam2), EM, seed, tol=TOL, max_nodes=MAXN)
        if ok_sol(sol):
            lam = lam2
            lams.append(lam); W.append(wall(sol)); sols.append(sol)
            seed = (sol.x, sol.y); step = min(dlam, step * 1.3); fails = 0
        else:
            step *= 0.5; fails += 1
            if fails > 12:
                break
    return np.array(lams), np.array(W), sols


def cross_fold(X, seed_sol, c_fold, lam_fold, dc=0.003, ntry=60):
    """Pin f''(0), step below c_fold, lambda unknown -> land on branch 2."""
    p = params(X, lam_fold)
    guess = (seed_sol.x, seed_sol.y); lam_g = lam_fold
    c = c_fold - dc; step = dc; last = None
    for _ in range(ntry):
        sol = db.solve_pinned(p, EM, guess, fpp_target=c, lam_guess=lam_g,
                              tol=TOL, pin_idx=2, max_nodes=MAXN)
        if ok_sol(sol) and (last is None or float(sol.p[0]) >= last[0] - 1e-6):
            lam_g = float(sol.p[0])
            guess = (sol.x, sol.y); last = (lam_g, sol)
            # once clearly onto branch 2 (lambda risen a bit above fold), stop
            if lam_g > lam_fold + 0.03:
                return sol, lam_g
            c -= step; step = min(dc, step * 1.2)
        else:
            step *= 0.5
            if step < 2e-4:
                break
            c = (last_c := c) + 0  # keep
            c -= step
    if last is not None:
        return last[1], last[0]
    return None, None


def trace_second(X, seed_sol, lam_seed, lam_target=0.0, dlam=0.01):
    """Natural continuation of branch 2 with lambda increasing to lam_target."""
    lam = lam_seed
    sol = seed_sol
    lams, W, sols = [lam], [wall(sol)], [sol]
    seed = (sol.x, sol.y); step = dlam
    while lam < lam_target + 1e-9:
        lam2 = min(lam + step, lam_target)
        sol = db.solve(params(X, lam2), 12.0, seed, tol=TOL, max_nodes=MAXN)
        if ok_sol(sol):
            lam = lam2
            lams.append(lam); W.append(wall(sol)); sols.append(sol)
            seed = (sol.x, sol.y); step = min(dlam, step * 1.3)
            if abs(lam - lam_target) < 1e-9:
                break
        else:
            step *= 0.5
            if step < 5e-5:
                break
    return np.array(lams), np.array(W), sols


def full_curve(X, verbose=True):
    lam1, W1, S1 = trace_first(X)
    i_fold = int(np.argmin(lam1))
    lam_fold = lam1[i_fold]; c_fold = W1[i_fold, 0]
    seed = S1[i_fold]
    s2seed, lam2seed = cross_fold(X, seed, c_fold, lam_fold)
    if s2seed is None:
        lam2, W2, S2 = np.array([]), np.zeros((0, 4)), []
    else:
        lam2, W2, S2 = trace_second(X, s2seed, lam2seed)
    if verbose:
        print("alpha=%.4f: first %d pts lam[%.4f,%.4f]; fold lam_c=%.5f c=%.5f"
              % (X, len(lam1), lam1.min(), lam1.max(), lam_fold, c_fold))
        if len(lam2):
            print("           second %d pts lam[%.4f,%.4f] reaches lam=%.4f"
                  % (len(lam2), lam2.min(), lam2.max(), lam2.max()))
    return dict(lam1=lam1, W1=W1, S1=S1, lam2=lam2, W2=W2, S2=S2,
                lam_fold=lam_fold, c_fold=c_fold)


if __name__ == "__main__":
    for X in (0.0,):
        r = full_curve(X)
        if len(r['lam2']):
            i2 = int(np.argmin(np.abs(r['lam2'])))
            i1 = int(np.argmin(np.abs(r['lam1'])))
            print("  lam~0: first f''=%.5f -t'=%.5f | second(lam=%.4f) f''=%.5f -t'=%.5f"
                  % (r['W1'][i1, 0], r['W1'][i1, 2], r['lam2'][i2],
                     r['W2'][i2, 0], r['W2'][i2, 2]))
