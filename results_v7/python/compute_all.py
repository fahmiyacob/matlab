"""
PROMPT 5 production run.  For alpha = 0, pi/4, pi/2 (Gr = 0.1, PROMPT-4 params,
sigma2 = 0) trace:
  * first (upper) branch: lambda = 1 -> fold -> lambda = 0
  * second (lower) branch: fold -> its termination (natural end near lambda~-0.3)
Save S-curve CSVs (lambda, f''(0), F'(0), -t'(0), tp(0)) for each branch and
clean velocity/temperature profiles (both branches, all alpha) at a common
lambda inside every dual-solution window.
"""
import os, warnings
import numpy as np
warnings.filterwarnings('ignore')
import dusty_bvp as db
import trace as T

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ALPHAS = [('0', 0.0), ('pi4', np.pi / 4), ('pi2', np.pi / 2)]
LAM_PROF = -0.43   # common lambda for the dual-profile figure (inside all folds)


def ok(sol):
    return (sol.status == 0 and np.all(np.isfinite(sol.y))
            and abs(sol.y[1, -1]) < 1e-4 and abs(sol.y[4, -1]) < 1e-4
            and sol.y[3, -1] > 1e-3)


def params(X, lam):
    p = dict(T.P4); p['X'] = X; p['lam'] = lam
    return p


def solve_fallback(X, lam, seed, boxes=((14., 1e-6), (14., 1e-5),
                                        (12., 1e-5), (16., 1e-5), (14., 1e-4))):
    best = None
    for EM, tol in boxes:
        sol = db.solve(params(X, lam), EM, seed, tol=tol, max_nodes=25000)
        if ok(sol):
            return sol
    return None


def trace_second_robust(X, seed_sol, lam_seed, lam_cap=0.0, dlam0=0.01):
    lam = lam_seed
    sol = seed_sol
    lams, W, sols = [lam], [T.wall(sol)], [sol]
    seed = (sol.x, sol.y); step = dlam0
    while lam < lam_cap - 1e-9:
        lam2 = min(lam + step, lam_cap)
        s = solve_fallback(X, lam2, seed)
        if s is not None:
            lam = lam2
            lams.append(lam); W.append(T.wall(s)); sols.append(s)
            seed = (s.x, s.y); step = min(dlam0, step * 1.25)
        else:
            step *= 0.5
            if step < 1.5e-4:
                break
    return np.array(lams), np.array(W), sols


def resolve_at(X, lam, sols, lams):
    """Clean profile exactly at lam, seeded from nearest traced solution."""
    j = int(np.argmin(np.abs(np.asarray(lams) - lam)))
    seed = (sols[j].x, sols[j].y)
    s = solve_fallback(X, lam, seed)
    return s


def save_curve(name, lams, W):
    path = os.path.join(OUT, name)
    hdr = 'lambda,fpp0,Fp0,-tp0,tp0'
    np.savetxt(path, np.column_stack([lams, W[:, 0], W[:, 1], W[:, 2], W[:, 3]]),
               delimiter=',', header=hdr, comments='')


def save_profile(name, sol):
    path = os.path.join(OUT, name)
    x = sol.x
    M = np.column_stack([x, sol.y[1], sol.y[4], sol.y[5], sol.y[7]])
    np.savetxt(path, M, delimiter=',',
               header='eta,fp,Fp,t,tp', comments='')


summary = {}
for tag, X in ALPHAS:
    print("=== alpha =", tag, "===", flush=True)
    lam1, W1, S1 = T.trace_first(X)
    i = int(np.argmin(lam1))
    lam_fold, c_fold = lam1[i], W1[i, 0]
    # keep only the descending (first-branch) part up to the fold, ordered
    order = np.argsort(-lam1[:i + 1])   # from lam=1 down to fold
    lam1o = lam1[:i + 1][order]; W1o = W1[:i + 1][order]
    S1o = [S1[:i + 1][k] for k in order]
    save_curve('task2_first_%s.csv' % tag, lam1o, W1o)
    print("  first: lam[%.4f,%.4f] fold lam_c=%.5f c=%.5f"
          % (lam1o.min(), lam1o.max(), lam_fold, c_fold), flush=True)

    s2seed, lam2seed = T.cross_fold(X, S1[i], c_fold, lam_fold)
    if s2seed is None:
        print("  second: could not cross fold", flush=True)
        summary[tag] = dict(lam_fold=lam_fold, lam_end=None)
        continue
    lam2, W2, S2 = trace_second_robust(X, s2seed, lam2seed)
    save_curve('task2_second_%s.csv' % tag, lam2, W2)
    lam_end = lam2.max()
    print("  second: lam[%.4f,%.4f] terminates near lam=%.4f (%d pts)"
          % (lam2.min(), lam2.max(), lam_end, len(lam2)), flush=True)

    # clean profiles at common lambda for both branches
    p1 = resolve_at(X, LAM_PROF, S1o, lam1o)
    p2 = resolve_at(X, LAM_PROF, S2, lam2)
    if p1 is not None:
        save_profile('prof_first_%s.csv' % tag, p1)
    if p2 is not None:
        save_profile('prof_second_%s.csv' % tag, p2)

    def wallrow(s):
        return None if s is None else (s.y[2, 0], s.y[4, 0], -s.y[6, 0], s.y[7, 0])
    summary[tag] = dict(lam_fold=lam_fold, c_fold=c_fold, lam_end=lam_end,
                        first0=wallrow(resolve_at(X, 0.0, S1o, lam1o)),
                        prof1=wallrow(p1), prof2=wallrow(p2))
    print("    profile@%.2f  first=%s  second=%s"
          % (LAM_PROF, summary[tag]['prof1'], summary[tag]['prof2']), flush=True)

np.save(os.path.join(OUT, 'summary.npy'), summary, allow_pickle=True)
print("\nSUMMARY")
for tag, X in ALPHAS:
    s = summary.get(tag, {})
    print(" alpha=%-4s fold lam_c=%s  second terminates ~%s"
          % (tag, round(s.get('lam_fold'), 4) if s.get('lam_fold') else None,
             round(s.get('lam_end'), 4) if s.get('lam_end') else None))
