"""Task 1: table of f''(0), F'(0), -t'(0), tp(0) vs sigma1.
lambda=1, phi1=phi2=Bv=BT=s=X=0, Pr=6.2, Ec=1, m=1, eps=1.
Unspecified params kept at the .m file defaults: sigma2=0.2, Gr=0.001, L=1.
"""
import numpy as np
from dusty_bvp import DEFAULTS, solve, guess1, guess2, outputs, is_physical

p = dict(DEFAULTS)
p.update(phi1=0.0, phi2=0.0, lam=1.0, Bv=0.0, BT=0.0, s=0.0, X=0.0,
         Pr=6.2, Ec=1.0, m=1.0, eps=1.0)

sigmas = [0.0, 0.1, 0.2, 0.5, 2.0, 5.0]

extra_guesses = []
def g3(x):
    e = np.exp(-x)
    return np.vstack([1 - e, 0.5 * e, -e, 1 + 0 * x, 0 * x, e, -e, 0 * x])
def g4(x):
    e = np.exp(-x)
    return np.vstack([0.5 * (1 - e), 2 * e, -2 * e, 0.5 + 0 * x, 0 * x, e, -e, 0 * x])
extra_guesses = [g3, g4]

rows = {}
for sig in sigmas:
    p['sigma1'] = sig
    found = []
    for gname, g, em in [('g1', guess1, 6), ('g2', guess2, 9), ('g3', g3, 10),
                         ('g4', g4, 12), ('g1b', guess1, 12), ('g2b', guess2, 14)]:
        try:
            sol = solve(p, em, g)
        except Exception as ex:
            print(sig, gname, 'EXC', ex)
            continue
        if is_physical(sol, em):
            o = outputs(sol)
            if not any(abs(o[0] - f[0][0]) < 1e-5 and abs(o[2] - f[0][2]) < 1e-5 for f in found):
                found.append((o, gname, em))
    rows[sig] = found
    print(f"sigma1={sig}: {len(found)} distinct solution(s)")
    for o, gname, em in found:
        print(f"   [{gname}, etaMax={em}] f''(0)={o[0]:.9f}  F'(0)={o[1]:.9f}  -t'(0)={o[2]:.9f}  tp(0)={o[3]:.9f}")

print("\n=== TABLE (first solution) ===")
print(f"{'sigma1':>7} {'f\"(0)':>15} {'F`(0)':>15} {'-t`(0)':>15} {'tp(0)':>15}")
for sig in sigmas:
    if rows[sig]:
        o = rows[sig][0][0]
        print(f"{sig:>7} {o[0]:>15.9f} {o[1]:>15.9f} {o[2]:>15.9f} {o[3]:>15.9f}")
