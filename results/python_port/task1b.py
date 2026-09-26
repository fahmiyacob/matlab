"""Task 1 verification: domain-convergence of the first solution,
and check whether the short-domain 'second' solution at sigma1=2 is genuine."""
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from dusty_bvp import DEFAULTS, solve, guess1, guess2, outputs, is_physical

p = dict(DEFAULTS)
p.update(phi1=0.0, phi2=0.0, lam=1.0, Bv=0.0, BT=0.0, s=0.0, X=0.0,
         Pr=6.2, Ec=1.0, m=1.0, eps=1.0)

def g4(x):
    e = np.exp(-x)
    return np.vstack([0.5 * (1 - e), 2 * e, -2 * e, 0.5 + 0 * x, 0 * x, e, -e, 0 * x])

sigmas = [0.0, 0.1, 0.2, 0.5, 2.0, 5.0]
print("Domain convergence of first solution (extend etaMax by continuation):")
final = {}
for sig in sigmas:
    p['sigma1'] = sig
    sol = solve(p, 12, g4)
    line = [f"sigma1={sig}:"]
    for em in [16, 20, 25, 30]:
        sol = solve(p, em, (sol.x, sol.y))
        o = outputs(sol)
        line.append(f"em{em}: f''={o[0]:.7f} -t'={o[2]:.7f} st={sol.status}")
    final[sig] = outputs(sol)
    print("  " + "  ".join(line))

print("\nFinal converged table (etaMax=30):")
print(f"{'sigma1':>7} {'f\"(0)':>15} {'F`(0)':>15} {'-t`(0)':>15} {'tp(0)':>15}")
for sig in sigmas:
    o = final[sig]
    print(f"{sig:>7} {o[0]:>15.9f} {o[1]:>15.9f} {o[2]:>15.9f} {o[3]:>15.9f}")

# Is the sigma1=2, etaMax=6 'second' solution real? Extend its domain.
p['sigma1'] = 2.0
s6 = solve(p, 6, guess1)
print("\nsigma1=2 short-domain solution at etaMax=6:", ["%.6f" % v for v in outputs(s6)])
sol = s6
for em in [8, 10, 12, 16, 20]:
    sol = solve(p, em, (sol.x, sol.y))
    o = outputs(sol)
    print(f"  extended to etaMax={em}: status={sol.status} f''(0)={o[0]:.7f} -t'(0)={o[2]:.7f}")
