"""Task 1: trace the second-solution branch in sigma1 by continuation
from the sigma1=2 second solution found with the file's OdeInit1 guess."""
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from dusty_bvp import DEFAULTS, solve, guess1, outputs

p = dict(DEFAULTS)
p.update(phi1=0.0, phi2=0.0, lam=1.0, Bv=0.0, BT=0.0, s=0.0, X=0.0,
         Pr=6.2, Ec=1.0, m=1.0, eps=1.0)

# seed: second solution at sigma1=2
p['sigma1'] = 2.0
sol = solve(p, 6, guess1)
sol = solve(p, 16, (sol.x, sol.y))
seed = (sol.x, sol.y)
print("seed sigma1=2:", ["%.6f" % v for v in outputs(sol)])

targets = [5.0, 0.5, 0.2, 0.1, 0.0]
results = {2.0: outputs(sol)}
for tgt in targets:
    # continuation from 2.0 toward tgt in steps
    cur = 2.0
    s = seed
    ok = True
    n = 30
    for sig in np.linspace(cur, tgt, n + 1)[1:]:
        p['sigma1'] = sig
        so = solve(p, 16, s)
        if so.status != 0 or not np.all(np.isfinite(so.y)):
            print(f"  continuation failed at sigma1={sig:.4f} en route to {tgt}")
            ok = False
            break
        s = (so.x, so.y)
    if ok:
        results[tgt] = outputs(so)
        th0 = so.y[5, 0]
        print(f"sigma1={tgt}: f''(0)={results[tgt][0]:.9f} F'(0)={results[tgt][1]:.9f} "
              f"-t'(0)={results[tgt][2]:.9f} tp(0)={results[tgt][3]:.9f}  theta(0)={th0:.4f}")
