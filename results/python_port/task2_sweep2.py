"""Task 2: full S-curve sweep, adaptive choice of pinned wall quantity.
Usage: python task2_sweep2.py <0|pi4|pi2>
Saves scurve_<X>.npy with rows (lam, f''(0), F'(0), -t'(0), tp(0)).
"""
import sys
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from dusty_bvp import DEFAULTS, solve, solve_pinned, guess1, outputs

XMAP = {'0': 0.0, 'pi4': np.pi / 4, 'pi2': np.pi / 2}
xname = sys.argv[1] if len(sys.argv) > 1 else '0'
X = XMAP[xname]
GR = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULTS['Gr']
if len(sys.argv) > 2:
    xname = f"{xname}_gr{sys.argv[2]}"

p = dict(DEFAULTS)
p.update(phi1=0.1, phi2=0.1, Bv=0.5, BT=0.5, s=1.0, Pr=6.2, Ec=1.0,
         m=1.0, eps=1.0, X=X, lam=1.0, Gr=GR)
EM = 10
PINS = [2, 4, 6, 7]  # f''(0), F'(0), t'(0), tp(0)
TOL = 1e-6
MAXN = 20000

if GR != DEFAULTS['Gr']:
    # ramp Gr up from the default via continuation
    p['Gr'] = DEFAULTS['Gr']
    sol = solve(p, EM, guess1)
    assert sol.status == 0, "Gr ramp start failed"
    for gr in np.linspace(DEFAULTS['Gr'], GR, 25)[1:]:
        p['Gr'] = gr
        sol = solve(p, EM, (sol.x, sol.y))
        assert sol.status == 0, f"Gr ramp failed at Gr={gr}"
else:
    sol = solve(p, EM, guess1)
assert sol.status == 0
lam = 1.0

def wallvec(sol, lam):
    y0 = sol.y[:, 0]
    return np.array([lam, y0[2], y0[4], y0[6], y0[7]])

records = [wallvec(sol, lam)]
seed = (sol.x, sol.y)

# bootstrap a second point (plain continuation, small step)
p['lam'] = lam - 0.01
so = solve(p, EM, seed)
assert so.status == 0
lam = p['lam']
seed = (so.x, so.y)
records.append(wallvec(so, lam))

step = 0.5          # fraction of last increment to extrapolate
consec_fail = 0
while len(records) < 3000:
    prev, last = records[-2], records[-1]
    d = last - prev
    scale = np.maximum(np.abs(last), 0.1)
    rel = np.abs(d) / scale
    # choose pin coordinate with largest relative change (among pinnable ones)
    order = np.argsort(-rel[1:])  # indices into PINS
    advanced = False
    for oi in order[:3]:
        pin_idx = PINS[oi]
        col = 1 + oi
        tgt = last[col] + d[col] * (1 + step)
        so = solve_pinned(p, EM, seed, tgt, last[0] + d[0] * (1 + step),
                          pin_idx=pin_idx, tol=TOL, max_nodes=MAXN)
        ok = (so.status == 0 and np.all(np.isfinite(so.y))
              and abs(so.y[1, -1]) < 1e-4 and abs(so.y[5, -1]) < 1e-4)
        if not ok:
            continue
        new = wallvec(so, so.p[0])
        # reject backtracking (angle with previous tangent)
        if np.dot((new - last) / scale, d / scale) < 0:
            continue
        # reject huge jumps
        if np.linalg.norm((new - last) / scale) > 8 * max(np.linalg.norm(d / scale), 1e-6):
            continue
        if abs(new[0] - last[0]) > 0.05:      # keep lambda resolution
            continue
        if abs(new[3] - last[3]) > max(5.0, 1.5 * abs(last[3])):  # tame t' jumps
            continue
        records.append(new)
        seed = (so.x, so.y)
        advanced = True
        break
    if not advanced:
        step /= 2
        consec_fail += 1
        if consec_fail > 40 or step < 1e-8:
            break
        continue
    consec_fail = 0
    step = min(step * 1.3, 1.0)
    lam_now = records[-1][0]
    if lam_now > 1.02 and records[-1][1] > records[0][1]:
        break  # branch 2 traced back past lambda = 1
    if abs(records[-1][3]) > 100:
        print("  stopping: thermal resonance runaway (|t'(0)|>100)", flush=True)
        break
    if len(records) % 25 == 0:
        np.save(f'scurve_{xname}.npy', np.array(records))
    if len(records) % 10 == 0:
        r = records[-1]
        print(f"  n={len(records)} lam={r[0]:.5f} f''={r[1]:.5f} F'={r[2]:.5f} t'={r[3]:.5f} tp={r[4]:.5f}",
              flush=True)

arr = np.array(records)
np.save(f'scurve_{xname}.npy', arr)
print(f"X={xname}: {len(arr)} points, lam range [{arr[:,0].min():.5f}, {arr[:,0].max():.5f}]")
i_c = np.argmin(arr[:, 0])
print(f"lambda_c = {arr[i_c,0]:.6f} at f''(0)={arr[i_c,1]:.6f} (index {i_c}/{len(arr)})")
r = arr[-1]
print(f"end: lam={r[0]:.5f} f''={r[1]:.5f} F'={r[2]:.5f} -t'={-r[3]:.5f} tp={r[4]:.5f}")
