"""Refine the fold tip of scurve_<X>.npy by pinning f''(0) (lambda unknown),
and extend branch 2 by pinning t'(0) until |t'(0)|~25.
Usage: python fold_refine.py <0|pi4|pi2>
Writes scurve_<X>_refined.npy
"""
import sys
import warnings
import numpy as np
warnings.filterwarnings('ignore')
from dusty_bvp import DEFAULTS, solve, solve_pinned, guess1

XMAP = {'0': 0.0, 'pi4': np.pi / 4, 'pi2': np.pi / 2}
xname = sys.argv[1]
X = XMAP[xname]

p = dict(DEFAULTS)
p.update(phi1=0.1, phi2=0.1, Bv=0.5, BT=0.5, s=1.0, Pr=6.2, Ec=1.0,
         m=1.0, eps=1.0, X=X, lam=1.0)
EM = 10

arr = np.load(f'scurve_{xname}.npy')
i_c = int(np.argmin(arr[:, 0]))
lam_tip = arr[i_c, 0]
fpp_hi = float(np.interp(lam_tip + 0.02, arr[:i_c + 1][::-1][:, 0], arr[:i_c + 1][::-1][:, 1]))
fpp_lo = float(arr[min(i_c + 2, len(arr) - 1), 1])
print(f"X={xname}: tracker fold lam={lam_tip:.5f}; refining f'' in [{fpp_lo:.4f}, {fpp_hi:.4f}]")

# seed: plain continuation on branch 1 down to lam_tip+0.02
sol = solve(p, EM, guess1)
seed = (sol.x, sol.y)
for lam in np.arange(1.0, lam_tip + 0.02 - 1e-9, -0.05).tolist() + [lam_tip + 0.02]:
    p['lam'] = lam
    so = solve(p, EM, seed)
    assert so.status == 0
    seed = (so.x, so.y)
lam_guess = lam_tip + 0.02

tip_pts = []
lam_min = np.inf
for tgt in np.linspace(so.y[2, 0], fpp_lo, 40):
    so2 = solve_pinned(p, EM, seed, tgt, lam_guess, tol=1e-8, max_nodes=40000)
    if so2.status != 0 or abs(so2.y[1, -1]) > 1e-4:
        print(f"  pin f''={tgt:.4f} failed, skip")
        continue
    seed = (so2.x, so2.y)
    lam_guess = so2.p[0]
    y0 = so2.y[:, 0]
    tip_pts.append((so2.p[0], y0[2], y0[4], y0[6], y0[7]))
    lam_min = min(lam_min, so2.p[0])
tip = np.array(tip_pts)
print(f"refined lambda_c = {tip[:,0].min():.6f} ({len(tip)} tip points)")

# extend branch 2 tail: pin t'(0) further along its trend
end = arr[-1]
# get a solution at the current branch-2 end: pin f'' at end f'' from the tip walk seed
so_end = solve_pinned(p, EM, seed, end[1], end[0], tol=1e-8, max_nodes=40000)
tail_pts = []
if so_end.status == 0:
    seed_t = (so_end.x, so_end.y)
    lam_g = so_end.p[0]
    t_cur = so_end.y[6, 0]
    dstep = -0.15 if arr[-1][3] < arr[-3][3] else 0.15
    for k in range(200):
        t_cur += dstep
        so3 = solve_pinned(p, EM, seed_t, t_cur, lam_g, pin_idx=6,
                           tol=1e-6, max_nodes=20000)
        if so3.status != 0 or abs(so3.y[1, -1]) > 1e-4:
            dstep *= 0.6
            t_cur = seed_t[1][6, 0]
            if abs(dstep) < 1e-3:
                break
            continue
        seed_t = (so3.x, so3.y)
        lam_g = so3.p[0]
        t_cur = so3.y[6, 0]
        y0 = so3.y[:, 0]
        tail_pts.append((lam_g, y0[2], y0[4], y0[6], y0[7]))
        dstep *= 1.3
        if abs(t_cur) > 25:
            break
tail = np.array(tail_pts) if tail_pts else np.zeros((0, 5))
print(f"branch-2 tail extended by {len(tail)} points; "
      f"end lam={tail[-1,0]:.5f} t'={tail[-1,3]:.3f}" if len(tail) else "no tail ext")

# merge: branch1 part of arr (lam > lam_tip+0.02), tip arc, branch2 part, tail
b1 = arr[:i_c + 1]
b2 = arr[i_c + 1:]
keep1 = b1[b1[:, 0] > lam_tip + 0.02]
keep2 = b2[b2[:, 1] < fpp_lo]  # f'' below refined range
merged = np.vstack([keep1, tip, keep2, tail])
np.save(f'scurve_{xname}_refined.npy', merged)
print(f"merged curve: {len(merged)} points, lambda_c={merged[:,0].min():.6f}")
