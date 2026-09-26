"""Task 3: velocity f'(eta) and temperature t(eta) profiles for BOTH solutions
at a lambda inside the dual region (X=0). Also F'(eta) and tp(eta).
Solid = first solution, dashed = second solution (as in the .m file).
"""
import os
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dusty_bvp import DEFAULTS, solve, solve_pinned, guess1

OUT = r"C:\CDC\Development\KnowledgeSharing\Auto Notify Engine - Python\Matlab\results"
os.makedirs(OUT, exist_ok=True)

LAM_TARGET = -0.42

p = dict(DEFAULTS)
p.update(phi1=0.1, phi2=0.1, Bv=0.5, BT=0.5, s=1.0, Pr=6.2, Ec=1.0,
         m=1.0, eps=1.0, X=0.0, lam=1.0)
EM = 10
PINS = [2, 4, 6, 7]

# ---------- branch 1 at LAM_TARGET (plain continuation) ----------
sol = solve(p, EM, guess1)
seed = (sol.x, sol.y)
for lam in np.arange(1.0, LAM_TARGET - 1e-9, -0.05).tolist() + [LAM_TARGET]:
    p['lam'] = lam
    so = solve(p, EM, seed)
    assert so.status == 0, f"branch1 failed at lam={lam}"
    seed = (so.x, so.y)
sol_b1 = so
print("branch1 at lam=%.3f: f''(0)=%.6f -t'(0)=%.6f" %
      (LAM_TARGET, sol_b1.y[2, 0], -sol_b1.y[6, 0]))

# ---------- branch 2 at LAM_TARGET (walk around the fold, adaptive pin) ----------
def wallvec(sol, lam):
    y0 = sol.y[:, 0]
    return np.array([lam, y0[2], y0[4], y0[6], y0[7]])

records = [wallvec(sol_b1, LAM_TARGET)]
p['lam'] = LAM_TARGET - 0.01
so = solve(p, EM, seed)
records.append(wallvec(so, p['lam']))
seed2 = (so.x, so.y)

sol_b2 = None
step = 0.5
fails = 0
passed_fold = False
while True:
    prev, last = records[-2], records[-1]
    d = last - prev
    scale = np.maximum(np.abs(last), 0.1)
    rel = np.abs(d) / scale
    order = np.argsort(-rel[1:])
    advanced = False
    for oi in order[:3]:
        pin_idx = PINS[oi]
        col = 1 + oi
        tgt = last[col] + d[col] * (1 + step)
        so = solve_pinned(p, EM, seed2, tgt, last[0] + d[0] * (1 + step),
                          pin_idx=pin_idx, tol=1e-6, max_nodes=20000)
        ok = (so.status == 0 and np.all(np.isfinite(so.y))
              and abs(so.y[1, -1]) < 1e-4 and abs(so.y[5, -1]) < 1e-4)
        if not ok:
            continue
        new = wallvec(so, so.p[0])
        if np.dot((new - last) / scale, d / scale) < 0:
            continue
        if abs(new[0] - last[0]) > 0.05 or \
           abs(new[3] - last[3]) > max(5.0, 1.5 * abs(last[3])):
            continue
        records.append(new)
        seed2 = (so.x, so.y)
        advanced = True
        break
    if not advanced:
        step /= 2
        fails += 1
        if fails > 40 or step < 1e-8:
            raise RuntimeError("branch2 walk failed")
        continue
    fails = 0
    step = min(step * 1.3, 1.0)
    lam_now, lam_last = records[-1][0], records[-2][0]
    if lam_now > lam_last:
        passed_fold = True
    if passed_fold and lam_now >= LAM_TARGET:
        # refine exactly onto LAM_TARGET with a plain solve at fixed lambda
        p['lam'] = LAM_TARGET
        so = solve(p, EM, seed2)
        assert so.status == 0
        sol_b2 = so
        break
    if abs(records[-1][3]) > 100:
        raise RuntimeError("hit thermal runaway before returning to target lam")

print("branch2 at lam=%.3f: f''(0)=%.6f -t'(0)=%.6f" %
      (LAM_TARGET, sol_b2.y[2, 0], -sol_b2.y[6, 0]))
assert abs(sol_b2.y[2, 0] - sol_b1.y[2, 0]) > 1e-3, "branches identical?!"

# ---------- verify BCs ----------
for nm, sb in [('first', sol_b1), ('second', sol_b2)]:
    ya, yb = sb.y[:, 0], sb.y[:, -1]
    lam, s_, s1_, s2_ = LAM_TARGET, p['s'], p['sigma1'], p['sigma2']
    print(f"{nm}: BC residuals: f(0)-s={ya[0]-s_:.2e}, "
          f"f'(0)-lam-sig1*f''(0)={ya[1]-lam-s1_*ya[2]:.2e}, "
          f"t(0)-1-sig2*t'(0)={ya[5]-1-s2_*ya[6]:.2e}, "
          f"f'(inf)={yb[1]:.2e}, F'(inf)={yb[4]:.2e}, "
          f"F(inf)-f(inf)={yb[3]-yb[0]:.2e}, t(inf)={yb[5]:.2e}, tp(inf)={yb[7]:.2e}")

np.save('prof_b1.npy', np.vstack([sol_b1.x, sol_b1.y]))
np.save('prof_b2.npy', np.vstack([sol_b2.x, sol_b2.y]))

# ---------- plots (styles follow the .m file figures) ----------
def prof_fig(idx, ylab, fname, c1, c2):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(sol_b1.x, sol_b1.y[idx], c1, lw=1.7, label='first solution')
    ax.plot(sol_b2.x, sol_b2.y[idx], c2, lw=1.5, label='second solution')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(ylab)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(rf'$\lambda={LAM_TARGET}$, $X=0$, $\varphi_1=\varphi_2=0.1$, '
                 rf'$\beta_v=\beta_T=0.5$, $s=1$')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=160)

prof_fig(1, r"$f\,'(\eta)$", 'fig_velocity_profile.png', 'r-', 'r--')
prof_fig(4, r"$F\,'(\eta)$", 'fig_dust_velocity_profile.png', 'b-', 'b--')
prof_fig(5, r"$t(\eta)$", 'fig_temperature_profile.png', 'y-', 'y--')
prof_fig(7, r"$t_p(\eta)$", 'fig_dust_temperature_profile.png', 'k-', 'k--')
print("profile figures saved")
