"""Build all deliverable figures from the saved S-curves.
Figures follow the style requested in the PROMPT (solid = first solution,
dashed = second solution).
"""
import os
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = r"C:\CDC\Development\KnowledgeSharing\Auto Notify Engine - Python\Matlab\results"
os.makedirs(OUT, exist_ok=True)

COL = {'0': 'r', 'pi4': 'b', 'pi2': 'g'}
LBL = {'0': 'X = 0', 'pi4': 'X = \u03c0/4', 'pi2': 'X = \u03c0/2'}

def split_branches(arr):
    """Split S-curve at the fold (min lambda); drop stray backtracking tail points."""
    i_c = int(np.argmin(arr[:, 0]))
    b1 = arr[:i_c + 1]
    b2 = arr[i_c:]
    # truncate branch 2 where lambda backtracks noticeably
    keep = [0]
    lam_max = b2[0, 0]
    for j in range(1, len(b2)):
        if b2[j, 0] < lam_max - 0.005:
            break
        lam_max = max(lam_max, b2[j, 0])
        keep.append(j)
    b2 = b2[keep]
    return b1, b2, arr[i_c, 0]

# ---- figures: f''(0) and -t'(0) vs lambda, with fold-region inset ----
LW = {'0': 4.0, 'pi4': 2.4, 'pi2': 1.2}
fig1, ax1 = plt.subplots(figsize=(7.5, 5.4))
fig2, ax2 = plt.subplots(figsize=(7.5, 5.4))
in1 = ax1.inset_axes([0.30, 0.06, 0.40, 0.40])
in2 = ax2.inset_axes([0.42, 0.42, 0.42, 0.42])
lamcs = {}
for xn in ['0', 'pi4', 'pi2']:
    f = f'scurve_{xn}_refined.npy'
    if not os.path.exists(f):
        f = f'scurve_{xn}.npy'
    if not os.path.exists(f):
        continue
    arr = np.load(f)
    b1, b2, lamc = split_branches(arr)
    lamcs[xn] = lamc
    c = COL[xn]
    lw = LW[xn]
    for ax, ins, col in [(ax1, in1, 1), (ax2, in2, 3)]:
        sgn = 1 if col == 1 else -1
        ax.plot(b1[:, 0], sgn * b1[:, col], c + '-', lw=lw,
                label=f'{LBL[xn]} (first solution)')
        ax.plot(b2[:, 0], sgn * b2[:, col], c + '--', lw=lw,
                label=f'{LBL[xn]} (second solution)')
        ins.plot(b1[:, 0], sgn * b1[:, col], c + '-', lw=lw)
        ins.plot(b2[:, 0], sgn * b2[:, col], c + '--', lw=lw)
        ins.plot([lamc], [sgn * arr[np.argmin(arr[:, 0]), col]], c + 'o', ms=5, mfc='w')
    print(f"{xn}: lambda_c = {lamc:.6f}, b2 extends to lam={b2[-1,0]:.4f}, "
          f"-t' range on b2: [{(-b2[:,3]).min():.3f}, {(-b2[:,3]).max():.3f}]")

for ins in (in1, in2):
    ins.set_xlim(-0.46, -0.37)
    ins.grid(alpha=0.3)
    ins.tick_params(labelsize=7)
in1.set_ylim(0.14, 0.36)
in2.set_ylim(1.30, 1.62)
in1.set_title('dual-solution region', fontsize=8)
in2.set_title('dual-solution region', fontsize=8)

note = (f"curves for X=0, π/4, π/2 nearly coincide (Gr=0.001)\n"
        f"λ_c = {lamcs['0']:.5f} (X=0), {lamcs['pi4']:.5f} (π/4), "
        f"{lamcs['pi2']:.5f} (π/2)")
ax1.set_xlabel(r'$\lambda$')
ax1.set_ylabel(r"$f''(0)$")
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(alpha=0.3)
ax1.set_title(r"$f''(0)$ vs $\lambda$;  $\varphi_1=\varphi_2=0.1$, $\beta_v=\beta_T=0.5$, $s=1$, Pr=6.2, Ec=1")
ax1.text(0.02, 0.98, note, transform=ax1.transAxes, fontsize=8, va='top')
fig1.tight_layout()
fig1.savefig(os.path.join(OUT, 'fig_fpp0_vs_lambda.png'), dpi=160)

ax2.set_xlabel(r'$\lambda$')
ax2.set_ylabel(r"$-t'(0)$")
ax2.legend(fontsize=8, loc='lower center')
ax2.grid(alpha=0.3)
ax2.set_title(r"$-t'(0)$ vs $\lambda$;  $\varphi_1=\varphi_2=0.1$, $\beta_v=\beta_T=0.5$, $s=1$, Pr=6.2, Ec=1")
ax2.text(0.02, 0.98, note, transform=ax2.transAxes, fontsize=8, va='top')
fig2.tight_layout()
fig2.savefig(os.path.join(OUT, 'fig_mtp0_vs_lambda.png'), dpi=160)

# ---- supplementary: Gr=1 so that the inclination effect is visible ----
if all(os.path.exists(f'scurve_{xn}_gr1.npy') for xn in ['0', 'pi4', 'pi2']):
    fig3, ax3 = plt.subplots(figsize=(7, 5.2))
    fig4, ax4 = plt.subplots(figsize=(7, 5.2))
    for xn in ['0', 'pi4', 'pi2']:
        arr = np.load(f'scurve_{xn}_gr1.npy')
        b1, b2, lamc = split_branches(arr)
        c = COL[xn]
        if len(b2) <= 3:  # no fold rounded: trace ended at thermal runaway
            ax3.plot(b1[:, 0], b1[:, 1], c + '-', lw=1.6,
                     label=f'{LBL[xn]} (first solution, ends at thermal runaway)')
            ax4.plot(b1[:, 0], -b1[:, 3], c + '-', lw=1.6,
                     label=f'{LBL[xn]} (first solution, ends at thermal runaway)')
        else:
            ax3.plot(b1[:, 0], b1[:, 1], c + '-', lw=1.6, label=f'{LBL[xn]} (first solution)')
            ax3.plot(b2[:, 0], b2[:, 1], c + '--', lw=1.4, label=f'{LBL[xn]} (second solution)')
            ax4.plot(b1[:, 0], -b1[:, 3], c + '-', lw=1.6, label=f'{LBL[xn]} (first solution)')
            ax4.plot(b2[:, 0], -b2[:, 3], c + '--', lw=1.4, label=f'{LBL[xn]} (second solution)')
        print(f"Gr=1 {xn}: trace min lambda = {lamc:.6f}")
    ax4.set_ylim(-30, 5)
    for ax, yl, fn in [(ax3, r"$f''(0)$", 'fig_fpp0_vs_lambda_Gr1.png'),
                       (ax4, r"$-t'(0)$", 'fig_mtp0_vs_lambda_Gr1.png')]:
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(yl)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title(yl + r" vs $\lambda$ (supplementary, Gr = 1 instead of 0.001)")
    fig3.tight_layout(); fig3.savefig(os.path.join(OUT, 'fig_fpp0_vs_lambda_Gr1.png'), dpi=160)
    fig4.tight_layout(); fig4.savefig(os.path.join(OUT, 'fig_mtp0_vs_lambda_Gr1.png'), dpi=160)

print("saved figures to", OUT)
