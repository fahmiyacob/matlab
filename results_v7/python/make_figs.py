"""PROMPT 5 figures: skin-friction & Nusselt vs lambda (both branches, 3 alpha),
and dual velocity/temperature profiles at a common lambda."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG = os.path.join(OUT, 'figures'); os.makedirs(FIG, exist_ok=True)

# Okabe-Ito colourblind-safe categorical hues, fixed order for the 3 inclinations
COL = {'0': '#0072B2', 'pi4': '#D55E00', 'pi2': '#009E73'}
LAB = {'0': r'$\alpha=0$', 'pi4': r'$\alpha=\pi/4$', 'pi2': r'$\alpha=\pi/2$'}
TAGS = ['0', 'pi4', 'pi2']
plt.rcParams.update({'font.size': 12, 'axes.grid': True,
                     'grid.alpha': 0.35, 'grid.linewidth': 0.6,
                     'axes.axisbelow': True, 'figure.dpi': 130})


def load(name):
    return np.loadtxt(os.path.join(OUT, name), delimiter=',', skiprows=1)


def curve(name):
    a = load(name)
    return a[:, 0], a[:, 1], a[:, 3]   # lambda, f''(0), -t'(0)


# ---------- Figure 1 & 2 : wall quantities vs lambda -----------------------
def wall_figure(col_idx, ylabel, fname, title):
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    for t in TAGS:
        l1, f1a, f1b = curve('task2_first_%s.csv' % t)
        y1 = f1a if col_idx == 1 else f1b
        ax.plot(l1, y1, '-', color=COL[t], lw=2, label=LAB[t] + ' (1st)')
        try:
            l2, f2a, f2b = curve('task2_second_%s.csv' % t)
            y2 = f2a if col_idx == 1 else f2b
            ax.plot(l2, y2, '--', color=COL[t], lw=2, label=LAB[t] + ' (2nd)')
            ax.plot(l2[np.argmax(l2)], y2[np.argmax(l2)], 'o',
                    color=COL[t], ms=6, mfc='white', mew=1.6)
        except Exception:
            pass
        # fold marker (min lambda of first branch)
        i = int(np.argmin(l1))
        ax.plot(l1[i], y1[i], 's', color=COL[t], ms=7, mfc=COL[t], mec='k', mew=0.6)
    ax.axvline(0, color='0.6', lw=0.8, ls=':')
    ax.set_xlabel(r'$\lambda$  (stretching $>0$ / shrinking $<0$)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=9, framealpha=0.9, loc='best')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)


# ---------- Figure 3 & 4 : dual profiles at common lambda ------------------
def profile_figure(cols, titles, ylabels, fname, suptitle):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    for ax, col, ttl, yl in zip(axes, cols, titles, ylabels):
        for t in TAGS:
            p1 = load('prof_first_%s.csv' % t)
            p2 = load('prof_second_%s.csv' % t)
            ax.plot(p1[:, 0], p1[:, col], '-', color=COL[t], lw=2,
                    label=LAB[t] + ' (1st)')
            ax.plot(p2[:, 0], p2[:, col], '--', color=COL[t], lw=2,
                    label=LAB[t] + ' (2nd)')
        ax.axhline(0, color='0.6', lw=0.8, ls=':')
        ax.set_xlabel(r'$\eta$'); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.set_xlim(0, None)
    axes[0].legend(ncol=3, fontsize=9, framealpha=0.9, loc='best')
    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    wall_figure(1, r"$f''(0)$   (skin-friction coefficient)",
                'fig_skinfriction_vs_lambda.png',
                r"Skin friction $f''(0)$ vs $\lambda$   "
                r"(square = fold, circle = 2nd-branch end)")
    wall_figure(3, r"$-t'(0)$   (Nusselt number / heat-transfer rate)",
                'fig_nusselt_vs_lambda.png',
                r"Heat transfer $-t'(0)$ vs $\lambda$  (Gr=0.1, $\sigma_2$=0)")
    profile_figure([1, 2],
                   [r"Nanofluid velocity $f'(\eta)$", r"Dust velocity $F'(\eta)$"],
                   [r"$f'(\eta)$", r"$F'(\eta)$"],
                   'fig_velocity_profiles.png',
                   r"Dual velocity profiles at $\lambda=-0.43$ "
                   r"(solid = 1st solution, dashed = 2nd solution)")
    profile_figure([3, 4],
                   [r"Nanofluid temperature $t(\eta)$", r"Dust temperature $t_p(\eta)$"],
                   [r"$t(\eta)$", r"$t_p(\eta)$"],
                   'fig_temperature_profiles.png',
                   r"Dual temperature profiles at $\lambda=-0.43$ "
                   r"(solid = 1st solution, dashed = 2nd solution)")
    print("figures written to", FIG)
