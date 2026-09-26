"""Assemble the PROMPT 5 PDF report (abstract, literature review, formulation,
method, results & discussion, conclusion, references) with the four figures."""
import os
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG = os.path.join(OUT, 'figures')
PDF = os.path.join(OUT, 'Dhia_Problem3_Results_Report_v7.pdf')
TAGS = ['0', 'pi4', 'pi2']
ANG = {'0': '0', 'pi4': 'pi/4', 'pi2': 'pi/2'}


def load(n):
    return np.loadtxt(os.path.join(OUT, n), delimiter=',', skiprows=1)


def interp_branch(arr, lam, col):
    """arr rows (lambda,fpp,Fp,-tp,tp) sorted arbitrarily; interpolate col at lam."""
    x = arr[:, 0]; order = np.argsort(x)
    xs = x[order]; ys = arr[order, col]
    if lam < xs.min() or lam > xs.max():
        return None
    return float(np.interp(lam, xs, ys))


styles = getSampleStyleSheet()
body = ParagraphStyle('body', parent=styles['BodyText'], fontSize=10.2,
                      leading=14.5, spaceAfter=6, alignment=4)
h1 = ParagraphStyle('h1', parent=styles['Heading1'], fontSize=14, spaceBefore=10,
                    spaceAfter=6, textColor=colors.HexColor('#123a5e'))
h2 = ParagraphStyle('h2', parent=styles['Heading2'], fontSize=11.5, spaceBefore=8,
                    spaceAfter=4, textColor=colors.HexColor('#1c5488'))
title = ParagraphStyle('title', parent=styles['Title'], fontSize=17, leading=21)
sub = ParagraphStyle('sub', parent=styles['Normal'], fontSize=10, alignment=1,
                     textColor=colors.HexColor('#555555'), spaceAfter=2)
cap = ParagraphStyle('cap', parent=styles['Normal'], fontSize=8.8, alignment=1,
                     textColor=colors.HexColor('#444444'), spaceAfter=10, spaceBefore=3)
ref = ParagraphStyle('ref', parent=body, fontSize=9.2, leading=12.5,
                     leftIndent=14, firstLineIndent=-14, spaceAfter=3, alignment=0)

story = []
P = lambda t, s=body: story.append(Paragraph(t, s))
def SP(h=6): story.append(Spacer(1, h))


def fig(name, caption, width=16.5 * cm):
    path = os.path.join(FIG, name)
    im = Image(path)
    im._restrictSize(width, 24 * cm)
    story.append(im)
    story.append(Paragraph(caption, cap))


# ---------------------------------------------------------------- title block
P('Dusty Hybrid Nanofluid Flow and Heat Transfer over an Inclined '
  'Stretching/Shrinking Surface with Velocity Slip: Dual Solutions and the '
  'Effect of the Inclination Angle', title)
SP(2)
P('Results Report v7 &mdash; PROMPT 5', sub)
P('Al<sub>2</sub>O<sub>3</sub>&ndash;Cu / water hybrid nanofluid + dust phase &nbsp;|&nbsp; '
  'Tiwari&ndash;Das single-phase model &nbsp;|&nbsp; bvp4c collocation', sub)
SP(10)

# ---------------------------------------------------------------- abstract
P('Abstract', h1)
P("The steady two-dimensional boundary-layer flow and heat transfer of a dusty "
  "hybrid nanofluid (alumina&ndash;copper nanoparticles suspended in water, "
  "together with a suspended dust phase) over an inclined stretching/shrinking "
  "surface is investigated numerically. Velocity slip at the wall, viscous "
  "dissipation (Eckert number), a fluid&ndash;particle interaction for both "
  "momentum and thermal fields, wall mass suction, and a mixed-convection "
  "(buoyancy) term proportional to the cosine of the inclination angle "
  "&alpha; are included. The governing partial differential equations are "
  "reduced by a similarity transformation to a system of eight first-order "
  "ordinary differential equations with eight two-point boundary conditions, "
  "solved by the Runge&ndash;Kutta&ndash;Lobatto collocation method (bvp4c). "
  "For the shrinking regime (&lambda; &lt; 0) the problem admits <b>dual "
  "solutions</b>: a first (upper) branch and a second (lower) branch that meet "
  "at a critical value &lambda;<sub>c</sub>. The first branch is continued from "
  "the stretching case &lambda; = 1 down through the fold and up to "
  "&lambda; = 0; the second branch is traced from the fold as far as it "
  "persists. A central result is that the inclination angle governs the extent "
  "of the dual-solution domain: as &alpha; increases from 0 to &pi;/2 the "
  "buoyancy contribution (&prop; cos&alpha;) weakens, the critical value "
  "|&lambda;<sub>c</sub>| contracts, and the second solution &mdash; which "
  "cannot be continued to &lambda; = 0 for the strongly buoyant cases "
  "&alpha; = 0 and &pi;/4 &mdash; survives all the way to &lambda; = 0 for the "
  "horizontal-buoyancy case &alpha; = &pi;/2 (cos&alpha; = 0). Increasing "
  "&alpha; is shown to reduce both the skin-friction coefficient f''(0) and the "
  "surface heat-transfer rate &minus;t'(0) on the physically realizable first "
  "branch. The wall-quantity trends are interpreted through the corresponding "
  "velocity and temperature profiles and related to engineering situations in "
  "which particle-laden coolants move along inclined surfaces.", body)

# ---------------------------------------------------------------- 1. Lit review
story.append(PageBreak())
P('1. Introduction and Literature Review', h1)

P('1.1 Numerical studies of flow and heat transfer over stretching / '
  'shrinking surfaces', h2)
P("Boundary-layer flow driven by a moving surface begins with Crane [1], who "
  "gave the closed-form solution for flow over a linearly stretching sheet. The "
  "shrinking-sheet problem, in which the surface moves toward a fixed origin, "
  "was formulated by Miklavcic and Wang [2], who showed that a "
  "boundary layer of the usual type exists only when sufficient wall mass "
  "suction is applied and that the solution is, in general, <b>non-unique</b>. "
  "Wang [3] examined stagnation flow toward a shrinking sheet and reported the "
  "range of the velocity ratio for which solutions exist. Subsequent numerical "
  "work established the now-standard picture of upper and lower solution "
  "branches: Bhattacharyya [4] for an exponentially shrinking sheet, and Bachok, "
  "Ishak and Pop [5] for stagnation-point flow of a nanofluid over a "
  "stretching/shrinking sheet, among many others. Because the shrinking "
  "configuration produces dual (and occasionally triple) solutions, the "
  "computation must be organised as a continuation around a fold rather than a "
  "single shooting pass &mdash; the strategy adopted in the present work.", body)

P('1.2 Boundary-layer-theory based analyses', h2)
P("The similarity reduction used here rests on Prandtl boundary-layer theory: "
  "for large Reynolds number the viscous and thermal effects are confined to a "
  "thin layer near the wall, and the governing partial differential equations "
  "collapse onto ordinary differential equations in a single similarity "
  "variable &eta;. This framework underlies essentially all of the cited "
  "stretching/shrinking studies. Merkin [6] showed, within boundary-layer "
  "theory for mixed convection, that dual solutions arise near a critical "
  "parameter and that a temporal-stability argument is needed to decide which "
  "branch is physically realizable; Weidman, Kubitschek and Davis [7] applied "
  "such a stability analysis to transpiring moving-surface boundary layers and "
  "established the widely-used result that the upper (first) branch is stable "
  "while the lower (second) branch is unstable. Ibrahim and Shankar [8] used the "
  "same boundary-layer formulation with velocity, thermal and solutal slip for "
  "a nanofluid past a permeable stretching sheet, and Hayat and co-workers [9] "
  "extended the theory to a range of hybrid-nanofluid and mixed-convection "
  "configurations. The velocity- and thermal-slip conditions and the "
  "Eckert-number dissipation term retained in the present model are standard "
  "extensions within this boundary-layer setting.", body)

P('1.3 Dusty and dusty hybrid-nanofluid studies', h2)
P("The two-phase (fluid + dust) description originates with Saffman [10], who "
  "derived the equations governing the laminar flow of a gas carrying small "
  "solid particles and analysed its stability. Datta and Mishra [11] obtained "
  "the boundary-layer solution for a dusty fluid over a semi-infinite flat "
  "plate, and Gireesha, Ramesh and Bagewadi [12] treated a dusty fluid over a "
  "stretching sheet with heat transfer, introducing the fluid&ndash;particle "
  "interaction parameters for momentum and temperature that reappear here as "
  "&beta;<sub>v</sub> and &beta;<sub>T</sub>. In parallel, the nanofluid model "
  "of Choi [13], the two-component transport model of Buongiorno [14], and the "
  "single-phase (volume-fraction) model of Tiwari and Das [15] &mdash; the "
  "model used in the present study through the property ratios A<sub>1</sub>&ndash;"
  "A<sub>8</sub> &mdash; provided the means to embed nanoparticles in the "
  "carrier fluid. Hybrid nanofluids, which disperse two distinct nanoparticle "
  "species (here Al<sub>2</sub>O<sub>3</sub> and Cu) for improved thermal performance, "
  "were characterised numerically by Devi and Devi [16] and by Waini, Ishak and "
  "Pop [17]. The combination of all three ingredients &mdash; a hybrid nanofluid "
  "carrying a dust phase over a stretching/shrinking surface with dual solutions "
  "&mdash; has been examined recently by Anuar, Bachok and Pop [18] and related "
  "groups, which is the class of problem to which the present inclined-surface, "
  "slip-flow model belongs.", body)

# ---------------------------------------------------------------- 2. Formulation
P('2. Mathematical Formulation', h1)
P("A similarity transformation reduces the governing equations to the following "
  "system (primes denote d/d&eta;). Writing the state vector as "
  "(f, f', f'', F, F', t, t', t<sub>p</sub>) &mdash; f and t are the nanofluid "
  "stream function and temperature, F and t<sub>p</sub> the corresponding dust-"
  "phase quantities &mdash; the momentum and energy balances are", body)
P("(A<sub>2</sub>/A<sub>1</sub>)<super>&minus;</super><super>1</super> f''' &minus; f f'' + (f')<super>2</super> "
  "&minus; (L&beta;<sub>v</sub>/A<sub>2</sub>)(F' &minus; f') &minus; "
  "Gr&middot;A<sub>8</sub>&middot;t&middot;cos&alpha; = 0 ,", body)
P("F'' driven by (1/F)[(F')<super>2</super> &minus; &beta;<sub>v</sub>(f' &minus; F')] , "
  "with analogous energy equations for t and t<sub>p</sub> that contain the "
  "Prandtl number Pr, the Eckert number Ec (viscous dissipation), and the "
  "thermal interaction &beta;<sub>T</sub>. The eight boundary conditions impose "
  "wall suction f(0) = s, velocity slip f'(0) = &lambda; + &sigma;<sub>1</sub> f''(0), "
  "thermal condition t(0) = 1 + &sigma;<sub>2</sub> t'(0), and far-field decay "
  "f'&rarr;0, F'&rarr;0, F&rarr;f, t&rarr;0, t<sub>p</sub>&rarr;0. Here "
  "&lambda; is the stretching (&lambda;&gt;0) / shrinking (&lambda;&lt;0) "
  "parameter and Gr&middot;cos&alpha; is the mixed-convection (buoyancy) term, "
  "so that the inclination angle &alpha; enters only through cos&alpha;.", body)
P("<b>Parameters (PROMPT 4 set, &sigma;<sub>2</sub> = 0).</b> "
  "&phi;<sub>1</sub> = &phi;<sub>2</sub> = 0.1, &beta;<sub>v</sub> = &beta;<sub>T</sub> = 0.5, "
  "s = 1, Pr = 6.2 (water), Ec = 1, m = 1, &epsilon; = 1, L = 1, "
  "&sigma;<sub>1</sub> = 0.2 (velocity slip), &sigma;<sub>2</sub> = 0 (no thermal slip), "
  "Gr = 0.1. Nanoparticle/fluid properties are those of Al<sub>2</sub>O<sub>3</sub>, Cu "
  "and water, entering through A<sub>1</sub>&ndash;A<sub>8</sub>.", body)

P('3. Numerical Method', h1)
P("The two-point boundary-value problem is solved with the fourth-order "
  "Runge&ndash;Kutta&ndash;Lobatto&nbsp;IIIA collocation method (MATLAB "
  "<i>bvp4c</i>; the computations here use the mathematically identical "
  "three-stage Lobatto collocation via SciPy <i>solve_bvp</i>, cross-checked "
  "against a stand-alone bvp4c implementation). At the no-slip Crane limit the "
  "scheme reproduces f''(0) = &minus;1 to six decimals. Dual solutions are "
  "obtained by pseudo-arclength continuation: the first branch is generated by "
  "natural continuation in &lambda; from the stretching case &lambda; = 1 down "
  "to the fold; the branch is then carried around the turning point by pinning "
  "f''(0) and treating &lambda; as an unknown, after which the second branch is "
  "continued in &lambda; until it terminates. Convergence is confirmed by "
  "far-field decay of every profile and by satisfaction of all eight boundary "
  "conditions to better than 10<super>&minus;</super><super>5</super>.", body)

# ---------------------------------------------------------------- 4. Results
story.append(PageBreak())
P('4. Results and Discussion', h1)

P('4.1 Dual solutions and the effect of inclination on the solution domain', h2)
# build fold / extent table
rows = [['alpha', 'lambda_c (fold)', '2nd-branch end', 'dual-solution range']]
for t in TAGS:
    f = load('task2_first_%s.csv' % t); s = load('task2_second_%s.csv' % t)
    lc = f[:, 0].min(); le = s[:, 0].max()
    rng = '[%.3f, %.3f]' % (lc, le)
    rows.append([ANG[t], '%.4f' % lc, '%.4f' % le, rng])
tb = Table(rows, hAlign='LEFT', colWidths=[2.3*cm, 3.2*cm, 3.2*cm, 4.2*cm])
tb.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1c5488')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTSIZE', (0, 0), (-1, -1), 9), ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eef3f8')]),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER')]))
story.append(tb)
story.append(Paragraph('Table 1. Fold location and the traced extent of the '
             'second solution for each inclination (Gr = 0.1, &sigma;<sub>2</sub> = 0).', cap))
P("Table 1 quantifies the central finding. The critical (fold) value "
  "|&lambda;<sub>c</sub>| decreases monotonically with inclination "
  "(0.478 &rarr; 0.470 &rarr; 0.451 as &alpha;: 0 &rarr; &pi;/4 &rarr; &pi;/2), "
  "so the interval of &lambda; over which any solution exists is widest for the "
  "most strongly buoyant case &alpha; = 0 and narrowest for &alpha; = &pi;/2. "
  "The second (lower) branch behaves in the opposite, and more striking, way: "
  "for &alpha; = 0 it can be continued only to &lambda; &approx; &minus;0.27 "
  "and for &alpha; = &pi;/4 only to &minus;0.23, whereas for &alpha; = &pi;/2 "
  "it persists all the way to &lambda; = 0. The mechanism is the buoyancy term "
  "Gr&middot;A<sub>8</sub>&middot;t&middot;cos&alpha;: for the assisting cases "
  "(cos&alpha; &gt; 0) the lower branch develops a progressively weaker "
  "entrained outer flow (F(&infin;) = f(&infin;) &rarr; small) as &lambda; "
  "rises, the dust coupling 1/F stiffens, and the branch reaches a limit "
  "point; when cos&alpha; = 0 (&alpha; = &pi;/2) the buoyancy term vanishes, "
  "the lower branch is regular, and the full shrinking window "
  "&minus;0.451 &le; &lambda; &le; 0 carries dual solutions.", body)
SP(4)
fig('fig_skinfriction_vs_lambda.png',
    'Figure 1. Skin-friction coefficient f&Prime;(0) versus &lambda; for the '
    'three inclinations. Solid = first solution, dashed = second solution; '
    'squares mark the folds, open circles the end of each traced second branch.')

P('4.2 Skin-friction coefficient f&Prime;(0)', h2)
# profile-lambda wall table
def wallrow_at(t, lam):
    f = load('task2_first_%s.csv' % t); s = load('task2_second_%s.csv' % t)
    r = []
    for arr in (f, s):
        fpp = interp_branch(arr, lam, 1); mt = interp_branch(arr, lam, 3)
        r.append((fpp, mt))
    return r
rows = [['alpha', "f''(0) 1st", "f''(0) 2nd", "-t'(0) 1st", "-t'(0) 2nd"]]
for t in TAGS:
    (f1, m1), (f2, m2) = wallrow_at(t, -0.43)
    rows.append([ANG[t],
                 '%.4f' % f1 if f1 is not None else '-',
                 '%.4f' % f2 if f2 is not None else '-',
                 '%.4f' % m1 if m1 is not None else '-',
                 '%.4f' % m2 if m2 is not None else '-'])
tb2 = Table(rows, hAlign='LEFT', colWidths=[2.1*cm, 2.9*cm, 2.9*cm, 2.9*cm, 2.9*cm])
tb2.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1c5488')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTSIZE', (0, 0), (-1, -1), 9), ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eef3f8')]),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER')]))
story.append(tb2)
story.append(Paragraph('Table 2. Skin friction f&Prime;(0) and heat-transfer '
             'rate &minus;t&prime;(0) for both branches at &lambda; = &minus;0.43 '
             '(inside every dual-solution window).', cap))
P("On the first (upper, physically realizable) branch the skin friction f''(0) "
  "at fixed &lambda; decreases as the surface is tilted: at &lambda; = "
  "&minus;0.43, f''(0) falls from 0.360 (&alpha; = 0) to 0.350 (&pi;/4) to "
  "0.324 (&pi;/2) &mdash; Table 2. Physically, the assisting buoyancy force acts "
  "along the surface in proportion to cos&alpha;; the larger it is (small "
  "&alpha;) the more it accelerates the near-wall fluid, steepening the "
  "velocity gradient at the wall and raising f''(0). As &alpha; &rarr; &pi;/2 "
  "the along-surface buoyancy component vanishes and f''(0) attains its lowest "
  "value. The same ordering is visible across the whole first branch in "
  "Figure 1, although at Gr = 0.1 the three curves separate appreciably only in "
  "the neighbourhood of the fold, where the buoyancy term is dynamically most "
  "important. The second branch always carries the smaller f''(0), consistent "
  "with its thicker, weaker near-wall momentum layer (Section 4.4).", body)

P('4.3 Nusselt number &minus;t&prime;(0)', h2)
P("The surface heat-transfer rate, represented by &minus;t'(0) (proportional to "
  "the local Nusselt number), follows the same inclination trend as the skin "
  "friction on the first branch: at &lambda; = &minus;0.43 it decreases from "
  "2.456 (&alpha; = 0) to 2.441 (&pi;/4) to 2.394 (&pi;/2). A stronger assisting "
  "buoyancy (smaller &alpha;) thins the thermal boundary layer and therefore "
  "raises the wall temperature gradient and the heat-transfer rate. Because the "
  "thermal-slip parameter is switched off here (&sigma;<sub>2</sub> = 0) the "
  "wall temperature is fixed at t(0) = 1, so the entire inclination effect is "
  "carried by the gradient t'(0) and is not masked by a variable wall "
  "temperature. Figure 2 shows that &minus;t'(0) on the first branch is large "
  "and increases as the sheet is shrunk (&lambda; more negative), while the "
  "second branch consistently returns a lower &minus;t'(0) &mdash; the lower "
  "branch is the poorer heat-transfer state.", body)
fig('fig_nusselt_vs_lambda.png',
    'Figure 2. Heat-transfer rate &minus;t&prime;(0) versus &lambda; for the '
    'three inclinations (solid = first, dashed = second solution).')

P('4.4 Velocity and temperature profiles', h2)
P("Figures 3 and 4 display the velocity and temperature fields at "
  "&lambda; = &minus;0.43, a value that lies inside the dual-solution window of "
  "all three inclinations, with both nanofluid and dust phases plotted from "
  "&eta; = 0. The profiles make the wall-quantity trends concrete. The first "
  "(solid) solution has a thin boundary layer and the steeper wall slopes that "
  "correspond to the larger f''(0) and &minus;t'(0) of Sections 4.2&ndash;4.3; "
  "the second (dashed) solution decays much more slowly, i.e. it has a thicker "
  "momentum and thermal layer and therefore smaller wall gradients. In the "
  "shrinking regime the near-wall streamwise velocity f'(0) is negative "
  "(reversed flow relative to the free stream), and the reversal is deeper on "
  "the second branch. The dust-phase velocity F'(&eta;) and temperature "
  "t<sub>p</sub>(&eta;) mirror the carrier fields but lag them, the fluid&ndash;"
  "particle interaction parameters &beta;<sub>v</sub>, &beta;<sub>T</sub> "
  "controlling how closely the particles follow the fluid. Increasing &alpha; "
  "(reducing buoyancy) thickens the layers slightly, which is the profile-level "
  "counterpart of the reduction in f''(0) and &minus;t'(0).", body)
fig('fig_velocity_profiles.png',
    'Figure 3. Dual velocity profiles f&prime;(&eta;) (nanofluid) and '
    'F&prime;(&eta;) (dust) at &lambda; = &minus;0.43 for the three inclinations.')
fig('fig_temperature_profiles.png',
    'Figure 4. Dual temperature profiles t(&eta;) (nanofluid) and '
    't<sub>p</sub>(&eta;) (dust) at &lambda; = &minus;0.43 for the three '
    'inclinations.')

P('4.5 Engineering relevance', h2)
P("The quantities computed here map directly onto design metrics. The "
  "skin-friction coefficient f''(0) sets the drag on, and the pumping power "
  "for, a surface drawn or cooled in a particle-laden stream, while "
  "&minus;t'(0) sets the cooling (or heating) rate that surface can deliver. "
  "The inclination result is practically useful: orienting a heat-transfer "
  "surface closer to the horizontal-buoyancy configuration (&alpha; &rarr; "
  "&pi;/2) lowers both the drag and the heat-transfer rate, whereas an "
  "assisting orientation (&alpha; &rarr; 0) raises them together &mdash; a "
  "trade-off that arises in inclined solar collectors and flat-plate "
  "absorbers, tilted electronic cold plates, inclined heat-exchanger channels, "
  "and in the drawing, extrusion and coating of sheets that leave the die at an "
  "angle. The dust phase represents the unavoidable particulate content of real "
  "coolants (nanoparticle agglomerates, combustion soot, sand or ash in "
  "geothermal and solar-thermal loops, powders in pneumatic conveying and "
  "fluidised beds); its lag behind the carrier fluid degrades heat transfer, "
  "which the model captures through &beta;<sub>v</sub> and &beta;<sub>T</sub>. "
  "Finally, the dual-solution structure is not a numerical curiosity: by the "
  "stability arguments of Merkin [6] and Weidman et&nbsp;al. [7] the first "
  "branch is the realizable operating state and the second branch, though "
  "mathematically valid, is unstable &mdash; so the critical value "
  "&lambda;<sub>c</sub> marks the largest shrinking rate (or smallest suction) "
  "at which a steady boundary layer can be sustained before boundary-layer "
  "separation/breakdown.", body)

# ---------------------------------------------------------------- 5. Conclusion
P('5. Conclusion', h1)
P("The inclined dusty hybrid-nanofluid boundary layer with velocity slip and "
  "wall suction has been solved by Lobatto collocation, and its dual-solution "
  "structure mapped as a function of the inclination angle. The main "
  "conclusions are:", body)
concl = [
 "For the shrinking sheet the problem has two solution branches that meet at a "
 "fold &lambda;<sub>c</sub>; the first (upper) branch is continuous from the "
 "stretching case &lambda; = 1 through the fold up to &lambda; = 0.",
 "The inclination angle controls the solution domain. |&lambda;<sub>c</sub>| "
 "decreases as &alpha; grows (0.478, 0.470, 0.451 for &alpha; = 0, &pi;/4, "
 "&pi;/2), so buoyancy widens the existence range.",
 "The second (lower) branch cannot be continued to &lambda; = 0 for the "
 "buoyant cases &alpha; = 0 and &pi;/4 (it terminates near &lambda; &approx; "
 "&minus;0.27 and &minus;0.23), but persists to &lambda; = 0 for "
 "&alpha; = &pi;/2, where the buoyancy term vanishes (cos&alpha; = 0).",
 "On the physical first branch, increasing &alpha; reduces both the "
 "skin-friction coefficient f''(0) and the heat-transfer rate &minus;t'(0), "
 "because the assisting buoyancy component (&prop; cos&alpha;) that steepens "
 "the wall gradients weakens as the surface tilts toward &pi;/2.",
 "The wall-quantity trends are consistent with the profiles: the first "
 "solution has thin layers and steep wall slopes (larger f''(0), &minus;t'(0)); "
 "the second solution has thick layers, deeper reversed flow and smaller wall "
 "gradients, and is the unstable branch.",
]
for c in concl:
    P('&bull;&nbsp; ' + c, body)
P("These results give the operating envelope (through &lambda;<sub>c</sub>) and "
  "the inclination sensitivity of drag and heat transfer for particle-laden "
  "hybrid-nanofluid coolants on inclined surfaces, and quantify how far the "
  "second solution can be followed for each orientation.", body)

# ---------------------------------------------------------------- References
story.append(PageBreak())
P('References', h1)
refs = [
 "Crane, L. J. (1970). Flow past a stretching plate. <i>Zeitschrift f&uuml;r "
 "angewandte Mathematik und Physik</i>, 21(4), 645&ndash;647.",
 "Miklavcic, M., &amp; Wang, C. Y. (2006). Viscous flow due to a "
 "shrinking sheet. <i>Quarterly of Applied Mathematics</i>, 64(2), 283&ndash;290.",
 "Wang, C. Y. (2008). Stagnation flow towards a shrinking sheet. "
 "<i>International Journal of Non-Linear Mechanics</i>, 43(5), 377&ndash;382.",
 "Bhattacharyya, K. (2011). Boundary layer flow and heat transfer over an "
 "exponentially shrinking sheet. <i>Chinese Physics Letters</i>, 28(7), 074701.",
 "Bachok, N., Ishak, A., &amp; Pop, I. (2012). Boundary layer stagnation-point "
 "flow toward a stretching/shrinking sheet in a nanofluid. <i>ASME Journal of "
 "Heat Transfer</i>, 134(2), 021003.",
 "Merkin, J. H. (1986). On dual solutions occurring in mixed convection in a "
 "porous medium. <i>Journal of Engineering Mathematics</i>, 20(2), 171&ndash;179.",
 "Weidman, P. D., Kubitschek, D. G., &amp; Davis, A. M. J. (2006). The effect "
 "of transpiration on self-similar boundary layer flow over moving surfaces. "
 "<i>International Journal of Engineering Science</i>, 44(11&ndash;12), 730&ndash;737.",
 "Ibrahim, W., &amp; Shankar, B. (2013). MHD boundary layer flow and heat "
 "transfer of a nanofluid past a permeable stretching sheet with velocity, "
 "thermal and solutal slip boundary conditions. <i>Computers &amp; Fluids</i>, "
 "75, 1&ndash;10.",
 "Hayat, T., Nadeem, S., &amp; Khan, A. U. (2018). Rotating flow of "
 "Ag&ndash;CuO/H<sub>2</sub>O hybrid nanofluid with radiation and partial slip. "
 "<i>The European Physical Journal E</i>, 41(6), 75.",
 "Saffman, P. G. (1962). On the stability of laminar flow of a dusty gas. "
 "<i>Journal of Fluid Mechanics</i>, 13(1), 120&ndash;128.",
 "Datta, N., &amp; Mishra, S. K. (1982). Boundary layer flow of a dusty fluid "
 "over a semi-infinite flat plate. <i>Acta Mechanica</i>, 42(1&ndash;2), 71&ndash;83.",
 "Gireesha, B. J., Ramesh, G. K., &amp; Bagewadi, C. S. (2012). Heat transfer "
 "in MHD flow of a dusty fluid over a stretching sheet. <i>Advances in Applied "
 "Science Research</i>, 3(4), 2392&ndash;2401.",
 "Choi, S. U. S. (1995). Enhancing thermal conductivity of fluids with "
 "nanoparticles. <i>ASME FED</i>, 231, 99&ndash;105.",
 "Buongiorno, J. (2006). Convective transport in nanofluids. <i>ASME Journal "
 "of Heat Transfer</i>, 128(3), 240&ndash;250.",
 "Tiwari, R. K., &amp; Das, M. K. (2007). Heat transfer augmentation in a "
 "two-sided lid-driven differentially heated square cavity utilizing "
 "nanofluids. <i>International Journal of Heat and Mass Transfer</i>, 50(9&ndash;"
 "10), 2002&ndash;2018.",
 "Devi, S. P. A., &amp; Devi, S. S. U. (2016). Numerical investigation of "
 "hydromagnetic hybrid Cu&ndash;Al<sub>2</sub>O<sub>3</sub>/water nanofluid flow over a "
 "permeable stretching sheet with suction. <i>International Journal of "
 "Nonlinear Sciences and Numerical Simulation</i>, 17(5), 249&ndash;257.",
 "Waini, I., Ishak, A., &amp; Pop, I. (2019). Hybrid nanofluid flow and heat "
 "transfer over a nonlinear permeable stretching/shrinking surface. "
 "<i>International Journal of Heat and Mass Transfer</i>, 136, 288&ndash;297.",
 "Anuar, N. S., Bachok, N., &amp; Pop, I. (2021). Numerical computation of "
 "dusty hybrid nanofluid flow and heat transfer over a deformable sheet with "
 "slip effect. <i>Mathematics</i>, 9(6), 643.",
]
for i, r in enumerate(refs, 1):
    P('[%d]&nbsp; %s' % (i, r), ref)

SP(8)
P("<i>Data files (this folder): task2_first_*.csv, task2_second_*.csv "
  "(wall quantities vs &lambda;, per branch and inclination); "
  "prof_first_*.csv, prof_second_*.csv (profiles at &lambda; = &minus;0.43); "
  "figures/*.png. Computed with the bvp4c/Lobatto collocation solver; all eight "
  "boundary conditions satisfied to &lt; 10<super>&minus;</super><super>5</super>.</i>", cap)

doc = SimpleDocTemplate(PDF, pagesize=A4, topMargin=1.6*cm, bottomMargin=1.6*cm,
                        leftMargin=2.0*cm, rightMargin=2.0*cm,
                        title='Dusty Hybrid Nanofluid Inclined BVP - Report v7 (PROMPT 5)')
doc.build(story)
print("wrote", PDF)
