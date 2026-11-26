"""
Step 22 - Main figures (Cell Press layout: 174 mm full width, Helvetica bold, panel letters bold capitals).

Reads only saved results; no decoding. Run after steps 16 (all regions and --compare), 19, 20 and 21.

    python 22_make_figures.py --regions CA3 CA1 DG ZI MOs VPM

Inputs
  processed_<REGION>/*/units.parquet, trials.parquet      unit positions, behaviour (Figure 1)
  results_<REGION>/prior_history_sessions.csv             step 16 (Figures 1, 2, 5)
  results_comparison/prior_history_unit_matched_*.csv     step 16 --compare (Figure 6)
  results_comparison/outcome_*.csv                        steps 19-21 (Figures 3-6)
Outputs (figures_paper/)
  Figure1.pdf ... Figure6.pdf (vector) and .png (600 dpi), figure_stats.txt (every number and test in the figures)

Statistics in the figures: cluster bootstrap over mice (10,000 resamples; mice drawn with replacement, all
their insertions kept), one-sided against 0 for single regions and two-sided for differences, Holm-corrected
within each panel. * p < 0.05, ** p < 0.01, *** p < 0.001.

Colours: hippocampal subfields in three distinct hues (Okabe-Ito colour-blind-safe set); comparison regions
in black and greys with different marker shapes, so that the hippocampus stands out and the figures print
legibly in greyscale.
"""
import argparse
import importlib
import json
import warnings

import numpy as np
import pandas as pd

import common as K
import config as C

warnings.filterwarnings('ignore')
m21 = importlib.import_module('21_final_checks')
ph = m21.ph

REGIONS = ('CA3', 'CA1', 'DG', 'ZI', 'MOs', 'VPM')
HIPPO = ('CA3', 'CA1', 'DG')
COL = {'CA3': '#D55E00', 'CA1': '#0072B2', 'DG': '#009E73', 'ZI': '#222222', 'MOs': '#6E6E6E', 'VPM': '#A8A8A8'}
MRK = {'CA3': 'o', 'CA1': 'o', 'DG': 'o', 'ZI': 's', 'MOs': '^', 'VPM': 'D'}
DOT = '#C8C8C8'                   # individual insertions
BLOCK_COL = {0.8: '#8C2D04', 0.2: '#225EA8'}
MM = 1 / 25.4
OUT = C.PROJECT_DIR / 'figures_paper'
RES = C.PROJECT_DIR / 'results_comparison'
STATS = []
FS, FS_TICK, FS_SMALL, FS_LETTER = 8, 7.5, 7, 11


# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    family = [f for f in ('Helvetica', 'Helvetica Neue', 'Arial', 'Liberation Sans', 'DejaVu Sans') if f in have]
    plt.rcParams.update({
        'font.family': family or ['sans-serif'], 'font.weight': 'bold', 'font.size': FS,
        'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'axes.labelsize': FS, 'axes.titlesize': FS,
        'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK, 'legend.fontsize': FS_SMALL,
        'axes.linewidth': 0.9, 'xtick.major.width': 0.9, 'ytick.major.width': 0.9,
        'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.major.pad': 2, 'ytick.major.pad': 2,
        'axes.labelpad': 3, 'lines.linewidth': 1.3, 'lines.markersize': 4.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.dpi': 150})
    return plt


def yticks(ax, n=3):
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10]))


def letter(ax, s, **_):
    """Panel letters are placed when the figure is saved: left edge of the panel (incl. y label), top of figure."""
    if not hasattr(ax.figure, '_letters'):
        ax.figure._letters = []
    ax.figure._letters.append((ax, s))


def place_letters(fig):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, s in getattr(fig, '_letters', []):
        bb = ax.get_tightbbox(rend).transformed(inv)
        fig.text(max(bb.x0 - 0.004, 0.002), 0.99, s, fontsize=FS_LETTER, fontweight='bold', va='top', ha='left')


def color_labels(ax, regs, anchor=(0.0, 1.03), sep=7):
    """Region names in their colours, in a row above the axes (instead of a legend box over the data)."""
    text_row(ax, [(r, COL[r]) for r in regs], anchor, sep)


def text_row(ax, items, anchor=(0.0, 1.03), sep=7):
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
    boxes = [TextArea(t, textprops=dict(color=c, fontsize=FS, fontweight='bold')) for t, c in items]
    ab = AnchoredOffsetbox(loc='lower left', child=HPacker(children=boxes, align='baseline', pad=0, sep=sep),
                           pad=0, borderpad=0, frameon=False, bbox_to_anchor=anchor, bbox_transform=ax.transAxes)
    ax.add_artist(ab)


def brackets(ax, items):
    """items: [(x0, x1, text)]. Extends the y axis upwards so that the brackets sit above all data, inside the
    axes; the left spine stops at the top tick within the data range."""
    items = [it for it in items if it[2]]
    if not items:
        return
    items = sorted(items, key=lambda t: abs(t[1] - t[0]))
    lo, hi = ax.get_ylim()
    ticks = [t for t in ax.get_yticks() if lo - 1e-9 <= t <= hi + 1e-9]
    f = 1 - 0.1 * len(items) - 0.04
    ax.set_ylim(lo, lo + (hi - lo) / f)
    ax.set_yticks(ticks)
    ax.spines['left'].set_bounds(lo, ticks[-1])
    for j, (x0, x1, text) in enumerate(items):
        bracket(ax, x0, x1, f + 0.03 + 0.1 * j, text)


def stars(p):
    if p is None or not np.isfinite(p):
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    place_letters(fig)
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png', dpi=600)


def region_axis(ax, regs, rotate=True):
    ax.set_xticks(range(len(regs)))
    ax.set_xticklabels(regs, rotation=90 if rotate else 0)
    ax.set_xlim(-0.6, len(regs) - 0.4)
    for lab, r in zip(ax.get_xticklabels(), regs):
        lab.set_color(COL[r] if r in HIPPO else 'k')


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def boot_one(df, col, reg, rng):
    b = m21.boot_region(df, col, [reg], rng)[reg]
    b = b[np.isfinite(b)]
    return (1 + (b <= 0).sum()) / (1 + len(b)), np.percentile(b, [2.5, 97.5])


def boot_diff(df, col, a, b, rng):
    bo = m21.boot_region(df, col, [a, b], rng)
    d = bo[a] - bo[b]
    d = d[np.isfinite(d)]
    p = min(1.0, 2 * min((1 + (d <= 0).sum()) / (1 + len(d)), (1 + (d >= 0).sum()) / (1 + len(d))))
    return p, np.percentile(d, [2.5, 97.5])


def test_regions(df, col, regs, rng, label):
    """One-sided bootstrap > 0 for each region, Holm-corrected; logs to STATS; returns {region: Holm p}."""
    regs = [r for r in regs if (df['region'] == r).any() and df.loc[df['region'] == r, col].notna().sum() >= 5]
    raw = {r: boot_one(df, col, r, rng) for r in regs}
    adj = dict(zip(regs, m21.holm([raw[r][0] for r in regs]))) if regs else {}
    for r in regs:
        g = df[df['region'] == r].dropna(subset=[col])
        STATS.append(f'{label} | {r}: mean {g[col].mean():+.4f}, 95% CI [{raw[r][1][0]:+.4f}, {raw[r][1][1]:+.4f}], '
                     f'{len(g)} insertions, {g["subject"].nunique()} mice; > 0 p={raw[r][0]:.3g}, Holm p={adj[r]:.3g}')
    return adj


def test_pairs(df, col, pairs, rng, label):
    res = [boot_diff(df, col, a, b, rng) for a, b in pairs]
    adj = m21.holm([r[0] for r in res]) if res else []
    out = {}
    for (a, b), (p, ci), pa in zip(pairs, res, adj):
        ma, mb = df.loc[df['region'] == a, col].mean(), df.loc[df['region'] == b, col].mean()
        STATS.append(f'{label} | {a} - {b}: {ma - mb:+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}], p={p:.3g}, Holm p={pa:.3g}')
        out[(a, b)] = pa
    return out


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def dots(ax, df, col, regs, rng, ylabel, ptab=None, chance=0.0, ylim=None):
    """Insertions as light grey dots, mean +/- s.e.m. in the region colour, stars above the axis area."""
    for i, r in enumerate(regs):
        v = df.loc[df['region'] == r, col].dropna().to_numpy()
        if len(v) == 0:
            continue
        x = i - 0.12 + rng.uniform(-0.14, 0.14, len(v))
        ax.scatter(x, v, s=3, color=DOT, lw=0, zorder=1, rasterized=True)
        ax.errorbar(i + 0.24, v.mean(), K.sem(v), fmt=MRK[r], ms=4.5, color=COL[r], mec=COL[r],
                    elinewidth=1.3, capsize=0, zorder=3)
    ax.axhline(chance, color='0.35', lw=0.7, ls=(0, (3, 2)), zorder=0)
    region_axis(ax, regs)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    yticks(ax)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ptab:
        for i, r in enumerate(regs):
            if r in ptab and stars(ptab[r]):
                ax.text(i, 1.0, stars(ptab[r]), transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                        fontsize=FS)


def bracket(ax, x0, x1, y, text):
    """Bracket at height y given in axes fraction (0-1)."""
    tr = ax.get_xaxis_transform()
    ax.plot([x0, x0, x1, x1], [y - 0.025, y, y, y - 0.025], color='k', lw=0.8, transform=tr, clip_on=False)
    ax.text((x0 + x1) / 2, y - 0.012, text, ha='center', va='bottom', fontsize=FS, transform=tr)


def read_sessions(regs):
    out = []
    for r in regs:
        f = C.PROJECT_DIR / f'results_{r}' / 'prior_history_sessions.csv'
        if f.exists():
            out.append(pd.read_csv(f).assign(region=r))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def corrected(df, key):
    return df[key] - df[f'{key}_null']


def region_legend(ax, regs, **kw):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[r], marker=MRK[r], ms=4, lw=1.3) for r in regs]
    return ax.legend(h, regs, handlelength=1.4, handletextpad=0.4, columnspacing=0.8, borderaxespad=0, **kw)


# ----------------------------------------------------------------------------
# Figure 1: task, recordings, behaviour
# ----------------------------------------------------------------------------
def figure1(plt, regs, sess, rng):
    fig = plt.figure(figsize=(174 * MM, 68 * MM))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.5, 1.15, 1.05, 1.05, 0.85], wspace=0.7, left=0.015, right=0.99,
                          bottom=0.2, top=0.84)
    # A: trial structure and analysis windows (from feedback of trial n-1 to the stimulus of trial n)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(2.4, 10.4)
    ax.set_ylim(0, 7.0)
    ax.axis('off')
    y = 3.6
    ax.plot([2.8, 10.0], [y, y], color='k', lw=1.0)
    for x, lab in ((4.0, 'Feedback\ntrial n\u22121'), (9.0, 'Stimulus\ntrial n')):
        ax.plot([x, x], [y - 0.25, y + 0.25], color='k', lw=1.3)
        ax.text(x, y + 0.5, lab, ha='center', va='bottom', fontsize=FS_SMALL, linespacing=1.1)
    ax.annotate('', xy=(8.9, y + 2.35), xytext=(4.1, y + 2.35), arrowprops=dict(arrowstyle='<->', lw=0.9, color='0.3'))
    ax.text(6.5, y + 2.5, 'Inter-trial interval', ha='center', va='bottom', fontsize=FS_SMALL, color='0.3')
    ax.add_patch(plt.Rectangle((4.1, y - 1.05), 1.3, 0.5, color='0.65', lw=0))
    ax.text(4.75, y - 1.3, 'After\nfeedback', ha='center', va='top', fontsize=FS_SMALL, linespacing=1.1)
    ax.add_patch(plt.Rectangle((7.9, y - 1.05), 1.0, 0.5, color=COL['CA1'], lw=0))
    ax.text(8.4, y - 1.3, 'Before\nstimulus', ha='center', va='top', fontsize=FS_SMALL, linespacing=1.1)
    letter(ax, 'A')

    # B: unit positions, sagittal projection (hippocampus)
    ax = fig.add_subplot(gs[1])
    have = False
    for r in HIPPO:
        pdir = C.PROJECT_DIR / f'processed_{r}'
        if not pdir.exists():
            continue
        ys, zs = [], []
        for p in pdir.iterdir():
            f = p / 'units.parquet'
            if f.exists():
                u = pd.read_parquet(f)
                if {'y', 'z'} <= set(u.columns):
                    ys.append(u['y'].to_numpy())
                    zs.append(u['z'].to_numpy())
        if ys:
            have = True
            ax.scatter(np.concatenate(ys) * 1e3, np.concatenate(zs) * 1e3, s=1.2, color=COL[r], lw=0, alpha=0.7,
                       rasterized=True)
    if have:
        ax.set_xlabel('AP from bregma (mm)')
        ax.set_ylabel('DV (mm)')
        ax.set_aspect('equal', adjustable='datalim')
        yticks(ax)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        color_labels(ax, HIPPO, sep=5)
    else:
        ax.text(0.5, 0.5, 'unit coordinates\nnot available', ha='center', va='center', transform=ax.transAxes)
    letter(ax, 'B', dx=-0.42)

    # C: data per region
    ax = fig.add_subplot(gs[2])
    rows = []
    for r in regs:
        g = sess[sess['region'] == r]
        if len(g):
            rows.append((r, int(g['n_units'].sum()), len(g), g['subject'].nunique()))
    for i, (r, nu, ni, nm) in enumerate(rows):
        ax.bar(i, nu, color=COL[r], width=0.72)
        ax.text(i, nu, f' {ni}/{nm}', ha='center', va='bottom', fontsize=FS_SMALL, rotation=90)
        STATS.append(f'Fig1C | {r}: {nu} units, {ni} insertions, {nm} mice (>= 5 units per insertion)')
    region_axis(ax, [r[0] for r in rows])
    ax.set_ylabel('Units')
    ax.set_ylim(0, max([r[1] for r in rows] + [1]) * 1.5)
    yticks(ax, 3)
    ax.text(0.0, 1.03, 'Insertions/mice', transform=ax.transAxes, ha='left', va='bottom', fontsize=FS_SMALL)
    letter(ax, 'C', dx=-0.48)

    # D, E: behaviour of the recorded sessions (hippocampal recordings; each session counted once)
    beh = load_behaviour()
    ax = fig.add_subplot(gs[3])
    if len(beh['psy']):
        psy = beh['psy']
        for blk, lab in ((0.8, 'P(left) 0.8'), (0.2, 'P(left) 0.2')):
            g = psy[psy['block'] == blk].groupby('contrast')['p_right']
            m, e = g.mean(), g.apply(K.sem)
            ax.errorbar(m.index * 100, m.values, e.values, color=BLOCK_COL[blk], marker='o', ms=3, lw=1.2,
                        capsize=0, label=lab)
        ax.set_xlabel('Contrast (%)')
        ax.set_ylabel('P(rightward choice)')
        ax.set_ylim(0, 1)
        ax.set_xticks([-100, 0, 100])
        ax.set_yticks([0, 0.5, 1])
        text_row(ax, [('P(left)', 'k'), ('0.8', BLOCK_COL[0.8]), ('0.2', BLOCK_COL[0.2])])
        STATS.append(f'Fig1D | psychometric curves from {psy["eid"].nunique()} sessions')
    letter(ax, 'D', dx=-0.42)
    ax = fig.add_subplot(gs[4])
    if len(beh['rep']):
        rep = beh['rep']
        for _, r in rep.iterrows():
            ax.plot([0, 1], [r['after_reward'], r['after_error']], color=DOT, lw=0.4)
        ax.errorbar([0, 1], [rep['after_reward'].mean(), rep['after_error'].mean()],
                    [K.sem(rep['after_reward']), K.sem(rep['after_error'])], color='k', marker='o', ms=4.5,
                    lw=1.3, capsize=0)
        from scipy import stats
        p = stats.wilcoxon(rep['after_reward'], rep['after_error']).pvalue
        STATS.append(f'Fig1E | P(repeat) after reward {rep["after_reward"].mean():.3f} vs after error '
                     f'{rep["after_error"].mean():.3f}, {len(rep)} sessions, Wilcoxon p={p:.3g}')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Reward', 'Error'])
        ax.set_xlabel('Previous trial')
        ax.set_ylabel('P(repeat choice)')
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(0.2, 1.0)
        ax.set_yticks([0.2, 0.6, 1.0])
        ax.text(0.5, 1.0, stars(p), transform=ax.transAxes, ha='center', va='bottom', fontsize=FS)
    letter(ax, 'E', dx=-0.62)
    save(fig, 'Figure1')
    plt.close(fig)


def load_behaviour():
    seen, psy, rep = set(), [], []
    for r in HIPPO:
        pdir = C.PROJECT_DIR / f'processed_{r}'
        if not pdir.exists():
            continue
        for p in sorted(pdir.iterdir()):
            if not (p / 'meta.json').exists():
                continue
            meta = json.loads((p / 'meta.json').read_text())
            eid = meta.get('eid', meta.get('pid'))
            if eid in seen:
                continue
            seen.add(eid)
            t = ph.rev.add_history(pd.read_parquet(p / 'trials.parquet'))
            t = t[t['choice'] != 0]
            rc = np.sign(t.loc[t['signed_contrast'] > 0, 'choice'].mean())
            if not np.isfinite(rc) or rc == 0:
                continue
            right = (t['choice'] == rc).astype(float)
            for blk in (0.2, 0.8):
                g = t[np.isclose(t['probabilityLeft'], blk)]
                for c, gg in g.groupby('signed_contrast'):
                    psy.append(dict(eid=eid, block=blk, contrast=c, p_right=right[gg.index].mean()))
            h = t[t['prev_choice'] != 0]
            stay = (h['choice'] == h['prev_choice'])
            rep.append(dict(eid=eid, after_reward=stay[h['prev_reward'] == 1].mean(),
                            after_error=stay[h['prev_reward'] == -1].mean()))
    return {'psy': pd.DataFrame(psy), 'rep': pd.DataFrame(rep).dropna()}


# ----------------------------------------------------------------------------
# Figure 2: block prior and choice history
# ----------------------------------------------------------------------------
def figure2(plt, regs, sess, rng):
    fig, axs = plt.subplots(1, 4, figsize=(174 * MM, 60 * MM), gridspec_kw=dict(wspace=0.62))
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.17, top=0.86)
    v = np.concatenate([corrected(sess[sess['region'].isin(regs)], k).dropna().to_numpy()
                        for k in ('block', 'block_prev', 'prev_choice')])
    lim = np.max(np.abs(v)) * 1.08 if len(v) else 0.1
    panels = (('block', 'Block side\n(acc. \u2212 null)', 'A', 'Fig2A block side, raw', (-lim, lim)),
              ('block_prev', 'Block side, previous\nchoice balanced', 'B',
               'Fig2B block side | previous choice balanced, raw', (-lim, lim)),
              ('prev_choice', 'Previous choice\n(acc. \u2212 null)', 'C', 'Fig2C previous choice, raw', (-lim, lim)),
              ('prev_reward_timed', 'Previous outcome\n(acc. \u2212 null)', 'D', 'Fig2D previous outcome, raw',
               None))
    for ax, (key, lab, L, slab, ylim) in zip(axs, panels):
        d = sess.assign(v=corrected(sess, key))
        ptab = test_regions(d, 'v', regs, rng, slab)
        dots(ax, d, 'v', regs, rng, lab, ptab, ylim=ylim)
        letter(ax, L, dx=-0.55)
    save(fig, 'Figure2')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: outcome information after feedback and before the next stimulus (hippocampus)
# ----------------------------------------------------------------------------
def figure3(plt, regs, rng):
    tc = pd.read_csv(RES / 'outcome_timecourse.csv') if (RES / 'outcome_timecourse.csv').exists() else None
    per = pd.read_csv(RES / 'outcome_persistence.csv') if (RES / 'outcome_persistence.csv').exists() else None
    hip = [r for r in regs if r in HIPPO]
    fig = plt.figure(figsize=(174 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.0, 0.95, 0.95, 0.95], wspace=0.62, left=0.085, right=0.99,
                          bottom=0.22, top=0.86)
    if tc is not None:
        tc['d'] = tc['acc'] - tc['null']
        sub = gs[0].subgridspec(1, 2, width_ratios=[15, 7], wspace=0.1)
        a1, a2 = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        for ax, align in ((a1, 'feedback'), (a2, 'stim')):
            ax.axvspan(*((0.05, 0.5) if align == 'feedback' else (-0.5, -0.1)), color='0.92', zorder=0, lw=0)
            for r in hip:
                g = tc[(tc['align'] == align) & (tc['region'] == r)]
                if g.empty:
                    continue
                m = g.groupby('t_center')['d'].mean()
                e = g.groupby('t_center')['d'].apply(K.sem)
                ax.fill_between(m.index, m - e, m + e, color=COL[r], alpha=0.2, lw=0)
                ax.plot(m.index, m.values, color=COL[r], lw=1.4, label=r)
            ax.axhline(0, color='0.35', lw=0.7, ls=(0, (3, 2)))
            ax.axvline(0, color='0.35', lw=0.7)
        a1.set_xlabel('From feedback\ntrial n\u22121 (s)')
        a2.set_xlabel('To stimulus\ntrial n (s)')
        a1.set_ylabel('Previous outcome\n(acc. \u2212 null)')
        lo = min(a1.get_ylim()[0], a2.get_ylim()[0])
        hi = max(a1.get_ylim()[1], a2.get_ylim()[1])
        for ax in (a1, a2):
            ax.set_ylim(lo, hi)
            yticks(ax)
        a2.set_yticklabels([])
        a1.set_xticks([0, 0.4])
        a1.set_xlim(-0.3, 0.5)
        a2.set_xlim(-0.5, 0.02)
        a2.set_xticks([-0.4, 0])
        color_labels(a1, hip)
        letter(a1, 'A', dx=-0.42)
    if per is not None:
        per = per[per['region'].isin(hip)]
        ax = fig.add_subplot(gs[1])
        for i, r in enumerate(hip):
            g = per[per['region'] == r]
            off = (i - 1) * 0.1
            m = [g['post_feedback'].mean(), g['pre_stimulus'].mean()]
            e = [K.sem(g['post_feedback']), K.sem(g['pre_stimulus'])]
            ax.errorbar(np.array([0, 1]) + off, m, e, color=COL[r], marker='o', ms=4, lw=1.3, capsize=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['After\nfeedback', 'Before\nstimulus'])
        ax.set_xlim(-0.35, 1.35)
        ax.axhline(0, color='0.35', lw=0.7, ls=(0, (3, 2)))
        ax.set_ylabel('Previous outcome\n(acc. \u2212 null)')
        yticks(ax)
        test_pairs(per, 'post_feedback', [('CA3', 'CA1'), ('CA3', 'DG'), ('CA1', 'DG')], rng,
                   'Fig3B after feedback, between subfields')
        pp = test_pairs(per, 'pre_stimulus', [('CA3', 'CA1'), ('CA3', 'DG'), ('CA1', 'DG')], rng,
                        'Fig3B before next stimulus, between subfields')
        letter(ax, 'B', dx=-0.62)
        ax = fig.add_subplot(gs[2])
        ptab = test_regions(per, 'pre_stimulus', hip, rng, 'Fig3C before next stimulus > 0')
        dots(ax, per, 'pre_stimulus', hip, rng, 'Before stimulus\n(acc. \u2212 null)', None)
        if ('CA1', 'DG') in pp:
            brackets(ax, [(hip.index('CA1'), hip.index('DG'), stars(pp[('CA1', 'DG')]))])
        letter(ax, 'C', dx=-0.62)
        ax = fig.add_subplot(gs[3])
        per = per.assign(drop=per['post_feedback'] - per['pre_stimulus'])
        pp = test_pairs(per, 'drop', [('CA3', 'CA1'), ('CA3', 'DG'), ('CA1', 'DG')], rng,
                        'Fig3D loss from feedback to next stimulus, between subfields (interaction)')
        for r in hip:
            g = per[per['region'] == r]
            STATS.append(f'Fig3D | {r}: loss {g["drop"].mean():+.4f} +/- {K.sem(g["drop"]):.4f}')
        dots(ax, per, 'drop', hip, rng, 'Loss, feedback to\nnext stimulus', None)
        if ('CA1', 'DG') in pp:
            brackets(ax, [(hip.index('CA1'), hip.index('DG'), stars(pp[('CA1', 'DG')]) or 'n.s.')])
        letter(ax, 'D', dx=-0.62)
        for a in fig.axes[-2:]:
            a.set_xticklabels(hip, rotation=0)
    save(fig, 'Figure3')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 4: memory of the previous outcome, not a slow state
# ----------------------------------------------------------------------------
def figure4(plt, regs, rng):
    lags = pd.read_csv(RES / 'outcome_lags.csv') if (RES / 'outcome_lags.csv').exists() else None
    xt = pd.read_csv(RES / 'outcome_crosstime.csv') if (RES / 'outcome_crosstime.csv').exists() else None
    lag2 = pd.read_csv(RES / 'outcome_lag2.csv') if (RES / 'outcome_lag2.csv').exists() else None
    fig = plt.figure(figsize=(174 * MM, 64 * MM))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 1.1, 1.45, 1.1], wspace=0.62, left=0.085, right=0.99,
                          bottom=0.2, top=0.78)
    if lags is not None:
        d = lags[lags['variant'] == 'me'].assign(c=lambda x: x['acc'] - x['null'])
        ax = fig.add_subplot(gs[0])
        lagv = sorted(d['lag'].unique())
        for k, r in enumerate(regs):
            g = d[d['region'] == r]
            if g.empty:
                continue
            m, e = g.groupby('lag')['c'].mean(), g.groupby('lag')['c'].apply(K.sem)
            x = np.array([lagv.index(v) for v in m.index]) + (k - (len(regs) - 1) / 2) * 0.06
            ax.errorbar(x, m.values, e.values, color=COL[r], marker=MRK[r], ms=3.5, lw=1.2, capsize=0,
                        mfc=COL[r] if r in HIPPO else 'white', mew=1.0)
        ax.axhline(0, color='0.35', lw=0.7, ls=(0, (3, 2)))
        ax.set_xticks(range(len(lagv)))
        ax.set_xticklabels([f'n{v:+d}'.replace('-', '\u2212') for v in lagv])
        ax.set_xlabel('Outcome of trial')
        ax.set_ylabel('Decoded from trial n\n(acc. \u2212 null)')
        yticks(ax)
        region_legend(ax, regs, ncol=3, loc='lower left', bbox_to_anchor=(0.0, 1.05))
        letter(ax, 'A', dx=-0.42, dy=1.32)
        w = d.pivot_table(index=['region', 'pid', 'subject'], columns='lag', values='c').reset_index()
        w['memory'] = w[-1] - w[1]
        ax = fig.add_subplot(gs[1])
        ptab = test_regions(w, 'memory', regs, rng, 'Fig4B memory index (n-1 minus n+1) > 0')
        test_pairs(w, 'memory', [(a, b) for i, a in enumerate(regs) for b in regs[i + 1:]], rng,
                   'Fig4B memory index, between regions')
        for r in regs:
            for lag in lagv:
                STATS.append(f'Fig4A | {r} lag {lag:+d}: {d[(d["region"] == r) & (d["lag"] == lag)]["c"].mean():+.4f}')
        dots(ax, w, 'memory', regs, rng, 'Memory index\n(n\u22121 minus n+1)', ptab)
        letter(ax, 'B', dx=-0.55, dy=1.32)
    if xt is not None:
        ax = fig.add_subplot(gs[2])
        combos = (('fb_fb', 'F\u2192F'), ('fb_stim', 'F\u2192P'), ('stim_fb', 'P\u2192F'), ('stim_stim', 'P\u2192P'))
        hip = [r for r in regs if r in HIPPO and (xt['region'] == r).any()]
        width = 0.8 / max(len(hip), 1)
        for k, r in enumerate(hip):
            g = xt[xt['region'] == r]
            vals = [g[c] - g[f'{c}_null'] for c, _ in combos]
            x = np.arange(len(combos)) + (k - (len(hip) - 1) / 2) * width
            ax.bar(x, [v.mean() for v in vals], width * 0.9, yerr=[K.sem(v) for v in vals], color=COL[r],
                   error_kw=dict(lw=1.0, capsize=0), label=r)
            dd = g.assign(**{c: g[c] - g[f'{c}_null'] for c, _ in combos})
            for c, lab in combos:
                test_regions(dd.assign(region=r), c, [r], rng, f'Fig4C cross-time {lab} > 0')
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels([c[1] for c in combos], fontsize=FS_SMALL)
        ax.set_xlabel('Train \u2192 test window')
        ax.set_ylabel('Previous outcome\n(acc. \u2212 null)')
        yticks(ax)
        color_labels(ax, hip)
        letter(ax, 'C', dx=-0.45, dy=1.32)
    if lag2 is not None:
        d = lag2[lag2['variant'] == 'me'].assign(c=lambda x: x['acc'] - x['null'])
        w = d.pivot_table(index=['region', 'pid', 'subject'], columns='lag', values='c').reset_index()
        if -2 in w and 2 in w:
            w['m2'] = w[-2] - w[2]
            ax = fig.add_subplot(gs[3])
            ptab = test_regions(w, 'm2', regs, rng, 'Fig4D n-2 minus n+2 (n-1 / n+1 balanced) > 0')
            test_regions(w.assign(n2=w[-2]), 'n2', regs, rng, 'Fig4D n-2 alone (n-1 balanced) > 0')
            dots(ax, w, 'm2', regs, rng, 'n\u22122 minus n+2\n(n\u00b11 balanced)', ptab)
            letter(ax, 'D', dx=-0.55, dy=1.32)
    save(fig, 'Figure4')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 5: movement, licking and pupil
# ----------------------------------------------------------------------------
def figure5(plt, regs, sess, rng):
    pup = pd.read_csv(RES / 'outcome_pupil.csv') if (RES / 'outcome_pupil.csv').exists() else None
    fig = plt.figure(figsize=(174 * MM, 64 * MM))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.5, 1.0, 1.0], wspace=0.62, left=0.085, right=0.99,
                          bottom=0.2, top=0.78)
    ax = fig.add_subplot(gs[0])
    if 'pupil_only_prev_reward_timed' in sess:
        d = sess.assign(v=sess['pupil_only_prev_reward_timed'] - 0.5)
        ptab = test_regions(d, 'v', regs, rng, 'Fig5A pupil alone, previous outcome (acc. - 0.5) > 0')
        dots(ax, d.assign(v=d['v'] + 0.5), 'v', regs, rng, 'Pupil alone\n(balanced accuracy)', ptab,
             chance=0.5)
    letter(ax, 'A', dx=-0.6, dy=1.32)
    ax = fig.add_subplot(gs[1])
    keys = ('prev_reward_timed', 'me_prev_reward_timed', 'state_prev_reward_timed')
    if all(k in sess and f'{k}_null' in sess for k in keys):
        from scipy import stats
        d = sess.assign(**{f'c{i}': corrected(sess, k) for i, k in enumerate(keys)}).dropna(subset=['c2'])
        for i, r in enumerate(regs):
            g = d[d['region'] == r]
            if g.empty:
                continue
            x = i + np.array([-0.25, 0, 0.25])
            m = [g[f'c{j}'].mean() for j in range(3)]
            e = [K.sem(g[f'c{j}']) for j in range(3)]
            ax.plot(x, m, color=COL[r], lw=1.2)
            ax.errorbar(x, m, e, fmt='none', ecolor=COL[r], elinewidth=1.1, capsize=0)
            for xx, mm, mk in zip(x, m, ('o', 's', 'D')):
                ax.plot(xx, mm, marker=mk, ms=3.8, color=COL[r], mfc=COL[r] if mk == 'o' else 'white', mew=1.1)
            p02 = stats.wilcoxon(g['c0'], g['c2']).pvalue if len(g) >= 5 else np.nan
            STATS.append(f'Fig5B | {r}: same {len(g)} insertions, raw {m[0]:+.4f}, +movement/licks {m[1]:+.4f}, '
                         f'+pupil {m[2]:+.4f}; raw vs +pupil Wilcoxon p={p02:.3g}')
        ax.axhline(0, color='0.35', lw=0.7, ls=(0, (3, 2)))
        region_axis(ax, regs, rotate=True)
        ax.set_ylabel('Previous outcome\n(acc. \u2212 null)')
        yticks(ax)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -0.08 * hi), hi)
        yticks(ax)
        ax.plot([], [], 'o', color='k', ms=3.8, label='Raw')
        ax.plot([], [], 's', color='k', mfc='white', ms=3.8, mew=1.1, label='Movement, licks regressed')
        ax.plot([], [], 'D', color='k', mfc='white', ms=3.8, mew=1.1, label='+ Pupil regressed')
        ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.03), handlelength=0.8, borderaxespad=0, ncol=1)
    letter(ax, 'B', dx=-0.38, dy=1.32)
    if pup is not None:
        ax = fig.add_subplot(gs[2])
        d = pup.assign(v=pup['strat'] - pup['strat_null'])
        ptab = test_regions(d, 'v', regs, rng, 'Fig5C pupil balanced by stratification > 0')
        dots(ax, d, 'v', regs, rng, 'Pupil balanced\n(acc. \u2212 null)', ptab)
        letter(ax, 'C', dx=-0.6, dy=1.32)
        ax = fig.add_subplot(gs[3])
        d = pup.assign(v=pup['gain'] - pup['gain_null'])
        ptab = test_regions(d, 'v', regs, rng, 'Fig5D gain of pupil + population over pupil (shift-corrected) > 0')
        for r in regs:
            g = pup[pup['region'] == r]
            if len(g):
                STATS.append(f'Fig5D | {r}: pupil alone {g["acc_pupil"].mean():.3f}, pupil + population '
                             f'{g["acc_pupil_pop"].mean():.3f}, raw gain {g["gain"].mean():+.4f}, shift-null gain '
                             f'{g["gain_null"].mean():+.4f}')
        dots(ax, d, 'v', regs, rng, 'Gain over pupil\n(acc. \u2212 null)', ptab)
        letter(ax, 'D', dx=-0.6, dy=1.32)
    save(fig, 'Figure5')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 6: brain-wide context and behaviour
# ----------------------------------------------------------------------------
def figure6(plt, regs, rng):
    um = sorted(RES.glob('prior_history_unit_matched_*.csv'))
    rate = pd.read_csv(RES / 'outcome_rate.csv') if (RES / 'outcome_rate.csv').exists() else None
    beh = pd.read_csv(RES / 'outcome_behaviour.csv') if (RES / 'outcome_behaviour.csv').exists() else None
    fig, axs = plt.subplots(1, 3, figsize=(174 * MM, 64 * MM), gridspec_kw=dict(wspace=0.55))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.2, top=0.9)
    pairs = [(a, b) for i, a in enumerate(regs) for b in regs[i + 1:]]

    def dg_brackets(ax, pp):
        items = []
        for b in ('CA1', 'CA3', 'ZI', 'MOs', 'VPM'):
            if b not in regs or 'DG' not in regs:
                continue
            p = pp.get((b, 'DG'), pp.get(('DG', b)))
            if p is not None and stars(p):
                items.append((regs.index('DG'), regs.index(b), stars(p)))
        brackets(ax, items)

    if um:
        d = pd.read_csv(um[-1])
        d = d[d['N'] == 5].assign(v=lambda x: x['prev_reward_ctrl'] - 0.5)
        pp = test_pairs(d, 'v', pairs, rng, 'Fig6A 5 units, controlled, between regions')
        dots(axs[0], d.assign(v=d['v'] + 0.5), 'v', regs, rng, 'Previous outcome,\n5 units (bal. acc.)', None,
             chance=0.5)
        dg_brackets(axs[0], pp)
    letter(axs[0], 'A', dx=-0.42, dy=1.25)
    if rate is not None:
        pp = test_pairs(rate, 'acc_hi', pairs, rng, 'Fig6B units >= 2 Hz, 5 units, between regions')
        for r in regs:
            g = rate[rate['region'] == r]
            if len(g):
                STATS.append(f'Fig6B | {r}: median unit rate {g["median_rate"].median():.2f} Hz; units >= 2 Hz '
                             f'{g["n_units_hi"].sum()}/{g["n_units"].sum()}')
        dots(axs[1], rate, 'acc_hi', regs, rng, 'Units \u2265 2 Hz,\n5 units (bal. acc.)', None, chance=0.5)
        dg_brackets(axs[1], pp)
    letter(axs[1], 'B', dx=-0.42, dy=1.25)
    if beh is not None:
        d = beh[(beh['cls'] == 'after_error') & (beh['outcome'] == 'log_rt')].assign(
            v=lambda x: x['r_score'] - x['r_null'])
        res = {}
        for r in regs:
            g = d[d['region'] == r].dropna(subset=['v'])
            if len(g) >= 5:
                p1, _ = boot_one(g, 'v', r, rng)
                p2, _ = boot_one(g.assign(v=-g['v']), 'v', r, rng)
                res[r] = min(1.0, 2 * min(p1, p2))
        adj = dict(zip(res, m21.holm(list(res.values())))) if res else {}
        for r in res:
            g = d[d['region'] == r]
            STATS.append(f'Fig6C | {r}: r(score, log RT | after error) - shift null {g["v"].mean():+.4f}, '
                         f'{len(g)} insertions, two-sided bootstrap p={res[r]:.3g}, Holm p={adj[r]:.3g}')
        dots(axs[2], d, 'v', regs, rng, 'Outcome score vs next\nRT after errors (r \u2212 null)', adj)
    letter(axs[2], 'C', dx=-0.42, dy=1.25)
    save(fig, 'Figure6')
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=list(REGIONS))
    ap.add_argument('--figures', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6])
    args = ap.parse_args()
    regs = [r for r in args.regions if (C.PROJECT_DIR / f'results_{r}' / 'prior_history_sessions.csv').exists()]
    plt = setup()
    rng = np.random.default_rng(C.RANDOM_SEED + 22)
    sess = read_sessions(regs)
    jitter = np.random.default_rng(0)
    if 1 in args.figures:
        figure1(plt, regs, sess, jitter)
    if 2 in args.figures:
        figure2(plt, regs, sess, rng)
    if 3 in args.figures:
        figure3(plt, regs, rng)
    if 4 in args.figures:
        figure4(plt, regs, rng)
    if 5 in args.figures:
        figure5(plt, regs, sess, rng)
    if 6 in args.figures:
        figure6(plt, regs, rng)
    OUT.mkdir(exist_ok=True)
    (OUT / 'figure_stats.txt').write_text('\n'.join(STATS), encoding='utf-8')
    print('\n'.join(STATS))
    print(f'figures written to {OUT}')


if __name__ == '__main__':
    main()
