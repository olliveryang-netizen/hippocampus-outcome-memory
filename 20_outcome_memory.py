"""
Step 20 - Is the pre-stimulus outcome signal a memory of the last trial, or a slow state?

Errors cluster in time (lapses of engagement), so activity that drifts with engagement can predict an
outcome without remembering it. In step 19 previous-outcome information was already above chance 250 ms
BEFORE the previous feedback, i.e. before the outcome was known. Three tests separate memory from state.
Run after step 19 (uses its CSV files and its loader).

  lags        Outcomes of trials n-2, n-1, n+1 and n+2 decoded from the same pre-stimulus window of trial n
              (-400 to -100 ms), each balanced within block side x choice of that same trial x tercile of
              the time since the feedback before trial n, raw and with movement and licks regressed out. A slow state predicts past and future outcomes equally;
              a memory predicts only the past. Test: lag -1 vs +1 and lag -2 vs +2 (paired, insertions and
              mice). Imposter null for every lag (keeps the slow structure of real label sequences, so
              chance alignment of two slow drifts is not counted as information).
  crosstime   Hippocampal insertions (feedback-aligned counts): a decoder trained 100-400 ms after the
              previous feedback is tested on the pre-stimulus window of the next trial, and the reverse.
              Above-null transfer means the pattern evoked by the outcome is still present before the next
              stimulus. Each window is z-scored within the session (no labels used). Imposter null.
              Transfer shows a shared population pattern; it does not by itself separate memory from a slow
              state (a state that also shapes the feedback response would transfer too) - that is test A.
  persistence From step 19's time course (no decoding): per insertion, outcome information before the
              previous feedback (baseline, -300 to -100 ms), after it (+50 to +500 ms) and before the next
              stimulus (-500 to -100 ms). Tests: pre-stimulus > baseline in each subfield, and subfield
              differences in the post-feedback level and in the pre-stimulus excess over baseline.
  sessions    Step 19's session-level behaviour table rebuilt with insertions matched by region (fixes a
              join in the first version of step 19 that could pair an insertion with the decoding accuracy
              of another region on the same probe). No decoding.

    python 20_outcome_memory.py --regions CA3 CA1 DG ZI MOs VPM
Outputs (results_comparison/): summary_20_outcome_memory.txt, outcome_lags.csv, outcome_crosstime.csv,
outcome_lags.png/.pdf
"""
import argparse
import importlib
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

import common as K
import config as C

warnings.filterwarnings('ignore')
fu = importlib.import_module('19_outcome_followup')
ph, dec = fu.ph, fu.dec

REGIONS = fu.REGIONS
PARTS = ('lags', 'crosstime', 'persistence', 'sessions')
LAGS = fu.LAGS
N_SUB, N_SUB_NULL, N_IMP = 10, 2, 30
N_IMP_XT = 20
FB_TRAIN = (0.10, 0.40)        # window after the previous feedback (s)
OUT = fu.OUT


# ----------------------------------------------------------------------------
def lag_targets(t, lag):
    """Outcome of trial n + lag, balanced within block side x choice of trial n + lag x tercile of the time
    since the feedback before trial n (the same balancing for every lag, so past and future are comparable;
    the time term removes the longer inter-trial interval after errors)."""
    y = t[f'out_{lag:+d}'].to_numpy()
    c = t[f'cho_{lag:+d}'].to_numpy()
    block = np.sign(0.5 - t['probabilityLeft'].to_numpy())
    gap = t['since_prev_fb'].to_numpy(float)
    terc = np.full(len(t), -1)
    fin = np.isfinite(gap)
    if fin.sum() > 30:
        terc[fin] = pd.qcut(gap[fin], 3, labels=False, duplicates='drop')
    ok = (c != 0) & (y != 0) & (terc >= 0)
    return np.where(ok, y, 0), (((block + 1) * 3 + (c + 1)) * 10 + terc + 1).astype(int)


def zscore_cols(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def window(counts, t0, t1):
    edges, _ = K.bin_centers()
    i0, i1 = (int(round((v - edges[0]) / C.BIN)) for v in (t0, t1))
    return np.sqrt(counts[:, :, i0:i1].sum(-1).T.astype(float))


def cross_time(Xa, Xb, y, strata, rng, n_sub):
    """Balanced accuracy for train/test combinations of window a (feedback) and b (pre-stimulus)."""
    out = {k: [] for k in ('fb_fb', 'fb_stim', 'stim_stim', 'stim_fb')}
    for _ in range(n_sub):
        idx = dec.matched_idx(y, strata, rng)
        yy = y[idx]
        if len(idx) < 4 * dec.N_FOLD or min((yy == 1).sum(), (yy == -1).sum()) < dec.N_FOLD:
            return {k: np.nan for k in out}
        pred = {k: np.empty_like(yy) for k in out}
        A, B = Xa[idx], Xb[idx]
        for tr, te in StratifiedKFold(dec.N_FOLD, shuffle=True, random_state=int(rng.integers(1e9))).split(A, yy):
            ma = dec.clf().fit(A[tr], yy[tr])
            mb = dec.clf().fit(B[tr], yy[tr])
            pred['fb_fb'][te], pred['fb_stim'][te] = ma.predict(A[te]), ma.predict(B[te])
            pred['stim_stim'][te], pred['stim_fb'][te] = mb.predict(B[te]), mb.predict(A[te])
        for k in out:
            out[k].append(balanced_accuracy_score(yy, pred[k]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def one_insertion(s, t_others, parts, seed):
    rng = np.random.default_rng(seed)
    t = s['t']
    res = {}
    if s['n_units'] < C.MIN_UNITS_POPULATION:
        return res
    if 'lags' in parts:
        rows = []
        for var, X in (('raw', s['X']), ('me', s['Xme'])):
            if X is None:
                continue
            for lag in LAGS:
                fn = (lambda tb, lag=lag: lag_targets(tb, lag))
                y, st = fn(t)
                real = ph.balanced_acc(X, y, st, rng, N_SUB)
                nl = fu.imposter_null(X, t_others[:N_IMP], fn, rng, N_SUB_NULL)
                rows.append(dict(pid=s['pid'], subject=s['subject'], variant=var, lag=lag, acc=real,
                                 null=np.median(nl) if len(nl) else np.nan))
        res['lags'] = rows
    if 'crosstime' in parts and s['fb'] is not None:
        Xa = zscore_cols(window(s['fb'], *FB_TRAIN))
        Xb = zscore_cols(s['X'])
        y, st = fu.timed_targets(t)
        real = cross_time(Xa, Xb, y, st, rng, N_SUB)
        nulls = []
        for t_o in t_others[:N_IMP_XT]:
            n = min(len(t), len(t_o))
            yo, so = fu.timed_targets(t_o.iloc[:n].reset_index(drop=True))
            nulls.append(cross_time(Xa[:n], Xb[:n], yo, so, rng, N_SUB_NULL))
        row = dict(pid=s['pid'], subject=s['subject'])
        for k, v in real.items():
            nl = np.array([d[k] for d in nulls], float)
            nl = nl[np.isfinite(nl)]
            row[k] = v
            row[f'{k}_null'] = np.median(nl) if len(nl) else np.nan
            row[f'{k}_p'] = fu.summary_p(v, nl)
        res['xt'] = row
    return res


# ----------------------------------------------------------------------------
def summarise_lags(df, L):
    L.append('=== A. Outcome of past and future trials decoded from the pre-stimulus window of trial n ===')
    L.append('corrected = real - imposter null; each lag balanced within block side x choice of that trial x '
             'time-since-feedback tercile. '
             'Memory: past > future; slow state: past = future.')
    df = df.assign(d=df['acc'] - df['null'])
    for var, lab in (('raw', 'raw activity'), ('me', 'movement and licks regressed out')):
        L.append(f'--- {lab}')
        for reg, g in df[df['variant'] == var].groupby('region', sort=False):
            parts = [f'{lag:+d}: {g.loc[g["lag"] == lag, "d"].mean():+.3f}' for lag in LAGS]
            line = f'  {reg}: ' + ', '.join(parts)
            w = g.pivot_table(index=['pid', 'subject'], columns='lag', values='d').reset_index()
            for a, b in ((-1, 1), (-2, 2)):
                if a in w and b in w:
                    x = w[[a, b, 'subject']].dropna()
                    if len(x) >= 5:
                        diff = x[a] - x[b]
                        dm = diff.groupby(x['subject']).mean()
                        p_i = stats.wilcoxon(diff, alternative='greater').pvalue
                        p_m = stats.wilcoxon(dm, alternative='greater').pvalue if len(dm) >= 5 else np.nan
                        line += (f' | lag {a:+d} minus {b:+d}: {diff.mean():+.3f} (n={len(x)}, p={p_i:.3g}; '
                                 f'{len(dm)} mice p={p_m:.3g})')
            fut = g.loc[g['lag'] == 1, ['d', 'subject']].dropna()
            if len(fut) >= 5:
                line += f' | future (+1) > 0: p={stats.wilcoxon(fut["d"], alternative="greater").pvalue:.3g}'
            L.append(line)


def summarise_xt(df, L):
    L.append('')
    L.append('=== B. Cross-time generalisation (hippocampus): feedback window '
             f'(+{FB_TRAIN[0] * 1000:.0f} to +{FB_TRAIN[1] * 1000:.0f} ms) <-> pre-stimulus window ===')
    L.append('corrected = real - imposter null; one-sided Wilcoxon > 0')
    for reg, g in df.groupby('region', sort=False):
        for k, lab in (('fb_fb', 'train feedback, test feedback'), ('fb_stim', 'train feedback, test pre-stimulus'),
                       ('stim_fb', 'train pre-stimulus, test feedback'),
                       ('stim_stim', 'train pre-stimulus, test pre-stimulus')):
            L.append(f'  {reg} {lab}: ' + fu.fmt_test(g[k] - g[f'{k}_null'], g['subject'])
                     + f'; insertions p < 0.05: {(g[f"{k}_p"] < 0.05).sum()}/{g[f"{k}_p"].notna().sum()}')


def summarise_persistence(regions, L):
    f = OUT / 'outcome_timecourse.csv'
    L.append('')
    L.append('=== C. Persistence (from step 19 time course; corrected accuracy in 100 ms windows) ===')
    if not f.exists():
        L.append('  outcome_timecourse.csv not found (run step 19 first)')
        return
    tc = pd.read_csv(f)
    tc['d'] = tc['acc'] - tc['null']
    fb, so = tc[tc['align'] == 'feedback'], tc[tc['align'] == 'stim']
    per = pd.concat([
        fb[fb['t_center'] < -0.1].groupby(['region', 'pid', 'subject'])['d'].mean().rename('baseline'),
        fb[fb['t_center'] > 0.05].groupby(['region', 'pid', 'subject'])['d'].mean().rename('post_feedback'),
        so.groupby(['region', 'pid', 'subject'])['d'].mean().rename('pre_stimulus')], axis=1).reset_index()
    per = per.dropna(subset=['baseline', 'post_feedback', 'pre_stimulus'])
    per['excess'] = per['pre_stimulus'] - per['baseline']
    per.to_csv(OUT / 'outcome_persistence.csv', index=False)
    L.append('baseline = before the previous feedback (centres -250 to -150 ms); post = +100 to +450 ms; '
             'pre-stimulus = -450 to -150 ms before the next stimulus')
    regs = [r for r in regions if (per['region'] == r).any()]
    for reg in regs:
        g = per[per['region'] == reg]
        L.append(f'  {reg} ({len(g)} insertions): baseline {g["baseline"].mean():+.3f}, post-feedback '
                 f'{g["post_feedback"].mean():+.3f}, pre-stimulus {g["pre_stimulus"].mean():+.3f}; '
                 f'pre-stimulus minus baseline ' + fu.fmt_test(g['excess'], g['subject']))
    L.append('  between subfields (Mann-Whitney on insertions; mouse means in brackets):')
    for col in ('post_feedback', 'excess'):
        for i, a in enumerate(regs):
            for b in regs[i + 1:]:
                ga, gb = per[per['region'] == a], per[per['region'] == b]
                p_i = stats.mannwhitneyu(ga[col], gb[col]).pvalue
                ma, mb = ga.groupby('subject')[col].mean(), gb.groupby('subject')[col].mean()
                p_m = stats.mannwhitneyu(ma, mb).pvalue if min(len(ma), len(mb)) >= 5 else np.nan
                L.append(f'    {col}: {a} {ga[col].mean():+.3f} vs {b} {gb[col].mean():+.3f}, '
                         f'p={p_i:.3g} [mice p={p_m:.3g}]')


def summarise_sessions(regions, L):
    f = OUT / 'outcome_behaviour_sessions.csv'
    L.append('')
    L.append('=== D. Session level (rebuilt): decoding accuracy (step 16, movement and licks regressed out) '
             'vs behaviour, Spearman across insertions ===')
    if not f.exists():
        L.append('  outcome_behaviour_sessions.csv not found (run step 19 first)')
        return
    ds = fu.session_table(pd.read_csv(f), regions)
    ds.to_csv(OUT / 'outcome_behaviour_sessions.csv', index=False)
    groups = [(r, [r]) for r in regions] + [('hippocampus (CA3+CA1+DG)', ['CA3', 'CA1', 'DG']),
                                             ('all regions', list(regions))]
    for name, regs in groups:
        g = ds[ds['region'].isin(regs)].dropna(subset=['dec'])
        if len(regs) > 1:
            g = g.groupby('pid', as_index=False).agg(dec=('dec', 'mean'),
                                                     post_error_slowing=('post_error_slowing', 'first'),
                                                     lose_switch_minus_win_switch=('lose_switch_minus_win_switch',
                                                                                   'first'))
        if len(g) < 8:
            continue
        for col in ('post_error_slowing', 'lose_switch_minus_win_switch'):
            r, p = stats.spearmanr(g['dec'], g[col], nan_policy='omit')
            L.append(f'  {name}: {col} rho={r:+.2f}, p={p:.3g} (n={len(g)} insertions)')


def plot_lags(df, regions):
    plt = K.setup_matplotlib()
    df = df[df['variant'] == 'me'].assign(d=lambda x: x['acc'] - x['null'])
    cols = dict(CA3='#1b4f72', CA1='#2e86c1', DG='#85c1e9', ZI='#c0392b', MOs='#d68910', VPM='#7d3c98')
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    for k, reg in enumerate(regions):
        g = df[df['region'] == reg]
        if g.empty:
            continue
        m = g.groupby('lag')['d'].mean()
        e = g.groupby('lag')['d'].apply(K.sem)
        x = m.index.to_numpy() + (k - len(regions) / 2) * 0.06
        ax.errorbar(x, m.values, e.values, color=cols.get(reg, 'k'), marker='o', ms=3, lw=1, capsize=0, label=reg)
    ax.axhline(0, color='0.6', lw=0.6, ls='--')
    ax.set_xticks(list(LAGS))
    ax.set_xticklabels(['n-2', 'n-1', 'n+1', 'n+2'])
    ax.set_xlabel('Outcome of trial')
    ax.set_ylabel('Decoded from trial n pre-stimulus\n(balanced acc. - null)')
    ax.legend(frameon=False, fontsize=6, loc='upper right')
    fig.tight_layout()
    fig.savefig(OUT / 'outcome_lags.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'outcome_lags.pdf', bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=list(REGIONS))
    ap.add_argument('--parts', nargs='+', default=list(PARTS), choices=PARTS)
    ap.add_argument('--jobs', type=int, default=-2, help='parallel workers (default: all cores but one)')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    regions = [r for r in args.regions if (C.PROJECT_DIR / f'processed_{r}').exists()]
    L = [f'=== Step 20: memory of the previous outcome vs slow state ({", ".join(regions)}) ===', '']
    heavy = [p for p in args.parts if p in ('lags', 'crosstime')]
    lags, xt = [], []
    if heavy:
        from joblib import Parallel, delayed
        t0 = time.time()
        for ri, reg in enumerate(regions):
            sess = fu.load_region(reg)
            print(f'{reg}: {len(sess)} insertions | elapsed {(time.time() - t0) / 60:.0f} min', flush=True)
            rng = np.random.default_rng([C.RANDOM_SEED + 20, ri])
            t_all = [s['t'] for s in sess]
            jobs = [delayed(one_insertion)(s, [t_all[k] for k in rng.permutation(len(sess)) if k != i], heavy,
                                           [C.RANDOM_SEED + 20, ri, i]) for i, s in enumerate(sess)]
            for r in Parallel(n_jobs=args.jobs, verbose=5)(jobs):
                lags += [dict(x, region=reg) for x in r.get('lags', [])]
                if 'xt' in r:
                    xt.append(dict(r['xt'], region=reg))
    if lags:
        df = pd.DataFrame(lags)
        df.to_csv(OUT / 'outcome_lags.csv', index=False)
        summarise_lags(df, L)
        plot_lags(df, regions)
    if xt:
        df = pd.DataFrame(xt)
        df.to_csv(OUT / 'outcome_crosstime.csv', index=False)
        summarise_xt(df, L)
    if 'persistence' in args.parts:
        summarise_persistence(regions, L)
    if 'sessions' in args.parts:
        summarise_sessions(regions, L)
    (OUT / 'summary_20_outcome_memory.txt').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
