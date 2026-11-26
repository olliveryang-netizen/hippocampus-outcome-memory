"""
Step 21 - Last checks before writing: two-trial memory and region comparisons with mice as the unit.

  lag2   Step 20 found outcome information two trials back (n-2) in CA3 and MOs. Outcomes are autocorrelated
         (errors cluster), so this could be the n-1 memory seen through that correlation. Here the outcome of
         n-2 is decoded with the outcome of n-1 also balanced (and, for symmetry, n+2 with n+1 balanced), on top
         of block side x choice of that trial x time-since-feedback tercile. Raw and movement/licks regressed
         out. Imposter null.
  stats  Region comparisons from the saved per-insertion results, with mice as the resampling unit (cluster
         bootstrap, 10,000 resamples: mice are drawn with replacement and all their insertions in a region are
         kept). One probe often passes through several regions and one mouse can carry several probes, so
         insertions are not independent across regions; the cluster bootstrap keeps that pairing. Holm
         correction within each family of comparisons. No decoding; needs the CSV files of steps 16, 19, 20.
           1. previous outcome, unit-matched (5 units, controlled)       results_comparison/prior_history_unit_matched_*.csv
           2. memory index: lag -1 minus lag +1 (movement/licks out)     results_comparison/outcome_lags.csv
           3. outcome information after the previous feedback            results_comparison/outcome_persistence.csv
           4. lag -2 (n-1 balanced) minus lag +2 (n+1 balanced)          from part lag2

    python 21_final_checks.py --regions CA3 CA1 DG ZI MOs VPM
Output: results_comparison/summary_21_final_checks.txt, outcome_lag2.csv
"""
import argparse
import importlib
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import config as C

warnings.filterwarnings('ignore')
fu = importlib.import_module('19_outcome_followup')
m20 = importlib.import_module('20_outcome_memory')
ph = fu.ph

REGIONS = fu.REGIONS
HIPPO = ('CA3', 'CA1', 'DG')
N_SUB, N_SUB_NULL, N_IMP = 10, 2, 30
N_BOOT = 10000
OUT = fu.OUT


# ----------------------------------------------------------------------------
def lag2_targets(t, lag):
    """Outcome of trial n + lag (lag = -2 or +2) with the outcome of the trial in between (n-1 or n+1)
    balanced as well."""
    y, s = m20.lag_targets(t, lag)
    o = t[f'out_{(-1 if lag < 0 else 1):+d}'].to_numpy()
    return np.where(o != 0, y, 0), (s * 3 + (o + 1)).astype(int)


def one_insertion(s, t_others, seed):
    rng = np.random.default_rng(seed)
    rows = []
    if s['n_units'] < C.MIN_UNITS_POPULATION:
        return rows
    for var, X in (('raw', s['X']), ('me', s['Xme'])):
        if X is None:
            continue
        for lag in (-2, 2):
            fn = (lambda tb, lag=lag: lag2_targets(tb, lag))
            y, st = fn(s['t'])
            real = ph.balanced_acc(X, y, st, rng, N_SUB)
            nl = fu.imposter_null(X, t_others[:N_IMP], fn, rng, N_SUB_NULL)
            rows.append(dict(pid=s['pid'], subject=s['subject'], variant=var, lag=lag, acc=real,
                             null=np.median(nl) if len(nl) else np.nan))
    return rows


# ----------------------------------------------------------------------------
def cluster_sums(df, col, reg):
    g = df[(df['region'] == reg)].dropna(subset=[col]).groupby('subject')[col].agg(['sum', 'count'])
    return g


def boot_region(df, col, regs, rng):
    """Cluster bootstrap over mice: returns {region: bootstrap means} using a common draw of mice, so that
    regions recorded in the same mice stay paired."""
    mice = np.array(sorted(df.loc[df['region'].isin(regs), 'subject'].dropna().unique()))
    idx = rng.integers(0, len(mice), (N_BOOT, len(mice)))
    out = {}
    for reg in regs:
        g = cluster_sums(df, col, reg).reindex(mice).fillna(0)
        s, n = g['sum'].to_numpy(), g['count'].to_numpy()
        with np.errstate(invalid='ignore', divide='ignore'):
            out[reg] = s[idx].sum(1) / n[idx].sum(1)
    return out


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj


def compare_family(df, col, regs, title, L, rng, one_sample=True):
    regs = [r for r in regs if (df['region'] == r).any() and df.loc[df['region'] == r, col].notna().sum() >= 5]
    if len(regs) < 1:
        return
    L.append(f'--- {title}')
    obs = {r: df.loc[df['region'] == r, col].mean() for r in regs}
    n_i = {r: df.loc[df['region'] == r, col].notna().sum() for r in regs}
    n_m = {r: df.loc[df['region'] == r].dropna(subset=[col])['subject'].nunique() for r in regs}
    if one_sample:
        res = []
        for r in regs:
            b = boot_region(df, col, [r], rng)[r]
            b = b[np.isfinite(b)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            res.append((r, (1 + (b <= 0).sum()) / (1 + len(b)), lo, hi))
        adj = holm([x[1] for x in res])
        for (r, p, lo, hi), pa in zip(res, adj):
            L.append(f'  {r}: {obs[r]:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}] ({n_i[r]} insertions, {n_m[r]} mice); '
                     f'> 0: p={p:.3g}, Holm p={pa:.3g}')
    pairs = [(a, b) for i, a in enumerate(regs) for b in regs[i + 1:]]
    res = []
    for a, b in pairs:
        bo = boot_region(df, col, [a, b], rng)
        d = bo[a] - bo[b]
        d = d[np.isfinite(d)]
        p = min(1.0, 2 * min((1 + (d <= 0).sum()) / (1 + len(d)), (1 + (d >= 0).sum()) / (1 + len(d))))
        lo, hi = np.percentile(d, [2.5, 97.5])
        res.append((a, b, obs[a] - obs[b], lo, hi, p))
    adj = holm([x[-1] for x in res])
    for (a, b, dd, lo, hi, p), pa in zip(res, adj):
        flag = ' *' if pa < 0.05 else ''
        L.append(f'    {a} - {b}: {dd:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}], p={p:.3g}, Holm p={pa:.3g}{flag}')


def summarise_lag2(df, L):
    L.append('=== A. Two trials back with the trial in between balanced ===')
    L.append('corrected = real - imposter null; one-sided tests: lag -2 > 0, and lag -2 minus lag +2 > 0 (paired)')
    df = df.assign(d=df['acc'] - df['null'])
    for var, lab in (('raw', 'raw activity'), ('me', 'movement and licks regressed out')):
        L.append(f'--- {lab}')
        for reg, g in df[df['variant'] == var].groupby('region', sort=False):
            w = g.pivot_table(index=['pid', 'subject'], columns='lag', values='d').reset_index()
            line = (f'  {reg}: n-2 ' + fu.fmt_test(w[-2], w['subject']) if -2 in w else f'  {reg}: n-2 missing')
            if -2 in w and 2 in w:
                x = w[[-2, 2, 'subject']].dropna()
                line += f' | n+2 {x[2].mean():+.3f} | n-2 minus n+2 ' + fu.fmt_test(x[-2] - x[2], x['subject'])
            L.append(line)


def run_stats(regions, lag2, L):
    rng = np.random.default_rng(C.RANDOM_SEED + 21)
    L.append('')
    L.append(f'=== B. Region comparisons, cluster bootstrap over mice ({N_BOOT} resamples), Holm within family ===')
    um = sorted(OUT.glob('prior_history_unit_matched_*.csv'))
    if um:
        d = pd.read_csv(um[-1])
        d = d[d['N'] == 5]
        compare_family(d, 'prev_reward_ctrl', regions,
                       f'1. previous outcome, 5 units, movement/licks/pupil controlled ({um[-1].name}; '
                       'balanced accuracy, chance 0.5)', L, rng, one_sample=False)
    else:
        L.append('  1. prior_history_unit_matched_*.csv not found (run step 16 --compare)')
    f = OUT / 'outcome_lags.csv'
    if f.exists():
        d = pd.read_csv(f)
        d = d[d['variant'] == 'me'].assign(c=lambda x: x['acc'] - x['null'])
        w = d.pivot_table(index=['region', 'pid', 'subject'], columns='lag', values='c').reset_index()
        w['memory'] = w[-1] - w[1]
        w['lag_m1'] = w[-1]
        compare_family(w, 'memory', regions, '2. memory index: lag -1 minus lag +1 (movement/licks regressed out)',
                       L, rng)
        compare_family(w, 'lag_m1', regions, '2b. previous outcome (lag -1) alone, corrected', L, rng)
    else:
        L.append('  2. outcome_lags.csv not found (run step 20)')
    f = OUT / 'outcome_persistence.csv'
    if f.exists():
        d = pd.read_csv(f)
        compare_family(d, 'post_feedback', [r for r in regions if r in HIPPO],
                       '3. outcome information +100 to +450 ms after the previous feedback (100 ms windows)', L, rng)
        compare_family(d, 'pre_stimulus', [r for r in regions if r in HIPPO],
                       '3b. same windows, -450 to -150 ms before the next stimulus', L, rng)
    else:
        L.append('  3. outcome_persistence.csv not found (run step 20)')
    if lag2 is not None and len(lag2):
        d = lag2[lag2['variant'] == 'me'].assign(c=lambda x: x['acc'] - x['null'])
        w = d.pivot_table(index=['region', 'pid', 'subject'], columns='lag', values='c').reset_index()
        if -2 in w and 2 in w:
            w['memory2'] = w[-2] - w[2]
            compare_family(w, 'memory2', regions,
                           '4. two-trial memory: lag -2 minus lag +2, n-1 / n+1 balanced (movement/licks out)', L, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=list(REGIONS))
    ap.add_argument('--parts', nargs='+', default=['lag2', 'stats'], choices=['lag2', 'stats'])
    ap.add_argument('--jobs', type=int, default=-2)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    regions = [r for r in args.regions if (C.PROJECT_DIR / f'processed_{r}').exists()]
    L = [f'=== Step 21: final checks ({", ".join(regions)}) ===', '']
    lag2 = None
    if 'lag2' in args.parts:
        from joblib import Parallel, delayed
        rows, t0 = [], time.time()
        for ri, reg in enumerate(regions):
            sess = fu.load_region(reg)
            print(f'{reg}: {len(sess)} insertions | elapsed {(time.time() - t0) / 60:.0f} min', flush=True)
            rng = np.random.default_rng([C.RANDOM_SEED + 21, ri])
            t_all = [s['t'] for s in sess]
            jobs = [delayed(one_insertion)(s, [t_all[k] for k in rng.permutation(len(sess)) if k != i],
                                           [C.RANDOM_SEED + 21, ri, i]) for i, s in enumerate(sess)]
            for r in Parallel(n_jobs=args.jobs, verbose=5)(jobs):
                rows += [dict(x, region=reg) for x in r]
        lag2 = pd.DataFrame(rows)
        lag2.to_csv(OUT / 'outcome_lag2.csv', index=False)
        summarise_lag2(lag2, L)
    elif (OUT / 'outcome_lag2.csv').exists():
        lag2 = pd.read_csv(OUT / 'outcome_lag2.csv')
    if 'stats' in args.parts:
        run_stats(regions, lag2, L)
    (OUT / 'summary_21_final_checks.txt').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
