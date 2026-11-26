"""
Step 16 - Block prior and trial history in pre-stimulus population activity.

For one region (ANALYSIS_REGION, e.g. CA1, CA3 or DG), activity in the quiescent pre-stimulus window
(-400 to -100 ms from stimulus onset; config.WINDOWS['iti']) is used to decode:

  block      side of the current biased block (probabilityLeft 0.2 vs 0.8; unbiased trials excluded).
             Null: pseudosessions - block sequences drawn from the task's generative process
             (90 unbiased trials, then alternating blocks with lengths from an exponential
             distribution with mean 60, truncated to 20-100 trials), decoded from the same neural data.
             This keeps slow drifts in the neural data and removes any link to the real blocks.
  block|prev block side with previous choice balanced within each block side (information about the
             block beyond the last choice). Pseudosession null as above.
  prev_choice   previous choice, balanced within block side x previous outcome.
  prev_reward   previous outcome, balanced within block side x previous choice.
  prev_reward_timed   previous outcome, additionally balanced for the time since the previous feedback
             (terciles within the session): after errors the inter-trial interval is longer, so without this
             balance the decoder could read elapsed time instead of the outcome.
             Null for the two history variables: imposter sessions (behaviour from other sessions).

Each decoder is run on raw activity, on activity with movement regressed out (prefix 'me_': video motion
energy of the whisker and body cameras and, after 17_licks.py, the number of licks), and on activity with
movement and pupil diameter regressed out (prefix 'state_', after 18_pupil.py). Pupil diameter alone is
also used to decode the previous outcome, to show how much of it is carried by arousal.

Accuracy is balanced accuracy, 5-fold stratified CV, L2 logistic regression (same decoder as step 04).
Per insertion: corrected accuracy = real - median(null); p = (1 + #null >= real) / (1 + n_null).

    python 16_prior_history.py              # current region
    python 16_prior_history.py --compare --regions CA3 CA1 DG ZI MOs VPM   # equal numbers of units

Outputs: results{_REGION}/prior_history_sessions.csv, results{_REGION}/summary_16_prior_history.txt,
         results_comparison/prior_history_unit_matched.csv, results_comparison/summary_16_compare.txt
"""
import argparse
import importlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import common as K
import config as C

warnings.filterwarnings('ignore')
dec = importlib.import_module('04_decoding')
fu = importlib.import_module('09_followup')
rev = importlib.import_module('10_reversal_history')

N_PSEUDO = 200          # pseudosessions per insertion (block decoders)
N_IMP = 50              # imposter pairings per insertion (history decoders)
N_SUB = 10              # balanced subsamples per decoder
N_SUB_NULL = 2          # subsamples per null sample
UNIT_N = (3, 5, 8)      # units per decoder for the matched comparison
N_SUBSETS = 10          # random unit subsets per insertion and N
COMPARE = ('CA3', 'CA1', 'DG')


# ----------------------------------------------------------------------------
def pseudo_blocks(n_trials, rng):
    """probabilityLeft for trials 0..n_trials-1 from the task's block generative process."""
    p = np.full(n_trials, 0.5)
    i, side = 90, rng.choice([0.2, 0.8])
    while i < n_trials:
        while True:
            L = rng.exponential(60)
            if 20 <= L <= 100:
                break
        p[i:i + int(L)] = side
        side = 1.0 - side
        i += int(L)
    return p


def load_sessions(pdir):
    sess = []
    for path in sorted(q for q in pdir.iterdir() if (q / 'meta.json').exists()):
        meta, t, u, b = K.load_insertion(path)
        t = rev.add_history(t)
        lut_fb = dict(zip(t['trial_number'], t['feedback_times']))
        t['since_prev_fb'] = t['stimOn_times'] - np.array([lut_fb.get(n - 1, np.nan) for n in t['trial_number']])
        ok = K.valid_trials(t) & (t['prev_choice'] != 0).to_numpy()
        if ok.sum() < 100 or len(u) < 1:
            continue
        X = np.sqrt(K.window_counts(b, 'iti')[:, ok].T)
        tt = t[ok].reset_index(drop=True)
        avail = lambda cols: [c for c in cols if c in tt.columns and tt[c].notna().mean() > 0.9]
        me_cols = avail(('me_whisker_iti', 'me_body_iti', 'licks_iti'))
        st_cols = avail(('pupil_iti',))
        Xme = residualise(X, tt, me_cols)
        Xst = residualise(X, tt, me_cols + st_cols) if st_cols else None
        pupil = tt['pupil_iti'].to_numpy(float) if st_cols else None
        sess.append(dict(pid=meta['pid'], subject=meta['subject'], X=X, Xme=Xme, Xst=Xst, pupil=pupil, t=tt,
                         trial_number=tt['trial_number'].to_numpy(), n_units=X.shape[1]))
    return sess


def residualise(X, tt, cols):
    """Neural activity with the given behavioural covariates regressed out (None if no covariates)."""
    if not cols:
        return None
    M = tt[cols].to_numpy(float)
    M = np.where(np.isfinite(M), M, np.nanmean(M, 0))
    M = np.column_stack([np.ones(len(M)), (M - M.mean(0)) / (M.std(0) + 1e-12)])
    return X - M @ np.linalg.lstsq(M, X, rcond=None)[0]


def balanced_acc(X, y, strata, rng, n_sub):
    """Mean balanced accuracy over n_sub subsamples balancing y within each stratum."""
    acc = []
    for _ in range(n_sub):
        idx = dec.matched_idx(y, strata, rng)
        if len(idx) < 4 * dec.N_FOLD:
            return np.nan
        acc.append(dec.cv_balacc(X[idx], y[idx], rng))
    return np.nanmean(acc)


def block_targets(pL, prev_choice):
    """Labels and strata for 'block' and 'block|prev'; trials with pL = 0.5 are dropped."""
    keep = pL != 0.5
    y = np.where(pL < 0.5, 1, -1)                      # 1 = right block (probabilityLeft 0.2)
    return keep, y, np.zeros_like(y), prev_choice


def decode_block(X, t, rng, n_sub, pL=None):
    pL = t['probabilityLeft'].to_numpy() if pL is None else pL
    keep, y, s0, s1 = block_targets(pL, t['prev_choice'].to_numpy())
    if keep.sum() < 60 or len(np.unique(y[keep])) < 2:
        return np.nan, np.nan
    a = balanced_acc(X[keep], y[keep], s0[keep], rng, n_sub)
    b = balanced_acc(X[keep], y[keep], s1[keep], rng, n_sub)
    return a, b


def history_targets(t_beh):
    block = np.sign(0.5 - t_beh['probabilityLeft'].to_numpy())          # -1, 0, +1
    pc = t_beh['prev_choice'].to_numpy()
    pr = t_beh['prev_reward'].to_numpy()
    gap = t_beh['since_prev_fb'].to_numpy()
    ok = np.isfinite(gap)
    terc = np.full(len(gap), -1)
    if ok.sum() > 30:
        terc[ok] = pd.qcut(gap[ok], 3, labels=False, duplicates='drop')
    return {'prev_choice': (pc, (block + 1) * 3 + (pr + 1)),
            'prev_reward': (pr, (block + 1) * 3 + (pc + 1)),
            'prev_reward_timed': (np.where(terc >= 0, pr, 0), ((block + 1) * 3 + (pc + 1)) * 10 + terc + 1)}


def decode_history(X, t_beh, rng, n_sub):
    return {k: balanced_acc(X, y, s, rng, n_sub) for k, (y, s) in history_targets(t_beh).items()}


# ----------------------------------------------------------------------------
def one_insertion(i, sess, seed):
    rng = np.random.default_rng([seed, i])
    s = sess[i]
    if True:
        row = dict(pid=s['pid'], subject=s['subject'], n_units=s['n_units'], n_trials=len(s['t']))
        n_tot = int(s['trial_number'].max()) + 1
        if s['pupil'] is not None:                     # does pupil size alone carry the previous outcome?
            y, st = history_targets(s['t'])['prev_reward_timed']
            P = np.nan_to_num(s['pupil'], nan=np.nanmean(s['pupil']))[:, None]
            row['pupil_only_prev_reward_timed'] = balanced_acc(P, y, st, rng, N_SUB)
        for pre, X in (('', s['X']), ('me_', s['Xme']), ('state_', s['Xst'])):
            if X is None:
                continue
            # block decoders with pseudosession null
            real = decode_block(X, s['t'], rng, N_SUB)
            null = np.array([decode_block(X, s['t'], rng, N_SUB_NULL,
                                          pL=pseudo_blocks(n_tot, rng)[s['trial_number']])
                             for _ in range(N_PSEUDO)])
            for j, name in enumerate(('block', 'block_prev')):
                nl = null[:, j][np.isfinite(null[:, j])]
                row[f'{pre}{name}'] = real[j]
                row[f'{pre}{name}_null'] = np.median(nl) if len(nl) else np.nan
                row[f'{pre}{name}_p'] = (1 + (nl >= real[j]).sum()) / (1 + len(nl)) if len(nl) else np.nan
            # history decoders with imposter null
            real_h = decode_history(X, s['t'], rng, N_SUB)
            others = [k for k in range(len(sess)) if k != i]
            null_h = {k: [] for k in real_h}
            for k in rng.choice(others, min(N_IMP, len(others)), replace=False):
                n = min(len(s['t']), len(sess[k]['t']))
                r = decode_history(X[:n], sess[k]['t'].iloc[:n], rng, N_SUB_NULL)
                for key in r:
                    null_h[key].append(r[key])
            for key, v in real_h.items():
                nl = np.array(null_h[key])
                nl = nl[np.isfinite(nl)]
                row[f'{pre}{key}'] = v
                row[f'{pre}{key}_null'] = np.median(nl) if len(nl) else np.nan
                row[f'{pre}{key}_p'] = (1 + (nl >= v).sum()) / (1 + len(nl)) if len(nl) else np.nan
    return row


def analyse_region(sess, seed, n_jobs):
    from joblib import Parallel, delayed
    use = [i for i, s in enumerate(sess) if s['n_units'] >= C.MIN_UNITS_POPULATION]
    rows = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one_insertion)(i, sess, seed) for i in use)
    return pd.DataFrame(rows)


def summarise(df, L):
    if 'pupil_only_prev_reward_timed' in df:
        v = df['pupil_only_prev_reward_timed'].dropna()
        if len(v) >= 5:
            L.append(f'--- pupil size alone, previous outcome (timed): {v.mean():.3f} (chance 0.5; {len(v)} insertions; '
                     f'Wilcoxon > 0.5 p={stats.wilcoxon(v - 0.5, alternative="greater").pvalue:.3g})')
    for pre, lab in (('', 'raw activity'), ('me_', 'motion energy and licks regressed out'),
                     ('state_', 'motion energy, licks and pupil regressed out')):
        L.append(f'--- {lab}')
        for key, txt in (('block', 'block side'), ('block_prev', 'block side, previous choice balanced'),
                         ('prev_choice', 'previous choice (block x previous outcome balanced)'),
                         ('prev_reward', 'previous outcome (block x previous choice balanced)'),
                         ('prev_reward_timed', 'previous outcome (also balanced for time since previous feedback)')):
            col = f'{pre}{key}'
            if col not in df:
                continue
            d = (df[col] - df[f'{col}_null']).dropna()
            if len(d) < 5:
                continue
            sub = df.loc[d.index, 'subject']
            dm = d.groupby(sub).mean()
            p_ins = stats.wilcoxon(d, alternative='greater').pvalue
            p_mouse = stats.wilcoxon(dm, alternative='greater').pvalue if len(dm) >= 5 else np.nan
            n_sig = int((df.loc[d.index, f'{col}_p'] < C.ALPHA).sum())
            L.append(f'  {txt}: real {df.loc[d.index, col].mean():.3f}, null {df.loc[d.index, f"{col}_null"].mean():.3f}, '
                     f'corrected {d.mean():+.3f} +/- {K.sem(d):.3f} ({len(d)} insertions; Wilcoxon > 0 p={p_ins:.3g}; '
                     f'{len(dm)} mice p={p_mouse:.3g}); insertions with p < {C.ALPHA}: {n_sig}/{len(d)} '
                     f'(binomial p={stats.binomtest(n_sig, len(d), C.ALPHA, alternative="greater").pvalue:.3g})')


# ----------------------------------------------------------------------------
def _matched_one(s, reg, seed, i):
    rng = np.random.default_rng([seed, i])
    y, st = history_targets(s['t'])['prev_reward_timed']
    Xc = s['Xst'] if s['Xst'] is not None else s['Xme'] if s['Xme'] is not None else s['X']
    rows = []
    for N in UNIT_N:
        if s['n_units'] < N:
            continue
        raw, ctl = [], []
        for _ in range(N_SUBSETS):
            cols = rng.choice(s['n_units'], N, replace=False)
            raw.append(balanced_acc(s['X'][:, cols], y, st, rng, N_SUB_NULL))
            ctl.append(balanced_acc(Xc[:, cols], y, st, rng, N_SUB_NULL))
        rows.append(dict(region=reg, pid=s['pid'], subject=s['subject'], N=N,
                         prev_reward=np.nanmean(raw), prev_reward_ctrl=np.nanmean(ctl)))
    return rows


def unit_matched(regions, n_jobs):
    """Previous-outcome decoding (time-balanced) with N randomly drawn units, raw and with movement, licks
    and pupil regressed out (whatever is available for the region)."""
    from joblib import Parallel, delayed
    out = []
    for k, reg in enumerate(regions):
        pdir = C.PROJECT_DIR / f'processed_{reg}'
        if not pdir.exists():
            continue
        sess = load_sessions(pdir)
        print(f'  {reg}: {len(sess)} insertions', flush=True)
        res = Parallel(n_jobs=n_jobs)(delayed(_matched_one)(s, reg, C.RANDOM_SEED + 160 + k, i)
                                      for i, s in enumerate(sess))
        out += [r for rr in res for r in rr]
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compare', action='store_true', help='CA3 vs CA1 vs DG with equal numbers of units')
    ap.add_argument('--regions', nargs='+', default=list(COMPARE), help='regions for --compare')
    ap.add_argument('--jobs', type=int, default=-2, help='parallel workers (default: all cores but one)')
    args = ap.parse_args()
    rng = np.random.default_rng(C.RANDOM_SEED + 16)
    if args.compare:
        out = C.PROJECT_DIR / 'results_comparison'
        out.mkdir(exist_ok=True)
        df = unit_matched(args.regions, args.jobs)
        df.to_csv(out / f'prior_history_unit_matched_{"_".join(args.regions)}.csv', index=False)
        L = ['=== Unit-count-matched comparison: previous outcome (time-balanced), pre-stimulus window ===',
             'balanced accuracy, chance 0.5; ctrl = movement energy, licks and pupil regressed out (where available)']
        pairs = [(a, b) for i, a in enumerate(args.regions) for b in args.regions[i + 1:]]
        for key in ('prev_reward', 'prev_reward_ctrl'):
            for N in UNIT_N:
                g = df[df['N'] == N]
                parts = [f'{r} {g.loc[g["region"] == r, key].mean():.3f} (n={g.loc[g["region"] == r, key].notna().sum()})'
                         for r in args.regions if (g['region'] == r).any()]
                L.append(f'  {key}, N = {N}: ' + ', '.join(parts))
                for a, b in pairs:
                    x = g.loc[g['region'] == a, key].dropna()
                    y = g.loc[g['region'] == b, key].dropna()
                    if len(x) >= 5 and len(y) >= 5:
                        L.append(f'    {a} vs {b}: Mann-Whitney p={stats.mannwhitneyu(x, y).pvalue:.3g}')
        (out / f'summary_16_compare_{"_".join(args.regions)}.txt').write_text('\n'.join(L), encoding='utf-8')
        print('\n'.join(L))
        return
    sess = load_sessions(C.PROCESSED_DIR)
    print(f'{C.REGION}: {len(sess)} insertions with >= 100 valid trials')
    df = analyse_region(sess, C.RANDOM_SEED + 16, args.jobs)
    df.to_csv(C.RESULTS_DIR / 'prior_history_sessions.csv', index=False)
    L = [f'=== {C.REGION}: block prior and trial history, pre-stimulus window '
         f'({C.WINDOWS["iti"][1] * 1000:.0f} to {C.WINDOWS["iti"][2] * 1000:.0f} ms from stimulus onset) ===',
         f'{len(df)} insertions (>= {C.MIN_UNITS_POPULATION} units), {df["subject"].nunique()} mice; '
         f'block null: {N_PSEUDO} pseudosessions; history null: {N_IMP} imposter sessions']
    summarise(df, L)
    (C.RESULTS_DIR / 'summary_16_prior_history.txt').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
