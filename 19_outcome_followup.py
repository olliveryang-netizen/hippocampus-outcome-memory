"""
Step 19 - Follow-up of the previous-outcome signal found in step 16.

Run after step 16 has been run for every region (it reads results_<REGION>/prior_history_sessions.csv).

  subset      Step 16 could regress pupil out only where pupil data exist (about half of the insertions), so
              its three blocks of results come from different insertions. Here raw, movement-controlled and
              movement + pupil-controlled decoding are compared on exactly the insertions that have pupil
              data (paired Wilcoxon). No new decoding.
  pupil       Two stricter tests of the arousal account (insertions with pupil data only).
              (1) Pupil balanced by stratification instead of linear regression: equal numbers of rewarded
                  and unrewarded previous trials inside every combination of previous choice x tercile of
                  time since the previous feedback x tercile of pre-stimulus pupil size (activity with
                  movement and licks regressed out). Imposter null.
              (2) Information beyond pupil: the same decoder trained on pupil alone and on pupil plus
                  population activity, on identical trials and folds. Null for the gain: population activity
                  circularly shifted across trials (keeps slow drifts, breaks the trial-by-trial link).
  timecourse  Previous-outcome decoding in 100 ms windows (50 ms steps) around the previous trial's feedback
              (insertions with feedback-aligned counts, i.e. the hippocampal ones) and before the current
              stimulus. After feedback the two outcomes differ in sensory input and behaviour (reward and
              licking vs the error tone), so that part shows when the signal appears, not that it is cognitive.
              Licks are deliberately not regressed out there: lick counts after feedback almost determine the
              outcome, and regressing a label-correlated covariate out of the whole session before
              cross-validation pushes accuracy below chance (Snoek et al., 2019, NeuroImage). Same trials, labels and balancing as the
              step-16 decoder 'previous outcome (also balanced for time since previous feedback)'.
              Imposter null.
  behaviour   Does the neural outcome signal relate to what the mouse does next? An out-of-fold decoder of
              the previous outcome (activity with movement and licks regressed out) gives every trial a
              score: how 'rewarded-like' the pre-stimulus state is. Within trials after an error, and
              separately within trials after a reward, the score is related to the current reaction time
              and to repeating the previous choice: partial Spearman correlation controlling for |contrast|,
              time since the previous feedback, trial number (linear and quadratic) and the mean reaction
              time of the preceding 10 trials; where pupil data exist, also with pupil size added.
              Null: the same correlations with scores from a decoder trained on circularly shifted activity
              (20 shifts), which absorbs any bias of the procedure and of slow drifts.
              Session level: decoding accuracy (step 16) vs post-error slowing and lose-switch tendency.
  rate        Pre-stimulus firing rates per region, and the unit-matched comparison (5 units) repeated with
              units firing >= 2 Hz only, to see whether the weaker DG signal reflects sparse firing.

    python 19_outcome_followup.py --regions CA3 CA1 DG ZI MOs VPM
    python 19_outcome_followup.py --regions CA3 CA1 DG ZI MOs VPM --parts subset      # seconds

Outputs (results_comparison/): summary_19_outcome_followup.txt, outcome_pupil.csv, outcome_timecourse.csv,
outcome_behaviour.csv, outcome_behaviour_sessions.csv, outcome_rate.csv, outcome_timecourse.png/.pdf
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
ph = importlib.import_module('16_prior_history')
dec = ph.dec

REGIONS = ('CA3', 'CA1', 'DG', 'ZI', 'MOs', 'VPM')
PARTS = ('subset', 'pupil', 'timecourse', 'behaviour', 'rate')
N_SUB, N_SUB_NULL = 10, 2
N_IMP = 50              # imposter sessions (pupil-stratified decoder)
N_SHIFT = 50            # circular shifts (information beyond pupil)
N_SUB_TC, N_IMP_TC = 5, 20
TC_W, TC_STEP = 0.10, 0.05
FB_STARTS = np.round(np.arange(-0.30, 0.40 + 1e-9, TC_STEP), 3)    # around the previous feedback
SO_STARTS = np.round(np.arange(-0.50, -0.20 + 1e-9, TC_STEP), 3)   # before the current stimulus
RATE_MIN, RATE_N, RATE_SUBSETS = 2.0, 5, 10
N_SHIFT_BEH = 20       # circular shifts for the behaviour analysis
LAGS = (-2, -1, 1, 2)   # trials before / after, used by step 20
MIN_CLASS = 30          # min trials per previous-outcome class for the behaviour analysis
OUT = C.PROJECT_DIR / 'results_comparison'


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_region(reg):
    """Like 16_prior_history.load_sessions, plus binned counts for the time course and per-unit rates."""
    pdir = C.PROJECT_DIR / f'processed_{reg}'
    sess = []
    for path in sorted(q for q in pdir.iterdir() if (q / 'meta.json').exists()):
        meta, t, u, b = K.load_insertion(path)
        t = ph.rev.add_history(t)
        lut_fb = dict(zip(t['trial_number'], t['feedback_times']))
        t['since_prev_fb'] = t['stimOn_times'] - np.array([lut_fb.get(n - 1, np.nan) for n in t['trial_number']])
        ok = K.valid_trials(t) & (t['prev_choice'] != 0).to_numpy()
        if ok.sum() < 100 or len(u) < 1:
            continue
        counts = K.window_counts(b, 'iti')[:, ok]
        X = np.sqrt(counts.T)
        tt = t[ok].reset_index(drop=True)
        row_of = {n: r for r, n in enumerate(t['trial_number'])}
        prev_rows = np.array([row_of[n - 1] for n in tt['trial_number']])
        if 'licks_post' in t.columns:
            tt['prev_licks_post'] = t['licks_post'].to_numpy(float)[prev_rows]
        lut_c = dict(zip(t['trial_number'], t['choice']))
        lut_r = dict(zip(t['trial_number'], t['feedbackType']))
        for lag in LAGS:                                  # choice and outcome of trial n + lag (0 = missing)
            tt[f'cho_{lag:+d}'] = [lut_c.get(n + lag, 0) for n in tt['trial_number']]
            tt[f'out_{lag:+d}'] = [lut_r.get(n + lag, 0) for n in tt['trial_number']]
        avail = lambda cols: [c for c in cols if c in tt.columns and tt[c].notna().mean() > 0.9]
        me_cols = avail(('me_whisker_iti', 'me_body_iti', 'licks_iti'))
        st_cols = avail(('pupil_iti',))
        sess.append(dict(
            pid=meta['pid'], subject=meta['subject'], t=tt, X=X, n_units=X.shape[1],
            Xme=ph.residualise(X, tt, me_cols), Xst=ph.residualise(X, tt, me_cols + st_cols) if st_cols else None,
            pupil=tt['pupil_iti'].to_numpy(float) if st_cols else None,
            so=b['so_counts'][:, ok, :], fb=b['fb_counts'][:, prev_rows, :] if 'fb_counts' in b else None,
            rate=counts.mean(1) / (C.WINDOWS['iti'][2] - C.WINDOWS['iti'][1])))
    return sess


def win_X(counts, start):
    """sqrt spike counts (trials x units) in [start, start + TC_W) of binned counts (units x trials x bins)."""
    edges, _ = K.bin_centers()
    i0 = int(round((start - edges[0]) / C.BIN))
    i1 = i0 + int(round(TC_W / C.BIN))
    return np.sqrt(counts[:, :, i0:i1].sum(-1).T.astype(float))


# ----------------------------------------------------------------------------
# Decoding helpers
# ----------------------------------------------------------------------------
def timed_targets(t):
    return ph.history_targets(t)['prev_reward_timed']


def pupil_targets(t):
    """Previous outcome with previous choice x time-since-feedback tercile x pupil tercile balanced."""
    pr = t['prev_reward'].to_numpy()
    pc = t['prev_choice'].to_numpy()
    gap = t['since_prev_fb'].to_numpy(float)
    pu = t['pupil_iti'].to_numpy(float) if 'pupil_iti' in t.columns else np.full(len(t), np.nan)
    ok = np.isfinite(gap) & np.isfinite(pu)
    s = np.full(len(t), -1)
    if ok.sum() > 60:
        tg = pd.qcut(gap[ok], 3, labels=False, duplicates='drop')
        tp = pd.qcut(pu[ok], 3, labels=False, duplicates='drop')
        s[ok] = (pc[ok] + 1) * 50 + tg * 10 + tp
    return np.where(s >= 0, pr, 0), s


def imposter_null(X, t_others, target_fn, rng, n_sub):
    out = []
    for t_o in t_others:
        n = min(len(X), len(t_o))
        y, s = target_fn(t_o.iloc[:n].reset_index(drop=True))
        out.append(ph.balanced_acc(X[:n], y, s, rng, n_sub))
    out = np.array(out, float)
    return out[np.isfinite(out)]


def pupil_gain(Xn, P, y, strata, rng, n_sub):
    """Balanced accuracy from pupil alone and from pupil + population, on identical subsamples and folds."""
    ap, an = [], []
    for _ in range(n_sub):
        idx = dec.matched_idx(y, strata, rng)
        yy = y[idx]
        if len(idx) < 4 * dec.N_FOLD or min((yy == 1).sum(), (yy == -1).sum()) < dec.N_FOLD:
            return np.nan, np.nan
        Z1, Z2 = P[idx], np.column_stack([P[idx], Xn[idx]])
        p1, p2 = np.empty_like(yy), np.empty_like(yy)
        for tr, te in StratifiedKFold(dec.N_FOLD, shuffle=True, random_state=int(rng.integers(1e9))).split(Z1, yy):
            p1[te] = dec.clf().fit(Z1[tr], yy[tr]).predict(Z1[te])
            p2[te] = dec.clf().fit(Z2[tr], yy[tr]).predict(Z2[te])
        ap.append(balanced_accuracy_score(yy, p1))
        an.append(balanced_accuracy_score(yy, p2))
    return float(np.mean(ap)), float(np.mean(an))


def summary_p(real, nl):
    return (1 + (nl >= real).sum()) / (1 + len(nl)) if len(nl) else np.nan


def partial_spearman(a, b, Z):
    a, b = stats.rankdata(a), stats.rankdata(b)
    Z = np.column_stack([np.ones(len(a)), Z])
    ra = a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    rb = b - Z @ np.linalg.lstsq(Z, b, rcond=None)[0]
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _z(v):
    v = np.asarray(v, float)
    v = np.where(np.isfinite(v), v, np.nanmean(v))
    return (v - v.mean()) / (v.std() + 1e-12)


# ----------------------------------------------------------------------------
# Per-insertion work
# ----------------------------------------------------------------------------
def one_insertion(s, t_others, t_others_pupil, parts, seed):
    rng = np.random.default_rng(seed)
    t = s['t']
    res = dict(pid=s['pid'], subject=s['subject'], n_units=s['n_units'])
    y, st = timed_targets(t)
    use_pop = s['n_units'] >= C.MIN_UNITS_POPULATION

    # --- pupil: stratification and information beyond pupil
    if 'pupil' in parts and use_pop and s['pupil'] is not None and len(t_others_pupil) >= 5:
        Xm = s['Xme'] if s['Xme'] is not None else s['X']
        yp, sp = pupil_targets(t)
        real = ph.balanced_acc(Xm, yp, sp, rng, N_SUB)
        nl = imposter_null(Xm, t_others_pupil[:N_IMP], pupil_targets, rng, N_SUB_NULL)
        P = np.nan_to_num(s['pupil'], nan=np.nanmean(s['pupil']))[:, None]
        a_p, a_np = pupil_gain(s['X'], P, y, st, rng, N_SUB)
        gains = []
        n = len(t)
        for _ in range(N_SHIFT):
            k = int(rng.integers(n // 10, n - n // 10))
            g = pupil_gain(np.roll(s['X'], k, axis=0), P, y, st, rng, N_SUB_NULL)
            gains.append(g[1] - g[0])
        gains = np.array(gains, float)
        gains = gains[np.isfinite(gains)]
        res['pupil'] = dict(pid=s['pid'], subject=s['subject'], strat=real,
                            strat_null=np.median(nl) if len(nl) else np.nan, strat_p=summary_p(real, nl),
                            acc_pupil=a_p, acc_pupil_pop=a_np, gain=a_np - a_p,
                            gain_null=np.median(gains) if len(gains) else np.nan,
                            gain_p=summary_p(a_np - a_p, gains))

    # --- time course
    if 'timecourse' in parts and use_pop:
        rows = []
        segs = [('stim', SO_STARTS, s['so'], None)]
        if s['fb'] is not None:
            segs.insert(0, ('feedback', FB_STARTS, s['fb'], None))
        for seg, starts, counts, cov in segs:
            for st0 in starts:
                Xw = win_X(counts, st0)
                if cov:
                    Xw = ph.residualise(Xw, t, cov)
                real = ph.balanced_acc(Xw, y, st, rng, N_SUB_TC)
                nl = imposter_null(Xw, t_others[:N_IMP_TC], timed_targets, rng, N_SUB_NULL)
                rows.append(dict(pid=s['pid'], subject=s['subject'], align=seg, t_center=st0 + TC_W / 2,
                                 acc=real, null=np.median(nl) if len(nl) else np.nan))
        res['tc'] = rows

    # --- behaviour
    if 'behaviour' in parts and use_pop:
        Xb = s['Xme'] if s['Xme'] is not None else s['X']
        pr = t['prev_reward'].to_numpy()
        lrt = np.log(np.clip(t['rt'].to_numpy(float), 1e-3, None))
        recent = pd.Series(lrt).shift(1).rolling(10, min_periods=3).mean().to_numpy()
        tn = _z(t['trial_number'])
        base = np.column_stack([_z(np.abs(t['signed_contrast'])), _z(np.log(t['since_prev_fb'].clip(lower=1e-3))),
                                tn, _z(tn ** 2), _z(recent)])
        outcomes = {'log_rt': lrt, 'stay': (t['choice'].to_numpy() == t['prev_choice'].to_numpy()).astype(float)}
        pu = _z(s['pupil']) if s['pupil'] is not None else None

        def oof_score(Z):
            sc = np.full(len(t), np.nan)
            if min((pr == 1).sum(), (pr == -1).sum()) >= 2 * dec.N_FOLD:
                for tr, te in StratifiedKFold(dec.N_FOLD, shuffle=True,
                                              random_state=int(rng.integers(1e9))).split(Z, pr):
                    sc[te] = dec.clf().fit(Z[tr], pr[tr]).decision_function(Z[te])
            return sc

        def corrs(sc, with_pupil):
            out = {}
            for cls, lab in ((-1, 'after_error'), (1, 'after_reward')):
                m = (pr == cls) & np.isfinite(sc) & np.isfinite(lrt)
                if m.sum() < MIN_CLASS:
                    continue
                for oname, o in outcomes.items():
                    r = dict(n=int(m.sum()), r_score=partial_spearman(sc[m], o[m], base[m]))
                    if with_pupil and pu is not None:
                        r['r_score_pupilctrl'] = partial_spearman(sc[m], o[m], np.column_stack([base[m], pu[m]]))
                        r['r_pupil'] = partial_spearman(pu[m], o[m], base[m])
                    out[(lab, oname)] = r
            return out

        real = corrs(oof_score(Xb), True)
        nulls = []
        n = len(t)
        for _ in range(N_SHIFT_BEH):       # scores from circularly shifted activity: analysis bias and drifts
            nulls.append(corrs(oof_score(np.roll(Xb, int(rng.integers(n // 10, n - n // 10)), axis=0)), False))
        rows = []
        for key, r in real.items():
            nl = np.array([d[key]['r_score'] for d in nulls if key in d], float)
            rows.append(dict(r, pid=s['pid'], subject=s['subject'], cls=key[0], outcome=key[1],
                             r_null=np.nanmedian(nl) if len(nl) else np.nan))
        res['beh'] = rows
        err, rew = pr == -1, pr == 1
        stay = outcomes['stay'].astype(bool)
        res['beh_sess'] = dict(pid=s['pid'], subject=s['subject'],
                               post_error_slowing=np.nanmedian(lrt[err]) - np.nanmedian(lrt[rew]),
                               lose_switch_minus_win_switch=(1 - stay[err].mean()) - (1 - stay[rew].mean()))

    # --- firing rate
    if 'rate' in parts:
        hi = np.where(s['rate'] >= RATE_MIN)[0]
        r = dict(pid=s['pid'], subject=s['subject'], median_rate=float(np.median(s['rate'])),
                 n_units=s['n_units'], n_units_hi=len(hi), acc_hi=np.nan)
        if len(hi) >= RATE_N:
            Xc = s['Xst'] if s['Xst'] is not None else s['Xme'] if s['Xme'] is not None else s['X']
            r['acc_hi'] = np.nanmean([ph.balanced_acc(Xc[:, rng.choice(hi, RATE_N, replace=False)], y, st, rng,
                                                      N_SUB_NULL) for _ in range(RATE_SUBSETS)])
        res['rate'] = r
    return res


# ----------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------
def fmt_test(d, subj, alternative='greater'):
    d = pd.Series(np.asarray(d, float), index=np.arange(len(d)))
    subj = pd.Series(np.asarray(subj), index=d.index)
    d = d.dropna()
    if len(d) < 5:
        return f'n={len(d)} (too few)'
    dm = d.groupby(subj[d.index]).mean()
    p_i = stats.wilcoxon(d, alternative=alternative).pvalue
    p_m = stats.wilcoxon(dm, alternative=alternative).pvalue if len(dm) >= 5 else np.nan
    side = 'one-sided' if alternative == 'greater' else 'two-sided'
    return f'{d.mean():+.3f} +/- {K.sem(d):.3f} ({len(d)} insertions, {side} Wilcoxon p={p_i:.3g}; {len(dm)} mice p={p_m:.3g})'


def summarise_subset(regions, L):
    L.append('=== A. Same insertions: raw vs movement + licks vs movement + licks + pupil (step-16 results) ===')
    L.append('corrected accuracy (real - null), only insertions with pupil data; paired Wilcoxon, two-sided')
    for reg in regions:
        f = C.PROJECT_DIR / f'results_{reg}' / 'prior_history_sessions.csv'
        if not f.exists():
            L.append(f'  {reg}: {f.name} not found (run step 16 first)')
            continue
        df = pd.read_csv(f)
        for key, txt in (('prev_reward_timed', 'previous outcome (timed)'), ('prev_choice', 'previous choice'),
                         ('block', 'block side')):
            cols = [f'{p}{key}' for p in ('', 'me_', 'state_')]
            if not all(c in df and f'{c}_null' in df for c in cols):
                continue
            d = pd.DataFrame({p: df[c] - df[f'{c}_null'] for p, c in zip(('raw', 'me', 'state'), cols)}).dropna()
            if len(d) < 5:
                continue
            p_rs = stats.wilcoxon(d['raw'], d['state']).pvalue
            p_ms = stats.wilcoxon(d['me'], d['state']).pvalue
            keep = d['state'].mean() / d['raw'].mean() if key == 'prev_reward_timed' and d['raw'].mean() > 0.01 else np.nan
            extra = f'; retained {100 * keep:.0f}%' if np.isfinite(keep) else ''
            L.append(f'  {reg} {txt}: raw {d["raw"].mean():+.3f}, +movement/licks {d["me"].mean():+.3f}, '
                     f'+pupil {d["state"].mean():+.3f} (n={len(d)}; raw vs +pupil p={p_rs:.3g}; '
                     f'movement vs +pupil p={p_ms:.3g}{extra})')


def summarise_pupil(df, L):
    L.append('')
    L.append('=== B. Pupil: balanced by stratification; information beyond pupil ===')
    L.append('B1 previous outcome, previous choice x time-since-feedback tercile x pupil tercile balanced '
             '(movement and licks regressed out), corrected = real - imposter null:')
    for reg, g in df.groupby('region', sort=False):
        L.append(f'  {reg}: real {g["strat"].mean():.3f}, null {g["strat_null"].mean():.3f}, corrected '
                 + fmt_test(g['strat'] - g['strat_null'], g['subject'])
                 + f'; insertions with p < 0.05: {(g["strat_p"] < 0.05).sum()}/{g["strat_p"].notna().sum()}')
    L.append('B2 information beyond pupil: balanced accuracy from pupil alone vs pupil + population '
             '(gain corrected by circular-shift null):')
    for reg, g in df.groupby('region', sort=False):
        L.append(f'  {reg}: pupil {g["acc_pupil"].mean():.3f}, pupil + population {g["acc_pupil_pop"].mean():.3f}, '
                 f'gain {g["gain"].mean():+.3f}, shift-null gain {g["gain_null"].mean():+.3f}, corrected gain '
                 + fmt_test(g['gain'] - g['gain_null'], g['subject'])
                 + f'; insertions with p < 0.05: {(g["gain_p"] < 0.05).sum()}/{g["gain_p"].notna().sum()}')


def summarise_tc(df, L):
    L.append('')
    L.append('=== C. Time course of previous-outcome information (100 ms windows; corrected = real - imposter null) ===')
    df = df.assign(d=df['acc'] - df['null'])
    for (align, reg), g in df.groupby(['align', 'region'], sort=False):
        parts = []
        for tc, h in g.groupby('t_center'):
            d = h['d'].dropna()
            p = stats.wilcoxon(d, alternative='greater').pvalue if len(d) >= 5 else np.nan
            parts.append(f'{tc * 1000:+.0f}: {d.mean():+.3f}{"*" if p < 0.05 else ""}')
        lab = 'ms from previous feedback' if align == 'feedback' else 'ms from current stimulus'
        L.append(f'  {reg}, {lab} ({g["pid"].nunique()} insertions): ' + ', '.join(parts))
    L.append('  (* one-sided Wilcoxon across insertions p < 0.05, uncorrected)')


def summarise_beh(df, ds, regions, L):
    L.append('')
    L.append('=== D. Neural outcome score and next-trial behaviour ===')
    L.append('partial Spearman r between the out-of-fold score (higher = more reward-like) and behaviour, within '
             'trials of one previous outcome; covariates |contrast|, time since previous feedback, trial number, '
             'recent RT. Null: scores from circularly shifted activity. Two-sided tests.')
    groups = [(r, [r]) for r in regions] + [('hippocampus (CA3+CA1+DG)', ['CA3', 'CA1', 'DG']),
                                             ('all regions', list(regions))]
    for name, regs in groups:
        g0 = df[df['region'].isin(regs)]
        if g0.empty:
            continue
        for (cls, oc), g in g0.groupby(['cls', 'outcome']):
            line = (f'  {name}, {cls}, {oc}: r {g["r_score"].mean():+.3f}, shift null {g["r_null"].mean():+.3f}, '
                    'r - null ' + fmt_test(g['r_score'] - g['r_null'], g['subject'], 'two-sided'))
            if 'r_score_pupilctrl' in g and g['r_score_pupilctrl'].notna().sum() >= 5:
                gp = g.dropna(subset=['r_score_pupilctrl'])
                line += ('; pupil-data insertions: r with pupil as covariate - null '
                         + fmt_test(gp['r_score_pupilctrl'] - gp['r_null'], gp['subject'], 'two-sided')
                         + '; pupil itself r ' + fmt_test(gp['r_pupil'], gp['subject'], 'two-sided'))
            L.append(line)
    if ds is not None and 'dec' in ds and len(ds):
        L.append('session level, Spearman across insertions: corrected decoding accuracy (step 16, movement and '
                 'licks regressed out) vs behaviour')
        for name, regs in groups:
            g = ds[ds['region'].isin(regs)].dropna(subset=['dec'])
            if len(regs) > 1:          # a probe can pass through several regions: one row per insertion
                g = g.groupby('pid', as_index=False).agg(dec=('dec', 'mean'),
                                                         post_error_slowing=('post_error_slowing', 'first'),
                                                         lose_switch_minus_win_switch=('lose_switch_minus_win_switch',
                                                                                       'first'))
            if len(g) < 8:
                continue
            for col in ('post_error_slowing', 'lose_switch_minus_win_switch'):
                r, p = stats.spearmanr(g['dec'], g[col], nan_policy='omit')
                L.append(f'  {name}: {col} rho={r:+.2f}, p={p:.3g} (n={len(g)})')


def session_table(ds, regions):
    """Session-level behaviour joined with the step-16 decoding accuracy of the same region and insertion."""
    ds = ds.drop(columns=[c for c in ('dec',) if c in ds]).drop_duplicates(['region', 'pid'])
    rows = []
    for reg in regions:
        f = C.PROJECT_DIR / f'results_{reg}' / 'prior_history_sessions.csv'
        if f.exists():
            d = pd.read_csv(f)
            c = 'me_prev_reward_timed' if 'me_prev_reward_timed' in d else 'prev_reward_timed'
            rows.append(pd.DataFrame(dict(pid=d['pid'], region=reg, dec=d[c] - d[f'{c}_null'])))
    return ds.merge(pd.concat(rows), on=['pid', 'region'], how='left') if rows and len(ds) else ds


def summarise_rate(df, regions, L):
    L.append('')
    L.append(f'=== E. Firing rates and units >= {RATE_MIN:g} Hz ===')
    for reg, g in df.groupby('region', sort=False):
        L.append(f'  {reg}: median unit rate {g["median_rate"].median():.1f} Hz (median over insertions); '
                 f'units >= {RATE_MIN:g} Hz: {g["n_units_hi"].sum()}/{g["n_units"].sum()} '
                 f'({100 * g["n_units_hi"].sum() / max(g["n_units"].sum(), 1):.0f}%); '
                 f'previous outcome, {RATE_N} units >= {RATE_MIN:g} Hz, controlled: {g["acc_hi"].mean():.3f} '
                 f'(n={g["acc_hi"].notna().sum()})')
    regs = [r for r in regions if (df['region'] == r).any()]
    for i, a in enumerate(regs):
        for b in regs[i + 1:]:
            x = df.loc[df['region'] == a, 'acc_hi'].dropna()
            y = df.loc[df['region'] == b, 'acc_hi'].dropna()
            if len(x) >= 5 and len(y) >= 5:
                L.append(f'    {a} vs {b}: Mann-Whitney p={stats.mannwhitneyu(x, y).pvalue:.3g}')


def plot_tc(df, regions):
    plt = K.setup_matplotlib()
    df = df.assign(d=df['acc'] - df['null'])
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), sharey=True, gridspec_kw=dict(width_ratios=[15, 7]))
    cols = dict(CA3='#1b4f72', CA1='#2e86c1', DG='#85c1e9', ZI='#c0392b', MOs='#d68910', VPM='#7d3c98')
    for ax, align, lab in ((axs[0], 'feedback', 'Time from previous feedback (s)'),
                           (axs[1], 'stim', 'Time from stimulus onset (s)')):
        for reg in regions:
            g = df[(df['align'] == align) & (df['region'] == reg)]
            if g.empty:
                continue
            m = g.groupby('t_center')['d'].mean()
            e = g.groupby('t_center')['d'].apply(K.sem)
            ax.plot(m.index, m.values, color=cols.get(reg, 'k'), lw=1.2, label=reg)
            ax.fill_between(m.index, m - e, m + e, color=cols.get(reg, 'k'), alpha=0.2, lw=0)
        ax.axhline(0, color='0.6', lw=0.6, ls='--')
        ax.axvline(0, color='0.6', lw=0.6)
        ax.set_xlabel(lab)
    axs[0].set_ylabel('Previous outcome\n(balanced acc. - null)')
    axs[1].legend(frameon=False, fontsize=6, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / 'outcome_timecourse.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'outcome_timecourse.pdf', bbox_inches='tight')
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
    L = [f'=== Step 19: follow-up of the previous-outcome signal ({", ".join(regions)}) ===', '']
    if 'subset' in args.parts:
        summarise_subset(regions, L)
    heavy = [p for p in args.parts if p != 'subset']
    collect = {k: [] for k in ('pupil', 'tc', 'beh', 'beh_sess', 'rate')}
    if heavy:
        from joblib import Parallel, delayed
        t_start = time.time()
        for ri, reg in enumerate(regions):
            sess = load_region(reg)
            t_all = [s['t'] for s in sess]
            t_pup = [s['t'] for s in sess if s['pupil'] is not None]
            print(f'{reg}: {len(sess)} insertions ({len(t_pup)} with pupil) | elapsed '
                  f'{(time.time() - t_start) / 60:.0f} min', flush=True)
            rng = np.random.default_rng([C.RANDOM_SEED + 19, ri])
            jobs = []
            for i, s in enumerate(sess):
                oth = [t_all[k] for k in rng.permutation(len(sess)) if k != i]
                othp = [t_all[k] for k in rng.permutation(len(sess)) if k != i and sess[k]['pupil'] is not None]
                jobs.append(delayed(one_insertion)(s, oth, othp, heavy, [C.RANDOM_SEED + 19, ri, i]))
            res = Parallel(n_jobs=args.jobs, verbose=5)(jobs)
            for r in res:
                for k in ('pupil', 'beh_sess', 'rate'):
                    if k in r:
                        collect[k].append(dict(r[k], region=reg))
                for k in ('tc', 'beh'):
                    collect[k] += [dict(x, region=reg) for x in r.get(k, [])]
    dfs = {k: pd.DataFrame(v) for k, v in collect.items()}
    if len(dfs['pupil']):
        dfs['pupil'].to_csv(OUT / 'outcome_pupil.csv', index=False)
        summarise_pupil(dfs['pupil'], L)
    if len(dfs['tc']):
        dfs['tc'].to_csv(OUT / 'outcome_timecourse.csv', index=False)
        summarise_tc(dfs['tc'], L)
        plot_tc(dfs['tc'], regions)
    if len(dfs['beh']):
        ds = session_table(dfs['beh_sess'], regions)
        dfs['beh'].to_csv(OUT / 'outcome_behaviour.csv', index=False)
        ds.to_csv(OUT / 'outcome_behaviour_sessions.csv', index=False)
        summarise_beh(dfs['beh'], ds, regions, L)
    if len(dfs['rate']):
        dfs['rate'].to_csv(OUT / 'outcome_rate.csv', index=False)
        summarise_rate(dfs['rate'], regions, L)
    (OUT / 'summary_19_outcome_followup.txt').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
