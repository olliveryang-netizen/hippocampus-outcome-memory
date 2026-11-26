"""
Step 18 - Add pupil diameter (arousal) to the processed trial tables.

For every processed insertion of the given regions, loads the smoothed pupil diameter of the left camera
(IBL video features, or computed from DLC if the features are missing), z-scores it within the session and
adds to trials.parquet:
  pupil_iti     mean pupil diameter in the pre-stimulus window (-400 to -100 ms from stimulus onset)
Sessions without usable pupil data get NaN.

With --clean, video files downloaded for this step are deleted after use (files already in the cache are
kept). Run once, before step 16:

    python 18_pupil.py --regions CA1 CA3 DG ZI MOs VPM --clean
"""
import argparse
import json
import logging

import numpy as np
import pandas as pd

import config as C

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger('pupil')
logging.getLogger('brainbox').setLevel(logging.ERROR)


def window_mean(times, vals, ev, t0, t1):
    lo = np.searchsorted(times, ev + t0)
    hi = np.searchsorted(times, ev + t1)
    out = np.full(len(ev), np.nan)
    for k, (a, b) in enumerate(zip(lo, hi)):
        if b > a and np.isfinite(ev[k]):
            seg = vals[a:b]
            if np.isfinite(seg).mean() > 0.5:
                out[k] = np.nanmean(seg)
    return out


def files_in(d):
    return set(f for f in d.rglob('*') if f.is_file()) if d is not None and d.exists() else set()


def load_pupil(one, eid):
    from brainbox.io.one import SessionLoader
    sl = SessionLoader(one=one, eid=eid)
    sl.load_pupil()
    p = sl.pupil
    if p is None or len(p) == 0 or 'pupilDiameter_smooth' not in p:
        return None
    v = p['pupilDiameter_smooth'].to_numpy(float)
    if np.isfinite(v).mean() < 0.5:
        return None
    v = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
    return p['times'].to_numpy(float), v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=['CA1', 'CA3', 'DG'])
    ap.add_argument('--clean', action='store_true', help='delete video files downloaded for this step')
    args = ap.parse_args()
    one = C.get_one()
    _, t0, t1 = C.WINDOWS['iti']
    paths = []
    for reg in args.regions:
        pdir = C.PROJECT_DIR / f'processed_{reg}'
        if pdir.exists():
            paths += [q for q in sorted(pdir.iterdir()) if (q / 'meta.json').exists()]
    by_eid = {}
    for q in paths:
        by_eid.setdefault(json.loads((q / 'meta.json').read_text())['eid'], []).append(q)
    log.info('%d sessions, %d insertions', len(by_eid), len(paths))
    n_ok = 0
    for i, (eid, qs) in enumerate(by_eid.items(), 1):
        done = all('pupil_iti' in pd.read_parquet(q / 'trials.parquet', columns=None).columns for q in qs)
        if done:
            continue
        try:
            sdir = one.eid2path(eid)
        except Exception:
            sdir = None
        before = files_in(sdir)
        try:
            pup = load_pupil(one, eid)
        except Exception as e:
            log.info('no pupil for %s: %r', eid, e)
            pup = None
        for q in qs:
            t = pd.read_parquet(q / 'trials.parquet')
            t['pupil_iti'] = (window_mean(*pup, t['stimOn_times'].to_numpy(), t0, t1)
                              if pup is not None else np.nan)
            t.to_parquet(q / 'trials.parquet')
        n_ok += pup is not None
        if args.clean and sdir is not None:
            new = files_in(sdir) - before
            for f in new:
                f.unlink(missing_ok=True)
        log.info('[%d/%d] %s pupil %s', i, len(by_eid), eid[:8], 'ok' if pup is not None else 'missing')
    log.info('sessions with pupil data: %d of %d', n_ok, len(by_eid))


if __name__ == '__main__':
    main()
