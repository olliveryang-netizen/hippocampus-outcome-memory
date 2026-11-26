"""
Step 17 - Add lick counts to the processed trial tables.

For every processed insertion of the given regions, loads the IBL lick times (detected from video) of the
session and adds two columns to trials.parquet:
  licks_iti     number of licks in the pre-stimulus window (-400 to -100 ms from stimulus onset)
  licks_post    number of licks in the 1 s after feedback
Sessions without lick data get NaN. Only the small lick file is downloaded. Run once, before step 16:

    python 17_licks.py --regions CA1 CA3 DG
"""
import argparse
import json
import logging

import numpy as np
import pandas as pd

import config as C

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger('licks')


def count(times, ev, t0, t1):
    lo = np.searchsorted(times, ev + t0)
    hi = np.searchsorted(times, ev + t1)
    out = (hi - lo).astype(float)
    out[~np.isfinite(ev)] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=['CA1', 'CA3', 'DG'])
    args = ap.parse_args()
    one = C.get_one()
    _, t0, t1 = C.WINDOWS['iti']
    cache = {}
    n_ok = n_missing = 0
    for reg in args.regions:
        pdir = C.PROJECT_DIR / f'processed_{reg}'
        for path in sorted(q for q in pdir.iterdir() if (q / 'meta.json').exists()):
            eid = json.loads((path / 'meta.json').read_text())['eid']
            if eid not in cache:
                try:
                    cache[eid] = np.sort(one.load_dataset(eid, 'licks.times.npy', collection='alf'))
                except Exception as e:
                    log.info('no lick data for %s: %r', eid, e)
                    cache[eid] = None
            f = path / 'trials.parquet'
            t = pd.read_parquet(f)
            lk = cache[eid]
            if lk is None:
                t['licks_iti'] = np.nan
                t['licks_post'] = np.nan
                n_missing += 1
            else:
                t['licks_iti'] = count(lk, t['stimOn_times'].to_numpy(), t0, t1)
                t['licks_post'] = count(lk, t['feedback_times'].to_numpy(), 0.0, 1.0)
                n_ok += 1
            t.to_parquet(f)
    log.info('insertions with lick data: %d, without: %d', n_ok, n_missing)


if __name__ == '__main__':
    main()
