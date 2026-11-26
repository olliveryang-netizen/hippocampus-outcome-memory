"""
Step 01 - Download IBL data for every insertion with good units in the chosen region(s) and save compact
processed files in processed_<REGION>/<pid>/:
  trials.parquet  one row per included trial: task variables, wheel kinematics,
                  video motion energy in each analysis window
  units.parquet   one row per unit: id, CCF coordinates (x=ML, y=AP, z=DV, metres from bregma), firing rate
  binned.npz      spike counts (units x trials x 20-ms bins, -0.5..0.5 s) aligned to stimulus onset,
                  movement onset and feedback, plus wheel velocity aligned to movement onset
  meta.json       session identifiers

A probe that passes through several regions is downloaded once (--regions). With --clean, the raw files
downloaded for an insertion are deleted as soon as it is processed; files that were already in the cache
(for example from an earlier project) are kept. The script can be stopped and restarted.

Run:  python 01_download_preprocess.py --regions CA1 CA3 DG --min-units 5 --dry-run   (estimate only)
      python 01_download_preprocess.py --regions CA1 CA3 DG --min-units 5 --clean     (download + process)
"""
import argparse
import json
import logging
import time
import traceback

import numpy as np
import pandas as pd

import config as C

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger('01')


def window_mean(times, values, event_times, t0, t1):
    """Mean of a continuous signal in [event+t0, event+t1) for every trial."""
    cs = np.concatenate([[0], np.cumsum(np.nan_to_num(values))])
    a = np.searchsorted(times, event_times + t0)
    b = np.searchsorted(times, event_times + t1)
    n = b - a
    out = np.full(len(event_times), np.nan)
    good = n > 0
    out[good] = (cs[b[good]] - cs[a[good]]) / n[good]
    return out


def binned_counts(spike_times, spike_clusters, unit_ids, event_times, edges):
    """Counts (units x trials x bins) using searchsorted on sorted spike times."""
    out = np.zeros((len(unit_ids), len(event_times), len(edges) - 1), dtype=np.uint16)
    abs_edges = event_times[:, None] + edges[None, :]
    for i, u in enumerate(unit_ids):
        st = np.sort(spike_times[spike_clusters == u])
        idx = np.searchsorted(st, abs_edges)
        out[i] = np.diff(idx, axis=1)
    return out


def trial_features(trials, mask, wheel):
    t = trials.loc[mask.values].copy()
    t['trial_number'] = t.index.to_numpy()
    t = t[t['choice'] != 0]
    t['signed_contrast'] = t['contrastRight'].fillna(0) - t['contrastLeft'].fillna(0)
    t['correct'] = (t['feedbackType'] == 1).astype(int)
    t['rt'] = t['firstMovement_times'] - t['stimOn_times']
    end = t['response_times'].where(t['response_times'].notna(), t['feedback_times'])
    t['move_end'] = end

    wt = wheel['times'].to_numpy()
    wv = wheel['velocity'].to_numpy()
    wp = wheel['position'].to_numpy()
    i0 = np.searchsorted(wt, t['firstMovement_times'].to_numpy())
    i1 = np.searchsorted(wt, end.to_numpy())
    peak, mean, disp = (np.full(len(t), np.nan) for _ in range(3))
    for k, (a, b) in enumerate(zip(i0, i1)):
        if b > a:
            seg = np.abs(wv[a:b])
            peak[k], mean[k] = np.nanmax(seg), np.nanmean(seg)
            disp[k] = wp[min(b, len(wp) - 1)] - wp[a]
    t['peak_speed'], t['mean_speed'], t['wheel_disp'] = peak, mean, disp
    t['move_dur'] = end - t['firstMovement_times']
    return t.reset_index(drop=True)


def process_insertion(one, row, targets):
    """Load one insertion once and save processed files for every region in targets {region: out_dir}."""
    from brainwidemap import load_good_units, load_trials_and_mask
    from brainbox.io.one import SessionLoader
    from iblatlas.atlas import BrainRegions

    eid, pid, pname = row['eid'], row['pid'], row['probe_name']
    targets = {r: d for r, d in targets.items() if not (d / pid / 'meta.json').exists()}
    if not targets:
        log.info('skip (done) %s', pid)
        return

    trials, mask = load_trials_and_mask(one, eid)
    sl = SessionLoader(one=one, eid=eid)
    sl.load_wheel()
    t = trial_features(trials, mask, sl.wheel)

    # video motion energy (optional: some sessions lack good video)
    # each camera is loaded separately, so a missing body camera does not discard the whisker (left) camera
    has_me = {}
    for cam, view, key, col in (('whisker', 'left', 'leftCamera', 'whiskerMotionEnergy'),
                                ('body', 'body', 'bodyCamera', 'bodyMotionEnergy')):
        try:
            sl.load_motion_energy(views=[view])
            me = sl.motion_energy.get(key)
            if me is None or len(me) == 0:
                continue
            times, vals = me['times'].to_numpy(), me[col].to_numpy()
            for w, (ev, t0, t1) in C.WINDOWS.items():
                ev_t = t['stimOn_times' if ev == 'stimOn' else 'firstMovement_times'].to_numpy()
                t[f'me_{cam}_{w}'] = window_mean(times, vals, ev_t, t0, t1)
            has_me[cam] = True
        except Exception as e:
            log.info('no %s motion energy for %s: %r', cam, eid, e)

    spikes, clusters = load_good_units(one, pid, eid=eid, pname=pname)
    clusters['Beryl'] = BrainRegions().id2acronym(clusters['atlas_id'], mapping='Beryl')
    edges = np.round(np.arange(C.EDGES[0], C.EDGES[1] + C.BIN / 2, C.BIN), 6)
    centers = (edges[:-1] + edges[1:]) / 2
    fm = t['firstMovement_times'].to_numpy()
    so = t['stimOn_times'].to_numpy()
    fb = np.nan_to_num(t['feedback_times'].to_numpy(), nan=-1e6)        # outcome-aligned counts
    wv_fm = np.interp(fm[:, None] + centers[None, :], sl.wheel['times'].to_numpy(),
                      sl.wheel['velocity'].to_numpy()).astype(np.float32)

    for region, pdir in targets.items():
        sel = clusters[clusters['Beryl'] == region]
        if len(sel) == 0:
            log.warning('no %s units in %s', region, pid)
            continue
        unit_ids = sel.index.to_numpy()
        keep = [c for c in ('x', 'y', 'z', 'depths', 'firing_rate', 'amp_median', 'uuids', 'cluster_id', 'acronym')
                if c in sel.columns]
        units = sel[keep].copy()
        units.insert(0, 'unit_index', unit_ids)
        units['eid'], units['pid'], units['subject'], units['lab'] = eid, pid, row['subject'], row['lab']
        out = pdir / pid
        out.mkdir(parents=True, exist_ok=True)
        t.to_parquet(out / 'trials.parquet')
        units.reset_index(drop=True).to_parquet(out / 'units.parquet')
        np.savez_compressed(out / 'binned.npz',
                            fm_counts=binned_counts(spikes['times'], spikes['clusters'], unit_ids, fm, edges),
                            so_counts=binned_counts(spikes['times'], spikes['clusters'], unit_ids, so, edges),
                            fb_counts=binned_counts(spikes['times'], spikes['clusters'], unit_ids, fb, edges),
                            edges=edges, wheel_vel_fm=wv_fm)
        meta = {k: str(row[k]) for k in ('eid', 'pid', 'subject', 'lab', 'date', 'probe_name')}
        meta.update(region=region, n_units=int(len(unit_ids)), n_trials=int(len(t)), motion_energy=has_me)
        (out / 'meta.json').write_text(json.dumps(meta, indent=1))
        log.info('saved %s %s: %d units, %d trials', region, pid, len(unit_ids), len(t))


# ----------------------------------------------------------------------------
# Multi-region inventory, disk estimate and cache clean-up
# ----------------------------------------------------------------------------
def multi_inventory(one, regions, min_units):
    """One row per insertion with >= min_units good units in at least one of the regions."""
    from brainwidemap import bwm_query, download_aggregate_tables
    from iblatlas.atlas import BrainRegions
    bwm_df = bwm_query(freeze=C.BWM_FREEZE)
    clus = pd.read_parquet(download_aggregate_tables(one, type='clusters'))
    clus = clus[clus['eid'].isin(bwm_df['eid'])].copy()
    clus['Beryl'] = BrainRegions().id2acronym(clus['atlas_id'], mapping='Beryl')
    clus = clus[clus['Beryl'].isin(regions) & (clus['label'] >= C.MIN_QC)]
    inv = clus.groupby(['eid', 'pid', 'Beryl']).size().unstack(fill_value=0).reset_index()
    for r in regions:
        if r not in inv:
            inv[r] = 0
    inv = inv[(inv[list(regions)] >= min_units).any(axis=1)]
    inv = inv.merge(bwm_df[['eid', 'pid', 'subject', 'lab', 'date', 'probe_name']], on=['eid', 'pid'], how='left')
    inv.to_csv(C.PROJECT_DIR / 'inventory.csv', index=False)
    for r in regions:
        k = inv[r] >= min_units
        log.info('%s: %d units in %d insertions (%d mice) with >= %d units',
                 r, inv.loc[k, r].sum(), k.sum(), inv.loc[k, 'subject'].nunique(), min_units)
    log.info('insertions to process: %d (%d sessions)', len(inv), inv['eid'].nunique())
    return inv


def session_dir(one, row):
    try:
        d = one.eid2path(row['eid'])
    except Exception:
        return None
    return d if d is not None and d.exists() else None


def probe_size(one, row):
    """Bytes of spike-sorting files already cached for this insertion (0 if not cached)."""
    d = session_dir(one, row)
    if d is None:
        return 0
    return sum(f.stat().st_size for f in (d / 'alf' / row['probe_name']).rglob('*') if f.is_file())


def estimate(one, inv):
    sizes = np.array([probe_size(one, r) for _, r in inv.iterrows()])
    cached = sizes > 0
    mean_gb = sizes[cached].mean() / 1e9 if cached.any() else 1.0
    n_new = int((~cached).sum())
    log.info('already cached: %d insertions; to download: %d', cached.sum(), n_new)
    log.info('typical size per insertion (from cache): %.2f GB -> about %.0f GB to download in total',
             mean_gb, n_new * mean_gb)
    log.info('with --clean, at most about %.1f GB is on disk at any time', 2 * mean_gb)


def new_files(one, row, before):
    d = session_dir(one, row)
    if d is None:
        return []
    return [f for f in d.rglob('*') if f.is_file() and f not in before]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', nargs='+', default=None,
                    help='process several regions in one pass, e.g. --regions CA1 CA3 DG (default: config REGION)')
    ap.add_argument('--min-units', type=int, default=1,
                    help='only insertions with at least this many good units in a region (e.g. 5)')
    ap.add_argument('--clean', action='store_true',
                    help='after each insertion, delete the raw files that were downloaded for it')
    ap.add_argument('--dry-run', action='store_true', help='list insertions and estimate the download only')
    ap.add_argument('--max', type=int, default=None)
    args = ap.parse_args()

    one = C.get_one()
    log.info('IBL cache folder: %s', C.CACHE_DIR)
    regions = args.regions or [C.REGION]
    inv = multi_inventory(one, regions, args.min_units)
    if args.dry_run:
        estimate(one, inv)
        return
    todo = inv if args.max is None else inv.head(args.max)
    targets_all = {r: C.PROJECT_DIR / f'processed_{r}' for r in regions}
    failed = []
    t_start, removed_gb = time.time(), 0.0
    for i, (_, row) in enumerate(todo.iterrows(), 1):
        targets = {r: d for r, d in targets_all.items() if row[r] >= args.min_units}
        d0 = session_dir(one, row)
        before = set(f for f in d0.rglob('*') if f.is_file()) if (args.clean and d0 is not None) else set()
        try:
            log.info('[%d/%d] %s %s (%s)', i, len(todo), row['subject'], row['probe_name'],
                     ', '.join(f'{r} {row[r]}' for r in targets))
            process_insertion(one, row, targets)
        except Exception as e:
            failed.append({'eid': row['eid'], 'pid': row['pid'], 'error': repr(e)})
            log.warning('FAILED %s: %r', row['pid'], e)
            log.debug(traceback.format_exc())
        if args.clean:
            gone = new_files(one, row, before)
            size = sum(f.stat().st_size for f in gone)
            for f in gone:
                f.unlink(missing_ok=True)
            removed_gb += size / 1e9
            log.info('removed %d downloaded files (%.2f GB)', len(gone), size / 1e9)
        el = time.time() - t_start
        eta = el / i * (len(todo) - i)
        hm = lambda x: f'{int(x // 3600)}h{int(x % 3600 // 60):02d}m'
        log.info('progress %d/%d (%.0f%%) | elapsed %s | remaining about %s | downloaded and removed so far %.1f GB',
                 i, len(todo), 100 * i / len(todo), hm(el), hm(eta), removed_gb)
    pd.DataFrame(failed, columns=['eid', 'pid', 'error']).to_csv(C.PROJECT_DIR / 'download_failures.csv', index=False)
    log.info('done. failed: %d', len(failed))


if __name__ == '__main__':
    main()
