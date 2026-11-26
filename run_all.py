"""
Run the analysis steps for one region. The data must already be in processed_<REGION>/
(made once for all regions with 01_download_preprocess.py --regions CA1 CA3 DG --min-units 5 --clean).

  python run_all.py --region CA3            # steps 02-07, 09, 10 and 16
  python run_all.py --region CA3 --only 16  # one step
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ['02_dataset_behavior.py', '03_single_unit_glm.py', '04_decoding.py', '05_subspace.py', '06_location.py',
         '07_imposter_null.py', '09_followup.py', '10_reversal_history.py', '16_prior_history.py']

ap = argparse.ArgumentParser()
ap.add_argument('--only', default=None, help='step number, e.g. 03')
ap.add_argument('--region', default='CA1', help='Beryl acronym, e.g. CA1 (default), CA3, DG')
args = ap.parse_args()
env = dict(os.environ, ANALYSIS_REGION=args.region)
print(f'Region: {args.region}')
if not (HERE / f'processed_{args.region}').exists():
    sys.exit(f'processed_{args.region}/ not found: run 01_download_preprocess.py first')

steps = [s for s in STEPS if s.startswith(args.only)] if args.only else STEPS
for s in steps:
    t0 = time.time()
    print(f'\n========== {s} ==========', flush=True)
    r = subprocess.run([sys.executable, str(HERE / s)], cwd=HERE, env=env)
    if r.returncode != 0:
        sys.exit(f'{s} failed (exit code {r.returncode}). Fix the error above, then rerun with --only {s[:2]} --region {args.region}')
    print(f'{s} finished in {(time.time() - t0) / 60:.1f} min', flush=True)
print(f'\nAll done. Tables and summary_*.txt in results_{args.region}/, quick-look figures in figures_{args.region}/.')
