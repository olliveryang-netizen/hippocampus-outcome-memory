"""
Shared settings for the hippocampus analyses (CA1, CA3, DG; any Beryl region can be analysed).

Processed data, results and figures are stored inside the folder that holds these scripts:

  <project>/processed_<REGION>/  compact per-insertion files made by 01_download_preprocess.py
  <project>/results_<REGION>/    tables (.csv) and summary numbers (.txt)
  <project>/figures_<REGION>/    quick-look figures made by the individual steps (.png, .pdf)
  <project>/results_comparison/  results that compare regions

Raw IBL files are cached in ibl_cache/. If the parent folder already has an ibl_cache/ (from an earlier
project), that cache is used so that files already downloaded are not downloaded again.
"""
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Region to analyse, e.g.  python run_all.py --region CA3  or  ANALYSIS_REGION=CA3 python 16_prior_history.py
REGION = os.environ.get('ANALYSIS_REGION', 'CA1')
_SUFFIX = f'_{REGION}'

CACHE_DIR = (Path(os.environ['IBL_CACHE']) if 'IBL_CACHE' in os.environ
             else PROJECT_DIR.parent / 'ibl_cache' if (PROJECT_DIR.parent / 'ibl_cache').exists()
             else PROJECT_DIR / 'ibl_cache')
PROCESSED_DIR = PROJECT_DIR / f'processed{_SUFFIX}'
RESULTS_DIR = PROJECT_DIR / f'results{_SUFFIX}'
FIGURES_DIR = PROJECT_DIR / f'figures{_SUFFIX}'
for _d in (CACHE_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(exist_ok=True)

# IBL brain-wide-map code (downloaded ZIP unpacked next to these scripts)
for _cand in (PROJECT_DIR / 'paper-brain-wide-map', PROJECT_DIR / 'paper-brain-wide-map-main',
              PROJECT_DIR.parent / 'paper-brain-wide-map', PROJECT_DIR.parent / 'paper-brain-wide-map-main'):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break

# macOS python.org builds sometimes lack root certificates -> use certifi's bundle
try:
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
except ImportError:
    pass

# ----------------------------------------------------------------------------
# Data selection
# ----------------------------------------------------------------------------
MIN_QC = 1.0                   # IBL "good unit" label
BWM_FREEZE = '2023_12_bwm_release'
MIN_UNITS_POPULATION = 5       # min units of the region per insertion for population analyses (decoding)

# ----------------------------------------------------------------------------
# Time binning (seconds)
# ----------------------------------------------------------------------------
BIN = 0.02
EDGES = (-0.5, 0.5)            # binned spike counts around each alignment event

# analysis windows: name -> (alignment event, start, end)
WINDOWS = {
    'iti':     ('stimOn',    -0.40, -0.10),   # quiescent period: block prior
    'premove': ('firstMove', -0.16,  0.00),   # decision -> action (main choice window)
    'move':    ('firstMove',  0.00,  0.20),   # early movement
}

# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
N_PERM = 2000                  # Freedman-Lane permutations per session (GLM); min p = 1/2001
N_PERM_DECODING = 200          # label permutations for decoding null
N_SUBSAMPLE = 50               # movement-matched resamples for decoding
N_SPEED_BINS = 5               # speed quantile bins for movement matching
ALPHA = 0.05
RANDOM_SEED = 2026


def get_one():
    """Connect to the public IBL database, caching files in the project folder."""
    from one.api import ONE
    return ONE(base_url='https://openalyx.internationalbrainlab.org',
               username='intbrainlab', password='international',
               cache_dir=CACHE_DIR, silent=True)


def save_fig(fig, name):
    """Save a figure as PNG (viewing) and PDF (journal submission)."""
    fig.savefig(FIGURES_DIR / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / f'{name}.pdf', bbox_inches='tight')


def write_summary(name, lines):
    """Write key numbers for the manuscript to results/<name>.txt and print them."""
    text = '\n'.join(lines)
    (RESULTS_DIR / f'{name}.txt').write_text(text, encoding='utf-8')
    print(text)
