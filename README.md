# Hippocampal outcome memory in the IBL Brain-Wide Map

Analysis code for the manuscript "Mouse CA1 and CA3 retain the previous outcome, but not choice history, between perceptual decisions".

The scripts analyze CA1, CA3 and dentate gyrus (DG) recordings from the International Brain Laboratory (IBL) Brain-Wide Map (data release `2023_12_bwm_release`). Zona incerta (ZI), secondary motor cortex (MOs) and ventral posteromedial thalamus (VPM) are analyzed in the same way for comparison. No new data were collected. All data are downloaded from the public IBL database (OpenAlyx) by the first script.

## Requirements

- Python 3 with the packages listed in `requirements.txt` (exported from the environment used for the paper). The main ones are ONE-api, ibllib, brainbox, numpy, scipy, pandas, scikit-learn, matplotlib and pyarrow.
- The IBL Brain-Wide Map analysis code (<https://github.com/int-brain-lab/paper-brain-wide-map>), unpacked next to these scripts or one folder up. `config.py` adds it to the Python path.
- Enough disk space for the raw files of one insertion at a time: with `--clean`, raw files are deleted after each insertion has been preprocessed.

```bash
python -m venv ibl_env
source ibl_env/bin/activate
pip install -r requirements.txt
```

## Reproducing the paper

Run the commands from this folder. Each step writes tables and a `summary_*.txt` file to `results_<REGION>/` or `results_comparison/`.

```bash
REGIONS="CA3 CA1 DG ZI MOs VPM"

# 1. download and preprocess (insertions with at least 5 units in a region)
python 01_download_preprocess.py --regions $REGIONS --min-units 5 --clean

# 2. licking and pupil covariates (run before step 16)
python 17_licks.py --regions $REGIONS
python 18_pupil.py --regions $REGIONS --clean

# 3. block prior, previous choice and previous outcome decoding, per region
for R in $REGIONS; do ANALYSIS_REGION=$R python 16_prior_history.py; done

# 4. comparison with equal numbers of units
python 16_prior_history.py --compare --regions $REGIONS

# 5. outcome analyses: time course, pupil, behavior, firing rate, lags, cross-temporal decoding
python 19_outcome_followup.py --regions $REGIONS
python 20_outcome_memory.py --regions $REGIONS
python 21_final_checks.py --regions $REGIONS

# 6. figures 1-6 and every number and test shown in them
python 22_make_figures.py --regions $REGIONS
```

Figures are written to `figures_paper/` as PDF (vector) and PNG (600 dpi). `figures_paper/figure_stats.txt` lists every value and statistical test shown in the figures.

Steps 02 to 13 (`run_all.py`) are exploratory analyses carried over from an earlier project. They are not needed for the figures in the paper.

## Files

| Script | What it does |
| --- | --- |
| `config.py`, `common.py` | Paths, analysis windows, shared helpers |
| `01_download_preprocess.py` | Downloads spikes, trials, wheel and video data; keeps binned counts for each insertion |
| `16_prior_history.py` | Block, previous-choice and previous-outcome decoders with stratified balancing, pseudosession and imposter nulls, and covariate regression |
| `17_licks.py`, `18_pupil.py` | Lick counts and pupil diameter for each trial |
| `19_outcome_followup.py` | Same-insertion controls, pupil stratification and gain, time course, behavior link, firing-rate control |
| `20_outcome_memory.py` | Past and future outcomes (lags), cross-temporal decoding, persistence |
| `21_final_checks.py` | Outcome two trials back with the intervening trial balanced; statistics summaries |
| `22_make_figures.py` | Main figures and `figure_stats.txt` |

## Statistics

Values in the figures are compared across mice with a cluster bootstrap (10,000 resamples of mice, all insertions of a resampled mouse kept), with Holm correction within each figure panel. Random seeds are fixed in `config.py`.

## Data

International Brain Laboratory et al. (2025). A brain-wide map of neural activity during complex behaviour. Nature. Data release `2023_12_bwm_release`, accessed through the ONE API from the public OpenAlyx server (<https://openalyx.internationalbrainlab.org>).

## License

MIT, see `LICENSE`.
