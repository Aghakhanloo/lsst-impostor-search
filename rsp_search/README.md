# DP2 candidate search on the Rubin Science Platform

Search Rubin Observatory / LSST difference-imaging data for **supernova (SN) impostors and luminous blue variable (LBV) eruptions** in nearby galaxies.

LBVs and SN impostors are massive stars that undergo non-terminal eruptions, brightening by magnitudes while the star survives. Only a few dozen are known. Rubin's wide, deep, multi-band coverage makes a systematic search possible for the first time. This package implements the candidate search used for Rubin Data Preview 2 (DP2) and is designed to be rerun on each new data release.

## Method

The search is **transient-first** over the whole footprint:

| Stage | What it does | Where |
|---|---|---|
| 1 | Select `DiaObject`s whose total flux (template + difference flux) varies by at least 1 mag in r or g, queried in RA strips | ADQL on the RSP TAP service |
| 2 | Associate each with a HECATE galaxy ([Kovlakas et al. 2021](https://doi.org/10.1093/mnras/stab1799)), by separation / galaxy radius or by projected offset in kpc; keep objects off the nucleus, and save those with no cataloged host for inspection | `astropy` cross-match |
| 3 | Retrieve forced photometry (`ForcedSourceOnDiaObject` joined to `Visit`) and apply the light-curve cuts | Python |

Selection criteria (all configurable in `SearchConfig`):

1. **Variability:** amplitude of at least 1 mag (5th to 95th percentile of significant forced fluxes in any band).
2. **Environment:** within 2 host radii of a HECATE galaxy, but not at its nucleus, to avoid AGN.
3. **Luminosity:** peak absolute magnitude between -17 and -7, using the HECATE distance.

## Quick start (Rubin Science Platform)

Requires a Rubin Science Platform account (Rubin data rights). From a terminal on the RSP Notebook aspect:

```bash
git clone https://github.com/Aghakhanloo/lsst-impostor-search.git
pip install --user -e lsst-impostor-search/rsp_search
```

Then open the notebooks:

- `notebooks/00_single_object_lightcurve.ipynb`: the DP2 light curve of one target (cone search, then forced photometry).
- `notebooks/02_full_pipeline.ipynb`: the whole-footprint search through the `impostorsearch` package, run in RA strips, with forced-photometry light-curve cuts and saved outputs.

```python
from impostorsearch import SearchConfig, load_hecate
from impostorsearch.queries import tap_client
from impostorsearch.pipeline import run_search

cfg = SearchConfig()
cands, metrics, phot = run_search(tap_client("dp2"), load_hecate(), cfg,
                                  ra_range=(50, 60), out_dir="outputs/test")
```

Outputs are written to `outputs/`: `candidates.ecsv`, `no_host.ecsv`, `all_matched_metrics.ecsv` (every host-matched object, with a pass/fail column per criterion), and `forced_photometry.fits`. Each file stores the full `SearchConfig` in its metadata.

## Tests

Automatic checks of the search code, run on made-up light curves where the right answer is known. They need no Rubin access and take a few seconds:

```bash
cd rsp_search
pip install -e ".[test]"
pytest
```

## Limitations

- Amplitudes use direct-image forced photometry, which includes host light, so they are lower limits.
- Absolute magnitudes are not corrected for extinction.
- Galaxy sizes from HECATE (`R1`) are assumed to be in arcmin; change `SearchConfig.hecate_r1_unit` if needed.

