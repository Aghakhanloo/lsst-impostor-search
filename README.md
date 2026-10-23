# lsst-impostor-search

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Rubin DP2](https://img.shields.io/badge/data-Rubin%20DP2-purple.svg)](https://doi.org/10.71929/rubin/3382528)

Tools to find **eruptive massive stars** (supernova impostors and luminous blue variable eruptions) in Rubin Observatory / LSST data, and to measure how often they erupt.

These stars brighten by several magnitudes in non-terminal eruptions and survive. Only a few dozen are known. Rubin's wide, deep, multi-band coverage makes a systematic search possible for the first time.

## What is in this repository

| Folder | What it does | Where it runs |
|---|---|---|
| [`rsp_search/`](rsp_search/) | Candidate search of Rubin Data Preview 2 (DP2): selects variable sources, assigns host galaxies and distances from HECATE, and retrieves forced-photometry light curves. Python package `impostorsearch` plus notebooks. | Rubin Science Platform |
| [`ml/`](ml/) | Machine-learning prototype: fine-tunes the pretrained ORACLE light-curve classifier to recognize simulated eruptions and compares it with training from scratch. | Laptop or GPU cluster (UVA Rivanna) |

Each folder has its own README with instructions.

## Quick start

**Candidate search (on the Rubin Science Platform):**

```bash
git clone https://github.com/Aghakhanloo/lsst-impostor-search.git
pip install --user -e lsst-impostor-search/rsp_search
```

Then open the notebooks in `rsp_search/notebooks/`.

**ML prototype (on a GPU cluster):**

```bash
git clone https://github.com/Aghakhanloo/lsst-impostor-search.git
cd lsst-impostor-search
sbatch ml/rivanna_job.slurm
```

## Data rights

DP2 data products are available only to Rubin data-rights holders. Query outputs are git-ignored and should not be committed. The ORACLE weights are downloaded by a script and are not redistributed here.

## Contact

Mojgan Aghakhanloo, University of Virginia: [mojgan.aghakhanloo@virginia.edu](mailto:mojgan.aghakhanloo@virginia.edu). Questions and bug reports are welcome as [GitHub issues](https://github.com/Aghakhanloo/lsst-impostor-search/issues).

## Citation

If you use this code, please cite it via `CITATION.cff` (Zenodo DOI to be added) and:

- Kovlakas, K., et al. 2021, MNRAS, 506, 1896 (HECATE): [https://doi.org/10.1093/mnras/stab1799](https://doi.org/10.1093/mnras/stab1799)
- Shah, V. G., et al. 2025 (ORACLE): [https://doi.org/10.3847/1538-4357/ae1130](https://doi.org/10.3847/1538-4357/ae1130), [arXiv:2501.01496](https://arxiv.org/abs/2501.01496)
- NSF-DOE Vera C. Rubin Observatory 2026, Legacy Survey of Space and Time Data Preview 2, [doi:10.71929/rubin/3382528](https://doi.org/10.71929/rubin/3382528)

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
