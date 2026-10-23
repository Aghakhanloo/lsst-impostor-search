"""End-to-end search: DiaObjects -> HECATE hosts -> forced photometry -> cuts."""
from pathlib import Path

import numpy as np

from .config import SearchConfig
from .queries import fetch_diaobjects, fetch_forced_photometry
from .hosts import standardize_hecate, match_to_hosts
from .lightcurves import compute_metrics
from .select import apply_cuts


def run_search(service, hecate_raw, cfg=None, ra_range=(0.0, 360.0), out_dir="outputs"):
    cfg = cfg or SearchConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    objects = fetch_diaobjects(service, cfg, ra_range=ra_range)
    print(f"Stage 1: {len(objects)} DiaObjects varying by >= {cfg.min_amplitude_mag} mag "
          f"in {'/'.join(cfg.variability_bands)}")

    hosts = standardize_hecate(hecate_raw, cfg)
    matched, no_host = match_to_hosts(objects, hosts, cfg)
    print(f"Stage 2: {len(matched)} within {cfg.host_max_norm_sep} R1 of a HECATE galaxy; "
          f"{int(np.sum(matched['pass_offset'])) if len(matched) else 0} offset from the nucleus; "
          f"{len(no_host)} with no catalogued host (saved for inspection)")

    phot = fetch_forced_photometry(service, matched["diaObjectId"], cfg)
    metrics = apply_cuts(compute_metrics(phot, matched, cfg), cfg)
    cands = metrics[metrics["is_candidate"]]
    cands.sort("peak_abs_mag")
    print(f"Stage 3: {len(cands)} candidates")

    for t in (metrics, cands):
        t.meta["search_config"] = cfg.as_meta()
    metrics.write(out / "all_matched_metrics.ecsv", overwrite=True)
    cands.write(out / "candidates.ecsv", overwrite=True)
    no_host.meta["search_config"] = cfg.as_meta()
    no_host.write(out / "no_host.ecsv", overwrite=True)
    if len(phot):
        phot.write(out / "forced_photometry.fits", overwrite=True)
    return cands, metrics, phot
