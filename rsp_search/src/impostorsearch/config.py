"""Search configuration.

All selection thresholds live here so a run is fully described by one
SearchConfig and can be written to the output metadata.
"""
from dataclasses import dataclass, field, asdict

import astropy.units as u


@dataclass
class SearchConfig:
    # --- data release ---
    schema: str = "dp2"                 # TAP schema on the Rubin Science Platform

    # --- stage 1: DiaObject pre-filter (run in ADQL) ---
    min_ndiasources: int = 5            # minimum number of difference-image detections, all bands
    min_ndata_band: int = 5             # minimum detections in each variability band
    variability_bands: tuple = ("r", "g")  # amplitude cut passes if ANY of these bands varies
    max_science_flux_njy: float = 1.45e6   # reject very bright/saturated sources (AB mag < 16)
    template_correction: bool = True    # estimate template flux as scienceFluxMean - psfFluxMean
    ra_strip_deg: float = 10.0          # whole-footprint query is split into RA strips

    # --- stage 2: host association (HECATE) ---
    offset_mode: str = "r1"             # "r1": separation / host radius; "kpc": projected offset
    host_max_norm_sep: float = 2.0      # (r1 mode) max separation in units of the host radius R1
    offset_min_kpc: float = 0.5         # (kpc mode) min projected offset from the nucleus
    offset_max_kpc: float = 50.0        # (kpc mode) max projected offset for association
    nucleus_min_sep_arcsec: float = 1.5 # reject objects closer than this to the host center
    nucleus_min_norm_sep: float = 0.05  # ... or closer than this fraction of R1 (AGN guard)
    hecate_r1_unit: u.Unit = u.arcmin   # unit of the HECATE size column (verify on first use)

    # --- stage 3: light-curve cuts (criteria 1 and 3) ---
    snr_min: float = 5.0                # per-epoch S/N for a flux to count
    min_good_epochs: int = 3            # per band, to measure an amplitude
    min_amplitude_mag: float = 1.0      # criterion 1: variation of at least 1 mag
    amp_percentiles: tuple = (5.0, 95.0)  # robust min/max of the flux distribution
    color_max_dt_days: float = 1.0      # g - r at peak is reported (not used as a cut); g and r paired within this window
    abs_mag_bright: float = -17.0       # criterion 3: -17 < M < -7
    abs_mag_faint: float = -7.0

    # --- bookkeeping ---
    forced_batch_size: int = 500        # diaObjectIds per forced-photometry query
    hecate_columns: dict = field(default_factory=lambda: {
        "ra": ["RA", "RAJ2000", "_RA"],
        "dec": ["DEC", "DEJ2000", "DE", "_DE"],
        "dist_mpc": ["D", "Dist"],
        "r1": ["R1", "r1"],
        "name": ["objname", "Name", "PGC"],
    })

    def as_meta(self):
        """Plain-dict version for table metadata."""
        d = asdict(self)
        d["hecate_r1_unit"] = str(self.hecate_r1_unit)
        return d
