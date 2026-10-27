"""Apply the three SN-impostor / LBV selection criteria."""
import numpy as np


def apply_cuts(metrics, cfg):
    """Add pass_* columns and the final is_candidate flag.

    1. amplitude >= cfg.min_amplitude_mag
    2. near a host galaxy but offset from its nucleus (pass_offset, set in match_to_hosts)
    3. cfg.abs_mag_bright < peak absolute magnitude < cfg.abs_mag_faint
    The g - r color at peak (color_gr_peak) is reported for inspection but not used as a cut.
    """
    t = metrics
    amp = np.asarray(t["amplitude_mag"], dtype=float)
    absmag = np.asarray(t["peak_abs_mag"], dtype=float)

    t["pass_amplitude"] = np.isfinite(amp) & (amp >= cfg.min_amplitude_mag)
    t["pass_absmag"] = np.isfinite(absmag) & (absmag > cfg.abs_mag_bright) & (absmag < cfg.abs_mag_faint)
    t["is_candidate"] = (t["pass_amplitude"] & np.asarray(t["pass_offset"], dtype=bool) & t["pass_absmag"])
    return t
