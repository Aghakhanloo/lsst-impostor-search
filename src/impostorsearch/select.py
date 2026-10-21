"""Apply the four SN-impostor / LBV selection criteria."""
import numpy as np


def apply_cuts(metrics, cfg):
    """Add pass_* columns and the final is_candidate flag.

    1. amplitude >= cfg.min_amplitude_mag
    2. blue at peak: g - r < cfg.max_color_gr (missing colour handled by cfg.require_color)
    3. near a host galaxy but offset from its nucleus (pass_offset, set in match_to_hosts)
    4. cfg.abs_mag_bright < peak absolute magnitude < cfg.abs_mag_faint
    """
    t = metrics
    amp = np.asarray(t["amplitude_mag"], dtype=float)
    col = np.asarray(t["color_gr_peak"], dtype=float)
    absmag = np.asarray(t["peak_abs_mag"], dtype=float)

    t["pass_amplitude"] = np.isfinite(amp) & (amp >= cfg.min_amplitude_mag)
    t["color_missing"] = ~np.isfinite(col)
    blue = np.isfinite(col) & (col < cfg.max_color_gr)
    t["pass_color"] = blue if cfg.require_color else (blue | ~np.isfinite(col))
    t["pass_absmag"] = np.isfinite(absmag) & (absmag > cfg.abs_mag_bright) & (absmag < cfg.abs_mag_faint)
    t["is_candidate"] = (t["pass_amplitude"] & t["pass_color"]
                         & np.asarray(t["pass_offset"], dtype=bool) & t["pass_absmag"])
    return t
