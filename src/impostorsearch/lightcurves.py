"""Light-curve metrics from ForcedSourceOnDiaObject photometry.

Fluxes are in nJy; AB magnitude m = -2.5 log10(f / nJy) + 31.4 (DP2 documentation).
Amplitudes use direct-image forced fluxes (psfFlux), which include underlying
host light, so measured amplitudes are conservative lower limits.
"""
import numpy as np
from astropy.table import Table

AB_ZP_NJY = 31.4
FLAG_COLUMNS = ("psfFlux_flag", "invalidPsfFlag",
                "pixelFlags_saturatedCenter", "pixelFlags_bad",
                "diff_PixelFlags_nodataCenter")


def njy_to_abmag(flux):
    flux = np.asarray(flux, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(flux > 0, -2.5 * np.log10(flux) + AB_ZP_NJY, np.nan)


def distance_modulus(dist_mpc):
    return 5.0 * np.log10(np.asarray(dist_mpc, dtype=float) * 1e6 / 10.0)


def clean_photometry(phot):
    """Drop flagged and non-finite rows."""
    ok = np.isfinite(np.asarray(phot["psfFlux"], dtype=float)) & \
         np.isfinite(np.asarray(phot["psfFluxErr"], dtype=float)) & \
         (np.asarray(phot["psfFluxErr"], dtype=float) > 0)
    for c in FLAG_COLUMNS:
        if c in phot.colnames:
            ok &= ~np.asarray(phot[c], dtype=bool)
    return phot[ok]


def _band_amplitude(flux, err, cfg):
    good = (flux > 0) & (flux / err >= cfg.snr_min)
    if good.sum() < cfg.min_good_epochs:
        return np.nan
    lo, hi = np.percentile(flux[good], cfg.amp_percentiles)
    return 2.5 * np.log10(hi / lo) if lo > 0 else np.nan


def object_metrics(rows, cfg):
    """Metrics for one object's photometry rows."""
    band = np.asarray(rows["band"]).astype(str)
    flux = np.asarray(rows["psfFlux"], dtype=float)
    err = np.asarray(rows["psfFluxErr"], dtype=float)
    mjd = np.asarray(rows["expMidptMJD"], dtype=float)

    amps = {b: _band_amplitude(flux[band == b], err[band == b], cfg) for b in np.unique(band)}
    finite = {b: a for b, a in amps.items() if np.isfinite(a)}
    amp_band = max(finite, key=finite.get) if finite else ""
    amplitude = finite.get(amp_band, np.nan)

    # peak: brightest significant epoch, preferring r, then g, i
    peak_band, peak_mag, peak_mjd = "", np.nan, np.nan
    for b in ("r", "g", "i", "z", "u", "y"):
        sel = (band == b) & (flux > 0) & (flux / err >= cfg.snr_min)
        if sel.any():
            k = np.argmax(np.where(sel, flux, -np.inf))
            peak_band, peak_mag, peak_mjd = b, float(njy_to_abmag(flux[k])), float(mjd[k])
            break

    # g - r at peak, from epochs paired within color_max_dt_days of the r peak
    color_gr = np.nan
    if peak_band == "r":
        g_sel = (band == "g") & (flux > 0) & (flux / err >= cfg.snr_min) & \
                (np.abs(mjd - peak_mjd) <= cfg.color_max_dt_days)
        if g_sel.any():
            kg = np.argmin(np.where(g_sel, np.abs(mjd - peak_mjd), np.inf))
            color_gr = float(njy_to_abmag(flux[kg])) - peak_mag

    return {
        "n_epochs": len(rows),
        "amp_band": amp_band,
        "amplitude_mag": amplitude,
        "peak_band": peak_band,
        "peak_mag": peak_mag,
        "peak_mjd": peak_mjd,
        "color_gr_peak": color_gr,
        "baseline_days": float(np.ptp(mjd)) if len(mjd) else np.nan,
    }


def compute_metrics(phot, matched, cfg):
    """Metrics for every matched object; adds peak absolute magnitude."""
    phot = clean_photometry(phot)
    ids = np.asarray(phot["diaObjectId"], dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    phot, ids = phot[order], ids[order]
    uniq, start = np.unique(ids, return_index=True)
    stop = np.append(start[1:], len(ids))

    records = []
    for oid, a, b in zip(uniq, start, stop):
        m = object_metrics(phot[a:b], cfg)
        m["diaObjectId"] = int(oid)
        records.append(m)
    if not records:
        return matched[:0]

    met = Table(rows=records)
    from astropy.table import join
    out = join(matched, met, keys="diaObjectId", join_type="inner")
    out["mu"] = distance_modulus(out["host_dist_mpc"])
    out["peak_abs_mag"] = np.asarray(out["peak_mag"], dtype=float) - np.asarray(out["mu"])
    return out
