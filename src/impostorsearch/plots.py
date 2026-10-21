"""Quick-look light-curve plot."""
import numpy as np
import matplotlib.pyplot as plt

from .lightcurves import clean_photometry, njy_to_abmag

COLORS = {"u": "#1f77b4", "g": "#2ca02c", "r": "#d62728",
          "i": "#ff7f0e", "z": "#9467bd", "y": "#8c564b"}


def plot_lightcurve(phot, dia_object_id, row=None, ax=None, snr_min=3.0):
    p = clean_photometry(phot[np.asarray(phot["diaObjectId"]) == dia_object_id])
    ax = ax or plt.subplots(figsize=(7, 3.5))[1]
    for b in "ugrizy":
        s = (np.asarray(p["band"]).astype(str) == b)
        f, e = np.asarray(p["psfFlux"][s], float), np.asarray(p["psfFluxErr"][s], float)
        good = (f > 0) & (f / e >= snr_min)
        if good.any():
            m = njy_to_abmag(f[good])
            me = 1.0857 * e[good] / f[good]
            ax.errorbar(np.asarray(p["expMidptMJD"][s])[good], m, yerr=me, fmt="o", ms=3,
                        color=COLORS[b], label=b)
    ax.invert_yaxis()
    ax.set_xlabel("MJD")
    ax.set_ylabel("AB mag (forced, direct image)")
    title = f"diaObjectId {dia_object_id}"
    if row is not None:
        title += (f"  |  host {row['host_name']}  D={row['host_dist_mpc']:.1f} Mpc"
                  f"  M_peak={row['peak_abs_mag']:.1f}")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, ncol=6)
    return ax
