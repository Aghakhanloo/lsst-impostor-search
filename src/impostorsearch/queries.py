"""ADQL queries against the Rubin Science Platform TAP service.

Table and column names follow the DP2 documentation (dp2.lsst.io):
DiaObject, ForcedSourceOnDiaObject, and Visit (for expMidptMJD).
"""
import numpy as np
from astropy.table import Table, vstack

BANDS = ("u", "g", "r", "i", "z", "y")


def tap_client(schema="dp2"):
    """Return an authenticated TAP client. Only works inside the RSP."""
    from lsst.rsp import RSPDiscovery
    return RSPDiscovery(schema).get_tap_client()


def run_query(service, adql):
    """Run an async TAP job and return an astropy Table."""
    job = service.submit_job(adql)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR"])
    if job.phase == "ERROR":
        job.raise_if_error()
    table = job.fetch_result().to_table()
    job.delete()
    return table


def _amplitude_condition(b, amp_ratio, template_correction):
    """ADQL condition: total flux at maximum >= amp_ratio x total flux at minimum.

    Total flux at an epoch = template flux + difference flux. The template flux is
    estimated as scienceFluxMean - psfFluxMean (mean science flux minus mean
    difference flux), or as scienceFluxMean alone if template_correction is False.
    """
    tmpl = f"({b}_scienceFluxMean - {b}_psfFluxMean)" if template_correction else f"{b}_scienceFluxMean"
    return (f"({b}_psfFluxNdata >= {{ndata}} AND ({tmpl} + {b}_psfFluxMin) > 0 AND "
            f"({tmpl} + {b}_psfFluxMax) > {amp_ratio:.6f} * ({tmpl} + {b}_psfFluxMin))")


def diaobject_strip_query(schema, ra_min, ra_max, min_ndiasources, cfg=None):
    """Stage 1: DiaObjects in one RA strip that vary by >= cfg.min_amplitude_mag.

    The amplitude cut runs inside the database, so only variable sources come back.
    With cfg=None only the detection-count cut is applied.
    """
    bands = cfg.variability_bands if cfg is not None else ("g", "r", "i")
    cols = ["diaObjectId", "ra", "dec", "nDiaSources"]
    for b in bands:
        cols += [f"{b}_psfFluxMax", f"{b}_psfFluxMin", f"{b}_psfFluxMean",
                 f"{b}_psfFluxNdata", f"{b}_scienceFluxMean"]
    where = [f"ra >= {ra_min:.6f}", f"ra < {ra_max:.6f}",
             f"nDiaSources >= {int(min_ndiasources)}"]
    if cfg is not None:
        ratio = 10 ** (cfg.min_amplitude_mag / 2.5)
        amp = " OR ".join(_amplitude_condition(b, ratio, cfg.template_correction)
                          .format(ndata=int(cfg.min_ndata_band)) for b in bands)
        where.append(f"({amp})")
        ref = bands[0]
        where.append(f"{ref}_scienceFluxMean < {cfg.max_science_flux_njy:.6g}")
    return f"SELECT {', '.join(cols)} FROM {schema}.DiaObject WHERE " + " AND ".join(where)


def forced_photometry_query(schema, dia_object_ids):
    """Stage 3: forced photometry (direct and difference image) plus MJD."""
    ids = ", ".join(str(int(i)) for i in dia_object_ids)
    return (
        "SELECT fs.diaObjectId, fs.visit, fs.band, "
        "fs.psfFlux, fs.psfFluxErr, fs.psfDiffFlux, fs.psfDiffFluxErr, "
        "fs.psfFlux_flag, fs.psfDiffFlux_flag, fs.invalidPsfFlag, "
        "fs.pixelFlags_saturatedCenter, fs.pixelFlags_bad, fs.diff_PixelFlags_nodataCenter, "
        "v.expMidptMJD "
        f"FROM {schema}.ForcedSourceOnDiaObject AS fs "
        f"JOIN {schema}.Visit AS v ON v.visit = fs.visit "
        f"WHERE fs.diaObjectId IN ({ids})"
    )


def fetch_diaobjects(service, cfg, ra_range=(0.0, 360.0), verbose=True):
    """Whole-footprint (or RA-limited) DiaObject pre-filter, strip by strip."""
    edges = np.arange(ra_range[0], ra_range[1], cfg.ra_strip_deg)
    parts = []
    for lo in edges:
        hi = min(lo + cfg.ra_strip_deg, ra_range[1])
        t = run_query(service, diaobject_strip_query(cfg.schema, lo, hi, cfg.min_ndiasources, cfg))
        if verbose:
            print(f"RA {lo:6.1f}-{hi:6.1f}: {len(t):8d} DiaObjects")
        if len(t):
            parts.append(t)
    return vstack(parts) if parts else Table()


def fetch_forced_photometry(service, dia_object_ids, cfg, verbose=True):
    """Forced photometry for many objects, in batches."""
    ids = np.unique(np.asarray(dia_object_ids, dtype=np.int64))
    parts = []
    for k in range(0, len(ids), cfg.forced_batch_size):
        batch = ids[k:k + cfg.forced_batch_size]
        parts.append(run_query(service, forced_photometry_query(cfg.schema, batch)))
        if verbose:
            print(f"forced photometry: {min(k + len(batch), len(ids))}/{len(ids)} objects")
    return vstack(parts) if parts else Table()


def cone_diaobject_query(schema, ra, dec, radius_arcsec=1.0):
    """DiaObjects within a small radius of a position (e.g. a known LBV or impostor)."""
    return (
        f"SELECT diaObjectId, ra, dec, nDiaSources FROM {schema}.DiaObject "
        f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {radius_arcsec / 3600.0:.10f})) = 1"
    )


def lightcurve_at_position(service, ra, dec, radius_arcsec=1.0, schema="dp2"):
    """Forced-photometry light curve(s) of the DiaObject(s) at a sky position."""
    objs = run_query(service, cone_diaobject_query(schema, ra, dec, radius_arcsec))
    if len(objs) == 0:
        return objs, objs
    return objs, run_query(service, forced_photometry_query(schema, objs["diaObjectId"]))
