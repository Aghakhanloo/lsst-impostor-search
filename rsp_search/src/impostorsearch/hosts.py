"""Host-galaxy association with the HECATE catalog (Kovlakas et al. 2021,
https://doi.org/10.1093/mnras/stab1799)."""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table

HECATE_VIZIER_ID = "J/MNRAS/506/1896"


def load_hecate(path=None):
    """Read HECATE from a local file, or download it from VizieR.

    A local copy (e.g. the CSV from hecate.ia.forth.gr) avoids repeated downloads.
    """
    if path is not None:
        return Table.read(path)
    from astroquery.vizier import Vizier
    viz = Vizier(columns=["**"], row_limit=-1)
    cats = viz.get_catalogs(HECATE_VIZIER_ID)
    return max(cats, key=len)          # the main galaxy table is the largest


def _pick(table, candidates, role):
    for name in candidates:
        if name in table.colnames:
            return name
    raise KeyError(
        f"No HECATE column found for '{role}'. Tried {candidates}; "
        f"available columns: {table.colnames}. Edit SearchConfig.hecate_columns."
    )


def standardize_hecate(table, cfg):
    """Return a table with columns host_name, host_ra, host_dec, host_dist_mpc, host_r1_arcsec."""
    cols = cfg.hecate_columns
    out = Table()
    name_col = None
    for c in cols["name"]:
        if c in table.colnames:
            name_col = c
            break
    out["host_name"] = table[name_col].astype(str) if name_col else np.arange(len(table)).astype(str)
    out["host_ra"] = np.asarray(table[_pick(table, cols["ra"], "ra")], dtype=float)
    out["host_dec"] = np.asarray(table[_pick(table, cols["dec"], "dec")], dtype=float)
    out["host_dist_mpc"] = np.asarray(table[_pick(table, cols["dist_mpc"], "dist_mpc")], dtype=float)
    if cfg.offset_mode == "r1":
        r1 = np.asarray(table[_pick(table, cols["r1"], "r1")], dtype=float) * cfg.hecate_r1_unit
        out["host_r1_arcsec"] = r1.to_value(u.arcsec)
    else:
        out["host_r1_arcsec"] = np.full(len(table), np.nan)
    good = np.isfinite(out["host_dist_mpc"]) & (out["host_dist_mpc"] > 0)
    if cfg.offset_mode == "r1":
        good &= np.isfinite(out["host_r1_arcsec"]) & (out["host_r1_arcsec"] > 0)
    return out[good]


def match_to_hosts(objects, hosts, cfg, max_search_deg=3.0):
    """Associate each DiaObject with a HECATE host (criterion 3).

    offset_mode "r1": host minimising separation / R1, kept if <= cfg.host_max_norm_sep.
    offset_mode "kpc": host minimising projected offset, kept if <= cfg.offset_max_kpc.
    Normalising by galaxy size or by distance (rather than a fixed angle) makes the cut
    mean the same thing for galaxies at 5 Mpc and at 100 Mpc.

    Returns (matched, no_host). no_host holds objects with no cataloged host nearby,
    kept for manual inspection: SN impostors in faint, uncataloged dwarfs end up there.
    """
    if len(objects) == 0 or len(hosts) == 0:
        return objects[:0], objects
    obj_c = SkyCoord(objects["ra"] * u.deg, objects["dec"] * u.deg)
    host_c = SkyCoord(hosts["host_ra"] * u.deg, hosts["host_dec"] * u.deg)
    dist = np.asarray(hosts["host_dist_mpc"], dtype=float)
    r1 = np.asarray(hosts["host_r1_arcsec"], dtype=float)

    if cfg.offset_mode == "r1":
        radius_deg = cfg.host_max_norm_sep * np.nanmax(r1) / 3600.0
    elif cfg.offset_mode == "kpc":
        radius_deg = np.degrees(cfg.offset_max_kpc / (np.min(dist) * 1e3))
    else:
        raise ValueError("offset_mode must be 'r1' or 'kpc'")
    radius_deg = min(radius_deg, max_search_deg)

    i_obj, i_host, sep, _ = search_around_sky(obj_c, host_c, radius_deg * u.deg)
    sep_arcsec = sep.to_value(u.arcsec)
    offset_kpc = dist[i_host] * 1e3 * sep.to_value(u.rad)
    if cfg.offset_mode == "r1":
        score = sep_arcsec / r1[i_host]
        keep = score <= cfg.host_max_norm_sep
    else:
        score = offset_kpc
        keep = offset_kpc <= cfg.offset_max_kpc
    i_obj, i_host, sep_arcsec, offset_kpc, score = (
        a[keep] for a in (i_obj, i_host, sep_arcsec, offset_kpc, score))

    # best host per object = smallest score
    order = np.lexsort((score, i_obj))
    i_obj, i_host, sep_arcsec, offset_kpc, score = (
        a[order] for a in (i_obj, i_host, sep_arcsec, offset_kpc, score))
    first = np.ones(len(i_obj), dtype=bool)
    first[1:] = i_obj[1:] != i_obj[:-1]
    i_obj, i_host, sep_arcsec, offset_kpc, score = (
        a[first] for a in (i_obj, i_host, sep_arcsec, offset_kpc, score))

    out = objects[i_obj]
    for c in hosts.colnames:
        out[c] = hosts[c][i_host]
    out["host_sep_arcsec"] = sep_arcsec
    out["host_offset_kpc"] = offset_kpc
    out["host_norm_sep"] = sep_arcsec / r1[i_host]
    off_nucleus = sep_arcsec >= cfg.nucleus_min_sep_arcsec
    if cfg.offset_mode == "r1":
        off_nucleus &= out["host_norm_sep"] >= cfg.nucleus_min_norm_sep
    else:
        off_nucleus &= offset_kpc >= cfg.offset_min_kpc
    out["pass_offset"] = np.asarray(off_nucleus)

    unmatched = np.ones(len(objects), dtype=bool)
    unmatched[i_obj] = False
    return out, objects[unmatched]
