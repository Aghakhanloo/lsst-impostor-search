"""Offline tests with synthetic data (no RSP access needed)."""
import numpy as np
import astropy.units as u
from astropy.table import Table

from impostorsearch import (SearchConfig, standardize_hecate, match_to_hosts,
                            compute_metrics, apply_cuts, njy_to_abmag, distance_modulus)
from impostorsearch.queries import diaobject_strip_query, forced_photometry_query


def mag_to_njy(m):
    return 10 ** (-0.4 * (np.asarray(m, float) - 31.4))


def make_hosts():
    # one galaxy at 10 Mpc (mu = 30) with R1 = 1 arcmin
    return Table({"objname": ["NGC_TEST"], "RA": [150.0], "DEC": [2.0],
                  "D": [10.0], "R1": [1.0]})


def make_objects():
    dec0, ra0 = 2.0, 150.0
    cosd = np.cos(np.deg2rad(dec0))
    off = lambda arcsec: ra0 + arcsec / 3600 / cosd
    return Table({
        "diaObjectId": np.array([1, 2, 3, 4, 5], dtype=np.int64),
        "ra": [off(30), off(30), off(0.5), off(400), off(40)],   # 4 is far from host, 3 at nucleus
        "dec": [dec0] * 5,
        "nDiaSources": [10] * 5,
    })


def lightcurve(oid, quiescent, peak, gr_peak=0.2, n=12):
    """r and g epochs; one bright epoch; g taken 0.3 d after the r peak."""
    rows = []
    mjd = np.arange(n) * 5.0
    rmag = np.full(n, quiescent)
    rmag[6] = peak
    for t, m in zip(mjd, rmag):
        f = mag_to_njy(m)
        rows.append((oid, "r", f, f / 50, t))
    gmag = rmag + 0.6
    gmag[6] = peak + gr_peak
    for t, m in zip(mjd + 0.3, gmag):
        f = mag_to_njy(m)
        rows.append((oid, "g", f, f / 50, t))
    return rows


def make_phot():
    rows = []
    rows += lightcurve(1, 22.5, 19.0)             # good impostor: amp 3.5, M_peak = -11
    rows += lightcurve(2, 22.5, 22.2)             # too small amplitude
    rows += lightcurve(3, 22.5, 19.0)             # at the nucleus: AGN guard
    rows += lightcurve(5, 22.5, 19.0, gr_peak=1.2)  # too red
    t = Table(rows=rows, names=("diaObjectId", "band", "psfFlux", "psfFluxErr", "expMidptMJD"))
    t["psfFlux_flag"] = False
    return t


def test_magnitude_conventions():
    assert np.isclose(njy_to_abmag(mag_to_njy(20.0)), 20.0)
    assert np.isclose(distance_modulus(10.0), 30.0)
    assert np.isnan(njy_to_abmag(-5.0))


def test_host_matching_and_offsets():
    cfg = SearchConfig()
    hosts = standardize_hecate(make_hosts(), cfg)
    assert np.isclose(hosts["host_r1_arcsec"][0], 60.0)
    m, no_host = match_to_hosts(make_objects(), hosts, cfg)
    assert list(no_host["diaObjectId"]) == [4]
    ids = list(m["diaObjectId"])
    assert 4 not in ids                       # 400" > 2 R1 = 120"
    po = dict(zip(m["diaObjectId"], m["pass_offset"]))
    assert po[1] and not po[3]                # 0.5" from nucleus is rejected


def test_full_selection():
    cfg = SearchConfig()
    m, _ = match_to_hosts(make_objects(), standardize_hecate(make_hosts(), cfg), cfg)
    res = apply_cuts(compute_metrics(make_phot(), m, cfg), cfg)
    r = {int(i): row for i, row in zip(res["diaObjectId"], res)}
    assert np.isclose(r[1]["peak_abs_mag"], -11.0, atol=0.01)
    assert np.isclose(r[1]["color_gr_peak"], 0.2, atol=0.01)
    assert r[1]["is_candidate"]
    assert not r[2]["pass_amplitude"] and not r[2]["is_candidate"]
    assert not r[3]["is_candidate"]
    assert not r[5]["pass_color"] and not r[5]["is_candidate"]


def test_query_text():
    q = diaobject_strip_query("dp2", 10, 20, 3)
    assert "FROM dp2.DiaObject" in q and "nDiaSources >= 3" in q
    q = diaobject_strip_query("dp2", 10, 20, 5, SearchConfig())
    assert "(r_scienceFluxMean - r_psfFluxMean)" in q and " OR " in q
    assert "2.511886 *" in q                      # 1 mag = flux ratio 10**0.4
    assert "r_scienceFluxMean < 1.45e+06" in q
    q = forced_photometry_query("dp2", [11, 12])
    assert "IN (11, 12)" in q and "JOIN dp2.Visit" in q


class _FakeJob:
    def __init__(self, table):
        self.table, self.phase = table, None
    def run(self): pass
    def wait(self, phases=None): self.phase = "COMPLETED"
    def raise_if_error(self): pass
    def fetch_result(self): return self
    def to_table(self): return self.table
    def delete(self): pass


class _FakeTAP:
    """Answers the two query types with the synthetic tables."""
    def submit_job(self, q):
        if "FROM dp2.DiaObject" in q:
            objs = make_objects()
            lo = float(q.split("ra >= ")[1].split()[0]); hi = float(q.split("ra < ")[1].split()[0])
            return _FakeJob(objs[(objs["ra"] >= lo) & (objs["ra"] < hi)])
        ids = [int(x) for x in q.split("IN (")[1].rstrip(")").split(",")]
        p = make_phot()
        return _FakeJob(p[np.isin(p["diaObjectId"], ids)])


def test_run_search_end_to_end(tmp_path):
    from impostorsearch.pipeline import run_search
    cands, metrics, phot = run_search(_FakeTAP(), make_hosts(), SearchConfig(),
                                      ra_range=(140.0, 160.0), out_dir=tmp_path)
    assert list(cands["diaObjectId"]) == [1]
    assert (tmp_path / "candidates.ecsv").exists()


def test_kpc_offset_mode():
    cfg = SearchConfig(offset_mode="kpc", offset_min_kpc=0.5, offset_max_kpc=10.0)
    hosts = standardize_hecate(make_hosts(), cfg)
    m, no_host = match_to_hosts(make_objects(), hosts, cfg)
    off = dict(zip(m["diaObjectId"], m["host_offset_kpc"]))
    assert np.isclose(off[1], 10e3 * np.deg2rad(30 / 3600), rtol=1e-3)   # 30" at 10 Mpc = 1.45 kpc
    po = dict(zip(m["diaObjectId"], m["pass_offset"]))
    assert po[1] and not po[3]                     # 0.5" = 0.02 kpc is nuclear
    assert 4 in list(no_host["diaObjectId"])       # 400" = 19 kpc > 10 kpc
