"""Simulated LSST-like light curves of SN impostors, supernovae, and AGN.

The models are deliberately simple and every parameter range is an assumption,
chosen to resemble published examples (e.g. the recurring outbursts of SN 2000ch
and AT 2016blu). The goal is a controlled test bed for a light-curve classifier,
not a population synthesis.

Classes
-------
0 impostor : LBV-like source with quasi-periodic outbursts (70%) or a single
             long giant eruption (30%), on top of slow S Dor-like wandering.
1 sn       : supernova at the position (Ia-like or II-like rise-and-decline).
2 agn      : stochastic (damped-random-walk) variability of a faint nucleus.

Each object gets a host distance modulus; the classifier also receives the
median absolute magnitude, mimicking a host-galaxy distance from HECATE.

Fluxes are in nJy (AB zero point 31.4). Noise is sky-dominated: the flux error
is the flux of a 5-sigma source at the single-visit depth m5, divided by 5.
"""
import numpy as np

BANDS = ("g", "r", "i")
CLASSES = ("impostor", "sn", "agn")
M5 = {"g": 24.8, "r": 24.4, "i": 23.9}     # assumed single-visit 5-sigma depths (AB)
ZP = 31.4                                   # AB zero point for nJy


def mag_to_njy(m):
    return 10.0 ** (-0.4 * (np.asarray(m, dtype=float) - ZP))


def cadence(rng, n_seasons=2, season_days=180.0, year=365.25, mean_gap_days=8.0):
    """Irregular per-band visit times over observing seasons with annual gaps."""
    t, b = [], []
    for k, band in enumerate(BANDS):
        for s in range(n_seasons):
            start = s * year + rng.uniform(0, 10)
            gaps = rng.exponential(mean_gap_days, size=int(3 * season_days / mean_gap_days) + 5)
            times = start + np.cumsum(gaps)
            times = times[times < start + season_days]
            t.append(times)
            b.append(np.full(len(times), k))
    t, b = np.concatenate(t), np.concatenate(b)
    order = np.argsort(t)
    return t[order], b[order]


def _slow_wander(rng, t, amp):
    """Smooth, slow magnitude wandering (S Dor-like), sum of long-period sinusoids."""
    out = np.zeros_like(t)
    for _ in range(3):
        period = rng.uniform(300, 3000)
        out += rng.normal(0, amp / np.sqrt(3)) * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
    return out


def impostor(rng, t, b, mu):
    """Returns per-epoch magnitudes and the parameters used."""
    m_q = rng.uniform(-10.0, -7.0) + mu              # quiescent apparent magnitude
    color_q = rng.uniform(-0.1, 0.4)                  # quiescent g - r
    band_offset = np.array([color_q, 0.0, -0.5 * color_q])[b]
    base_flux = mag_to_njy(m_q + band_offset + _slow_wander(rng, t, 0.3))

    periodic = rng.uniform() < 0.7
    burst_color = rng.uniform(-0.25, 0.15)            # outbursts are bluer
    burst_off = np.array([burst_color, 0.0, -0.5 * burst_color])[b]
    burst_flux = np.zeros_like(t)
    if periodic:
        period = rng.uniform(40.0, 400.0)
        amp0 = rng.uniform(1.0, 3.5)                  # outburst amplitude above quiescence (mag)
        t_first = rng.uniform(-period, 0.0)
        for tc in np.arange(t_first, t.max() + period, period):
            tc = tc + rng.normal(0, 0.05 * period)
            amp = amp0 * rng.uniform(0.5, 1.2)
            width = rng.uniform(4.0, 30.0)
            peak_flux = mag_to_njy(m_q - amp) - mag_to_njy(m_q)
            for _ in range(rng.integers(1, 4)):       # irregular, multi-peaked outbursts
                tsub = tc + rng.uniform(-width, width)
                sig = width * rng.uniform(0.2, 0.5)
                burst_flux += peak_flux * rng.uniform(0.4, 1.0) * np.exp(-0.5 * ((t - tsub) / sig) ** 2)
        params = dict(subtype="periodic", period=period, amplitude=amp0)
    else:
        peak_abs = rng.uniform(-14.0, -11.0)
        amp0 = (m_q - mu) - peak_abs
        t0 = rng.uniform(-60.0, t.max())
        rise, tau = rng.uniform(10.0, 60.0), rng.uniform(30.0, 300.0)
        peak_flux = mag_to_njy(peak_abs + mu) - mag_to_njy(m_q)
        shape = np.where(t < t0, np.exp(-0.5 * ((t - t0) / rise) ** 2), np.exp(-(t - t0) / tau))
        burst_flux = peak_flux * shape
        params = dict(subtype="giant", period=np.nan, amplitude=amp0)
    flux = base_flux + burst_flux * 10 ** (-0.4 * burst_off)
    params["template"] = mag_to_njy(m_q + band_offset)      # quiescent star sits in the template
    return flux, params


def supernova(rng, t, b, mu):
    u = rng.uniform()
    if u < 0.4:
        peak_abs, tau_fall, subtype = rng.normal(-19.2, 0.3), rng.uniform(15.0, 40.0), "Ia-like"
    elif u < 0.75:
        peak_abs, tau_fall, subtype = rng.uniform(-18.0, -15.5), rng.uniform(40.0, 120.0), "II-like"
    else:   # slow, faint interacting SNe: the hardest to tell from giant eruptions
        peak_abs, tau_fall, subtype = rng.uniform(-18.0, -14.0), rng.uniform(80.0, 300.0), "IIn-like"
    tau_rise = rng.uniform(1.0, 8.0)
    t0 = rng.uniform(-30.0, t.max())
    shape = np.exp(-(t - t0) / tau_fall) / (1.0 + np.exp(-(t - t0) / tau_rise))
    tg = np.linspace(t0 - 50, t0 + 50, 2001)
    norm = np.max(np.exp(-(tg - t0) / tau_fall) / (1.0 + np.exp(-(tg - t0) / tau_rise)))
    color = rng.uniform(-0.2, 0.1) + 0.5 * np.clip((t - t0) / 50.0, 0, 2)   # reddens with time
    off = np.select([b == 0, b == 2], [color, -0.5 * color], 0.0)
    flux = mag_to_njy(peak_abs + mu) * (shape / norm) * 10 ** (-0.4 * off)
    return flux, dict(subtype=subtype, period=np.nan, amplitude=np.nan, template=np.zeros_like(t))


def agn(rng, t, b, mu):
    """Background quasar seen through the host: damped random walk in magnitude.

    Its apparent magnitude is independent of the host distance, so the absolute
    magnitude computed with the host distance is wrong, as it would be in real data.
    """
    mean_app = rng.uniform(20.0, 23.5)
    tau = rng.uniform(30.0, 500.0)
    sf_inf = rng.uniform(0.15, 0.8)                   # asymptotic structure function (mag)
    sigma = sf_inf / np.sqrt(2.0)
    tu = np.unique(t)
    x = np.empty(len(tu))
    x[0] = rng.normal(0, sigma)
    for k in range(1, len(tu)):
        a = np.exp(-(tu[k] - tu[k - 1]) / tau)
        x[k] = a * x[k - 1] + rng.normal(0, sigma * np.sqrt(1 - a * a))
    dm = x[np.searchsorted(tu, t)] * np.array([1.2, 1.0, 0.8])[b]   # bluer bands vary more
    color = rng.uniform(0.0, 0.6)
    off = np.array([color, 0.0, -0.5 * color])[b]
    flux = mag_to_njy(mean_app + off + dm)
    return flux, dict(subtype="agn", period=np.nan, amplitude=2 * sf_inf,
                      template=mag_to_njy(mean_app + off))


MODELS = (impostor, supernova, agn)


def observe(rng, flux_true, b, host_flux):
    """Add host light at the position and sky-dominated Gaussian noise."""
    sigma = np.array([mag_to_njy(M5[bb]) / 5.0 for bb in BANDS])[b]
    flux = flux_true + host_flux + rng.normal(0, sigma)
    return flux, sigma


def luptitude(flux, sigma):
    """asinh magnitude: well behaved at low and negative flux (Lupton et al. 1999, https://doi.org/10.1086/301004)."""
    soft = 1.042 * sigma
    return -2.5 / np.log(10) * (np.arcsinh(flux / (2 * soft)) + np.log(soft)) + ZP


def detected(flux_true, b, n_min=2, snr=5.0):
    """At least n_min epochs differ from the object's median flux by >= snr sigma,
    as a source must to appear in the difference-imaging (DiaObject) catalog."""
    sigma = np.array([mag_to_njy(M5[bb]) / 5.0 for bb in BANDS])[b]
    return np.sum(np.abs(flux_true - np.median(flux_true)) / sigma >= snr) >= n_min


def simulate_object(rng, label, max_tries=100):
    for _ in range(max_tries):
        t, b = cadence(rng)
        mu = rng.uniform(28.5, 33.0)                  # 5 to 40 Mpc
        flux_true, params = MODELS[label](rng, t, b, mu)
        if detected(flux_true, b):
            break
    host_flux = mag_to_njy(rng.uniform(22.0, 26.0))
    flux, sigma = observe(rng, flux_true, b, host_flux)
    template = params.pop("template")
    diff_flux = flux - host_flux - template            # difference-image flux (alert-like)
    lup = luptitude(flux, sigma)
    lup_err = 1.0857 * sigma / np.sqrt(flux ** 2 + (2 * 1.042 * sigma) ** 2)   # d(lup)/d(flux) x sigma
    return dict(t=t - t.min(), band=b, lup=lup, lup_err=lup_err, mu=mu, label=label,
                diff_flux=diff_flux, sigma=sigma, total_flux=flux, **params)


def make_dataset(n_per_class, seed=0, max_len=256):
    """Padded arrays ready for the classifier."""
    rng = np.random.default_rng(seed)
    n = n_per_class * len(CLASSES)
    labels = np.repeat(np.arange(len(CLASSES)), n_per_class)
    rng.shuffle(labels)
    T = np.zeros((n, max_len), np.float32)
    B = np.zeros((n, max_len), np.int64)
    X = np.zeros((n, max_len, 2), np.float32)
    M = np.zeros((n, max_len), bool)
    ctx = np.zeros((n, 2), np.float32)
    meta = {k: np.full(n, np.nan) for k in ("mu", "amplitude", "period")}
    subtype = np.empty(n, dtype=object)
    for i, lab in enumerate(labels):
        o = simulate_object(rng, int(lab))
        L = min(len(o["t"]), max_len)
        med = np.median(o["lup"][o["band"] == 1]) if np.any(o["band"] == 1) else np.median(o["lup"])
        T[i, :L] = o["t"][:L]
        B[i, :L] = o["band"][:L]
        X[i, :L, 0] = (o["lup"][:L] - med)              # shape relative to the median
        X[i, :L, 1] = np.minimum(o["lup_err"][:L], 2.0)
        M[i, :L] = True
        ctx[i] = [(med - o["mu"] + 12.0) / 4.0,           # median absolute mag (scaled)
                  (o["mu"] - 30.75) / 1.3]                # distance modulus (scaled)
        for k in meta:
            meta[k][i] = o[k]
        subtype[i] = o["subtype"]
    return dict(t=T, band=B, x=X, mask=M, ctx=ctx, y=labels.astype(np.int64),
                subtype=subtype, **meta)
