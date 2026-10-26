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

How to read this file: each "def" is a function (a reusable recipe), and
"rng" is the random number generator used to draw every random value.
"""
# NumPy: Python's library for arrays of numbers ("np" is its short name).
import numpy as np

# ---- Fixed settings ----
BANDS = ("g", "r", "i")                     # the three filters simulated (index 0, 1, 2)
CLASSES = ("impostor", "sn", "agn")         # class labels 0, 1, 2
# Single-visit 5-sigma point-source depths (AB): median expected over the survey
# (baseline_v2.0 cadence, 30 s visits), PSTN-054 Table 4, https://pstn-054.lsst.io/PSTN-054.pdf
M5 = {"g": 24.50, "r": 24.03, "i": 23.41}
# AB zero point for fluxes in nJy: 2.5 log10(3631 Jy / 1 nJy) = 31.40 (exact, from the AB definition)
ZP = 31.4


def mag_to_njy(m):
    """Convert AB magnitude to flux in nJy."""
    return 10.0 ** (-0.4 * (np.asarray(m, dtype=float) - ZP))


def cadence(rng, n_seasons=2, season_days=180.0, year=365.25, mean_gap_days=8.0):
    """Irregular per-band visit times over observing seasons with annual gaps."""
    t, b = [], []                                    # lists to collect times and bands
    for k, band in enumerate(BANDS):                 # loop over g, r, i (k = 0, 1, 2)
        for s in range(n_seasons):                   # loop over observing seasons
            start = s * year + rng.uniform(0, 10)    # season start, one year apart
            # random gaps between visits, about mean_gap_days on average
            gaps = rng.exponential(mean_gap_days, size=int(3 * season_days / mean_gap_days) + 5)
            times = start + np.cumsum(gaps)          # add up the gaps to get visit times
            times = times[times < start + season_days]   # keep only visits inside the season
            t.append(times)
            b.append(np.full(len(times), k))         # band index for each of these visits
    t, b = np.concatenate(t), np.concatenate(b)      # join all bands and seasons
    order = np.argsort(t)                            # sort all visits by time
    return t[order], b[order]


def _slow_wander(rng, t, amp):
    """Smooth, slow magnitude wandering (S Dor-like), sum of long-period sinusoids."""
    out = np.zeros_like(t)
    for _ in range(3):                               # add three slow sine waves
        period = rng.uniform(300, 3000)              # period of 300 to 3000 days
        out += rng.normal(0, amp / np.sqrt(3)) * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
    return out


def impostor(rng, t, b, mu):
    """Returns per-epoch magnitudes and the parameters used."""
    # --- the quiescent star ---
    m_q = rng.uniform(-10.0, -7.0) + mu              # quiescent apparent magnitude
    color_q = rng.uniform(-0.1, 0.4)                  # quiescent g - r
    band_offset = np.array([color_q, 0.0, -0.5 * color_q])[b]   # brightness offset per band
    base_flux = mag_to_njy(m_q + band_offset + _slow_wander(rng, t, 0.3))   # star + slow wandering

    # --- the eruptions ---
    periodic = rng.uniform() < 0.7                    # 70% periodic outbursts, 30% one giant eruption
    burst_color = rng.uniform(-0.25, 0.15)            # outbursts are bluer
    burst_off = np.array([burst_color, 0.0, -0.5 * burst_color])[b]
    burst_flux = np.zeros_like(t)                     # start with no outburst light
    if periodic:
        period = rng.uniform(40.0, 400.0)             # orbital period (days)
        amp0 = rng.uniform(1.0, 3.5)                  # outburst amplitude above quiescence (mag)
        t_first = rng.uniform(-period, 0.0)           # time of the first periastron
        for tc in np.arange(t_first, t.max() + period, period):   # one outburst per orbit
            tc = tc + rng.normal(0, 0.05 * period)    # each outburst a bit early or late
            amp = amp0 * rng.uniform(0.5, 1.2)        # each outburst a bit weaker or stronger
            width = rng.uniform(4.0, 30.0)            # outburst duration scale (days)
            peak_flux = mag_to_njy(m_q - amp) - mag_to_njy(m_q)   # extra flux at peak
            for _ in range(rng.integers(1, 4)):       # irregular, multi-peaked outbursts
                tsub = tc + rng.uniform(-width, width)   # time of this sub-peak
                sig = width * rng.uniform(0.2, 0.5)      # width of this sub-peak
                burst_flux += peak_flux * rng.uniform(0.4, 1.0) * np.exp(-0.5 * ((t - tsub) / sig) ** 2)
        params = dict(subtype="periodic", period=period, amplitude=amp0)
    else:
        peak_abs = rng.uniform(-14.0, -11.0)          # peak absolute magnitude of the giant eruption
        amp0 = (m_q - mu) - peak_abs                  # amplitude above quiescence (mag)
        t0 = rng.uniform(-60.0, t.max())              # time of peak
        rise, tau = rng.uniform(10.0, 60.0), rng.uniform(30.0, 300.0)   # rise and fade times (days)
        peak_flux = mag_to_njy(peak_abs + mu) - mag_to_njy(m_q)
        # Gaussian rise before the peak, exponential fade after it
        shape = np.where(t < t0, np.exp(-0.5 * ((t - t0) / rise) ** 2), np.exp(-(t - t0) / tau))
        burst_flux = peak_flux * shape
        params = dict(subtype="giant", period=np.nan, amplitude=amp0)
    flux = base_flux + burst_flux * 10 ** (-0.4 * burst_off)   # total = star + eruptions
    params["template"] = mag_to_njy(m_q + band_offset)      # quiescent star sits in the template
    return flux, params


def supernova(rng, t, b, mu):
    """Supernova light curve: fast rise, exponential decline, reddening with time."""
    u = rng.uniform()                                 # pick a subtype at random
    if u < 0.4:      # 40%: Ia-like, bright and fast
        peak_abs, tau_fall, subtype = rng.normal(-19.2, 0.3), rng.uniform(15.0, 40.0), "Ia-like"
    elif u < 0.75:   # 35%: II-like, fainter and slower
        peak_abs, tau_fall, subtype = rng.uniform(-18.0, -15.5), rng.uniform(40.0, 120.0), "II-like"
    else:   # slow, faint interacting SNe: the hardest to tell from giant eruptions
        peak_abs, tau_fall, subtype = rng.uniform(-18.0, -14.0), rng.uniform(80.0, 300.0), "IIn-like"
    tau_rise = rng.uniform(1.0, 8.0)                  # rise time (days)
    t0 = rng.uniform(-30.0, t.max())                  # explosion time
    shape = np.exp(-(t - t0) / tau_fall) / (1.0 + np.exp(-(t - t0) / tau_rise))   # rise-and-decline shape
    # normalize the shape so its peak equals 1
    tg = np.linspace(t0 - 50, t0 + 50, 2001)
    norm = np.max(np.exp(-(tg - t0) / tau_fall) / (1.0 + np.exp(-(tg - t0) / tau_rise)))
    color = rng.uniform(-0.2, 0.1) + 0.5 * np.clip((t - t0) / 50.0, 0, 2)   # reddens with time
    off = np.select([b == 0, b == 2], [color, -0.5 * color], 0.0)            # offset per band
    flux = mag_to_njy(peak_abs + mu) * (shape / norm) * 10 ** (-0.4 * off)
    return flux, dict(subtype=subtype, period=np.nan, amplitude=np.nan, template=np.zeros_like(t))


def agn(rng, t, b, mu):
    """Background quasar seen through the host: damped random walk in magnitude.

    Its apparent magnitude is independent of the host distance, so the absolute
    magnitude computed with the host distance is wrong, as it would be in real data.
    """
    mean_app = rng.uniform(20.0, 23.5)                # mean apparent magnitude
    tau = rng.uniform(30.0, 500.0)                    # how long it "remembers" its brightness (days)
    sf_inf = rng.uniform(0.15, 0.8)                   # asymptotic structure function (mag)
    sigma = sf_inf / np.sqrt(2.0)                     # typical size of the variations
    tu = np.unique(t)                                 # distinct observation times
    x = np.empty(len(tu))
    x[0] = rng.normal(0, sigma)                       # random starting offset
    for k in range(1, len(tu)):                       # step forward in time
        a = np.exp(-(tu[k] - tu[k - 1]) / tau)        # how much of the previous offset is kept
        x[k] = a * x[k - 1] + rng.normal(0, sigma * np.sqrt(1 - a * a))   # drift + random kick
    dm = x[np.searchsorted(tu, t)] * np.array([1.2, 1.0, 0.8])[b]   # bluer bands vary more
    color = rng.uniform(0.0, 0.6)
    off = np.array([color, 0.0, -0.5 * color])[b]
    flux = mag_to_njy(mean_app + off + dm)
    return flux, dict(subtype="agn", period=np.nan, amplitude=2 * sf_inf,
                      template=mag_to_njy(mean_app + off))


# The three recipes in label order: 0 = impostor, 1 = supernova, 2 = AGN.
MODELS = (impostor, supernova, agn)


def observe(rng, flux_true, b, host_flux):
    """Add host light at the position and sky-dominated Gaussian noise."""
    sigma = np.array([mag_to_njy(M5[bb]) / 5.0 for bb in BANDS])[b]   # noise per band from the depth
    flux = flux_true + host_flux + rng.normal(0, sigma)               # true flux + host + noise
    return flux, sigma


def luptitude(flux, sigma):
    """asinh magnitude: well behaved at low and negative flux (Lupton et al. 1999, https://doi.org/10.1086/301004)."""
    soft = 1.042 * sigma                              # softening: where it switches from log to linear
    return -2.5 / np.log(10) * (np.arcsinh(flux / (2 * soft)) + np.log(soft)) + ZP


def detected(flux_true, b, n_min=2, snr=5.0):
    """At least n_min epochs differ from the object's median flux by >= snr sigma,
    as a source must to appear in the difference-imaging (DiaObject) catalog."""
    sigma = np.array([mag_to_njy(M5[bb]) / 5.0 for bb in BANDS])[b]
    return np.sum(np.abs(flux_true - np.median(flux_true)) / sigma >= snr) >= n_min


def simulate_object(rng, label, max_tries=100):
    """Make one detectable object of the given class (0, 1, or 2)."""
    for _ in range(max_tries):                        # try until the object is detectable
        t, b = cadence(rng)                           # observation times and bands
        mu = rng.uniform(28.5, 33.0)                  # 5 to 40 Mpc
        flux_true, params = MODELS[label](rng, t, b, mu)   # run the right recipe
        if detected(flux_true, b):
            break
    host_flux = mag_to_njy(rng.uniform(22.0, 26.0))   # host-galaxy light at the position
    flux, sigma = observe(rng, flux_true, b, host_flux)   # add host light and noise
    template = params.pop("template")
    diff_flux = flux - host_flux - template            # difference-image flux (alert-like)
    lup = luptitude(flux, sigma)                       # asinh magnitudes
    lup_err = 1.0857 * sigma / np.sqrt(flux ** 2 + (2 * 1.042 * sigma) ** 2)   # d(lup)/d(flux) x sigma
    # return everything as a dictionary (a set of named values); times start at 0
    return dict(t=t - t.min(), band=b, lup=lup, lup_err=lup_err, mu=mu, label=label,
                diff_flux=diff_flux, sigma=sigma, total_flux=flux, **params)


def make_dataset(n_per_class, seed=0, max_len=256):
    """Padded arrays ready for the classifier."""
    rng = np.random.default_rng(seed)                  # random generator; same seed gives same data
    n = n_per_class * len(CLASSES)                     # total number of objects
    labels = np.repeat(np.arange(len(CLASSES)), n_per_class)   # equal numbers of each class
    rng.shuffle(labels)                                # in random order
    # empty arrays with max_len slots per object (shorter light curves leave slots empty)
    T = np.zeros((n, max_len), np.float32)             # times
    B = np.zeros((n, max_len), np.int64)               # bands
    X = np.zeros((n, max_len, 2), np.float32)          # magnitude and its error
    M = np.zeros((n, max_len), bool)                   # mask: True where a slot holds a real point
    ctx = np.zeros((n, 2), np.float32)                 # extra host information per object
    meta = {k: np.full(n, np.nan) for k in ("mu", "amplitude", "period")}   # true parameters
    subtype = np.empty(n, dtype=object)
    for i, lab in enumerate(labels):                   # make each object
        o = simulate_object(rng, int(lab))
        L = min(len(o["t"]), max_len)                  # number of points to keep
        # median r-band magnitude (or all bands if no r data)
        med = np.median(o["lup"][o["band"] == 1]) if np.any(o["band"] == 1) else np.median(o["lup"])
        T[i, :L] = o["t"][:L]
        B[i, :L] = o["band"][:L]
        X[i, :L, 0] = (o["lup"][:L] - med)              # shape relative to the median
        X[i, :L, 1] = np.minimum(o["lup_err"][:L], 2.0)   # error, capped at 2 mag
        M[i, :L] = True
        ctx[i] = [(med - o["mu"] + 12.0) / 4.0,           # median absolute mag (scaled)
                  (o["mu"] - 30.75) / 1.3]                # distance modulus (scaled)
        for k in meta:
            meta[k][i] = o[k]
        subtype[i] = o["subtype"]
    return dict(t=T, band=B, x=X, mask=M, ctx=ctx, y=labels.astype(np.int64),
                subtype=subtype, **meta)
