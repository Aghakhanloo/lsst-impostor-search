"""Simulated LSST-like light curves of SN impostors, supernovae, and AGN.

The models are deliberately simple and every parameter range is an assumption,
chosen to resemble published examples (e.g. the recurring outbursts of SN 2000ch
and AT 2016blu). The goal is a controlled test bed for a light-curve classifier,
not a population synthesis.

Classes
-------
0 SN impostor : LBV-like source with quasi-periodic outbursts (70%) or a single
             long giant eruption (30%), on top of S Dor-like variability (0.1-2 mag
             over years to decades, redder when brighter).
1 sn       : supernova at the position (Ia-like or II-like rise-and-decline).
2 agn      : stochastic (damped-random-walk) variability of a faint nucleus.

Each object gets a host distance modulus; the classifier also receives the
median absolute magnitude, mimicking a host-galaxy distance from HECATE.

Fluxes are in nJy (AB zero point 31.4). Noise is sky-dominated: the flux error
is the flux of a 5-sigma source at the single-visit depth m5, divided by 5.

How to read this file: each "def" is a function (a reusable recipe), and
"rng" is the random number generator used to draw every random value.

Where the numbers come from (each is also marked in the code):
  [measured]   taken from a published source, linked next to it
  [exact]      follows from a definition
  [assumption] a choice made for this test, not a measurement; change it freely
Main sources: Rubin depths, PSTN-054 (https://pstn-054.lsst.io/PSTN-054.pdf);
SN 2000ch outbursts, Aghakhanloo et al. 2023a (https://arxiv.org/abs/2212.00113);
AT 2016blu outbursts, Aghakhanloo et al. 2023b (https://arxiv.org/abs/2212.09708);
AGN damped random walk, MacLeod et al. 2010 (https://doi.org/10.1088/0004-637X/721/2/1014).
"""
# NumPy: Python's library for arrays of numbers ("np" is its short name).
import numpy as np

# ---- Fixed settings ----
BANDS = ("g", "r", "i")                     # the three filters simulated (index 0, 1, 2)
CLASSES = ("impostor", "sn", "agn")         # class labels 0, 1, 2
# [measured] Single-visit 5-sigma point-source depths (AB): median expected over the survey
# (baseline_v2.0 cadence, 30 s visits), PSTN-054 Table 4, https://pstn-054.lsst.io/PSTN-054.pdf
M5 = {"g": 24.50, "r": 24.03, "i": 23.41}
# [exact] AB zero point for fluxes in nJy: 2.5 log10(3631 Jy / 1 nJy) = 31.40 (exact, from the AB definition)
ZP = 31.4


def mag_to_njy(m):
    """Convert AB magnitude to flux in nJy."""
    return 10.0 ** (-0.4 * (np.asarray(m, dtype=float) - ZP))


def cadence(rng, n_seasons=2, season_days=180.0, year=365.25, mean_gap_days=8.0):
    """Irregular per-band visit times over observing seasons with annual gaps.

    [assumption] 2 seasons of 180 d and one visit per band every ~8 d on average.
    The real LSST cadence varies across the sky and will replace this once
    available (PSTN-054 describes the simulated baseline cadence).
    """
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


def _sdor_variation(rng, t):
    """S Dor-like variability: slow brightness changes over years to decades.

    Based on van Genderen (2001, https://www.aanda.org/articles/aa/full/2001/05/aads1896/aads1896.orig.html):
    [measured] S Dor variations reach up to ~2.5 mag; "strong-active" stars vary by
    more than 0.5 mag and weak-active ones by less. Short S Dor phases last years
    (< 10 yr) and long phases decades.
    [assumption] Total range drawn from 0.1 to 2 mag, split at random between a
    short cycle (1-10 yr) and a long cycle (10-30 yr); both are sinusoids.
    Returns the magnitude change dm (negative = brighter).
    """
    total = rng.uniform(0.1, 2.0)                    # [assumption] total range (mag), weak to strong-active
    frac_short = rng.uniform()                       # share of the range in the short cycle
    dm = np.zeros_like(t)
    for share, (p_min, p_max) in ((frac_short, (365.0, 3650.0)),          # short cycle: 1-10 yr
                                  (1 - frac_short, (3650.0, 10950.0))):   # long cycle: 10-30 yr
        period = rng.uniform(p_min, p_max)
        phase = rng.uniform(0, 2 * np.pi)            # random point in the cycle when observations start
        dm += 0.5 * total * share * np.sin(2 * np.pi * t / period + phase)   # half-range = amplitude
    return dm


def impostor(rng, t, b, mu):
    """Returns per-epoch magnitudes and the parameters used."""
    # --- the quiescent star ---
    m_q = rng.uniform(-10.0, -7.0) + mu              # [assumption] quiescent abs. mag -10 to -7, made apparent with mu
    color_q = rng.uniform(-0.1, 0.4)                  # [assumption] quiescent g - r
    band_offset = np.array([color_q, 0.0, -0.5 * color_q])[b]   # brightness offset per band
    dm_sdor = _sdor_variation(rng, t)                 # slow S Dor-like brightness change
    # [measured] S Dor variables get redder as they brighten and bluer as they fade
    # (van Genderen 2001, https://www.aanda.org/articles/aa/full/2001/05/aads1896/aads1896.orig.html);
    # [assumption] g - r changes by 0.1 to 0.3 mag per mag of brightening.
    reddening = -rng.uniform(0.1, 0.3) * dm_sdor      # dm < 0 (brighter) -> redder (larger g - r)
    sdor_color = np.select([b == 0, b == 2], [reddening, -0.5 * reddening], 0.0)   # applied per band
    base_flux = mag_to_njy(m_q + band_offset + dm_sdor + sdor_color)   # star with S Dor variability

    # --- the eruptions ---
    periodic = rng.uniform() < 0.7                    # [assumption] 70% periodic outbursts, 30% one giant eruption
    # [assumption] no preferred outburst color: g - r offset of -0.15 to +0.15 mag. The spectrum of
    # AT 2016blu stays fairly consistent between quiescence and eruptions (https://arxiv.org/abs/2412.13117).
    burst_color = rng.uniform(-0.15, 0.15)
    burst_off = np.array([burst_color, 0.0, -0.5 * burst_color])[b]
    burst_flux = np.zeros_like(t)                     # start with no outburst light
    if periodic:
        # [assumption] range chosen to bracket the measured periods of AT 2016blu
        # (113 +/- 2 d, https://arxiv.org/abs/2212.09708) and SN 2000ch (200.7 +/- 2 d, https://arxiv.org/abs/2212.00113)
        period = rng.uniform(40.0, 400.0)             # orbital period (days)
        amp0 = rng.uniform(1.0, 3.5)                  # amplitude (mag); [assumption], wider than AT 2016blu's 1-2 mag (https://arxiv.org/abs/2212.09708)
        t_first = rng.uniform(-period, 0.0)           # time of the first periastron
        for tc in np.arange(t_first, t.max() + period, period):   # one outburst per orbit
            tc = tc + rng.normal(0, 0.05 * period)    # each outburst a bit early or late
            amp = amp0 * rng.uniform(0.5, 1.2)        # each outburst a bit weaker or stronger
            width = rng.uniform(4.0, 30.0)            # duration scale (d); AT 2016blu outbursts last 1-4 weeks (https://arxiv.org/abs/2212.09708)
            peak_flux = mag_to_njy(m_q - amp) - mag_to_njy(m_q)   # extra flux at peak
            for _ in range(rng.integers(1, 4)):       # irregular, multi-peaked outbursts
                tsub = tc + rng.uniform(-width, width)   # time of this sub-peak
                sig = width * rng.uniform(0.2, 0.5)      # width of this sub-peak
                burst_flux += peak_flux * rng.uniform(0.4, 1.0) * np.exp(-0.5 * ((t - tsub) / sig) ** 2)
        params = dict(subtype="periodic", period=period, amplitude=amp0)
    else:
        peak_abs = rng.uniform(-14.0, -11.0)          # [assumption] peak abs. mag of the giant eruption
        amp0 = (m_q - mu) - peak_abs                  # amplitude above quiescence (mag)
        t0 = rng.uniform(-60.0, t.max())              # time of peak
        rise, tau = rng.uniform(10.0, 60.0), rng.uniform(30.0, 300.0)   # [assumption] rise and fade times (days)
        peak_flux = mag_to_njy(peak_abs + mu) - mag_to_njy(m_q)
        # Gaussian rise before the peak, exponential fade after it
        shape = np.where(t < t0, np.exp(-0.5 * ((t - t0) / rise) ** 2), np.exp(-(t - t0) / tau))
        burst_flux = peak_flux * shape
        params = dict(subtype="giant", period=np.nan, amplitude=amp0)
    flux = base_flux + burst_flux * 10 ** (-0.4 * burst_off)   # total = star + eruptions
    params["template"] = mag_to_njy(m_q + band_offset)      # quiescent star sits in the template
    return flux, params


def supernova(rng, t, b, mu):
    """Supernova light curve: fast rise, exponential decline, reddening with time.

    [assumption] Peak magnitudes, timescales, and subtype fractions are rough
    typical values for each type, not fitted to a published sample.
    """
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
    """Background quasar seen through the host: damped random walk in magnitude
    (the model of MacLeod et al. 2010, https://doi.org/10.1088/0004-637X/721/2/1014).

    Its apparent magnitude is independent of the host distance, so the absolute
    magnitude computed with the host distance is wrong, as it would be in real data.
    """
    mean_app = rng.uniform(20.0, 23.5)                # [assumption] mean apparent magnitude
    tau = rng.uniform(30.0, 500.0)                    # [assumption] how long it "remembers" its brightness (days)
    sf_inf = rng.uniform(0.15, 0.8)                   # [assumption] asymptotic structure function (mag)
    sigma = sf_inf / np.sqrt(2.0)                     # typical size of the variations
    tu = np.unique(t)                                 # distinct observation times
    x = np.empty(len(tu))
    x[0] = rng.normal(0, sigma)                       # random starting offset
    for k in range(1, len(tu)):                       # step forward in time
        a = np.exp(-(tu[k] - tu[k - 1]) / tau)        # how much of the previous offset is kept
        x[k] = a * x[k - 1] + rng.normal(0, sigma * np.sqrt(1 - a * a))   # drift + random kick
    dm = x[np.searchsorted(tu, t)] * np.array([1.2, 1.0, 0.8])[b]   # bluer bands vary more ([assumption] factors)
    color = rng.uniform(0.0, 0.6)
    off = np.array([color, 0.0, -0.5 * color])[b]
    flux = mag_to_njy(mean_app + off + dm)
    return flux, dict(subtype="agn", period=np.nan, amplitude=2 * sf_inf,
                      template=mag_to_njy(mean_app + off))


# The three recipes in label order: 0 = SN impostor, 1 = supernova, 2 = AGN.
MODELS = (impostor, supernova, agn)


def observe(rng, flux_true, b, host_flux):
    """Add host light at the position and sky-dominated Gaussian noise."""
    sigma = np.array([mag_to_njy(M5[bb]) / 5.0 for bb in BANDS])[b]   # noise per band from the depth
    flux = flux_true + host_flux + rng.normal(0, sigma)               # true flux + host + noise
    return flux, sigma


def luptitude(flux, sigma):
    """asinh magnitude: well behaved at low and negative flux (Lupton et al. 1999, https://doi.org/10.1086/301004)."""
    soft = 1.042 * sigma                              # softening (Lupton et al. 1999): where it switches from log to linear
    return -2.5 / np.log(10) * (np.arcsinh(flux / (2 * soft)) + np.log(soft)) + ZP


def detected(flux_true, b, n_min=2, snr=5.0):
    """At least n_min epochs differ from the object's median flux by >= snr sigma,
    as a source must to appear in the difference-imaging (DiaObject) catalog.

    [assumption] 5 sigma matches Rubin's detection threshold for difference-image
    sources; requiring 2 epochs is a choice made here.
    """
    sigma = np.array([mag_to_njy(M5[bb]) / 5.0 for bb in BANDS])[b]
    return np.sum(np.abs(flux_true - np.median(flux_true)) / sigma >= snr) >= n_min


def simulate_object(rng, label, max_tries=100):
    """Make one detectable object of the given class (0, 1, or 2)."""
    for _ in range(max_tries):                        # try until the object is detectable
        t, b = cadence(rng)                           # observation times and bands
        mu = rng.uniform(28.5, 33.0)                  # [assumption] distance modulus 28.5-33 = 5 to 40 Mpc
        flux_true, params = MODELS[label](rng, t, b, mu)   # run the right recipe
        if detected(flux_true, b):
            break
    host_flux = mag_to_njy(rng.uniform(22.0, 26.0))   # [assumption] host-galaxy light at the position (22-26 mag)
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
