"""Search Rubin/LSST difference-imaging data for SN impostor and LBV candidates."""
from .config import SearchConfig
from .hosts import load_hecate, standardize_hecate, match_to_hosts
from .lightcurves import compute_metrics, njy_to_abmag, distance_modulus
from .select import apply_cuts

__version__ = "0.1.0"
__all__ = ["SearchConfig", "load_hecate", "standardize_hecate", "match_to_hosts",
           "compute_metrics", "njy_to_abmag", "distance_modulus", "apply_cuts"]
