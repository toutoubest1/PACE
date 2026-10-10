from .estimator import pace, pace_fit, apply_nuisance, final_fit
from .old_rlpd import old_rlpd, old_lpd
from . import competitors


def fit_pace(Xobs, y, M, penalty="mcp", seed=12345):
    """PACE-CF5: 5-fold cross-fitted weights, concentration-step
    pilot, cell flags at |z|>3.5, pattern-adjusted leverage weight, pattern-standardised bisquare
    residual weight, weighted pairwise-Hajek moments, eigenvalue-clipped final fit, MCP with
    1-SE moment cross-validation.  Xobs: n x p array (missing entries arbitrary, e.g. 0),
    M: boolean n x p observation mask (True = observed), y: length-n response."""
    return pace(Xobs, y, M, mode="crossfit", cf_folds=5, penalty=penalty, seed=seed)
