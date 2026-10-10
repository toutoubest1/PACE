"""Method registry: name -> callable(D) returning a coefficient vector."""
from pace import pace, old_rlpd, old_lpd, fit_pace
from pace import competitors as C

F = lambda D: (D["Xobs"], D["y"], D["M"])

METHODS = {
    # competitors
    "Naive-Lasso": lambda D: C.naive_lasso(*F(D)),
    "MICE-Lasso": lambda D: C.mice_lasso(*F(D)),
    "Huber-Lasso": lambda D: C.huber_lasso(*F(D)),
    "SparseLTS": lambda D: C.sparse_lts(*F(D)),
    "Filt-MICE-SLTS": lambda D: C.filter_mice_slts(*F(D)),
    "IPW-LPD-Lasso": lambda D: C.ipw_lpd_lasso(*F(D)),
    # rejected manuscript (defects fixed)
    "OLD-LPD-Lasso": lambda D: old_lpd(*F(D), "lasso")[0],
    "OLD-RLPD-Lasso": lambda D: old_rlpd(*F(D), "lasso")[0],
    "OLD-RLPD-MCP": lambda D: old_rlpd(*F(D), "mcp")[0],
    "OLD-RLPD-MCP-absfloor": lambda D: old_rlpd(*F(D), "mcp", eps_mode="absolute")[0],
    #final estimator and its theory-aligned versions
    "PACE": lambda D: pace(*F(D), mode="full"),
    "PACE-split": lambda D: pace(*F(D), mode="split"),
    "PACE-crossfit": lambda D: pace(*F(D), mode="crossfit"),
    "PACE-crossfit-ls": lambda D: pace(*F(D), mode="crossfit", local_scale=True),
    "PACE-SC": lambda D: pace(*F(D), pilot="screen_cstep"),
    "PACE-CF5": lambda D: pace(*F(D), mode="crossfit", cf_folds=5),   # == fit_pace (frozen)
    "PACE-SC-CF5": lambda D: pace(*F(D), mode="crossfit", cf_folds=5, pilot="screen_cstep"),
    "PACE-crossfit5": lambda D: pace(*F(D), mode="crossfit", cf_folds=5),
    "PACE-crossfit5-ls": lambda D: pace(*F(D), mode="crossfit", cf_folds=5, local_scale=True),
    "PACE-split-ls": lambda D: pace(*F(D), mode="split", local_scale=True),
    "PACE-halfsample": lambda D: pace(D["Xobs"][100:], D["y"][100:], D["M"][100:]),
    # pilot variants (validation round 2)
    "PACE-pilot-screen": lambda D: pace(*F(D), pilot="screen"),
    "PACE-pilot-screen+cstep": lambda D: pace(*F(D), pilot="screen+cstep"),
    "PACE-pilot-screen_cstep": lambda D: pace(*F(D), pilot="screen_cstep"),
    "PACE-pilot-screen_cstep-K38": lambda D: pace(*F(D), pilot="screen_cstep", screen_K=38),
    "PACE-pilot-screen_cstep+cstep": lambda D: pace(*F(D), pilot="screen_cstep+cstep"),
    # ablations / candidates
    "cand-R": lambda D: pace(*F(D), cell="none", stab="lpd"),
    "cand-W": lambda D: pace(*F(D), cell="winsor", tau=2.5, stab="lpd"),
    "cand-C": lambda D: pace(*F(D), stab="lpd"),
    "cand-C-Lasso": lambda D: pace(*F(D), penalty="lasso", stab="lpd"),
    "PACE-clip-everywhere": lambda D: pace(*F(D), stab="clip"),
    "PACE-noY": lambda D: pace(*F(D), use_wy=False),
    "PACE-noX": lambda D: pace(*F(D), use_wx=False),
    "PACE-zerofill-dist": lambda D: pace(*F(D), wx_mode="zf"),
    "PACE-oldweight-noY": lambda D: pace(*F(D), wx_mode="old", use_wy=False, cell="none"),
    "PACE-Lasso": lambda D: pace(*F(D), penalty="lasso"),
}
