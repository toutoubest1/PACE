"""Regimes, dimensions and seed blocks used in the paper.  Seed for replicate r of a block = SEED0 + r."""
N = 200
P_MAIN = 300
REGIMES = {
    "none":      dict(contam="none"),
    "xonly":     dict(contam="x"),
    "xlev":      dict(contam="xlev"),
    "yonly":     dict(contam="y"),
    "xy":        dict(contam="xy"),
    "t2err":     dict(contam="none", err="t2"),
    "cell6":     dict(contam="cell"),
    "cell4":     dict(contam="cell4"),
    "xy_m10":    dict(contam="xy", mrate=0.1),
    "xy_m30":    dict(contam="xy", mrate=0.3),
    "xy_m40":    dict(contam="xy", mrate=0.4),
    "xy_hetcol": dict(contam="xy", miss="hetcol"),
    "xy_hetrow": dict(contam="xy", miss="hetrow"),
    "xy_mar":    dict(contam="xy", miss="mar"),
    "xy_mnar":   dict(contam="xy", miss="mnar"),
}
# common defaults of sim.generate.gen: s=10, rho=0.5, eps=0.1 (row contamination fraction),
# mrate=0.2 MCAR, cell_rate=0.02, beta ~ U[1,2] with random signs, xshift=3, yshift=15.
SEEDS = {
    "dev_round1": 0,        # development of candidates (decision report 1)
    "confirm_round1": 1000, # held-out confirmation of report 1
    "dev_pilot": 2000,      # validation round 2: pilot / cross-fitting design (development)
    "final": 7000,          # FINAL held-out seeds; never used for any design decision
}
PGRID = {100: 20, 300: 20, 500: 10, 1000: 10}   # p -> replicates in the dimension study
PGRID_REGIMES = ["none", "yonly", "xy", "cell6"]
