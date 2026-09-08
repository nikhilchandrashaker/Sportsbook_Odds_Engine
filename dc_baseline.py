"""
Dixon-Coles baseline for the Football Charts dataset.

Round 1 of the challenge: beat this baseline's log-loss on held-out matches.
Round 2 (coming soon): beat the *market* at football-charts.com. We're building a
public leaderboard where you submit probabilities for upcoming matches before
kickoff and they're scored against the closing line, the price that has beaten
almost everyone who ever tried.

This is a deliberately small, readable Dixon-Coles: team attack/defence
strengths + home advantage + the low-score correlation term (tau), fit by
maximum likelihood per league on the training seasons, evaluated by multiclass
log-loss on the most recent season. Everything runs on pandas/numpy/scipy.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

CSV = "football_charts_matches.csv"
MAX_GOALS = 10


def tau(hg, ag, lh, la, rho):
    """Dixon-Coles low-score correlation correction."""
    out = np.ones_like(hg, dtype=float)
    out[(hg == 0) & (ag == 0)] = 1 - lh[(hg == 0) & (ag == 0)] * la[(hg == 0) & (ag == 0)] * rho
    out[(hg == 0) & (ag == 1)] = 1 + lh[(hg == 0) & (ag == 1)] * rho
    out[(hg == 1) & (ag == 0)] = 1 + la[(hg == 1) & (ag == 0)] * rho
    out[(hg == 1) & (ag == 1)] = 1 - rho
    return out


def fit_league(train):
    teams = sorted(set(train.home_team) | set(train.away_team))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hg = train.home_goals.to_numpy()
    ag = train.away_goals.to_numpy()
    hi = train.home_team.map(idx).to_numpy()
    ai = train.away_team.map(idx).to_numpy()

    # params: n attack, n defence, home_adv, rho
    def negll(p):
        atk, dfn = p[:n], p[n:2 * n]
        home, rho = p[2 * n], p[2 * n + 1]
        lh = np.exp(atk[hi] - dfn[ai] + home)
        la = np.exp(atk[ai] - dfn[hi])
        ll = (poisson.logpmf(hg, lh) + poisson.logpmf(ag, la)
              + np.log(np.clip(tau(hg, ag, lh, la, rho), 1e-9, None)))
        return -ll.sum()

    p0 = np.concatenate([np.zeros(2 * n), [0.25, -0.05]])
    # sum-to-zero on attack keeps it identifiable
    cons = {'type': 'eq', 'fun': lambda p: p[:n].sum()}
    res = minimize(negll, p0, constraints=cons, method='SLSQP',
                   options={'maxiter': 200, 'ftol': 1e-6})
    atk, dfn = res.x[:n], res.x[n:2 * n]
    return idx, atk, dfn, res.x[2 * n], res.x[2 * n + 1]


def match_1x2(idx, atk, dfn, home_adv, rho, h, a):
    """Return (P_home, P_draw, P_away) for one fixture."""
    if h not in idx or a not in idx:
        return None
    lh = np.exp(atk[idx[h]] - dfn[idx[a]] + home_adv)
    la = np.exp(atk[idx[a]] - dfn[idx[h]])
    hs = poisson.pmf(np.arange(MAX_GOALS + 1), lh)
    as_ = poisson.pmf(np.arange(MAX_GOALS + 1), la)
    m = np.outer(hs, as_)
    # apply tau to the four low-score cells
    m[0, 0] *= 1 - lh * la * rho
    m[0, 1] *= 1 + lh * rho
    m[1, 0] *= 1 + la * rho
    m[1, 1] *= 1 - rho
    m /= m.sum()
    ph = np.tril(m, -1).sum()   # home more goals
    pa = np.triu(m, 1).sum()    # away more goals
    pd = np.trace(m)
    return ph, pd, pa


def main():
    df = pd.read_csv(CSV)
    df = df.dropna(subset=["home_goals", "away_goals"])
    # Seasons aren't aligned across leagues (summer "2025" vs winter
    # "2024-2025"), so hold out each LEAGUE's own most recent season and train
    # on its earlier ones. Clean per-league walk-forward-style split.
    skey = lambda s: int(str(s).split("-")[0])
    print("Per-league holdout: train on a league's earlier seasons, "
          "test on its most recent.\n")

    y_true, probs = [], []
    for lg, g in df.groupby("league"):
        seasons = sorted(g.season.unique(), key=skey)
        if len(seasons) < 2:
            continue
        test_season = seasons[-1]
        train = g[g.season != test_season]
        test = g[g.season == test_season]
        if len(train) < 100 or len(test) < 20:
            continue
        try:
            model = fit_league(train)
        except Exception:
            continue
        for _, r in test.iterrows():
            p = match_1x2(*model, r.home_team, r.away_team)
            if p is None:
                continue
            probs.append(p)
            y_true.append({"H": 0, "D": 1, "A": 2}[r.result])

    probs = np.clip(np.array(probs), 1e-6, 1)
    probs /= probs.sum(axis=1, keepdims=True)
    y = np.array(y_true)
    logloss = -np.mean(np.log(probs[np.arange(len(y)), y]))
    # a naive baseline: the base rates of H/D/A in the test set
    rates = np.bincount(y, minlength=3) / len(y)
    naive = -np.mean(np.log(rates[y]))

    print(f"Matches scored: {len(y):,}")
    print(f"Dixon-Coles 1X2 log-loss: {logloss:.4f}")
    print(f"Naive base-rate log-loss: {naive:.4f}  (beat DC by beating {logloss:.4f})")
    print("\nRound 1: beat that DC number.")
    print("Round 2 (coming soon): beat the MARKET at football-charts.com — the real bar.")


if __name__ == "__main__":
    main()
