# Football Charts — Match Results & Goal Timing

76,928 matches across 91 leagues, the 3 most recent completed
seasons of each. Data by [football-charts.com](https://www.football-charts.com).

Most public football datasets cover the big five leagues and stop at the final
score. This one goes 91 leagues deep, from the Premier League to the
Baltic and Nordic lower tiers, and includes the **minute of every goal** — the
timing columns most datasets skip.

**No betting odds are included.** Results are the ground truth here. The market
is the real test, and that lives at football-charts.com (see the challenge
below).

## Columns

| column | meaning |
|---|---|
| league | short league code (e.g. `premier`, `spain1`) |
| league_name | human-readable league name |
| country | country |
| season | season string; summer leagues `"2025"`, winter `"2024-2025"` |
| match_date | date of the match (YYYY-MM-DD) |
| time | kickoff time if known (UTC) |
| home_team / away_team | team names (stable within a league) |
| ht_result | half-time score, `"H:A"` |
| ft_result | full-time score, `"H:A"` |
| home_goals / away_goals | full-time goals, integers |
| total_goals | home_goals + away_goals |
| result | `H` home win, `D` draw, `A` away win |
| first_goal_time | minute of the first goal; blank if the match was goalless or the timing was not recorded |
| all_goal_times | minutes of all goals, e.g. `"12,45,78"`; blank if goalless or timing not recorded |

Seasons included: 2022-2023, 2023, 2023-2024, 2024-2025, 2024, 2025-2026, 2025, 2026.

## The challenge

**Round 1 — beat a Dixon-Coles baseline (here).** You have the results and the
goal timing. Fit a model, score it with log-loss against what actually
happened, and try to beat the Dixon-Coles starter (see the accompanying
notebook). No sign-up, fully offline.

**Round 2 — beat the market (coming soon at football-charts.com).** Beating a
baseline model is the easy part. The betting market prices all of these matches
and is very hard to beat. We're building a public model leaderboard where you
submit probabilities for upcoming matches, before kickoff, and they're scored
against the closing line and published as a timestamped, honestly-settled
record. No tips, no claims, just receipts. Watch football-charts.com.

## Provenance & licence

Match results and goal times compiled by football-charts.com from public
sources. Free to use for research and modelling with attribution to
football-charts.com. No odds or derived market data are included in this file.
