"""Paired statistics shared by evaluation and diagnostic commands."""

from math import comb


def mcnemar(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value for discordant pairs."""
    n = wins + losses
    if not n:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2**n)
