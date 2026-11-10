"""Exact count sampling and designs for unknown-reliability repetition.

This module defines the new study. The known-scale baseline model.py is
preserved with its original source hash.
"""

from functools import lru_cache
import math
import numpy as np
from scipy.stats import binom


def pilot_estimates(sums, m):
    """Reliabilities from the absolute means of one retained sign per channel."""
    return np.maximum(np.abs(sums) / m, 1 / math.sqrt(m))


def adaptive_depths(ahat, m):
    # The cap is redundant in exact arithmetic and protects integer rounding.
    return np.minimum(m, np.maximum(1, np.ceil(ahat ** -2))).astype(np.int64)


def dyadic_schedule(budget):
    m = budget // 8
    stages = int(math.log2(budget))
    block_budget = (budget - 2 * m) // stages
    schedule = [[2**j, block_budget // (2 ** (j + 1))]
                for j in range(stages)]
    schedule = [(r, n) for r, n in schedule if n > 0]
    used = 2 * m + sum(2 * r * n for r, n in schedule)
    r, n = schedule[0]
    schedule[0] = (r, n + (budget - used) // 2)
    return schedule


def posterior_response(counts, depth, reliability):
    if reliability == 1.0:
        return np.sign(2 * counts - depth)
    return np.tanh(np.arctanh(reliability) * (2 * counts - depth))


@lru_cache(maxsize=256)
def gamma(depth, reliability):
    k = np.arange(depth + 1)
    p = binom.pmf(k, depth, (1 + reliability) / 2)
    # The square is invariant under the latent sign.
    return float(p @ posterior_response(k, depth, reliability) ** 2)


@lru_cache(maxsize=256)
def cross_law(r1, r2, a1, a2, q, posterior=False):
    """The exact law needed for a sign-invariant pair statistic.

    Fix S1=+1; reflecting both received count vectors leaves the statistic
    unchanged. S2 then equals the latent sign product with mean q.
    """
    k1, k2 = np.arange(r1 + 1), np.arange(r2 + 1)
    p1 = binom.pmf(k1, r1, (1 + a1) / 2)
    p2 = binom.pmf(k2, r2, (1 + a2) / 2)
    p = np.outer(p1, (1 + q) / 2 * p2 + (1 - q) / 2 * p2[::-1])
    if posterior:
        x1 = posterior_response(k1, r1, a1)
        x2 = posterior_response(k2, r2, a2)
    else:
        x1, x2 = (2 * k1 - r1) / r1, (2 * k2 - r2) / r2
    values = np.outer(x1, x2).ravel()
    prob = p.ravel()
    prob /= prob.sum()
    return prob, values


def draw_cross_means(rng, r1, r2, a1, a2, q, n, trials,
                     posterior=False):
    """Exact group-statistic samples; no normal approximation."""
    if n == 0:
        return np.zeros(trials)
    if (r1 + 1) * (r2 + 1) <= 4096:
        prob, values = cross_law(r1, r2, a1, a2, q, posterior)
        out = np.empty(trials)
        chunk = max(1, 1_000_000 // len(prob))
        for lo in range(0, trials, chunk):
            size = min(chunk, trials - lo)
            hist = rng.multinomial(n, prob, size=size)
            out[lo:lo + size] = hist @ values / n
        return out
    out = np.empty(trials)
    chunk = max(1, 1_000_000 // n)
    for lo in range(0, trials, chunk):
        size = min(chunk, trials - lo)
        c1 = rng.binomial(r1, (1 + a1) / 2, size=(size, n))
        c2 = rng.binomial(r2, (1 + a2) / 2, size=(size, n))
        signs = 2 * rng.binomial(1, (1 + q) / 2, size=(size, n)) - 1
        if posterior:
            x1 = posterior_response(c1, r1, a1)
            x2 = posterior_response(c2, r2, a2)
        else:
            x1, x2 = (2 * c1 - r1) / r1, (2 * c2 - r2) / r2
        out[lo:lo + size] = np.mean(signs * x1 * x2, axis=1)
    return out


def rho_from_raw(raw):
    return np.sin(np.pi / 2 * np.clip(raw, -1, 1))


def summarize(estimates, rho):
    loss = (estimates - rho) ** 2
    return {
        "mse": float(loss.mean()),
        "mcse": float(loss.std(ddof=1) / math.sqrt(len(loss))),
        "bias": float(estimates.mean() - rho),
        "sample_variance": float(estimates.var(ddof=1)),
    }
