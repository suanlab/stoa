"""Exact small-sample tests, implemented here because scipy is not available.

The paper reports one comparison it cannot resolve --- whether a query-agnostic placement
signal beats random selection on LoCoMo --- and the honest treatment of a null result is to
say how much data would have been needed to see the effect, not merely that we did not see
it. That requires an exact test (the samples are tens of questions with success rates near
0.1, where the normal approximation is not trustworthy) and a power curve computed under
the same test.

Everything is exact enumeration over the hypergeometric distribution, in log space so the
factorials do not overflow. Deterministic; the only randomness is in `power_fisher`, which
takes an explicit seed.
"""
from __future__ import annotations

import math
from functools import lru_cache


@lru_cache(maxsize=None)
def _lfact(n: int) -> float:
    return math.lgamma(n + 1)


def _log_hypergeom(a: int, b: int, c: int, d: int) -> float:
    """log P(this exact 2x2 table | fixed margins), the Fisher null."""
    n = a + b + c + d
    return (_lfact(a + b) + _lfact(c + d) + _lfact(a + c) + _lfact(b + d)
            - _lfact(n) - _lfact(a) - _lfact(b) - _lfact(c) - _lfact(d))


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the table [[a, b], [c, d]].

    Two-sided by the conventional definition: sum the probability of every table with the
    same margins that is no more likely than the observed one. This is the same convention
    scipy uses, so a number reported here is comparable to one a reviewer recomputes.
    """
    n1, n2 = a + b, c + d
    k = a + c
    obs = _log_hypergeom(a, b, c, d)
    tol = 1e-9
    total = 0.0
    lo, hi = max(0, k - n2), min(k, n1)
    for x in range(lo, hi + 1):
        lp = _log_hypergeom(x, n1 - x, k - x, n2 - (k - x))
        if lp <= obs + tol:
            total += math.exp(lp)
    return min(1.0, total)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p for paired binary outcomes.

    `b` and `c` are the DISCORDANT counts: b = A correct while B is wrong, c = the reverse.
    Concordant pairs carry no information about which arm is better and are excluded by
    construction.

    This is the right test when every arm answers the SAME questions, which is how the
    LoCoMo comparison is built. Treating those arms as independent samples (Fisher) throws
    away the pairing and is needlessly conservative -- with question difficulty varying far
    more than the arms do, most of the variance is shared and should be differenced out.

    Under the null the discordant pairs split 50/50, so this is a two-sided binomial test on
    `b` successes in `b + c` trials at p = 0.5.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Log space. The direct form -- sum(comb(n, i)) / 2**n -- overflows for n above ~1000,
    # which is reachable both from a large discordant count and from the power simulation's
    # upper bracket. It raised OverflowError on the first real LoCoMo run.
    log_half_n = n * math.log(0.5)
    terms = [_lfact(n) - _lfact(i) - _lfact(n - i) + log_half_n for i in range(k + 1)]
    m = max(terms)
    tail = math.exp(m) * sum(math.exp(t - m) for t in terms)
    return min(1.0, 2.0 * tail)


def n_for_power_mcnemar(p_disc: float, p_favour: float, alpha: float, target: float = 0.80,
                        lo: int = 10, hi: int = 4000,
                        trials: int = 1500, seed: int = 0) -> int | None:
    """Paired-design sample size: questions needed for `target` power under McNemar.

    `p_disc` is the fraction of questions on which the two arms disagree, and `p_favour` the
    share of those disagreements favouring the better arm. Both are read off a pilot run
    rather than assumed -- which is the whole point of running the pilot.

    `alpha` is REQUIRED and has no default. Sizing a study at 0.05 while declaring
    significance at a Bonferroni-corrected threshold understates the requirement by 50-65%,
    and a default made that easy to do by omission. Pass the threshold you will actually
    judge the result against.
    """
    import random as _random

    def power(n: int) -> float:
        rng = _random.Random(seed)
        rej = 0
        for _ in range(trials):
            b = c = 0
            for _ in range(n):
                if rng.random() < p_disc:
                    if rng.random() < p_favour:
                        b += 1
                    else:
                        c += 1
            if mcnemar_exact(b, c) < alpha:
                rej += 1
        return rej / trials

    if p_favour == 0.5 or p_disc == 0.0:
        return None
    if power(hi) < target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if power(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def power_fisher(p1: float, p2: float, n1: int, n2: int, alpha: float = 0.05,
                 trials: int = 4000, seed: int = 0) -> float:
    """P(reject at `alpha`) when the true rates are `p1`, `p2` and the arms have n1, n2 draws.

    Simulated rather than enumerated: exact enumeration is O(n1 n2) Fisher tests, each itself
    O(n), which is minutes per point at the sample sizes we care about. `trials` draws give a
    standard error of about 0.008 on the power estimate, which is far finer than the decision
    it feeds ("do we need 30 questions or 300?").
    """
    import random as _random
    rng = _random.Random(seed)
    reject = 0
    for _ in range(trials):
        a = sum(1 for _ in range(n1) if rng.random() < p1)
        c = sum(1 for _ in range(n2) if rng.random() < p2)
        if fisher_exact(a, n1 - a, c, n2 - c) < alpha:
            reject += 1
    return reject / trials


def n_for_power(p1: float, p2: float, target: float = 0.80, alpha: float = 0.05,
                lo: int = 10, hi: int = 4000, seed: int = 0) -> int | None:
    """Smallest equal per-arm n reaching `target` power, or None if `hi` does not reach it.

    Bisection on a simulated, therefore noisy, power curve: the returned n is accurate to
    within the granularity that matters (an order of magnitude), not to the last question.
    """
    if p1 == p2:
        return None
    if power_fisher(p1, p2, hi, hi, alpha, trials=1500, seed=seed) < target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if power_fisher(p1, p2, mid, mid, alpha, trials=1500, seed=seed) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def p_order_reversal(p1: float, p2: float, n_a: int, n_b: int,
                     trials: int = 20_000, seed: int = 0) -> float:
    """P(two independent runs of sizes `n_a` and `n_b` rank the two arms oppositely).

    This is what a reader needs in order to judge a reversal between two runs. If it is
    large, the reversal is not evidence of anything and neither run's ordering should be
    reported as a finding.
    """
    import random as _random
    rng = _random.Random(seed)
    rev = 0
    for _ in range(trials):
        def diff(n):
            a = sum(1 for _ in range(n) if rng.random() < p1)
            c = sum(1 for _ in range(n) if rng.random() < p2)
            return a - c
        if diff(n_a) * diff(n_b) < 0:
            rev += 1
    return rev / trials
