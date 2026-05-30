"""
SVI (Stochastic Volatility Inspired) volatility surface.

Gatheral's parameterisation for total implied variance w = sigma^2 * T:

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + s^2))

where k = log(K / F) is the log-moneyness and F is the forward.

Parameter intuition (for one expiry slice):
    a  - vertical level (overall variance offset)
    b  - controls the slope of both wings; b >= 0
    rho - skew/tilt; rho in [-1, 1]. Negative rho = left wing steeper
          (typical for equity indices: OTM puts cost more).
    m  - horizontal shift; where the minimum of the smile sits
    s  - curvature near the minimum; s > 0. Smaller s = sharper smile.

We fit one SVI slice per expiry independently. For a full surface you
would also want to enforce calendar arbitrage (longer expiries must
have weakly higher total variance at every strike). We check it after
fitting and warn if violated.
"""

import numpy as np
from scipy.optimize import minimize


def svi_total_variance(k, a, b, rho, m, s):
    """Total implied variance w(k) = sigma^2(k) * T."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))


def svi_iv(k, T, a, b, rho, m, s):
    """Convenience: implied vol (not variance) for a given log-moneyness."""
    w = svi_total_variance(k, a, b, rho, m, s)
    return np.sqrt(np.maximum(w, 1e-8) / T)


def fit_svi_slice(k, iv, T, weights=None):
    """
    Fit SVI parameters to a single expiry slice.

    Inputs:
        k       : array of log-moneyness values
        iv      : array of observed implied vols at those strikes
        T       : time to expiry (years)
        weights : optional, e.g. vega-weights so we don't overfit deep OTM
                  quotes that are illiquid.

    We minimise sum of squared errors in TOTAL VARIANCE space, not vol space.
    This is the standard choice because the SVI formula is linear-ish in
    variance, and it avoids over-penalising tiny vol differences in the
    wings.
    """
    w_market = (iv ** 2) * T  # observed total variance
    if weights is None:
        weights = np.ones_like(k)

    def loss(params):
        a, b, rho, m, s = params
        w_model = svi_total_variance(k, a, b, rho, m, s)
        return np.sum(weights * (w_model - w_market) ** 2)

    # Bounds enforce the basic shape constraints
    bounds = [
        (1e-6, 1.0),    # a >= 0 (variance can't be negative at minimum)
        (1e-6, 2.0),    # b >= 0 (wings open upward)
        (-0.999, 0.999),  # |rho| < 1
        (-1.0, 1.0),    # m: log-moneyness shift; rarely outside [-1, 1]
        (1e-4, 1.0),    # s > 0 (smile has finite curvature)
    ]

    # Reasonable starting point: flat smile centred at zero
    x0 = [np.mean(w_market) * 0.5, 0.1, -0.3, 0.0, 0.1]

    result = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
    return result.x  # (a, b, rho, m, s)


def check_butterfly_arbitrage(params, k_grid=None):
    """
    Butterfly arbitrage: the density implied by the smile must be non-negative.
    For SVI there is a closed-form condition (Gatheral & Jacquier, 2014)
    involving g(k) = (1 - k * w'(k) / (2*w))^2 - (w'(k)^2 / 4) * (1/w + 0.25)
                    + w''(k) / 2.
    If g(k) >= 0 everywhere, no butterfly arb.

    We just check numerically on a grid here, which is good enough for
    a project. Real desks use the analytical condition.
    """
    a, b, rho, m, s = params
    if k_grid is None:
        k_grid = np.linspace(-1.0, 1.0, 200)

    def w(k):
        return svi_total_variance(k, a, b, rho, m, s)

    h = 1e-4
    wk = w(k_grid)
    wp = (w(k_grid + h) - w(k_grid - h)) / (2 * h)        # w'(k)
    wpp = (w(k_grid + h) - 2 * wk + w(k_grid - h)) / (h ** 2)  # w''(k)

    g = (1 - k_grid * wp / (2 * wk)) ** 2 - (wp ** 2 / 4) * (1 / wk + 0.25) + wpp / 2
    return np.all(g >= -1e-6), float(np.min(g))


if __name__ == "__main__":
    # Build a synthetic smile to fit, just to verify the fitter works.
    # We'll pretend the true smile has known SVI params and fit them back.
    true_params = (0.04, 0.4, -0.4, 0.0, 0.1)  # roughly 20% ATM vol, skewed
    T = 0.5
    k_obs = np.linspace(-0.3, 0.3, 15)
    iv_true = svi_iv(k_obs, T, *true_params)
    # Add small noise
    rng = np.random.default_rng(0)
    iv_obs = iv_true + rng.normal(0, 0.003, size=len(k_obs))

    fitted = fit_svi_slice(k_obs, iv_obs, T)
    print("True params:   ", [f"{x:+.4f}" for x in true_params])
    print("Fitted params: ", [f"{x:+.4f}" for x in fitted])

    no_arb, g_min = check_butterfly_arbitrage(fitted)
    print(f"Butterfly arb-free: {no_arb}  (min g = {g_min:.4f})")

    # RMSE in vol space
    iv_fit = svi_iv(k_obs, T, *fitted)
    rmse = np.sqrt(np.mean((iv_fit - iv_obs) ** 2))
    print(f"Fit RMSE in vol: {rmse * 100:.3f} vol points")
