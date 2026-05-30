"""
Black-Scholes pricing, Greeks, and implied volatility solver.

The whole module is built around one identity:
    BS_price(S, K, T, r, sigma) = market_price
We can either go forward (price an option given sigma) or backward
(solve for sigma given a market price). Going backward is what gives
us "implied volatility".
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def d1(S, K, T, r, sigma, q=0.0):
    """
    The d1 term shows up everywhere in BS. Intuitively it measures how
    far in-the-money the option is, scaled by total volatility sqrt(T)*sigma.
    q is the continuous dividend yield (0 for non-dividend stocks).
    """
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma, q=0.0):
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Black-Scholes price of a European option.

    Interpretation of the formula for a call:
        Price = S * N(d1) - K * exp(-rT) * N(d2)
    First term: present value of receiving the stock IF exercised.
    Second term: present value of paying the strike IF exercised.
    N(d2) is the risk-neutral probability of finishing in-the-money.
    """
    if T <= 0:
        # At expiry, option is worth max(S-K, 0) for call, max(K-S, 0) for put
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    D1 = d1(S, K, T, r, sigma, q)
    D2 = d2(S, K, T, r, sigma, q)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-D2) - S * np.exp(-q * T) * norm.cdf(-D1)


def bs_delta(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Delta: dPrice/dSpot. For a call, this is N(d1), which is bounded in [0, 1].
    Interpretation: if you're long 1 call with delta 0.6, your position behaves
    like being long 0.6 shares of stock. So you hedge by SHORTING 0.6 shares.
    """
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    D1 = d1(S, K, T, r, sigma, q)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(D1)
    return np.exp(-q * T) * (norm.cdf(D1) - 1.0)


def bs_gamma(S, K, T, r, sigma, q=0.0):
    """
    Gamma: d2Price/dSpot2 = dDelta/dSpot.
    How fast your delta changes as the stock moves. High gamma = need to
    re-hedge often. Gamma is highest for at-the-money options near expiry.
    """
    if T <= 0:
        return 0.0
    D1 = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(D1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, sigma, q=0.0):
    """
    Vega: dPrice/dSigma. NOT a Greek letter, despite the name.
    Sensitivity to volatility. Used in the implied-vol Newton solver,
    and to understand vol exposure of a position.
    """
    if T <= 0:
        return 0.0
    D1 = d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(D1) * np.sqrt(T)


def bs_theta(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Theta: dPrice/dT. Time decay. Almost always negative for long options:
    you lose value as time passes, holding all else equal. This is the
    "rent" you pay for optionality.
    """
    if T <= 0:
        return 0.0
    D1 = d1(S, K, T, r, sigma, q)
    D2 = d2(S, K, T, r, sigma, q)
    first = -S * np.exp(-q * T) * norm.pdf(D1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        return first - r * K * np.exp(-r * T) * norm.cdf(D2) + q * S * np.exp(-q * T) * norm.cdf(D1)
    return first + r * K * np.exp(-r * T) * norm.cdf(-D2) - q * S * np.exp(-q * T) * norm.cdf(-D1)


def implied_vol(market_price, S, K, T, r, option_type="call", q=0.0):
    """
    Solve for sigma such that bs_price(...) == market_price.

    We use Brent's method (a bracketed root-finder) because it's robust:
    no derivatives needed, and it can't diverge. We just need a lower and
    upper bound that bracket the true vol. 1e-5 to 5.0 (i.e. 500% vol)
    covers any realistic option.

    Why robust matters: in real data you'll get garbage quotes that are
    arbitrage-violating (e.g. price below intrinsic value). Brent will
    fail gracefully; Newton's method on these would explode.
    """
    # Sanity check: price must be above intrinsic value (no-arbitrage)
    if option_type == "call":
        intrinsic = max(S - K * np.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0.0)

    if market_price < intrinsic - 1e-6:
        return np.nan  # arbitrage violation, garbage quote

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type, q) - market_price

    try:
        return brentq(objective, 1e-5, 5.0, xtol=1e-6)
    except ValueError:
        # Brent couldn't bracket a root: quote is outside the model's range
        return np.nan


if __name__ == "__main__":
    # Quick sanity check: a 1-year ATM call at 20% vol, 5% rates, no divs
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    price = bs_price(S, K, T, r, sigma)
    print(f"Call price:    {price:.4f}")
    print(f"Delta:         {bs_delta(S, K, T, r, sigma):.4f}")
    print(f"Gamma:         {bs_gamma(S, K, T, r, sigma):.4f}")
    print(f"Vega:          {bs_vega(S, K, T, r, sigma):.4f}")
    print(f"Theta (daily): {bs_theta(S, K, T, r, sigma) / 365:.4f}")

    # Round-trip: get price, recover sigma
    iv = implied_vol(price, S, K, T, r)
    print(f"\nRecovered IV:  {iv:.6f}  (input was {sigma})")
