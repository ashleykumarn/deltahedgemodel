"""
Delta-hedged P&L simulation.

The trade we simulate: sell 1 European call at time 0 at the market's
implied vol. Re-hedge delta daily by holding -delta shares of stock and
financing the position at the risk-free rate. At expiry, settle the
option payoff and unwind the hedge.

Key result we will demonstrate empirically:
    Total P&L ~ (1/2) * sum_t [ Gamma_t * S_t^2 * (sigma_realised^2 - sigma_implied^2) * dt ]

So if the realised vol of the underlying turns out HIGHER than what
we sold the option for, we lose money. Lower, we make money.
"""

import numpy as np
import pandas as pd
from black_scholes import bs_price, bs_delta, bs_gamma


def simulate_gbm_path(S0, mu, sigma, T, n_steps, seed=None):
    """
    Simulate a Geometric Brownian Motion stock path.
    dS = mu*S*dt + sigma*S*dW

    This is the "true" world. We will then trade an option on this stock
    using a possibly DIFFERENT implied vol, so we can study what happens
    when realised != implied.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal(n_steps)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_prices = np.log(S0) + np.cumsum(log_returns)
    return np.concatenate([[S0], np.exp(log_prices)])


def delta_hedge_short_call(path, K, T, r, sigma_implied, q=0.0):
    """
    Sell 1 call at time 0, delta-hedge daily, mark-to-market.

    Bookkeeping (the part that trips most people up):
        cash account: starts at +premium received
        stock position: -delta shares (short stock to hedge the long-delta call we sold)
        Each rebalance: buy/sell shares so position = -delta_new
        Cash account accrues interest at rate r between rebalances.

    Returns a DataFrame with the per-step state and a final P&L.
    """
    n_steps = len(path) - 1
    dt = T / n_steps

    # Time to expiry at each step (decreases to 0)
    times = np.linspace(T, 0, n_steps + 1)

    records = []
    S0 = path[0]

    # Step 0: sell the call, get premium, set up initial hedge
    premium = bs_price(S0, K, times[0], r, sigma_implied, "call", q)
    delta0 = bs_delta(S0, K, times[0], r, sigma_implied, "call", q)

    # We sold the call. To hedge, we need to be long delta shares (since
    # short call has delta = -delta). Cash to buy delta0 shares: delta0 * S0.
    cash = premium - delta0 * S0  # premium received, paid for hedge shares
    shares = delta0

    for i in range(1, n_steps + 1):
        S = path[i]
        T_left = times[i]

        # Accrue interest on cash position over dt
        cash *= np.exp(r * dt)

        # Compute new delta and rebalance
        if T_left > 0:
            new_delta = bs_delta(S, K, T_left, r, sigma_implied, "call", q)
            gamma = bs_gamma(S, K, T_left, r, sigma_implied, q)
            option_value = bs_price(S, K, T_left, r, sigma_implied, "call", q)
        else:
            # At expiry
            new_delta = 1.0 if S > K else 0.0
            gamma = 0.0
            option_value = max(S - K, 0.0)

        # Trade to get to new_delta shares: buy (new_delta - shares) shares
        trade = new_delta - shares
        cash -= trade * S
        shares = new_delta

        # Portfolio value = cash + shares * S - option_liability
        # (we are SHORT the call, so option_value is a liability)
        pnl = cash + shares * S - option_value

        records.append({
            "step": i,
            "S": S,
            "T_left": T_left,
            "delta": new_delta,
            "gamma": gamma,
            "option_value": option_value,
            "shares_held": shares,
            "cash": cash,
            "pnl": pnl,
        })

    df = pd.DataFrame(records)
    df.attrs["premium"] = premium
    df.attrs["S0"] = S0
    df.attrs["K"] = K
    df.attrs["T"] = T
    df.attrs["sigma_implied"] = sigma_implied
    return df


def realised_vol(path, T):
    """
    Realised volatility of a path. Standard estimator on log returns,
    annualised.
    """
    log_returns = np.diff(np.log(path))
    n_steps = len(log_returns)
    dt = T / n_steps
    # Variance per step, annualised
    return np.sqrt(np.sum(log_returns ** 2) / T)


def run_monte_carlo(S0, K, T, r, sigma_realised, sigma_implied,
                    mu=0.05, n_steps=252, n_paths=1000, seed=0):
    """
    Run many independent paths and collect terminal P&Ls.
    We use sigma_realised to simulate the stock path (the "true" world),
    but we price and hedge the option using sigma_implied (what the market
    thinks). The expected P&L should be positive if implied > realised
    (we sold expensive vol).
    """
    rng = np.random.default_rng(seed)
    pnls = []
    for i in range(n_paths):
        path = simulate_gbm_path(S0, mu, sigma_realised, T, n_steps,
                                 seed=rng.integers(0, 2**31))
        df = delta_hedge_short_call(path, K, T, r, sigma_implied)
        pnls.append(df["pnl"].iloc[-1])
    return np.array(pnls)


if __name__ == "__main__":
    # Experiment: sell an ATM call at 25% implied, but realised is only 20%.
    # We should make money on average.
    print("=" * 60)
    print("Experiment 1: implied 25%, realised 20% (we sold expensive)")
    print("=" * 60)
    pnls = run_monte_carlo(
        S0=100, K=100, T=0.25, r=0.05,
        sigma_realised=0.20, sigma_implied=0.25,
        n_steps=63, n_paths=1000, seed=42,
    )
    print(f"Mean P&L: {pnls.mean():+.4f}")
    print(f"Std  P&L: {pnls.std():.4f}")
    print(f"Sharpe (per trade): {pnls.mean() / pnls.std():.3f}")

    print()
    print("=" * 60)
    print("Experiment 2: implied 20%, realised 30% (we sold too cheap)")
    print("=" * 60)
    pnls2 = run_monte_carlo(
        S0=100, K=100, T=0.25, r=0.05,
        sigma_realised=0.30, sigma_implied=0.20,
        n_steps=63, n_paths=1000, seed=42,
    )
    print(f"Mean P&L: {pnls2.mean():+.4f}")
    print(f"Std  P&L: {pnls2.std():.4f}")

    print()
    print("=" * 60)
    print("Experiment 3: implied = realised = 20% (fair price)")
    print("=" * 60)
    pnls3 = run_monte_carlo(
        S0=100, K=100, T=0.25, r=0.05,
        sigma_realised=0.20, sigma_implied=0.20,
        n_steps=63, n_paths=1000, seed=42,
    )
    print(f"Mean P&L: {pnls3.mean():+.4f}")
    print(f"Std  P&L: {pnls3.std():.4f}")
    print("(should be ~0 mean; the std is purely hedging error from discrete rebalancing)")
