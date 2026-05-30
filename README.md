# Delta-Hedged Volatility Model

A Python implementation of European option pricing, SVI volatility surface fitting, and a discrete delta-hedging simulator that empirically demonstrates the gamma–vol P&L relationship.

The project shows that the P&L of a delta-hedged short option position is driven by the gap between implied and realised volatility, with variance proportional to dollar gamma. This is the central identity underpinning volatility trading.

## What's in here

| File | Purpose |
|---|---|
| `black_scholes.py` | Closed-form pricing, full Greeks (delta, gamma, vega, theta), and a Brent-method implied vol solver robust to arbitrage-violating quotes |
| `svi_surface.py` | Gatheral's SVI parameterisation fitted per expiry slice in total-variance space, with a butterfly-arbitrage check on the implied density |
| `delta_hedge.py` | Monte Carlo simulator: sell a call, hedge delta daily, attribute terminal P&L against (σ_realised² − σ_implied²) |

## Theory

**Pricing.** Black-Scholes prices a European option given five inputs (S, K, T, r, σ). Implied volatility is the σ that reproduces a market price — recovered numerically with Brent's method.

**The surface.** Implied vol varies by strike (smile/skew) and by expiry (term structure). The SVI parameterisation models total variance per expiry as

w(k) = a + b · (ρ(k − m) + √((k − m)² + s²))

where k = log(K/F). The five parameters (a, b, ρ, m, s) each have a geometric interpretation: level, wing slope, tilt, horizontal shift, and curvature. Fitting in variance space (not vol space) is standard because the formula is well-behaved there and avoids overweighting tiny vol differences in the wings.

**Hedging P&L.** For a delta-hedged short call, the per-step P&L decomposes as

ΔP&L ≈ ½ · Γ · S² · (σ_realised² − σ_implied²) · Δt

So profit on a short option position requires implied vol to exceed realised vol, weighted by dollar gamma (Γ·S²). The residual P&L when implied = realised is pure discrete-hedging error, which shrinks as 1/√n_rebalances.

## Results

Monte Carlo over 1000 paths of a 3-month ATM call, daily rebalancing:

| Scenario | Mean P&L | Std P&L | Comment |
|---|---|---|---|
| Sold at 25% IV, realised 20% | +1.01 | 0.55 | Selling expensive vol pays off |
| Sold at 20% IV, realised 30% | −1.97 | 1.10 | Short gamma in high-vol regime is brutal |
| Sold at 20% IV, realised 20% | +0.02 | 0.43 | Fair-value baseline; residual is discretisation error |

The Sharpe per trade in the favourable scenario is 1.84. Halving rebalance frequency materially widens the P&L distribution, illustrating the cost of discrete hedging.

## Running

```bash
git clone git@github.com:ashleykumarn/deltahedgemodel.git
cd deltahedgemodel
pip install numpy scipy pandas

python black_scholes.py    # sanity checks on pricer and IV solver
python svi_surface.py      # synthetic SVI fit with arbitrage check
python delta_hedge.py      # MC P&L for three implied/realised scenarios
```

Python 3.10+. No other dependencies.

## What I'd extend next

- Pull a live option chain (e.g. SPY via `yfinance`) and fit the full surface across expiries; enforce calendar arbitrage between slices
- Vega-weight the SVI loss so deep OTM illiquid quotes don't dominate
- Backtest a weekly short-straddle hedged daily over 2020–2024 to quantify the equity variance risk premium and its drawdowns
- Add transaction costs (1 bp per share) and study the impact on Sharpe and capacity
- Replace constant-vol GBM with a Heston path to test hedging performance when the model is mis-specified
