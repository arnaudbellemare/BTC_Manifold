import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import altair as alt
from arch import arch_model
from geomstats.geometry.riemannian_metric import RiemannianMetric
import geomstats.geometry.euclidean
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy.sparse.linalg import eigs
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")
st.title("BTC/USD Price Analysis on a Volatility-Weighted Manifold")
st.write("""
This application models the Bitcoin market as a 2D geometric space (manifold) of (Time, Log-Price).
The geometry is warped by GARCH volatility, with geodesics indicating trending/mean-reverting paths and critical levels derived geometrically.
- **High Volatility (Yellow areas):** Manifold is 'stretched,' indicating high risk.
- **Low Volatility (Dark areas):** Manifold is 'flat,' indicating calmer markets.
- **Geodesic (Red Line):** Straightest path through the manifold, showing idealized price movement.
- **S/R Grid:** Green (support) and red (resistance) lines show probable price levels.
""")

# --- Geometric Modeling Class ---
class VolatilityMetric(RiemannianMetric):
    def __init__(self, sigma, mu, t, T):
        self.dim = 2
        space = geomstats.geometry.euclidean.Euclidean(dim=self.dim)
        super().__init__(space=space)
        self.sigma_interp = interp1d(t, sigma, bounds_error=False, fill_value="extrapolate")
        self.mu_interp = interp1d(t, mu, bounds_error=False, fill_value="extrapolate")
        self.t = t
        self.T = T

    def metric_matrix(self, base_point):
        t_val = base_point[0]
        sigma_val = max(self.sigma_interp(t_val), 1e-6)
        mu_val = self.mu_interp(t_val)
        return np.array([[1.0, mu_val], [mu_val, sigma_val**2]])

    def christoffel_symbols(self, base_point):
        t_val = base_point[0]
        sigma_val = max(self.sigma_interp(t_val), 1e-6)
        mu_val = self.mu_interp(t_val)
        eps = 1e-6
        t_plus, t_minus = min(t_val + eps, self.T), max(t_val - eps, 0)
        d_sigma_dt = (self.sigma_interp(t_plus) - self.sigma_interp(t_minus)) / (2 * eps)
        d_mu_dt = (self.mu_interp(t_plus) - self.mu_interp(t_minus)) / (2 * eps)
        gamma = np.zeros((2, 2, 2))
        gamma[1, 0, 1] = (1 / sigma_val) * d_sigma_dt
        gamma[1, 1, 0] = gamma[1, 0, 1]
        gamma[0, 1, 1] = -sigma_val * d_sigma_dt
        gamma[0, 0, 1] = d_mu_dt
        return gamma

class CustomManifold(geomstats.geometry.manifold.Manifold):
    def __init__(self, dim, metric):
        super().__init__(dim=dim, metric=metric)

# --- Helper Functions ---
@st.cache_data
def fetch_kraken_data(symbol, timeframe, start_date, end_date):
    exchange = ccxt.kraken()
    since = int(start_date.timestamp() * 1000)
    limit = int((end_date - start_date).total_seconds() / 3600) + 24
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].dropna()
        if len(df) >= 50:
            st.success(f"Fetched {len(df)} data points for {symbol}.")
            return df
    except Exception as e:
        st.warning(f"Could not fetch data: {e}")
    st.error("Failed to fetch data. Using simulated data.")
    sim_t = pd.date_range(start=start_date, end=end_date, freq='h')
    sim_prices = 70000 + np.cumsum(np.random.normal(0, 500, len(sim_t)))
    return pd.DataFrame({'datetime': sim_t, 'close': sim_prices})

@st.cache_data
def fit_garch(returns, p=1, q=1):
    try:
        model = arch_model(returns, vol='Garch', p=p, q=q, mean='Constant').fit(disp='off')
        return model.conditional_volatility / 100, model.params.get('mu', 0.0) / 100
    except:
        st.warning("GARCH failed. Using constant volatility.")
        return np.full(len(returns), returns.std() / 100), returns.mean() / 100

def compute_geodesic(metric, t, p0, pT, T):
    manifold = CustomManifold(dim=2, metric=metric)
    geodesic = metric.geodesic(initial_point=np.array([0.0, p0]), end_point=np.array([T, pT]))
    points = geodesic(np.linspace(0, 1, len(t)))
    return pd.DataFrame({"Time": points[:, 0], "Price": points[:, 1], "Path": "Geodesic"})

def compute_single_path(p0, T, N, dt, drift, diffusion, rng_seed):
    np.random.seed(rng_seed)
    path = np.zeros((N, 2))
    path[0, :] = [0.0, p0]
    price_std = 10.0  # Clamp log-price to ±10 std deviations
    for j in range(N - 1):
        pos = path[j, :]
        dW = np.random.normal(0, np.sqrt(dt))
        stoch_term = np.array([0.0, diffusion[j] * dW])
        next_pos = pos + drift[j] * dt + stoch_term
        next_pos[1] = np.clip(next_pos[1], p0 - price_std, p0 + price_std)
        path[j + 1, :] = next_pos
    return path[:, 1]

def simulate_paths_manifold(metric, p0, T, N, n_paths, subsample_factor=0.2):
    N_coarse = max(int(N * subsample_factor), 50)
    dt = T / (N_coarse - 1)
    t_coarse = np.linspace(0, T, N_coarse)
    
    sigma = np.array([metric.sigma_interp(ti) for ti in t_coarse])
    mu = np.array([metric.mu_interp(ti) for ti in t_coarse])
    det = 1.0 * sigma**2 - mu**2
    det = np.maximum(det, 1e-6)
    g_inv_22 = 1.0 / det
    drift = np.vstack([np.ones(N_coarse), mu]).T
    diffusion = np.sqrt(g_inv_22)
    
    # Validate sigma and mu
    if np.any(~np.isfinite(sigma)) or np.any(~np.isfinite(mu)):
        st.warning("Invalid GARCH parameters detected. Using constant values.")
        sigma = np.full_like(sigma, np.mean(sigma[np.isfinite(sigma)]))
        mu = np.full_like(mu, np.mean(mu[np.isfinite(mu)]))
        det = 1.0 * sigma**2 - mu**2
        det = np.maximum(det, 1e-6)
        g_inv_22 = 1.0 / det
        drift = np.vstack([np.ones(N_coarse), mu]).T
        diffusion = np.sqrt(g_inv_22)
    
    try:
        paths_coarse = np.array(Parallel(n_jobs=-1, backend='loky')(
            delayed(compute_single_path)(p0, T, N_coarse, dt, drift, diffusion, i)
            for i in range(n_paths)
        ))
    except Exception as e:
        st.warning(f"Parallel execution failed: {e}. Falling back to sequential.")
        paths_coarse = np.zeros((n_paths, N_coarse))
        for i in range(n_paths):
            paths_coarse[i, :] = compute_single_path(p0, T, N_coarse, dt, drift, diffusion, i)
    
    t = np.linspace(0, T, N)
    paths = np.zeros((n_paths, N))
    for i in range(n_paths):
        if np.any(~np.isfinite(paths_coarse[i])):
            st.warning(f"Path {i} contains invalid values. Replacing with initial price.")
            paths[i, :] = p0
        else:
            interp = interp1d(t_coarse, paths_coarse[i], bounds_error=False, fill_value=p0)
            paths[i, :] = interp(t)
    
    return paths, t

def compute_critical_levels(metric, t_grid, p_grid, n_levels=4):
    N_t, N_p = len(t_grid), len(p_grid)
    laplacian = np.zeros((N_t * N_p, N_t * N_p))
    for i in range(N_t):
        for j in range(N_p):
            idx = i * N_p + j
            point = [t_grid[i], p_grid[j]]
            g = metric.metric_matrix(point)
            g_inv = np.linalg.inv(g)
            laplacian[idx, idx] = -2 * np.sum(g_inv)
            if j > 0: laplacian[idx, idx - 1] = g_inv[1, 1]
            if j < N_p - 1: laplacian[idx, idx + 1] = g_inv[1, 1]
    _, eigenvectors = eigs(laplacian, k=n_levels, which='SM')
    levels = []
    for ev in eigenvectors.T:
        ev_grid = ev.reshape(N_t, N_p)
        peak_idx = np.argmax(np.abs(ev_grid[-1, :]))
        levels.append(p_grid[peak_idx])
    return np.array(levels)

def classify_regime(geodesic_df, prices, times):
    deviation = np.abs(geodesic_df['Price'].values - np.mean(prices))
    if np.any(~np.isfinite(deviation)):
        return "Unknown (Invalid Data)"
    if np.mean(deviation[-len(deviation)//4:]) > np.std(prices):
        return "Trending"
    return "Mean Reverting"

def visualize_manifold(metric, t_grid, p_grid):
    valid_t = [ti for ti in t_grid if np.isfinite(metric.sigma_interp(ti))]
    if not valid_t:
        st.warning("Invalid volatility data for heatmap. Using constant value.")
        SCALING_FACTOR = 1000
    else:
        SCALING_FACTOR = 1 / np.max([metric.sigma_interp(ti)**2 for ti in valid_t]) * 1000
    g_pp_values = []
    for t_val in t_grid:
        cost = metric.metric_matrix([t_val, 0])[1, 1]
        for p_val in p_grid:
            g_pp_values.append({'Time': t_val, 'Price': p_val, 'Cost': cost * SCALING_FACTOR})
    g_df = pd.DataFrame(g_pp_values)
    return alt.Chart(g_df).mark_rect().encode(
        x='Time:Q', y=alt.Y('Price:Q', scale=alt.Scale(zero=False)),
        color=alt.Color('Cost:Q', scale=alt.Scale(scheme='viridis'),
                        legend=alt.Legend(title=f"Cost (σ² × {SCALING_FACTOR})"))
    ).properties(title="Market Manifold Geometry")

# --- Main Application ---
st.sidebar.header("Model Parameters")
days_history = st.sidebar.slider("Historical Data (Days)", 7, 90, 30)
n_paths = st.sidebar.slider("Simulated Paths", 500, 5000, 1000, step=100)
n_display_paths = st.sidebar.slider("Displayed Paths", 10, 200, 50, step=10)
epsilon_factor = st.sidebar.slider("Probability Range Factor", 0.1, 1.0, 0.25, step=0.05)
garch_p = st.sidebar.slider("GARCH p", 1, 3, 1)
garch_q = st.sidebar.slider("GARCH q", 1, 3, 1)
subsample_factor = st.sidebar.slider("Time Subsample Factor", 0.1, 1.0, 0.2, step=0.1)

end_date = pd.Timestamp.now(tz='UTC')
start_date = end_date - pd.Timedelta(days=days_history)
df = fetch_kraken_data('BTC/USD', '1h', start_date, end_date)

if df is not None and len(df) > 50 and df['close'].std() > 1e-6:
    prices = np.log(df['close'].values)
    times = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds() / 3600
    T = times.iloc[-1]
    N = len(prices)
    p0 = prices[0]
    returns = 100 * df['close'].pct_change().dropna()

    with st.spinner("Fitting GARCH model..."):
        sigma, mu = fit_garch(returns, garch_p, garch_q)
        sigma = np.pad(sigma, (N - len(sigma), 0), mode='edge')
        mu = np.full(N, mu)

    metric = VolatilityMetric(sigma, mu, times, T)
    if np.any(metric.sigma_interp(times) < 1e-6):
        st.warning("GARCH volatility too low; clamping may affect geometry.")

    with st.spinner("Simulating price paths..."):
        paths, t = simulate_paths_manifold(metric, p0, T, N, n_paths, subsample_factor)
        paths = np.exp(paths)  # Convert back to price

    col1, col2 = st.columns([2, 1])
    with col1:
        final_prices = np.exp(paths[:, -1])
        if np.any(~np.isfinite(final_prices)):
            st.warning("Final prices contain NaNs or Infs. Filtering invalid values.")
            valid_mask = np.isfinite(final_prices)
            final_prices = final_prices[valid_mask]
            if len(final_prices) < 10:
                st.error("Too few valid prices for KDE. Using historical quantiles for S/R.")
                final_prices = np.exp(prices)
        
        try:
            kde = gaussian_kde(final_prices)
            price_grid = np.linspace(final_prices.min(), final_prices.max(), 500)
            u = kde(price_grid)
            u /= np.trapz(u, price_grid)
        except:
            st.warning("KDE failed. Using uniform distribution.")
            u = np.ones_like(price_grid) / (price_grid.max() - price_grid.min())

        critical_levels = np.exp(compute_critical_levels(metric, t, np.linspace(prices.min(), prices.max(), 50)))
        support_levels = critical_levels[critical_levels <= np.median(critical_levels)]
        resistance_levels = critical_levels[critical_levels > np.median(critical_levels)]

        epsilon = epsilon_factor * np.mean([np.sqrt(metric.metric_matrix([T, np.log(p)])[1, 1]) for p in price_grid if np.isfinite(np.log(p))])
        def get_hit_prob(level_list):
            probs = []
            for level in level_list:
                if np.isfinite(np.log(level)):
                    mask = (price_grid >= level - epsilon) & (price_grid <= level + epsilon)
                    raw_prob = np.trapz(u[mask], price_grid[mask])
                    volume_element = np.sqrt(np.abs(np.linalg.det(metric.metric_matrix([T, np.log(level)]))))
                    probs.append(raw_prob * volume_element)
                else:
                    probs.append(0.0)
            total_prob = sum(probs)
            return [p / total_prob for p in probs] if total_prob > 0 else [0] * len(probs)
        
        support_probs = get_hit_prob(support_levels)
        resistance_probs = get_hit_prob(resistance_levels)

        with st.spinner("Computing geodesic path..."):
            geodesic_df = compute_geodesic(metric, t, p0, prices[-1], T)
            geodesic_df['Price'] = np.exp(geodesic_df['Price'])

        regime = classify_regime(geodesic_df, np.exp(prices), times)
        st.write(f"Market Regime: {regime}")

        path_data = [{"Time": t[j], "Price": paths[i, j], "Path": f"Path_{i}"} 
                     for i in range(min(n_paths, n_display_paths)) for j in range(N) if np.isfinite(paths[i, j])]
        plot_df = pd.concat([pd.DataFrame(path_data), geodesic_df])
        support_df = pd.DataFrame({"Price": support_levels})
        resistance_df = pd.DataFrame({"Price": resistance_levels})
        base = alt.Chart(plot_df).encode(x=alt.X("Time:Q", title="Time (hours)"), 
                                        y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False)))
        main_chart = (base.mark_line(opacity=0.2).encode(color=alt.value('gray'), detail='Path:N').transform_filter(alt.datum.Path != "Geodesic") +
                      base.mark_line(strokeWidth=3, color="red").transform_filter(alt.datum.Path == "Geodesic") +
                      alt.Chart(support_df).mark_rule(stroke="green", strokeWidth=1.5).encode(y="Price:Q") +
                      alt.Chart(resistance_df).mark_rule(stroke="red", strokeWidth=1.5).encode(y="Price:Q")
                     ).properties(title="Price Paths, Geodesic, and S/R Grid", height=500).interactive()
        st.altair_chart(main_chart, use_container_width=True)

    with col2:
        viz_p_grid = np.linspace(np.exp(prices).min(), np.exp(prices).max(), 50)
        manifold_heatmap = visualize_manifold(metric, t, viz_p_grid)
        history_df = pd.DataFrame({'Time': times, 'Price': np.exp(prices)})
        history_line = alt.Chart(history_df).mark_line(color='white', strokeWidth=2.5).encode(x='Time:Q', y='Price:Q')
        st.altair_chart((manifold_heatmap + history_line).properties(height=300).interactive(), use_container_width=True)

        st.subheader("Analysis Summary")
        st.metric("Expected Final Price", f"${np.mean(final_prices[np.isfinite(final_prices)]):,.2f}")
        st.write("**Support Levels:**")
        if support_levels.size > 0:
            st.dataframe(pd.DataFrame({'Level': support_levels, 'Hit Probability': support_probs}).style.format({'Level': '${:,.2f}', 'Hit Probability': '{:.1%}'}))
        else:
            st.write("No distinct support levels.")
        st.write("**Resistance Levels:**")
        if resistance_levels.size > 0:
            st.dataframe(pd.DataFrame({'Level': resistance_levels, 'Hit Probability': resistance_probs}).style.format({'Level': '${:,.2f}', 'Hit Probability': '{:.1%}'}))
        else:
            st.write("No distinct resistance levels.")

        current_price = np.exp(prices[-1])
        signals = []
        for level, prob in zip(support_levels, support_probs):
            if abs(current_price - level) < epsilon and prob > 0.3:
                signals.append(f"Buy at ${level:.2f} (Support, {prob:.1%} probability)")
        for level, prob in zip(resistance_levels, resistance_probs):
            if abs(current_price - level) < epsilon and prob > 0.3:
                signals.append(f"Sell at ${level:.2f} (Resistance, {prob:.1%} probability)")
        if signals:
            st.write("**Trading Signals:**")
            for signal in signals:
                st.write(signal)
else:
    st.error("Insufficient or invalid data for analysis.")
