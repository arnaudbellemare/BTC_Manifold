import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import altair as alt
from arch import arch_model
from geomstats.geometry.riemannian_metric import RiemannianMetric
import geomstats.geometry.euclidean
from geomstats.numerics.geodesic import ExpODESolver
from geomstats.numerics.ivp import ScipySolveIVP
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
        self.exp_solver = ExpODESolver(space=space, integrator=ScipySolveIVP(method='RK45', point_ndim=2))

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
        sigma = np.clip(model.conditional_volatility / 100, 1e-6, 1.0)
        mu = np.clip(model.params.get('mu', 0.0) / 100, -0.1, 0.1)
        return sigma, mu
    except:
        st.warning("GARCH failed. Using constant volatility.")
        return np.full(len(returns), np.clip(returns.std() / 100, 1e-6, 1.0)), np.clip(returns.mean() / 100, -0.1, 0.1)

def rdog_geodesic(metric, initial_point, end_point, n_steps=100, max_iter=100, epsilon=1e-6, zeta_kappa=1.0):
    t = np.linspace(0, 1, n_steps)
    path = []
    x = np.array(initial_point, dtype=np.float64)
    x0 = np.array(end_point, dtype=np.float64)
    v = (x0 - x) / 1.0  # Initial velocity
    for _ in range(max_iter):
        g = metric.log(x0, x)
        if not np.isfinite(g).all():
            st.warning("Non-finite logarithm in RDoG. Adjusting position.")
            x = (x + x0) / 2
            g = metric.log(x0, x)
        rt = np.sqrt(np.sum((x - x0)**2))
        rt_bar = max(epsilon, rt)
        eta_t = rt_bar / np.sqrt(zeta_kappa * max(metric.metric_matrix(x)[1, 1], 1e-6))
        x_new = metric.exp(x, -eta_t * g)
        if not np.isfinite(x_new).all():
            st.warning("Non-finite values in RDoG. Reducing step size.")
            eta_t *= 0.5
            x_new = metric.exp(x, -eta_t * g)
        x = x_new
        v = metric.parallel_transport(v, x, -eta_t * g)
        path.append(x.copy())
        if rt < epsilon:
            break
    path = np.array(path)
    if path.shape[0] < n_steps:
        path = np.vstack([path, np.tile(path[-1], (n_steps - path.shape[0], 1))])
    path = path[:n_steps]
    return pd.DataFrame({"Time": path[:, 0], "Price": path[:, 1], "Path": "Geodesic"})

def compute_single_path(p0, T, N, dt, drift, diffusion, rng_seed):
    np.random.seed(rng_seed)
    path = np.zeros((N, 2))
    path[0, :] = [0.0, p0]
    price_std = 5.0
    for j in range(N - 1):
        pos = path[j, :]
        dW = np.random.normal(0, np.sqrt(dt))
        stoch_term = np.array([0.0, diffusion[j] * dW])
        next_pos = pos + drift[j] * dt + stoch_term
        next_pos[1] = np.clip(next_pos[1], p0 - price_std, p0 + price_std)
        path[j + 1, :] = next_pos
    return path[:, 1]

def simulate_paths_manifold(metric, p0, T, N, n_paths, subsample_factor=0.2):
    N_coarse = max(int(N * subsample_factor), 100)
    dt = T / (N_coarse - 1)
    t_coarse = np.linspace(0, T, N_coarse)
    
    sigma = np.array([metric.sigma_interp(ti) for ti in t_coarse])
    mu = np.array([metric.mu_interp(ti) for ti in t_coarse])
    sigma = np.clip(sigma, 1e-6, 1.0)
    mu = np.clip(mu, -0.1, 0.1)
    det = 1.0 * sigma**2 - mu**2
    det = np.maximum(det, 1e-6)
    g_inv_22 = 1.0 / det
    drift = np.vstack([np.ones(N_coarse), mu]).T
    diffusion = np.sqrt(g_inv_22)
    
    if np.any(~np.isfinite(sigma)) or np.any(~np.isfinite(mu)):
        st.warning("Invalid GARCH parameters detected. Using constant values.")
        sigma = np.full_like(sigma, np.clip(np.mean(sigma[np.isfinite(sigma)]), 1e-6, 1.0))
        mu = np.full_like(mu, np.clip(np.mean(mu[np.isfinite(mu)]), -0.1, 0.1))
        det = 1.0 * sigma**2 - mu**2
        det = np.maximum(det, 1e-6)
        g_inv_22 = 1.0 / det
        drift = np.vstack([np.ones(N_coarse), mu]).T
        diffusion = np.sqrt(g_inv_22)
    
    try:
        paths_coarse = np.array(Parallel(n_jobs=-1)(
            delayed(compute_single_path)(p0, T, N_coarse, dt, drift, diffusion, i)
            for i in range(n_paths)
        ))
    except Exception as e:
        st.warning(f"Parallel execution failed: {e}. Falling back to sequential.")
        paths_coarse = np.zeros((n_paths, N_coarse))
        for i in range(n_paths):
            paths_coarse[i, :] = compute_single_path(p0, T, N_coarse, dt, drift, diffusion, i)
    
    paths_coarse = np.where(np.isfinite(paths_coarse), paths_coarse, p0)
    
    t = np.linspace(0, T, N)
    paths = np.zeros((n_paths, N))
    for i in range(n_paths):
        interp = interp1d(t_coarse, paths_coarse[i], bounds_error=False, fill_value=p0)
        paths[i, :] = np.clip(interp(t), p0 - 5.0, p0 + 5.0)
    
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
    try:
        _, eigenvectors = eigs(laplacian, k=n_levels, which='SM')
        levels = []
        for ev in eigenvectors.T:
            ev_grid = ev.reshape(N_t, N_p)
            peak_idx = np.argmax(np.abs(ev_grid[-1, :]))
            levels.append(p_grid[peak_idx])
        return np.array(levels)
    except:
        st.warning("Laplacian eigenvalue computation failed. Using historical quantiles.")
        return np.quantile(p_grid, np.linspace(0.2, 0.8, n_levels))

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

# --- Main Analyzer Class ---
class MarketManifoldAnalyzer:
    def __init__(self, symbol, timeframe, start_date, end_date, n_paths, n_display_paths, epsilon_factor, garch_p, garch_q, subsample_factor):
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.n_paths = n_paths
        self.n_display_paths = n_display_paths
        self.epsilon_factor = epsilon_factor
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.subsample_factor = subsample_factor
        self.df = None
        self.prices = None
        self.times = None
        self.T = None
        self.N = None
        self.p0 = None
        self.metric = None
        self.paths = None
        self.t = None
        self.final_prices = None
        self.price_grid = None
        self.u = None
        self.support_levels = None
        self.resistance_levels = None
        self.support_probs = None
        self.resistance_probs = None
        self.geodesic_df = None
        self.regime = None

    def fetch_and_process_data(self):
        self.df = fetch_kraken_data(self.symbol, self.timeframe, self.start_date, self.end_date)
        if self.df is not None and len(self.df) > 50 and self.df['close'].std() > 1e-6:
            self.prices = np.log(self.df['close'].values)
            self.times = (self.df['datetime'] - self.df['datetime'].iloc[0]).dt.total_seconds() / 3600
            self.T = self.times.iloc[-1]
            self.N = len(self.prices)
            self.p0 = self.prices[0]
            return True
        return False

    def simulate_and_analyze(self):
        if not self.fetch_and_process_data():
            st.error("Insufficient or invalid data for analysis.")
            return False
        
        with st.spinner("Fitting GARCH model..."):
            returns = 100 * self.df['close'].pct_change().dropna()
            sigma, mu = fit_garch(returns, self.garch_p, self.garch_q)
            sigma = np.pad(sigma, (self.N - len(sigma), 0), mode='edge')
            mu = np.full(self.N, mu)

        self.metric = VolatilityMetric(sigma, mu, self.times, self.T)
        if np.any(self.metric.sigma_interp(self.times) < 1e-6):
            st.warning("GARCH volatility too low; clamping may affect geometry.")

        with st.spinner("Simulating price paths..."):
            self.paths, self.t = simulate_paths_manifold(self.metric, self.p0, self.T, self.N, self.n_paths, self.subsample_factor)
            self.paths = np.where(np.isfinite(self.paths), self.paths, self.p0)
            self.paths = np.exp(self.paths)

        self.final_prices = self.paths[:, -1]
        if np.any(~np.isfinite(self.final_prices)):
            st.warning("Final prices contain NaNs or Infs. Filtering invalid values.")
            valid_mask = np.isfinite(self.final_prices)
            self.final_prices = self.final_prices[valid_mask]
            if len(self.final_prices) < 10:
                st.error("Too few valid prices for KDE. Using historical quantiles for S/R.")
                self.final_prices = np.exp(self.prices)
        
        try:
            kde = gaussian_kde(self.final_prices)
            self.price_grid = np.linspace(self.final_prices.min(), self.final_prices.max(), 500)
            self.u = kde(self.price_grid)
            self.u /= np.trapz(self.u, self.price_grid)
        except:
            st.warning("KDE failed. Using uniform distribution.")
            self.u = np.ones_like(self.price_grid) / (self.price_grid.max() - self.price_grid.min())

        critical_levels = np.exp(compute_critical_levels(self.metric, self.t, np.linspace(self.prices.min(), self.prices.max(), 50)))
        self.support_levels = critical_levels[critical_levels <= np.median(critical_levels)]
        self.resistance_levels = critical_levels[critical_levels > np.median(critical_levels)]

        epsilon = self.epsilon_factor * np.mean([np.sqrt(self.metric.metric_matrix([self.T, np.log(p)])[1, 1]) 
                                                for p in self.price_grid if np.isfinite(np.log(p))])
        def get_hit_prob(level_list):
            probs = []
            for level in level_list:
                if np.isfinite(np.log(level)):
                    mask = (self.price_grid >= level - epsilon) & (self.price_grid <= level + epsilon)
                    raw_prob = np.trapz(self.u[mask], self.price_grid[mask])
                    volume_element = np.sqrt(np.abs(np.linalg.det(self.metric.metric_matrix([self.T, np.log(level)]))))
                    probs.append(raw_prob * volume_element)
                else:
                    probs.append(0.0)
            total_prob = sum(probs)
            return [p / total_prob for p in probs] if total_prob > 0 else [0] * len(probs)
        
        self.support_probs = get_hit_prob(self.support_levels)
        self.resistance_probs = get_hit_prob(self.resistance_levels)

        with st.spinner("Computing geodesic path with RDoG..."):
            self.geodesic_df = rdog_geodesic(self.metric, [0.0, self.p0], [self.T, self.prices[-1]], n_steps=self.N)

        self.regime = classify_regime(self.geodesic_df, np.exp(self.prices), self.times)
        return True

    def visualize(self, n_display_paths):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"Market Regime: {self.regime}")
            path_data = [{"Time": self.t[j], "Price": self.paths[i, j], "Path": f"Path_{i}"} 
                         for i in range(min(self.n_paths, n_display_paths)) for j in range(self.N) 
                         if np.isfinite(self.paths[i, j])]
            plot_df = pd.concat([pd.DataFrame(path_data), self.geodesic_df])
            support_df = pd.DataFrame({"Price": self.support_levels})
            resistance_df = pd.DataFrame({"Price": self.resistance_levels})
            base = alt.Chart(plot_df).encode(x=alt.X("Time:Q", title="Time (hours)"), 
                                            y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False)))
            main_chart = (base.mark_line(opacity=0.2).encode(color=alt.value('gray'), detail='Path:N').transform_filter(alt.datum.Path != "Geodesic") +
                          base.mark_line(strokeWidth=3, color="red").transform_filter(alt.datum.Path == "Geodesic") +
                          alt.Chart(support_df).mark_rule(stroke="green", strokeWidth=1.5).encode(y="Price:Q") +
                          alt.Chart(resistance_df).mark_rule(stroke="red", strokeWidth=1.5).encode(y="Price:Q")
                         ).properties(title="Price Paths, Geodesic, and S/R Grid", height=500).interactive()
            st.altair_chart(main_chart, use_container_width=True)

        with col2:
            viz_p_grid = np.linspace(np.exp(self.prices).min(), np.exp(self.prices).max(), 50)
            manifold_heatmap = visualize_manifold(self.metric, self.t, viz_p_grid)
            history_df = pd.DataFrame({'Time': self.times, 'Price': np.exp(self.prices)})
            history_line = alt.Chart(history_df).mark_line(color='white', strokeWidth=2.5).encode(x='Time:Q', y='Price:Q')
            st.altair_chart((manifold_heatmap + history_line).properties(height=300).interactive(), use_container_width=True)

            st.subheader("Analysis Summary")
            st.metric("Expected Final Price", f"${np.mean(self.final_prices[np.isfinite(self.final_prices)]):,.2f}")
            st.write("**Support Levels:**")
            if self.support_levels.size > 0:
                st.dataframe(pd.DataFrame({'Level': self.support_levels, 'Hit Probability': self.support_probs})
                             .style.format({'Level': '${:,.2f}', 'Hit Probability': '{:.1%}'}))
            else:
                st.write("No distinct support levels.")
            st.write("**Resistance Levels:**")
            if self.resistance_levels.size > 0:
                st.dataframe(pd.DataFrame({'Level': self.resistance_levels, 'Hit Probability': self.resistance_probs})
                             .style.format({'Level': '${:,.2f}', 'Hit Probability': '{:.1%}'}))
            else:
                st.write("No distinct resistance levels.")

            current_price = np.exp(self.prices[-1])
            signals = []
            for level, prob in zip(self.support_levels, self.support_probs):
                if abs(current_price - level) < self.epsilon_factor and prob > 0.3:
                    signals.append(f"Buy at ${level:.2f} (Support, {prob:.1%} probability)")
            for level, prob in zip(self.resistance_levels, self.resistance_probs):
                if abs(current_price - level) < self.epsilon_factor and prob > 0.3:
                    signals.append(f"Sell at ${level:.2f} (Resistance, {prob:.1%} probability)")
            if signals:
                st.write("**Trading Signals:**")
                for signal in signals:
                    st.write(signal)

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

analyzer = MarketManifoldAnalyzer(
    symbol='BTC/USD',
    timeframe='1h',
    start_date=start_date,
    end_date=end_date,
    n_paths=n_paths,
    n_display_paths=n_display_paths,
    epsilon_factor=epsilon_factor,
    garch_p=garch_p,
    garch_q=garch_q,
    subsample_factor=subsample_factor
)

if analyzer.simulate_and_analyze():
    analyzer.visualize(n_display_paths)
