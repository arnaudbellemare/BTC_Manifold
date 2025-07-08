import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import altair as alt
from arch import arch_model
from geomstats.geometry.riemannian_metric import RiemannianMetric
import geomstats.geometry.euclidean
from geomstats.information_geometry.normal import BrownianMotion
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy.sparse.linalg import eigs
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")
st.title("BTC/USD Price Analysis on a Volatility-Weighted Manifold")

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

@st.cache_data
def fit_garch(returns, p=1, q=1):
    model = arch_model(returns, vol='Garch', p=p, q=q).fit(disp='off')
    return model.conditional_volatility / 100, model.params.get('mu', 0.0)

def compute_geodesic(metric, t, p0, pT, T):
    manifold = CustomManifold(dim=2, metric=metric)
    geodesic = metric.geodesic(initial_point=np.array([0.0, p0]), end_point=np.array([T, pT]))
    points = geodesic(np.linspace(0, 1, len(t)))
    return pd.DataFrame({"Time": points[:, 0], "Price": points[:, 1], "Path": "Geodesic"})

def simulate_paths_manifold(metric, p0, T, N, n_paths):
    manifold = CustomManifold(dim=2, metric=metric)
    bm = BrownianMotion(manifold)
    paths = []
    for _ in range(n_paths):
        path = bm(np.array([0.0, p0]), T, n_points=N)
        paths.append(path[:, 1])
    return np.array(paths), np.linspace(0, T, N)

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
    if np.mean(deviation[-len(deviation)//4:]) > np.std(prices):
        return "Trending"
    return "Mean Reverting"

# Main logic (abridged)
st.sidebar.header("Model Parameters")
days_history = st.sidebar.slider("Historical Data (Days)", 7, 90, 30)
n_paths = st.sidebar.slider("Simulated Paths", 500, 10000, 2000, step=100)
n_display_paths = st.sidebar.slider("Displayed Paths", 10, 200, 50, step=10)
epsilon_factor = st.sidebar.slider("Probability Range Factor", 0.1, 1.0, 0.25, step=0.05)
garch_p = st.sidebar.slider("GARCH p", 1, 3, 1)
garch_q = st.sidebar.slider("GARCH q", 1, 3, 1)

end_date = pd.Timestamp.now(tz='UTC')
start_date = end_date - pd.Timedelta(days=days_history)
df = fetch_kraken_data('BTC/USD', '1h', start_date, end_date)

if df is not None and len(df) > 50 and df['close'].std() > 1e-6:
    prices = df['close'].values
    times = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds() / 3600
    T = times.iloc[-1]
    N = len(prices)
    p0 = prices[0]
    returns = 100 * df['close'].pct_change().dropna()
    
    sigma, mu = fit_garch(returns, garch_p, garch_q)
    sigma = np.pad(sigma, (N - len(sigma), 0), mode='edge')
    mu = np.full(N, mu)

    metric = VolatilityMetric(sigma, mu, times, T)
    paths, t = simulate_paths_manifold(metric, p0, T, N, n_paths)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        final_prices = paths[:, -1]
        kde = gaussian_kde(final_prices)
        price_grid = np.linspace(final_prices.min(), final_prices.max(), 500)
        u = kde(price_grid)
        u /= np.trapz(u, price_grid)
        
        critical_levels = compute_critical_levels(metric, t, price_grid)
        support_levels = critical_levels[critical_levels <= np.median(critical_levels)]
        resistance_levels = critical_levels[critical_levels > np.median(critical_levels)]
        
        epsilon = epsilon_factor * np.mean([np.sqrt(metric.metric_matrix([T, p])[1, 1]) for p in price_grid])
        support_probs = get_hit_prob(support_levels)
        resistance_probs = get_hit_prob(resistance_levels)
        
        geodesic_df = compute_geodesic(metric, t, p0, prices[-1], T)
        regime = classify_regime(geodesic_df, prices, times)
        st.write(f"Market Regime: {regime}")
        
        # Plotting and signals (use original plotting code, updated S/R levels)
        path_data = [{"Time": t[j], "Price": paths[i, j], "Path": f"Path_{i}"} for i in range(min(n_paths, n_display_paths)) for j in range(N)]
        plot_df = pd.concat([pd.DataFrame(path_data), geodesic_df])
        support_df = pd.DataFrame({"Price": support_levels})
        resistance_df = pd.DataFrame({"Price": resistance_levels})
        base = alt.Chart(plot_df).encode(x=alt.X("Time:Q"), y=alt.Y("Price:Q", scale=alt.Scale(zero=False)))
        main_chart = (base.mark_line(opacity=0.2).encode(color=alt.value('gray'), detail='Path:N').transform_filter(alt.datum.Path != "Geodesic") +
                      base.mark_line(strokeWidth=3, color="red").transform_filter(alt.datum.Path == "Geodesic") +
                      alt.Chart(support_df).mark_rule(stroke="green", strokeWidth=1.5).encode(y="Price:Q") +
                      alt.Chart(resistance_df).mark_rule(stroke="red", strokeWidth=1.5).encode(y="Price:Q")
                     ).properties(title="Price Paths, Geodesic, and S/R Grid", height=500).interactive()
        st.altair_chart(main_chart, use_container_width=True)
    
    with col2:
        SCALING_FACTOR = 1 / np.max(sigma**2) * 1000
        manifold_heatmap = visualize_manifold(metric, t, price_grid)  # Use original visualize_manifold
        history_line = alt.Chart(pd.DataFrame({'Time': times, 'Price': prices})).mark_line(color='white', strokeWidth=2.5).encode(x='Time:Q', y='Price:Q')
        st.altair_chart((manifold_heatmap + history_line).properties(height=300).interactive(), use_container_width=True)
        
        st.subheader("Analysis Summary")
        st.metric("Expected Final Price", f"${np.mean(final_prices):,.2f}")
        st.write("**Support Levels:**")
        if support_levels.size > 0:
            st.dataframe(pd.DataFrame({'Level': support_levels, 'Hit Probability': support_probs}).style.format({'Level': '${:,.2f}', 'Hit Probability': '{:.1%}'}))
        st.write("**Resistance Levels:**")
        if resistance_levels.size > 0:
            st.dataframe(pd.DataFrame({'Level': resistance_levels, 'Hit Probability': resistance_probs}).style.format({'Level': '${:,.2f}', 'Hit Probability': '{:.1%}'}))
        
        current_price = prices[-1]
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
