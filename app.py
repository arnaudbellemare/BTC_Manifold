import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import altair as alt
from arch import arch_model
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.euclidean import Euclidean
from geomstats.numerics.geodesic import ExpODESolver
from geomstats.numerics.ivp import ScipySolveIVP
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.signal import find_peaks
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import time
import warnings
warnings.filterwarnings("ignore")

st.title("BTC/USD Rigorous Riemannian Manifold Analysis (July 1-7, 2025)")

# Advanced Riemannian metric with full implementations
class AdvancedRiemannianMetric(RiemannianMetric):
    def __init__(self, sigma, t, T, prices, variances):
        super().__init__(space=Euclidean(dim=2), default_point_type="vector")
        self.sigma = sigma
        self.t = t
        self.T = T
        self.prices = prices
        self.variances = variances
        self.exp_solver = ExpODESolver(self)

    def metric_matrix(self, base_point):
        t_val, p_val = base_point
        idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        sigma_val = max(self.sigma[idx], 1e-3)
        var_val = np.interp(t_val, self.t, self.variances.mean(axis=0))
        price_dist, bin_edges = np.histogram(self.prices, bins=50, density=True)
        p_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fisher_info = 1.0 / (np.interp(p_val, p_centers, price_dist) + 1e-6)
        return np.array([[1.0, 0.0], [0.0, (sigma_val**2 * var_val + fisher_info)]])

    def christoffel_symbols(self, base_point, epsilon=1e-6):
        g_inv = np.linalg.inv(self.metric_matrix(base_point))
        christoffels = np.zeros((2, 2, 2))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    partial_derivatives = np.zeros(2)
                    for l in range(2):
                        h = np.zeros(2)
                        h[l] = epsilon
                        g_plus = self.metric_matrix(base_point + h)
                        g_minus = self.metric_matrix(base_point - h)
                        
                        term1_deriv_vec = np.zeros(2); term1_deriv_vec[j] = epsilon
                        term2_deriv_vec = np.zeros(2); term2_deriv_vec[i] = epsilon
                        
                        dg_ik_dj = (self.metric_matrix(base_point + term1_deriv_vec)[i,k] - self.metric_matrix(base_point - term1_deriv_vec)[i,k]) / (2*epsilon)
                        dg_jk_di = (self.metric_matrix(base_point + term2_deriv_vec)[j,k] - self.metric_matrix(base_point - term2_deriv_vec)[j,k]) / (2*epsilon)
                        
                        term3_deriv_vec = np.zeros(2); term3_deriv_vec[k] = epsilon
                        dg_ij_dk = (self.metric_matrix(base_point + term3_deriv_vec)[i,j] - self.metric_matrix(base_point - term3_deriv_vec)[i,j]) / (2*epsilon)
                        
                        christoffels[i, j, k] += 0.5 * g_inv[k, l] * (dg_ik_dj + dg_jk_di - dg_ij_dk)
        return christoffels

    def ricci_curvature(self, base_point, epsilon=1e-6):
        christoffels = self.christoffel_symbols(base_point)
        ricci = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    h = np.zeros(2); h[k] = epsilon
                    gamma_plus = self.christoffel_symbols(base_point + h)
                    gamma_minus = self.christoffel_symbols(base_point - h)
                    d_gamma_k = (gamma_plus - gamma_minus) / (2 * epsilon)
                    
                    term1 = d_gamma_k[i, j, k]
                    term2 = d_gamma_k[i, k, j]
                    
                    term3 = 0
                    term4 = 0
                    for l in range(2):
                        term3 += christoffels[i, j, l] * christoffels[l, k, k]
                        term4 += christoffels[i, k, l] * christoffels[l, j, k]
                    
                    ricci[i, j] += term1 - term2 + term3 - term4
        return ricci

    def log(self, point, base_point):
        point = np.asarray(point)
        base_point = np.asarray(base_point)
        
        def objective(velocity):
            geodesic = self.exp_solver.solve(velocity, base_point)
            final_point = geodesic[-1]
            return final_point - point
        
        initial_guess = point - base_point
        sol = root(objective, initial_guess, method='hybr', tol=1e-5)
        
        if not sol.success:
            # Fallback for failed convergence
            warnings.warn("Logarithm map solver did not converge. Returning linear approximation.")
            return point - base_point
            
        return sol.x
        
@st.cache_data
def fetch_kraken_data(symbols, timeframe, start_date, end_date, limit):
    exchange = ccxt.kraken()
    try:
        exchange.load_markets()
    except Exception as e:
        st.warning(f"Failed to load Kraken markets: {e}")
        return None

    since = int(start_date.timestamp() * 1000)
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].dropna()
            if len(df) >= 10:
                st.write(f"Success: Fetched {len(df)} data points for {symbol}")
                return df
        except Exception as e:
            st.warning(f"Error fetching {symbol}: {e}")
    
    st.error("Failed to fetch valid data. Using simulated data for demonstration.")
    sim_t = pd.date_range(start=start_date, periods=168, freq='h')
    sim_prices = 70000 + np.cumsum(np.random.normal(0, 500, 168))
    return pd.DataFrame({'datetime': sim_t, 'close': sim_prices, 'timestamp': sim_t.astype(np.int64) // 10**6})

# Parameters
st.sidebar.header("Parameters")
n_paths = st.sidebar.slider("Number of Simulated Paths", 100, 2000, 1000, step=100)
n_display_paths = st.sidebar.slider("Number of Paths to Display", 10, 200, 50, step=10)

# Fetch and Prepare Data
df = fetch_kraken_data(['BTC/USD'], '1h', pd.to_datetime("2025-07-01"), pd.to_datetime("2025-07-07 23:59:59"), 168)
prices = df['close'].values
times = (df['timestamp'] - df['timestamp'].iloc[0]) / (1000 * 3600)
T = times.iloc[-1]
N = len(prices)
p0 = prices[0]
returns = np.log(prices[1:] / prices[:-1])
mu = np.mean(returns) * 24 * 365 # Annualized log-return drift

# Volatility Modeling
v0 = np.var(returns) if len(returns) > 0 else 0.01**2
kappa, theta, xi, rho = 0.1, v0, 0.1, -0.3
sigma = np.sqrt(v0) * np.ones(N)
try:
    garch_model = arch_model(returns * 100, vol='Garch', p=1, q=1).fit(disp='off')
    sigma = np.concatenate(([garch_model.conditional_volatility[0]], garch_model.conditional_volatility)) / 100
except Exception:
    st.warning("GARCH failed. Using default volatility.")

# Heston Simulation
def simulate_paths(p0, mu, v0, kappa, theta, xi, rho, T, N, n_paths):
    dt = T / (N - 1)
    t = np.linspace(0, T, N)
    paths = np.zeros((n_paths, N)); paths[:, 0] = p0
    variances = np.zeros((n_paths, N)); variances[:, 0] = v0
    for j in range(1, N):
        z1 = np.random.normal(size=n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(size=n_paths)
        variances[:, j] = np.maximum(variances[:, j-1] + kappa * (theta - variances[:, j-1]) * dt + xi * np.sqrt(variances[:, j-1]) * np.sqrt(dt) * z2, 1e-6)
        paths[:, j] = paths[:, j-1] * np.exp((mu - 0.5 * variances[:, j-1]) * dt + np.sqrt(variances[:, j-1]) * np.sqrt(dt) * z1)
    return paths, t, variances

with st.spinner("Simulating Heston paths..."):
    paths, t, variances = simulate_paths(p0, mu / (365*24), v0, kappa, theta, xi, rho, T, N, n_paths)

# Fokker-Planck with Laplace-Beltrami Operator
nt, n_p_steps = 200, 400
t_grid = np.linspace(0, T, nt)
p_grid = np.linspace(paths.min() * 0.9, paths.max() * 1.1, n_p_steps)
dt_fd, dp = t_grid[1] - t_grid[0], p_grid[1] - p_grid[0]

u = np.zeros((nt, n_p_steps))
sigma_init = np.sqrt(v0) * p0 if v0 > 0 else np.std(prices[:10])
u[0, :] = np.exp(-((p_grid - p0)**2) / (2 * sigma_init**2))
u[0, :] /= np.trapz(u[0, :], p_grid)

I = diags([1], [0], shape=(n_p_steps, n_p_steps), format="csr")
D_fwd = (1 / dp) * diags([-1, 1], [0, 1], shape=(n_p_steps, n_p_steps), format="csr")
D_bwd = (1 / dp) * diags([-1, 1], [-1, 0], shape=(n_p_steps, n_p_steps), format="csr")

metric_fp = AdvancedRiemannianMetric(sigma, t, T, prices, variances)

for i in range(nt - 1):
    current_time = t_grid[i]
    g_p = np.array([metric_fp.metric_matrix([current_time, p]) for p in p_grid])
    g_inv_pp = 1 / g_p[:, 1, 1]
    sqrt_det_g = np.sqrt(g_p[:, 1, 1])
    
    # Laplace-Beltrami Operator: Δf = (1/√g) ∂p (√g g^{pp} ∂p f)
    L_LB = D_bwd @ diags(sqrt_det_g * g_inv_pp) @ D_fwd
    
    drift_coeff = (mu / (365*24)) * p_grid
    L_drift = diags(drift_coeff) @ D_fwd # Using forward difference for drift
    
    L = 0.5 * diags(p_grid**2) @ L_LB + L_drift
    
    LHS = I - 0.5 * dt_fd * L
    RHS = (I + 0.5 * dt_fd * L) @ u[i, :]
    
    u[i + 1, :] = spsolve(LHS, RHS)
    u[i + 1, :] = np.maximum(u[i + 1, :], 0)
    norm = np.trapz(u[i + 1, :], p_grid)
    if norm > 1e-9: u[i + 1, :] /= norm

u_density = u[-1, :]

# S/R levels from Effective Potential
V = -np.log(u_density + 1e-20)
V -= V.min() # Normalize potential
potential_minima_idx, _ = find_peaks(-V, distance=int(n_p_steps/20), height=-np.log(0.8))
stable_levels = p_grid[potential_minima_idx]

# Geodesic Calculation
metric = AdvancedRiemannianMetric(sigma, t, T, prices, variances)
def geodesic_equation(s, y):
    pos = y[:2]; vel = y[2:]
    gamma = metric.christoffel_symbols(pos)
    accel = -np.einsum('ijk,j,k->i', gamma, vel, vel)
    return np.concatenate([vel, accel])

try:
    initial_point = np.array([0.0, p0])
    initial_velocity = np.array([1.0, (prices[-1] - p0) / T])
    y0 = np.concatenate([initial_point, initial_velocity])
    sol = solve_ivp(geodesic_equation, [0, T], y0, t_eval=t, rtol=1e-5, atol=1e-7)
    geodesic_df = pd.DataFrame({"Time": sol.y[0, :], "Price": sol.y[1, :], "Path": "Geodesic"})
except Exception as e:
    st.error(f"Geodesic computation failed: {e}. Using linear approximation.")
    geodesic_df = pd.DataFrame({"Time": t, "Price": p0 + ((prices[-1] - p0) / T) * t, "Path": "Geodesic"})

# Trending/Reversion Analysis using true geodesic distance
distances = [metric.dist(np.array([t[i], paths[j,i]]), np.array([geodesic_df["Time"].iloc[i], geodesic_df["Price"].iloc[i]])) for j in range(n_paths) for i in range(N-1, N)]
distances = np.array(distances)
trending_threshold = np.percentile(distances, 25)
reversion_threshold = np.percentile(distances, 75)
trending_paths = np.sum(distances < trending_threshold)
reversion_paths = np.sum(distances > reversion_threshold)

# Visualization
path_data = [{"Time": t[j], "Price": paths[i, j], "Path": f"Path_{i}"} for i in range(min(n_paths, n_display_paths)) for j in range(N)]
plot_df = pd.concat([pd.DataFrame(path_data), geodesic_df])
stable_levels_df = pd.DataFrame({"Price": stable_levels})

base = alt.Chart(plot_df).encode(x=alt.X("Time:Q", title="Time (hours)"), y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False)))
path_lines = base.mark_line(opacity=0.1).encode(color=alt.value('gray'), detail='Path:N').transform_filter(alt.datum.Path != "Geodesic")
geodesic_line = base.mark_line(strokeWidth=3).encode(color=alt.value('red')).transform_filter(alt.datum.Path == "Geodesic")
level_lines = alt.Chart(stable_levels_df).mark_rule(stroke="blue", strokeWidth=1.5, strokeDash=[6, 3]).encode(y="Price:Q")

chart = (path_lines + geodesic_line + level_lines).properties(
    title="BTC/USD Price Dynamics on a Riemannian Manifold", width=800, height=500
).interactive()

st.altair_chart(chart, use_container_width=True)
st.write(f"**Stable Price Levels (Attractors):** {[f'${x:,.2f}' for x in stable_levels]}")
st.write(f"**Trending Paths:** {trending_paths} ({trending_paths/n_paths:.1%}) (Paths near the geodesic)")
st.write(f"**Reversion Paths:** {reversion_paths} ({reversion_paths/n_paths:.1%}) (Paths deviating from the geodesic)")
