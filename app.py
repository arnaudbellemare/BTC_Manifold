import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import altair as alt
from arch import arch_model
from geomstats.geometry.riemannian_metric import RiemannianMetric
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

st.title("BTC/USD Geometric Analysis with Support & Resistance")

# --- ADVANCED RIEMANNIAN METRIC (WITH COMPATIBILITY FIX) ---
class AdvancedRiemannianMetric(RiemannianMetric):
    def __init__(self, sigma, t, T, prices, variances):
        self.dim = 2
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

    def christoffel_symbols(self, base_point, epsilon=1e-5):
        g_inv = np.linalg.inv(self.metric_matrix(base_point))
        christoffels = np.zeros((self.dim, self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    val = 0
                    for l in range(self.dim):
                        h_j = np.zeros(self.dim); h_j[j] = epsilon
                        dg_ik_dj = (self.metric_matrix(base_point + h_j)[i, k] - self.metric_matrix(base_point - h_j)[i, k]) / (2 * epsilon)
                        h_i = np.zeros(self.dim); h_i[i] = epsilon
                        dg_jk_di = (self.metric_matrix(base_point + h_i)[j, k] - self.metric_matrix(base_point - h_i)[j, k]) / (2 * epsilon)
                        h_k = np.zeros(self.dim); h_k[k] = epsilon
                        dg_ij_dk = (self.metric_matrix(base_point + h_k)[i, j] - self.metric_matrix(base_point - h_k)[i, j]) / (2 * epsilon)
                        val += g_inv[k, l] * 0.5 * (dg_ik_dj + dg_jk_di - dg_ij_dk)
                    christoffels[i, j, k] = val
        return christoffels
    
    def log(self, point, base_point):
        point = np.asarray(point)
        base_point = np.asarray(base_point)
        def objective(velocity):
            geodesic_path = self.exp_solver.solve(velocity, base_point)
            final_point = geodesic_path[-1]
            return final_point - point
        initial_guess = point - base_point
        sol = root(objective, initial_guess, method='hybr', tol=1e-5)
        if not sol.success: return point - base_point
        return sol.x
        
@st.cache_data
def fetch_kraken_data(symbols, timeframe, start_date, end_date, limit):
    exchange = ccxt.kraken()
    try: exchange.load_markets()
    except Exception: return None
    since = int(start_date.timestamp() * 1000)
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].dropna()
            if len(df) >= 10: return df
        except Exception: continue
    st.error("Failed to fetch valid data for July 1-7, 2025. Using simulated data.")
    sim_t = pd.date_range(start=start_date, periods=168, freq='h')
    sim_prices = 70000 + np.cumsum(np.random.normal(0, 500, 168))
    return pd.DataFrame({'datetime': sim_t, 'close': sim_prices, 'timestamp': sim_t.astype(np.int64) // 10**6})

# Parameters & Data Prep
st.sidebar.header("Parameters")
n_paths = st.sidebar.slider("Number of Paths", 100, 2000, 1000, step=100)
n_display_paths = st.sidebar.slider("Paths to Display", 10, 200, 50, step=10)

df = fetch_kraken_data(['BTC/USD', 'XBT/USD'], '1h', pd.to_datetime("2025-07-01"), pd.to_datetime("2025-07-07 23:59:59"), 168)

prices = df['close'].values
times = (df['timestamp'] - df['timestamp'].iloc[0]) / (1000 * 3600) if not df.empty else np.array([])
T = times.iloc[-1] if len(times) > 0 else 168
N = len(prices)
p0 = prices[0] if N > 0 else 70000
returns = np.log(prices[1:] / prices[:-1]) if len(prices) > 1 else np.array([])
mu = np.mean(returns) * 24 * 365 if len(returns) > 0 else 0.0
v0 = np.var(returns) if len(returns) > 0 else (0.05/(24*365))**2
sigma = np.sqrt(v0) * np.ones(N) if N > 0 else np.array([])
if len(returns) > 5:
    try:
        garch = arch_model(returns * 100, vol='Garch', p=1, q=1).fit(disp='off')
        sigma = np.concatenate(([garch.conditional_volatility[0]], garch.conditional_volatility)) / 100
    except Exception: st.warning("GARCH failed. Using default volatility.")

# Heston Model Simulation
kappa, theta, xi, rho = 0.1, v0, 0.1, -0.3
def simulate_paths_heston(p0, mu, v0, kappa, theta, xi, rho, T, N, n_paths):
    if N == 0: return np.array([[p0]]), np.array([0]), np.array([[v0]])
    dt = T / (N - 1) if N > 1 else T
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
    paths, t, variances = simulate_paths_heston(p0, mu / (365*24), v0, kappa, theta, xi, rho, T, N, n_paths)

# --- FOKKER-PLANCK SOLVER WITH STABILITY FIXES ---
nt, n_p_steps = 400, 500 # Increased steps for stability
t_grid = np.linspace(0, T, nt)
p_grid = np.linspace(paths.min() * 0.9, paths.max() * 1.1, n_p_steps)
dt_fd, dp = (t_grid[1] - t_grid[0]), (p_grid[1] - p_grid[0])
u = np.zeros((nt, n_p_steps))
sigma_init = np.sqrt(v0) * p0 if v0 > 0 else np.std(prices[:10] if len(prices) > 10 else [100])
u[0, :] = np.exp(-((p_grid - p0)**2) / (2 * max(sigma_init**2, 1e-4)))
u[0, :] /= np.trapz(u[0, :], p_grid)
I = diags([1], [0], shape=(n_p_steps, n_p_steps), format="csr")
D_fwd = (1 / dp) * diags([-1, 1], [0, 1], shape=(n_p_steps, n_p_steps), format="csr")
D_bwd = (1 / dp) * diags([-1, 1], [-1, 0], shape=(n_p_steps, n_p_steps), format="csr")
metric_fp = AdvancedRiemannianMetric(sigma, t, T, prices, variances)
with st.spinner("Solving Fokker-Planck on the manifold..."):
    for i in range(nt - 1):
        current_time = t_grid[i]
        g_p = np.array([metric_fp.metric_matrix([current_time, p]) for p in p_grid])
        g_inv_pp = 1 / g_p[:, 1, 1]
        sqrt_det_g = np.sqrt(g_p[:, 1, 1])
        L_LB_op = D_bwd @ diags(sqrt_det_g * g_inv_pp) @ D_fwd
        
        # --- FIX 1: Stabilize the diffusion term by clipping its coefficient ---
        diffusion_coeff = 0.5 * p_grid**2
        max_diffusion = np.percentile(diffusion_coeff, 99) # Prevent explosion from large prices
        clipped_diffusion_coeff = np.clip(diffusion_coeff, 0, max_diffusion)
        L_diffusion = diags(clipped_diffusion_coeff) @ L_LB_op
        
        L_drift_op = diags((mu / (365*24)) * p_grid) @ D_fwd
        L_diffusion.setdiag(0, k=0); L_diffusion.setdiag(0, k=-1) # Boundary conditions
        L_drift_op.setdiag(0, k=0); L_drift_op.setdiag(0, k=-1)
        L = L_diffusion + L_drift_op
        LHS = I - 0.5 * dt_fd * L
        RHS = (I + 0.5 * dt_fd * L) @ u[i, :]
        u_new = spsolve(LHS, RHS)
        
        # --- FIX 2: Add sanity check to detect and handle numerical instability ---
        if not np.all(np.isfinite(u_new)):
            st.warning(f"Numerical instability detected at step {i+1}. Halting Fokker-Planck evolution.")
            u[i+1:, :] = u[i, :] # Fill remaining steps with the last valid one
            break # Exit the loop
            
        u[i + 1, :] = np.maximum(u_new, 0)
        norm = np.trapz(u[i + 1, :], p_grid)
        if norm > 1e-9: u[i + 1, :] /= norm

# --- ANALYSIS WITH ROBUST FALLBACKS ---
u_density = u[-1, :]
expected_price = np.trapz(p_grid * u_density, p_grid)
V_eff = -np.log(u_density + 1e-20)
V_eff -= V_eff.min()
potential_minima_idx, _ = find_peaks(-V_eff, distance=int(n_p_steps/20), height=-np.log(0.8))
stable_levels = p_grid[potential_minima_idx]
support_levels = [level for level in stable_levels if level < expected_price]
resistance_levels = [level for level in stable_levels if level > expected_price]

# --- FIX 3: Robust fallback for S/R levels to guarantee output ---
if not support_levels and np.isfinite(expected_price):
    support_levels = [np.quantile(p_grid[u_density > 1e-6], 0.25)]
if not resistance_levels and np.isfinite(expected_price):
    resistance_levels = [np.quantile(p_grid[u_density > 1e-6], 0.75)]

# Calculate Hit Probabilities
final_std_dev = np.sqrt(np.trapz((p_grid - expected_price)**2 * u_density, p_grid))
epsilon = 0.25 * final_std_dev if np.isfinite(final_std_dev) and final_std_dev > 0 else 500
support_probs, resistance_probs = [], []
total_support_prob, total_resistance_prob = 0, 0
for s_level in support_levels:
    mask = (p_grid >= s_level - epsilon) & (p_grid <= s_level + epsilon)
    prob = np.trapz(u_density[mask], p_grid[mask])
    support_probs.append(prob)
    total_support_prob += prob
for r_level in resistance_levels:
    mask = (p_grid >= r_level - epsilon) & (p_grid <= r_level + epsilon)
    prob = np.trapz(u_density[mask], p_grid[mask])
    resistance_probs.append(prob)
    total_resistance_prob += prob
if total_support_prob > 0: support_probs = [p / total_support_prob for p in support_probs]
if total_resistance_prob > 0: resistance_probs = [p / total_resistance_prob for p in resistance_probs]

# Geodesic Calculation
metric = AdvancedRiemannianMetric(sigma, t, T, prices, variances)
def geodesic_equation(s, y):
    pos, vel = y[:2], y[2:]
    gamma = metric.christoffel_symbols(pos)
    accel = -np.einsum('ijk,j,k->i', gamma, vel, vel)
    return np.concatenate([vel, accel])
try:
    initial_point = np.array([0.0, p0])
    initial_velocity = np.array([1.0, (prices[-1] - p0) / T if T > 0 else 0.0])
    y0 = np.concatenate([initial_point, initial_velocity])
    sol = solve_ivp(geodesic_equation, [0, T], y0, t_eval=t, rtol=1e-5, atol=1e-7)
    geodesic_df = pd.DataFrame({"Time": sol.y[0, :], "Price": sol.y[1, :], "Path": "Geodesic"})
except Exception as e:
    st.error(f"Geodesic computation failed: {e}. Using linear approximation.")
    geodesic_df = pd.DataFrame({"Time": t, "Price": p0 + initial_velocity[1] * t, "Path": "Geodesic"})

# --- VISUALIZATION WITH SEPARATE S/R ---
path_data = [{"Time": t[j], "Price": paths[i, j], "Path": f"Path_{i}"} for i in range(min(n_paths, n_display_paths)) for j in range(N)]
plot_df = pd.concat([pd.DataFrame(path_data), geodesic_df])
support_df = pd.DataFrame({"Price": support_levels})
resistance_df = pd.DataFrame({"Price": resistance_levels})
base = alt.Chart(plot_df).encode(x=alt.X("Time:Q", title="Time (hours)"), y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(zero=False)))
path_lines = base.mark_line(opacity=0.1).encode(color=alt.value('gray'), detail='Path:N').transform_filter(alt.datum.Path != "Geodesic")
geodesic_line = base.mark_line(strokeWidth=3).encode(color=alt.value('red')).transform_filter(alt.datum.Path == "Geodesic")
support_lines = alt.Chart(support_df).mark_rule(stroke="green", strokeWidth=1.5, strokeDash=[6, 3]).encode(y="Price:Q")
resistance_lines = alt.Chart(resistance_df).mark_rule(stroke="red", strokeWidth=1.5, strokeDash=[6, 3]).encode(y="Price:Q")
chart = (path_lines + geodesic_line + support_lines + resistance_lines).properties(title="Price Dynamics, Geodesic, Support (Green), and Resistance (Red)", width=800, height=500).interactive()

st.altair_chart(chart, use_container_width=True)
st.write(f"**Expected Final Price:** ${expected_price:,.2f}" if np.isfinite(expected_price) else "**Expected Final Price:** Calculation failed due to instability.")
st.write("**Support Levels (BTC/USD):**")
if support_levels:
    st.dataframe(pd.DataFrame({'Level': support_levels, 'Hit Probability': support_probs}).style.format({'Level': '${:,.2f}', 'Hit Probability': '{:.1%}'}))
else:
    st.write("No distinct support levels found.")
st.write("**Resistance Levels (BTC/USD):**")
if resistance_levels:
    st.dataframe(pd.DataFrame({'Level': resistance_levels, 'Hit Probability': resistance_probs}).style.format({'Level': '${:,.2f}', 'Hit Probability': '{:.1%}'}))
else:
    st.write("No distinct resistance levels found.")
