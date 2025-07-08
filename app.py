import streamlit as st
import ccxt
import numpy as np  # Explicitly import NumPy
import pandas as pd
import altair as alt
from arch import arch_model
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.euclidean import Euclidean
from geomstats.numerics.geodesic import ExpODESolver
from geomstats.numerics.ivp import ScipySolveIVP
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import time
import warnings
warnings.filterwarnings("ignore")

st.title("BTC/USD Advanced Riemannian Manifold Analysis (July 1-7, 2025)")

# Advanced Riemannian metric with volatility, correlation, and Fisher information
class AdvancedRiemannianMetric(RiemannianMetric):
    def __init__(self, sigma, t, T, prices, variances):
        super().__init__(space=Euclidean(dim=2))
        self.sigma = sigma
        self.t = t
        self.T = T
        self.prices = prices
        self.variances = variances
        self.exp_solver = ExpODESolver(space=Euclidean(dim=2), integrator=ScipySolveIVP())
        self.fisher_cache = {}

    def metric_matrix(self, base_point):
        t_val, p_val = base_point
        idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        sigma_val = max(self.sigma[idx], 0.01)
        var_val = np.interp(t_val, self.t, self.variances.mean(axis=0))
        # Fisher information from price distribution
        price_dist = np.histogram(self.prices, bins=50, density=True)[0]
        fisher_info = 1.0 / (np.interp(p_val, np.linspace(min(self.prices), max(self.prices), 50), price_dist) + 1e-6)
        return np.diag([1.0, (sigma_val**2 * var_val + fisher_info)])

    def christoffel_symbols(self, base_point):
        t_val, p_val = base_point
        idx = int(np.clip(t_val / self.T * (len(self.sigma) - 1), 0, len(self.sigma) - 1))
        sigma_val = max(self.sigma[idx], 0.01)
        d_sigma = np.gradient(self.sigma, self.t / self.T * (len(self.sigma) - 1))[idx] / (self.T / len(self.sigma)) if idx > 0 and idx < len(self.sigma) - 1 else 0.0
        g = self.metric_matrix(base_point)
        g_inv = np.linalg.inv(g)
        gamma = np.zeros((2, 2, 2))
        gamma[1, 1, 0] = 0.5 * g_inv[0, 0] * d_sigma * p_val / sigma_val
        gamma[1, 0, 1] = gamma[1, 1, 0]
        gamma[0, 1, 1] = gamma[1, 1, 0]
        return gamma

    def ricci_curvature(self, base_point):
        gamma = self.christoffel_symbols(base_point)
        g = self.metric_matrix(base_point)
        g_inv = np.linalg.inv(g)
        R = np.zeros((2, 2))
        R[1, 1] = np.sum([g_inv[i, j] * (gamma[i, 1, 1] - gamma[i, 1, 0]) for i in range(2) for j in range(2)])
        return R

# Fetch Kraken data
@st.cache_data
def fetch_kraken_data(symbols, timeframe, start_date, end_date, limit):
    exchange = ccxt.kraken()
    try:
        markets = exchange.load_markets()
        available_symbols = list(markets.keys())
        st.write(f"Available Kraken symbols: {available_symbols[:10]}... (total {len(available_symbols)})")
    except Exception as e:
        st.warning(f"Failed to load Kraken markets: {e}")
        return None

    since = int(start_date.timestamp() * 1000)
    until = int(end_date.timestamp() * 1000)
    st.write(f"Fetching data: from {start_date} to {end_date}, timeframe={timeframe}, limit={limit}")

    for symbol in symbols:
        if symbol not in available_symbols:
            st.warning(f"Symbol {symbol} not in Kraken markets")
            continue
        for attempt in range(3):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
                if len(df) >= 10 and df['timestamp'].notnull().all() and df['close'].notnull().all() and df['close'].gt(0).all():
                    st.write(f"Success: Fetched {len(df)} data points for {symbol}")
                    st.write(f"Sample: {df[['datetime', 'close']].head(3).to_dict('records')}")
                    return df
                else:
                    st.warning(f"Invalid data for {symbol}: len={len(df)}")
            except ccxt.NetworkError as e:
                st.warning(f"Network error for {symbol} (attempt {attempt+1}): {e}")  # Fixed syntax
                time.sleep(5)
            except Exception as e:
                st.warning(f"Error for {symbol} (attempt {attempt+1}): {e}")
    st.error("Failed to fetch valid data from Kraken. Using simulated data.")
    t = np.linspace(0, 168, 168)
    prices = np.array([df['close'].iloc[0]] * len(t))  # Start with actual initial price
    prices = prices + np.cumsum(np.random.normal(0, 1000, 168))  # Pure stochastic evolution
    df = pd.DataFrame({"timestamp": (t * 3600 * 1000).astype(int), "close": prices})
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Parameters
st.sidebar.header("Parameters")
n_paths = st.sidebar.slider("Number of Simulated Paths", 1000, 5000, 2000, step=500)
n_display_paths = st.sidebar.slider("Number of Paths to Display", 50, 200, 100, step=25)
epsilon = st.sidebar.slider("Probability Integration Range ($)", 1000, 10000, 5000, step=1000)

# Fetch data
symbols = ['BTC/USD']
timeframe = '1h'
limit = 168
start_date = pd.to_datetime("2025-07-01")
end_date = pd.to_datetime("2025-07-07 23:59:59")
df = fetch_kraken_data(symbols, timeframe, start_date, end_date, limit)

# Validate DataFrame
if df is None or df.empty or len(df) < 10:
    st.error(f"No valid data: df={'None' if df is None else f'len={len(df)}'}")
    st.stop()
if not df['timestamp'].notnull().all() or not df['close'].notnull().all() or not df['close'].gt(0).all():
    st.error(f"Invalid data: timestamps={df['timestamp'].notnull().all()}, close={df['close'].notnull().all()}, positive={df['close'].gt(0).all()}")
    st.stop()

# Prepare data
prices = df['close'].values
times = (df['timestamp'] - df['timestamp'].iloc[0]) / (1000 * 3600)
if len(times) < 2 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(prices)) or not np.all(prices > 0):
    st.error(f"Invalid data: times={len(times)}, prices={len(prices)}, finite={np.all(np.isfinite(times)) and np.all(np.isfinite(prices))}")
    st.stop()

# Heston model parameters
returns = np.diff(prices) / prices[:-1]
v0 = np.var(returns)  # Initial variance
kappa = 0.1  # Mean reversion rate
theta = v0  # Long-term variance
xi = 0.1  # Volatility of variance
rho = -0.3  # Correlation between price and variance
sigma = np.sqrt(v0 * np.ones(len(prices)))  # Initial volatility
if len(returns) > 5:
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1)
        res = model.fit(disp='off')
        sigma = res.conditional_volatility
        sigma = np.pad(sigma, (0, 1), mode='edge')
    except Exception as e:
        st.warning(f"GARCH failed: {e}. Using Heston parameters.")

p0 = prices[0]  # Use actual initial price from data
T = times.iloc[-1]
N = len(prices)
mu = np.mean(returns) * N / T if len(returns) > 0 else 0.0

# Simulate paths with Heston model
def simulate_paths(p0, mu, v0, kappa, theta, xi, rho, sigma, T, N, n_paths):
    dt = T / (N - 1)
    t = np.linspace(0, T, N)
    paths = np.zeros((n_paths, N))
    variances = np.zeros((n_paths, N))
    paths[:, 0] = p0
    variances[:, 0] = v0
    for j in range(1, N):
        z1 = np.random.normal(0, np.sqrt(dt), n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), n_paths)
        variances[:, j] = np.clip(variances[:, j-1] + kappa * (theta - variances[:, j-1]) * dt + xi * np.sqrt(variances[:, j-1]) * z2, 1e-6, np.inf)
        paths[:, j] = paths[:, j-1] + mu * paths[:, j-1] * dt + np.sqrt(variances[:, j-1]) * paths[:, j-1] * z1
    return paths, t, variances

with st.spinner("Simulating Heston paths..."):
    paths, t, variances = simulate_paths(p0, mu, v0, kappa, theta, xi, rho, sigma, T, N, n_paths)

# 2D Fokker-Planck with finite differences
# 2D Fokker-Planck with finite differences
nt = 400  # Number of time steps
np = 800  # Number of price steps
t_grid = np.linspace(0, T, nt)  # Time grid
p_grid = np.linspace(min(prices) * 0.95, max(prices) * 1.05, np)  # Price grid
dt_fd = T / (nt - 1)
dp = (max(prices) * 1.05 - min(prices) * 0.95) / (np - 1)

# Initial condition
u = np.zeros((nt, np))
sigma_init = np.sqrt(v0) * p0
u[0, :] = np.exp(-((p_grid - p0)**2) / (2 * sigma_init**2))
u[0, :] /= np.trapz(u[0, :], p_grid)

# Crank-Nicolson scheme with Laplace-Beltrami
r = dt_fd / (2 * dp**2)
A = diags([1, -2, 1], [-1, 0, 1], shape=(np, np)).toarray()
A[0, 0] = A[-1, -1] = 1  # Neumann boundary
A = r * A
I = diags([1], [0], shape=(np, np)).toarray()

for i in range(nt - 1):
    idx = int(i * (len(sigma) - 1) / (nt - 1))
    sigma_t = max(sigma[idx], 0.01)
    var_t = np.interp(t_grid[i], t, variances.mean(axis=0))
    metric = AdvancedRiemannianMetric(sigma[:idx+1], t[:i+1], t_grid[i], prices[:i+1], variances[:, :i+1])
    g = np.array([metric.metric_matrix([t_grid[i], p]) for p in p_grid])
    det_g = np.array([np.linalg.det(g_k) for g_k in g])
    g_inv = np.array([np.linalg.inv(g_k) for g_k in g])
    laplace_term = np.array([np.sum(g_inv[k] * np.gradient(np.gradient(u[i, :], dp), dp)) for k in range(np)])
    b = u[i, :] + 0.5 * dt_fd * (0.5 * var_t * p_grid**2 * laplace_term - mu * p_grid * np.gradient(u[i, :], dp))
    A_eff = I - r * var_t * p_grid**2 * np.diag(det_g)
    u[i + 1, :] = spsolve(csr_matrix(A_eff), b)
    u[i + 1, :] = np.clip(u[i + 1, :], 1e-10, 1e10)
    u[i + 1, :] /= np.trapz(u[i + 1, :], p_grid)

u_density = u[-1, :]

# Identify support and resistance with curvature
du = np.gradient(u_density, dp)
d2u = np.gradient(du, dp)
ricci = np.array([metric.ricci_curvature(np.array([T, p]))[1, 1] for p in p_grid])
support_idx = np.where((du > 0) & (d2u < 0) & (u_density > 0.1 * u_density.max()) & (ricci < 0))[0]
resistance_idx = np.where((du < 0) & (d2u > 0) & (u_density > 0.1 * u_density.max()) & (ricci > 0))[0]
support_levels = p_grid[support_idx]
resistance_levels = p_grid[resistance_idx]
if len(support_levels) == 0 or len(resistance_levels) == 0:
    st.warning("Insufficient distinct levels. Using density peaks with curvature fallback.")
    peaks, _ = find_peaks(u_density, height=0.1 * u_density.max(), distance=10)
    levels = p_grid[peaks]
    support_levels = levels[(du[peaks] > 0) & (ricci[peaks] < 0)]
    resistance_levels = levels[(du[peaks] < 0) & (ricci[peaks] > 0)]

# Heat kernel and probabilities
def heat_kernel(t, x, y, metric):
    dist = metric.dist(x, y)
    return (4 * np.pi * t)**(-1.0) * np.exp(-dist**2 / (4 * t))

support_probs = []
resistance_probs = []
total_support_prob = 0.0
total_resistance_prob = 0.0

for sr in support_levels:
    prob = 0.0
    for p in p_grid:
        if abs(p - sr) <= epsilon:
            kernel = heat_kernel(T, [0, p0], [T, p], metric)
            det_g = np.linalg.det(metric.metric_matrix([T, p]))
            if det_g > 0:
                prob += kernel * np.sqrt(det_g) * dp
    support_probs.append(prob)
    total_support_prob += prob

for rr in resistance_levels:
    prob = 0.0
    for p in p_grid:
        if abs(p - rr) <= epsilon:
            kernel = heat_kernel(T, [0, p0], [T, p], metric)
            det_g = np.linalg.det(metric.metric_matrix([T, p]))
            if det_g > 0:
                prob += kernel * np.sqrt(det_g) * dp
    resistance_probs.append(prob)
    total_resistance_prob += prob

# Normalize probabilities
support_probs = [p / total_support_prob if total_support_prob > 0 else 1.0 / len(support_levels) for p in support_probs]
resistance_probs = [p / total_resistance_prob if total_resistance_prob > 0 else 1.0 / len(resistance_levels) for p in resistance_probs]

# Geodesic with curvature
def geodesic_equation(t, y, metric):
    pos = y[:2]
    vel = y[2:]
    gamma = metric.christoffel_symbols(pos)
    accel = -np.einsum('ijk,vk->vj', gamma, np.outer(vel, vel))
    return np.concatenate([vel, accel.flatten()])

try:
    initial_point = np.array([0.0, p0])
    delta_p = prices[-1] - prices[0]
    initial_tangent = np.array([T / N, delta_p / N, 0.0, 0.0])  # Normalized initial velocity
    sol = solve_ivp(geodesic_equation, [0, 1], np.concatenate([initial_point, initial_tangent[2:]]),
                    args=(metric,), method='RK45', t_eval=np.linspace(0, 1, N), rtol=1e-8, atol=1e-10)
    geodesic_points = np.vstack([t * sol.y[0, :], initial_point[1] + sol.y[1, :] * T]).T
    geodesic_df = pd.DataFrame({
        "Time": geodesic_points[:, 0],
        "Price": geodesic_points[:, 1],
        "Path": "Geodesic"
    })
except Exception as e:
    st.error(f"Geodesic computation failed: {e}. Using linear approximation.")
    geodesic_df = pd.DataFrame({
        "Time": t,
        "Price": p0 + (delta_p / T) * t,
        "Path": "Geodesic"
    })

# Trending/Reversion analysis
def geodesic_distance(metric, path, geodesic):
    return np.mean([metric.dist(np.array([t[i], path[i]]), np.array([geodesic["Time"].iloc[i], geodesic["Price"].iloc[i]])) for i in range(N)])

distances = [geodesic_distance(metric, paths[i], geodesic_df) for i in range(n_paths)]
trending_threshold = np.percentile(distances, 25)
reversion_threshold = np.percentile(distances, 75)
trending_paths = np.sum(np.array(distances) < trending_threshold)
reversion_paths = np.sum(np.array(distances) > reversion_threshold)

# Prepare data for Altair
path_data = []
for i in range(min(n_paths, n_display_paths)):
    for j in range(N):
        path_data.append({"Time": t[j], "Price": paths[i, j], "Path": f"Path_{i}"})
path_df = pd.DataFrame(path_data)
plot_df = pd.concat([path_df, geodesic_df])

support_df = pd.DataFrame({"Price": support_levels, "Level": [f"Support_{i}" for i in range(len(support_levels))]})
resistance_df = pd.DataFrame({"Price": resistance_levels, "Level": [f"Resistance_{i}" for i in range(len(resistance_levels))]})

# Altair chart
base = alt.Chart(plot_df).encode(
    x=alt.X("Time:Q", title="Time (hours)", scale=alt.Scale(domain=[0, T])),
    y=alt.Y("Price:Q", title="BTC/USD Price", scale=alt.Scale(domain=[min(plot_df["Price"]) * 0.95, max(plot_df["Price"]) * 1.05])),
    color=alt.Color("Path:N", legend=alt.Legend(title="Path Type", orient="bottom"))
)

paths = base.mark_line(opacity=0.1).transform_filter(
    alt.datum.Path != "Geodesic"
)

geodesic = base.mark_line(strokeWidth=3, color="red").transform_filter(
    alt.datum.Path == "Geodesic"
)

support_lines = alt.Chart(support_df).mark_rule(
    stroke="green", strokeWidth=2, strokeDash=[5, 5]
).encode(
    y="Price:Q",
    tooltip=["Price:Q", alt.value("Support Level")]
).interactive()

resistance_lines = alt.Chart(resistance_df).mark_rule(
    stroke="red", strokeWidth=2, strokeDash=[5, 5]
).encode(
    y="Price:Q",
    tooltip=["Price:Q", alt.value("Resistance Level")]
).interactive()

chart = (paths + geodesic + support_lines + resistance_lines).properties(
    title="BTC/USD Price Paths, Geodesic, Support, and Resistance Levels",
    width=1200,
    height=800
).interactive()

st.altair_chart(chart, use_container_width=True)
st.write("**Support Levels (BTC/USD):**", support_levels)
st.write("**Support Hit Probabilities:**", dict(zip(support_levels, support_probs)))
st.write("**Resistance Levels (BTC/USD):**", resistance_levels)
st.write("**Resistance Hit Probabilities:**", dict(zip(resistance_levels, resistance_probs)))
st.write(f"**Trending Paths:** {trending_paths} (Distance < {trending_threshold:.2f})")
st.write(f"**Reversion Paths:** {reversion_paths} (Distance > {reversion_threshold:.2f})")

# MATLAB validation script
matlab_script = """
syms t p
sigma = @(t) interp1(t_data, sigma_data, t); % Replace with Python t, sigma
fisher = @(p) 1 ./ (hist(p_data, 50) + 1e-6); % Replace with Python prices
g11 = 1;
g22 = sigma(t)^2 + fisher(p);
g = [g11 0; 0 g22];
g_inv = inv(g);
% Christoffel symbols
christoffel = sym(zeros(2,2,2));
d_sigma = diff(sigma(t), t);
christoffel(2,2,1) = 0.5 * g_inv(1,1) * d_sigma * p / sigma(t);
% Geodesic equation
syms gamma_dot1 gamma_dot2
geodesic_eq = christoffel(2,2,1) * gamma_dot1^2;
% Ricci curvature (approximate)
ricci = simplify(diff(g_inv, t) - christoffel * christoffel');
disp('Metric Matrix:'); disp(g);
disp('Christoffel Symbols:'); disp(christoffel);
disp('Ricci Curvature:'); disp(ricci);
% Numerical geodesic
tspan = [0 1];
y0 = [0; p0; 1; 0]; % [t, p, dt, dp]
options = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
[t_sol, y_sol] = ode45(@(t,y) [y(3); y(4); -christoffel(2,2,1)*y(4)^2], tspan, y0, options);
plot(t_sol, y_sol(:,2), 'LineWidth', 2);
xlabel('Time'); ylabel('Price'); title('Geodesic Path');
grid on;
"""
st.write("**MATLAB Validation Script:**")
st.code(matlab_script, language="matlab")
