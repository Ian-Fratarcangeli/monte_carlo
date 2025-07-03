import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from data import get_historical_prices, calculate_log_returns, set_fred_api_key, get_risk_free_rate
from blackscholes import BlackScholesCall, BlackScholesPut

st.title("Monte Carlo Option Pricing Dashboard")

# API Key setup (hide in real app)
set_fred_api_key("09076f364bb3c14142005a12bb7ef7a3")

# Sidebar for user input
st.sidebar.header("Simulation Parameters")
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
S0 = st.sidebar.number_input("Initial Stock Price (S0)", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (years, T)", value=1.0, step=0.5)
Nsteps = st.sidebar.number_input("Number of Steps", value=10000, step=1000)
n_sim = st.sidebar.number_input("Number of Simulations", value=1000, step=100)
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

# Fetch data
st.header(f"Fetched Historical Data for {symbol}")
data_load_state = st.text('Loading data...')
apple = get_historical_prices(symbol=symbol)
data_load_state.text('')
if apple.empty:
    st.error(f"No data found for symbol: {symbol}")
    st.stop()

# Calculate log returns
log_returns, mean_return, std_return = calculate_log_returns(apple)

# Parameters for simulation
r = get_risk_free_rate()
sigma = std_return * np.sqrt(252)  # Annualized volatility
mu = mean_return
interval = [0.0, T]
dt = (interval[1] - interval[0]) / Nsteps
t_axis = np.linspace(interval[0], interval[1], int(Nsteps) + 1)

st.sidebar.markdown(f"**Risk-free rate (3M T-bill):** {r:.4f}")
st.sidebar.markdown(f"**Annualized Volatility:** {sigma:.4f}")

# Geometric Brownian Motion Monte Carlo Simulation
st.header("Geometric Brownian Motion Monte Carlo Simulation")

Z = np.random.normal(loc=0, scale=1, size=(int(n_sim), int(Nsteps)))
X = np.zeros((int(n_sim), int(Nsteps) + 1))
X[:, 0] = S0
for i in range(int(Nsteps)):
    X[:, i + 1] = X[:, i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i])

fig1, ax1 = plt.subplots(figsize=(10, 6))
for path in range(min(20, int(n_sim))):
    ax1.plot(t_axis, X[path, :], alpha=0.5)
ax1.set_title("Geometric Brownian Motion Monte Carlo Sample Paths")
ax1.set_xlabel("Time (years)")
ax1.set_ylabel("Asset Value")
st.pyplot(fig1)

# Monte Carlo Option Price
if option_type == 'call':
    value = np.maximum(X[:, -1] - K, 0)
else:
    value = np.maximum(K - X[:, -1], 0)
monte_carlo_estimate = np.mean(value) * np.exp(-r * T)
st.write(f"**Monte Carlo {option_type.capitalize()} Estimate:** {monte_carlo_estimate:.4f}")


# Heston Model Simulation
st.header("Heston Model Monte Carlo Simulation")
#params
v0 = sigma**2      # Initial variance
k = 4    # Speed of mean reversion
theta = sigma**2     # Long-term variance
sigma_v = 0.05        # Vol of vol
rho = -0.5       # Correlation
mu = np.array([0,0]) # Mean for the correlated Brownian motions
cov = np.array([[1, rho] , [rho , 1]]) # Covariance matrix for the correlated Brownian motions


S = np.zeros((int(n_sim), int(Nsteps) + 1))
v = np.zeros((int(n_sim), int(Nsteps) + 1))
S[:, 0] = S0
v[:, 0] = v0

for t in range(int(Nsteps)):
    B_t = np.random.multivariate_normal(mu, cov, size=n_sim)

    # Asset price for Heston Model
    S[:, t + 1] = S[:, t] * np.exp((r - 0.5 * v[:, t]) * dt + np.sqrt(v[:, t] * dt) * B_t[:, 0])
    # Variance for Heston Model
    v[:, t + 1] = v[:, t] + k * (theta - v[:, t]) * dt + sigma_v * np.sqrt(v[:, t] * dt) * B_t[:, 1]
    v[:, t + 1] = np.maximum(v[:, t + 1], 0)  # make sure variance is positive

fig2, ax2 = plt.subplots(figsize=(10, 6))
for path in range(min(20, int(n_sim))):
    ax2.plot(t_axis, S[path, :], alpha=0.5)
ax2.set_title("Heston Model Sample Paths")
ax2.set_xlabel("Time (years)")
ax2.set_ylabel("Asset Value")
st.pyplot(fig2)

# Heston Option Price
if option_type == 'call':
    heston_value = np.maximum(S[:, -1] - K, 0)
else:
    heston_value = np.maximum(K - S[:, -1], 0)
heston_estimate = np.mean(heston_value) * np.exp(-r * T)
st.write(f"**Heston {option_type.capitalize()} Estimate:** {heston_estimate:.4f}")

# Black-Scholes Analytical Price
st.header("Black-Scholes Analytical Price")
if option_type == 'call':
    call = BlackScholesCall(S=S0, K=K, T=T, r=r, sigma=sigma, q=0)
    st.write(f"**Black-Scholes Call Price:** {call.price():.4f}")
else:
    put = BlackScholesPut(S=S0, K=K, T=T, r=r, sigma=sigma, q=0)
    st.write(f"**Black-Scholes Put Price:** {put.price():.4f}")