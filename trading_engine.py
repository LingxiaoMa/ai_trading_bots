import io
import os
import math
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
plt.style.use('seaborn-v0_8-whitegrid')

DATA_FILENAME = 'BTC-Daily.csv'

# --- Data Loading ---

def load_and_split_data(filepath=None, split_date='2020-01-01'):
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), DATA_FILENAME)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset {filepath} not found.")

    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    train_df = df[df['date'] < split_date].reset_index(drop=True)
    test_df = df[df['date'] >= split_date].reset_index(drop=True)
    return train_df, test_df


# --- Trading Engine ---

def pad(P, N):
    if N <= 1:
        return P.copy()
    padding = 2.0 * P[0] - np.flip(P[1:N])
    return np.concatenate([padding, P])


def sma_kernel(N):
    return np.ones(N) / N


def lma_kernel(N):
    return (1.0 - np.arange(N) / N) * (2.0 / (N + 1))


def ema_kernel(N, alpha):
    alpha = float(np.clip(alpha, 1e-4, 1 - 1e-4))
    w = alpha * (1.0 - alpha) ** np.arange(N)
    return w / max(w.sum(), 1e-12)


def apply_filter(P, N, kernel):
    N = max(2, int(N))
    padded = pad(P, N)
    out = np.convolve(padded, kernel, 'valid')
    return out[: len(P)]


def apply_bounds(candidate, lb, ub):
    return np.clip(candidate, lb, ub)


def build_composite_line(prices, w1, w2, w3, d1, d2, d3, alpha):
    d1, d2, d3 = int(d1), int(d2), int(d3)
    s1 = apply_filter(prices, d1, sma_kernel(d1))
    s2 = apply_filter(prices, d2, lma_kernel(d2))
    s3 = apply_filter(prices, d3, ema_kernel(d3, alpha))

    w_sum = abs(w1) + abs(w2) + abs(w3)
    if w_sum < 1e-12:
        return s1
    return (abs(w1) * s1 + abs(w2) * s2 + abs(w3) * s3) / w_sum


def generate_signals(prices, params):
    high_line = build_composite_line(prices, *params[0:7])
    low_line = build_composite_line(prices, *params[7:14])
    diff = high_line - low_line
    sign_diff = np.sign(diff)

    buy_signals = np.zeros(len(prices), dtype=bool)
    sell_signals = np.zeros(len(prices), dtype=bool)

    for t in range(1, len(prices)):
        delta = sign_diff[t] - sign_diff[t - 1]
        if delta >= 1.5:
            buy_signals[t] = True
        elif delta <= -1.5:
            sell_signals[t] = True

    return buy_signals, sell_signals, high_line, low_line


def evaluate_fitness(params, prices, initial_cash=1000.0, fee=0.03):
    buy_signals, sell_signals, _, _ = generate_signals(prices, params)
    cash, btc = float(initial_cash), 0.0
    for t in range(len(prices)):
        price = float(prices[t])
        if buy_signals[t] and cash > 0:
            btc = (cash * (1.0 - fee)) / price
            cash = 0.0
        elif sell_signals[t] and btc > 0:
            cash = btc * price * (1.0 - fee)
            btc = 0.0
    if btc > 0:
        cash = btc * prices[-1] * (1.0 - fee)
    return cash


def simulate_portfolio_curve(params, prices, initial_cash=1000.0, fee=0.03):
    buy_signals, sell_signals, _, _ = generate_signals(prices, params)
    cash, btc = float(initial_cash), 0.0
    portfolio = []
    for t in range(len(prices)):
        price = float(prices[t])
        if buy_signals[t] and cash > 0:
            btc = (cash * (1.0 - fee)) / price
            cash = 0.0
        elif sell_signals[t] and btc > 0:
            cash = btc * price * (1.0 - fee)
            btc = 0.0
        portfolio.append(cash + btc * price)
    return np.array(portfolio)


def get_bounds(high_limit=False, low_limit=False):
    high_base = [
        (0.01, 1.0),
        (0.01, 1.0),
        (0.01, 1.0),
        (2.0, 40.0),
        (2.0, 40.0),
        (2.0, 40.0),
        (0.01, 0.99),
    ]
    low_base = [
        (0.01, 1.0),
        (0.01, 1.0),
        (0.01, 1.0),
        (21.0, 100.0),
        (21.0, 100.0),
        (21.0, 100.0),
        (0.01, 0.99),
    ]

    if high_limit:
        high_base[3] = (2.0, 20.0)
        high_base[4] = (2.0, 20.0)
        high_base[5] = (2.0, 20.0)
    if low_limit:
        low_base[3] = (21.0, 60.0)
        low_base[4] = (21.0, 60.0)
        low_base[5] = (21.0, 60.0)

    return high_base + low_base


def random_search(prices, bounds, max_evals, seed=GLOBAL_SEED, fitness_fn=None):
    if fitness_fn is None:
        fitness_fn = evaluate_fitness
    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    best_nest = rng.uniform(lb, ub)
    best_fitness = fitness_fn(best_nest, prices)
    history = [best_fitness]

    for _ in range(max_evals - 1):
        candidate = rng.uniform(lb, ub)
        f = fitness_fn(candidate, prices)
        if f > best_fitness:
            best_nest, best_fitness = candidate.copy(), f
        history.append(best_fitness)
    return best_nest, best_fitness, history


def simulated_annealing(prices, bounds, max_evals, initial_temp=10000.0, cooling_rate=0.98, seed=GLOBAL_SEED, fitness_fn=None):
    if fitness_fn is None:
        fitness_fn = evaluate_fitness
    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    current_state = rng.uniform(lb, ub)
    current_fitness = fitness_fn(current_state, prices)
    best_state, best_fitness = current_state.copy(), current_fitness
    temp = initial_temp
    history = [best_fitness]

    for _ in range(max_evals - 1):
        neighbor = current_state + rng.normal(0, (ub - lb) * 0.05)
        neighbor = apply_bounds(neighbor, lb, ub)
        neighbor_fitness = fitness_fn(neighbor, prices)

        if neighbor_fitness > current_fitness:
            current_state, current_fitness = neighbor, neighbor_fitness
            if current_fitness > best_fitness:
                best_state, best_fitness = current_state.copy(), best_fitness
        else:
            diff = neighbor_fitness - current_fitness
            if rng.random() < math.exp(max(diff / temp, -700)):
                current_state, current_fitness = neighbor, neighbor_fitness

        temp *= cooling_rate
        history.append(best_fitness)

    return best_state, best_fitness, history


def antlion_optimizer(prices, bounds, n_agents, max_iter, seed=GLOBAL_SEED, fitness_fn=None):
    if fitness_fn is None:
        fitness_fn = evaluate_fitness
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    antlions = rng.uniform(lb, ub, (n_agents, dim))
    al_fitness = np.array([fitness_fn(a, prices) for a in antlions])
    best_idx = np.argmax(al_fitness)
    elite, elite_fitness = antlions[best_idx].copy(), al_fitness[best_idx]
    history = []

    for t in range(max_iter):
        I = 10 ** (1 * (t / max_iter)) if t > max_iter * 0.1 else 1
        c, d = lb / I, ub / I
        ants = np.zeros_like(antlions)
        for i in range(n_agents):
            shifted_fit = al_fitness - np.min(al_fitness) + 1e-5
            selected_idx = rng.choice(n_agents, p=shifted_fit / shifted_fit.sum())
            selected_al = antlions[selected_idx]
            walk_towards_al = selected_al + rng.uniform(-1, 1, dim) * (d - c)
            walk_towards_elite = elite + rng.uniform(-1, 1, dim) * (d - c)
            ants[i] = apply_bounds((walk_towards_al + walk_towards_elite) / 2, lb, ub)

        ant_fitness = np.array([fitness_fn(a, prices) for a in ants])
        combined = np.vstack((antlions, ants))
        combined_fit = np.concatenate((al_fitness, ant_fitness))
        top_idx = np.argsort(combined_fit)[-n_agents:]
        antlions, al_fitness = combined[top_idx], combined_fit[top_idx]

        if al_fitness[-1] > elite_fitness:
            elite, elite_fitness = antlions[-1].copy(), al_fitness[-1]
        history.extend([elite_fitness] * n_agents)

    return elite, elite_fitness, history


def cuckoo_search(prices, bounds, n_nests, max_iter, pa=0.25, seed=GLOBAL_SEED, fitness_fn=None):
    if fitness_fn is None:
        fitness_fn = evaluate_fitness
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    nests = rng.uniform(lb, ub, (n_nests, dim))
    fitness = np.array([fitness_fn(n, prices) for n in nests])
    best_idx = np.argmax(fitness)
    best_nest, best_fitness = nests[best_idx].copy(), fitness[best_idx]
    history = []

    beta = 1.5
    sigma = (gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

    for _ in range(max_iter):
        for i in range(n_nests):
            u = rng.normal(0, sigma, dim)
            v = rng.normal(0, 1, dim)
            step = u / (np.abs(v) ** (1 / beta))
            stepsize = 0.05 * step * (nests[i] - best_nest)
            new_nest = apply_bounds(nests[i] + stepsize * rng.standard_normal(dim), lb, ub)
            f_new = fitness_fn(new_nest, prices)
            j = rng.integers(0, n_nests)
            if f_new > fitness[j]:
                nests[j], fitness[j] = new_nest, f_new
                if f_new > best_fitness:
                    best_nest, best_fitness = new_nest.copy(), f_new
            history.append(best_fitness)

        K = rng.random((n_nests, dim)) < pa
        step_discovery = rng.random((n_nests, dim)) * (nests[rng.permutation(n_nests)] - nests[rng.permutation(n_nests)])
        new_nests_discovery = apply_bounds(nests + K * step_discovery, lb, ub)

        for i in range(n_nests):
            f_new_disc = fitness_fn(new_nests_discovery[i], prices)
            if f_new_disc > fitness[i]:
                nests[i], fitness[i] = new_nests_discovery[i], f_new_disc
                if f_new_disc > best_fitness:
                    best_nest, best_fitness = nests[i].copy(), f_new_disc
            history.append(best_fitness)

    return best_nest, best_fitness, history


def choose_algorithm(method, prices, bounds, budget):
    if method == 'CS':
        n_nests = min(20, max(5, budget // 40))
        max_iter = max(2, budget // (2 * n_nests))
        return cuckoo_search(prices, bounds, n_nests=n_nests, max_iter=max_iter)
    if method == 'ALO':
        n_agents = min(20, max(5, budget // 30))
        max_iter = max(2, budget // n_agents)
        return antlion_optimizer(prices, bounds, n_agents=n_agents, max_iter=max_iter)
    if method == 'SA':
        return simulated_annealing(prices, bounds, max_evals=budget)
    if method == 'RS':
        return random_search(prices, bounds, max_evals=budget)
    raise ValueError(f"Unknown algorithm: {method}")


def figure_to_base64(fig):
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format='png', dpi=120)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def plot_convergence(history, algo_name):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(len(history)), history, color='#2980b9', linewidth=2)
    ax.set_title(f'{algo_name} Convergence Trajectory', fontsize=12, fontweight='bold')
    ax.set_xlabel('Objective Function Evaluations')
    ax.set_ylabel('Best Portfolio Value (USD)')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    return fig


def plot_portfolio_curve(params, dates, prices, algo_name):
    curve = simulate_portfolio_curve(params, prices)
    benchmark = prices / prices[0] * 1000.0
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, curve, label=f'{algo_name} Strategy', color='#27ae60', linewidth=2)
    ax.plot(dates, benchmark, label='Buy & Hold Baseline', color='#7f8c8d', linestyle='--', linewidth=1.5)
    ax.set_title(f'{algo_name} Backtest Results (Out-of-Sample Test Set)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (USD)')
    ax.legend(frameon=True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    fig.autofmt_xdate()
    return fig


def plot_trade_signals(params, dates, prices, algo_name):
    buy_sigs, sell_sigs, high_line, low_line = generate_signals(prices, params)
    buy_idx = np.where(buy_sigs)[0]
    sell_idx = np.where(sell_sigs)[0]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, prices, label='BTC Close Price', color='black', alpha=0.4, linewidth=1)
    ax.plot(dates, high_line, label='HIGH Composite Line', color='#2980b9', linewidth=1.5)
    ax.plot(dates, low_line, label='LOW Composite Line', color='#e67e22', linewidth=1.5)

    if buy_idx.size > 0:
        ax.scatter(dates[buy_idx], prices[buy_idx], marker='^', color='green', s=60, label=f'Buy Signals ({len(buy_idx)})')
    if sell_idx.size > 0:
        ax.scatter(dates[sell_idx], prices[sell_idx], marker='v', color='red', s=60, label=f'Sell Signals ({len(sell_idx)})')

    ax.set_title(f'{algo_name} Trading Signal Visualization', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend(frameon=True)
    fig.autofmt_xdate()
    return fig
