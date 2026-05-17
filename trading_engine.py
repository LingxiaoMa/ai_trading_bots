import io
import os
import math
import base64
import matplotlib
matplotlib.use('Agg')
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


def particle_swarm_optimisation(prices, bounds, max_evaluations, swarm_size=20, inertia_weight=0.7,
                                cognitive_coefficient=1.5, social_coefficient=1.5,
                                seed=GLOBAL_SEED, fitness_fn=None):
    if fitness_fn is None:
        fitness_fn = evaluate_fitness
    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    search_range = ub - lb

    positions = rng.uniform(lb, ub, size=(swarm_size, len(bounds)))
    velocities = rng.uniform(-0.1 * search_range, 0.1 * search_range, size=(swarm_size, len(bounds)))

    particle_fitness = np.array([fitness_fn(p, prices) for p in positions])
    personal_best_pos = positions.copy()
    personal_best_fit = particle_fitness.copy()
    best_idx = np.argmax(personal_best_fit)
    global_best_pos = personal_best_pos[best_idx].copy()
    global_best_fit = personal_best_fit[best_idx]
    history = [global_best_fit]
    evals = swarm_size

    while evals < max_evaluations:
        for i in range(swarm_size):
            if evals >= max_evaluations:
                break
            r1, r2 = rng.random(len(bounds)), rng.random(len(bounds))
            velocities[i] = (inertia_weight * velocities[i]
                             + cognitive_coefficient * r1 * (personal_best_pos[i] - positions[i])
                             + social_coefficient * r2 * (global_best_pos - positions[i]))
            positions[i] = apply_bounds(positions[i] + velocities[i], lb, ub)
            f = fitness_fn(positions[i], prices)
            evals += 1
            if f > personal_best_fit[i]:
                personal_best_pos[i] = positions[i].copy()
                personal_best_fit[i] = f
            if f > global_best_fit:
                global_best_pos = positions[i].copy()
                global_best_fit = f
            history.append(global_best_fit)

    return global_best_pos, global_best_fit, history


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
    if method == 'PSO':
        swarm_size = min(20, max(5, budget // 30))
        return particle_swarm_optimisation(prices, bounds, max_evaluations=budget, swarm_size=swarm_size)
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


# --- Multi-Segment Fitness ---

def get_train_segments(train_df, train_prices):
    seg_dates = train_df['date']
    segments = [
        train_prices[(seg_dates < '2017-01-01').values],
        train_prices[((seg_dates >= '2017-01-01') & (seg_dates < '2018-01-01')).values],
        train_prices[((seg_dates >= '2018-01-01') & (seg_dates < '2019-01-01')).values],
        train_prices[(seg_dates >= '2019-01-01').values],
    ]
    labels = ['2014-2016 (Sideways)', '2017 (Bull)', '2018 (Bear)', '2019 (Recovery)']
    return segments, labels


def evaluate_fitness_multi(params, prices, train_segments=None, initial_cash=1000.0, fee=0.03):
    if train_segments is None:
        return evaluate_fitness(params, prices, initial_cash, fee)
    scores = [evaluate_fitness(params, seg, initial_cash, fee) for seg in train_segments]
    return np.mean(scores)


# --- Run All Algorithms Comparison ---

def run_comparison(prices, bounds, budget, fitness_fn, train_segments=None):
    results = {}
    for method, name in [('CS', 'Cuckoo Search'), ('ALO', 'Antlion Optimizer'), ('SA', 'Simulated Annealing'), ('PSO', 'Particle Swarm Optimisation')]:
        if method == 'CS':
            n_nests = min(20, max(5, budget // 40))
            max_iter = max(2, budget // (2 * n_nests))
            params, train_fit, history = cuckoo_search(prices, bounds, n_nests=n_nests, max_iter=max_iter, fitness_fn=fitness_fn)
        elif method == 'ALO':
            n_agents = min(20, max(5, budget // 30))
            max_iter = max(2, budget // n_agents)
            params, train_fit, history = antlion_optimizer(prices, bounds, n_agents=n_agents, max_iter=max_iter, fitness_fn=fitness_fn)
        elif method == 'SA':
            params, train_fit, history = simulated_annealing(prices, bounds, max_evals=budget, fitness_fn=fitness_fn)
        elif method == 'PSO':
            swarm_size = min(20, max(5, budget // 30))
            params, train_fit, history = particle_swarm_optimisation(prices, bounds, max_evaluations=budget, swarm_size=swarm_size, fitness_fn=fitness_fn)
        results[name] = {'params': params.tolist(), 'train': float(train_fit), 'history': [float(h) for h in history]}
    return results


# --- Period-based Evaluation ---

def evaluate_on_periods(params, prices, dates, period_defs):
    period_results = {}
    for label, (start, end) in period_defs.items():
        mask = (dates >= np.datetime64(start)) & (dates < np.datetime64(end))
        if mask.sum() == 0:
            period_results[label] = None
            continue
        seg_prices = prices[mask]
        val = evaluate_fitness(params, seg_prices)
        bh_val = 1000.0 * (seg_prices[-1] / seg_prices[0])
        period_results[label] = {'algo': float(val), 'bh': float(bh_val)}
    return period_results


# --- Comparison Plotting ---

ALGO_COLORS = {
    'Cuckoo Search': '#27ae60',
    'Antlion Optimizer': '#8e44ad',
    'Simulated Annealing': '#f39c12',
    'Particle Swarm Optimisation': '#577590',
}


def plot_combined_convergence(comparison_results, budget, mode_label):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in comparison_results.items():
        color = ALGO_COLORS.get(name, '#333333')
        hist = data['history']
        x = np.linspace(0, budget, len(hist))
        ax.plot(x, hist, label=f"{name} (${data['train']:,.0f})", color=color, linewidth=2)
    ax.set_title(f'Convergence Comparison — {mode_label}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Objective Function Evaluations')
    ax.set_ylabel('Best Portfolio Value (USD)')
    ax.legend(frameon=True, facecolor='white')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    ax.grid(True, alpha=0.3)
    return fig


def plot_combined_backtest(comparison_results, dates, prices, bh_value, mode_label):
    fig, ax = plt.subplots(figsize=(12, 5))
    benchmark = prices / prices[0] * 1000.0
    ax.plot(dates, benchmark, label=f'Buy & Hold (${bh_value:,.0f})', color='#7f8c8d',
            linestyle='--', linewidth=2, alpha=0.8)
    for name, data in comparison_results.items():
        color = ALGO_COLORS.get(name, '#333333')
        curve = simulate_portfolio_curve(np.array(data['params']), prices)
        test_val = evaluate_fitness(np.array(data['params']), prices)
        ax.plot(dates, curve, label=f"{name} (${test_val:,.0f})", color=color, linewidth=1.8)
    ax.set_title(f'Out-of-Sample Backtest Comparison — {mode_label}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (USD)')
    ax.legend(frameon=True, facecolor='white', fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    fig.autofmt_xdate()
    return fig


def plot_single_vs_multi(single_data, multi_data, bh_value):
    fig, ax = plt.subplots(figsize=(10, 5))
    algos = ['Cuckoo Search', 'Antlion Optimizer', 'Simulated Annealing', 'Particle Swarm Optimisation']
    x = np.arange(len(algos))
    width = 0.25

    single_vals = [single_data.get(a, {}).get('test', 0) for a in algos]
    multi_vals = [multi_data.get(a, {}).get('test', 0) for a in algos]

    bars1 = ax.bar(x - width, single_vals, width, label='Single-Segment Test', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x, multi_vals, width, label='Multi-Segment Test', color='#2ecc71', alpha=0.8)
    ax.axhline(y=bh_value, color='#7f8c8d', linestyle='--', linewidth=2, label=f'Buy & Hold (${bh_value:,.0f})')

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=9)
    ax.set_title('Single-Segment vs Multi-Segment: Out-of-Sample Performance', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Portfolio Value (USD)')
    ax.legend(frameon=True, facecolor='white', fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))

    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, f'${bar.get_height():,.0f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, f'${bar.get_height():,.0f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
    return fig


def plot_stress_test(all_stress_data, mode_label):
    period_labels = list(list(all_stress_data.values())[0].keys()) if all_stress_data else []
    algos = list(all_stress_data.keys())
    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(period_labels))
    n_algos = len(algos)
    width = 0.7 / n_algos

    for i, algo in enumerate(algos):
        vals = [all_stress_data[algo].get(p, {}).get('algo', 0) or 0 for p in period_labels]
        offset = (i - (n_algos - 1) / 2) * width
        color = ALGO_COLORS.get(algo, '#333333')
        ax.bar(x + offset, vals, width, label=algo, color=color, alpha=0.85)

    bh_vals = [all_stress_data[algos[0]].get(p, {}).get('bh', 0) or 0 for p in period_labels]
    ax.bar(x + (n_algos / 2) * width, bh_vals, width, label='Buy & Hold', color='#7f8c8d', alpha=0.5, hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(period_labels, fontsize=8)
    ax.set_title(f'Multi-Period Stress Test — {mode_label}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Portfolio Value (USD)')
    ax.legend(frameon=True, facecolor='white', fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    return fig


def plot_bh_range(dates, prices, start_date, end_date, algo_results, best_algo_name):
    mask = (dates >= start_date) & (dates <= end_date)
    if mask.sum() < 2:
        return None
    seg_prices = prices[mask]
    seg_dates = dates[mask]
    bh_curve = 1000.0 * seg_prices / seg_prices[0]
    bh_final = float(bh_curve[-1])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(range(len(seg_dates)), 1000, bh_curve, alpha=0.08, color='#7f8c8d')
    ax.plot(seg_dates, bh_curve, label=f'Buy & Hold (${bh_final:,.0f})', color='#7f8c8d', linestyle='--', linewidth=2.5)

    for name, data in algo_results.items():
        curve = simulate_portfolio_curve(np.array(data['params']), seg_prices)
        final_val = float(curve[-1])
        if name == best_algo_name:
            color = ALGO_COLORS.get(name, '#27ae60')
            ax.plot(seg_dates, curve, label=f'{name} (${final_val:,.0f})', color=color, linewidth=2.5, zorder=5)
        else:
            color = ALGO_COLORS.get(name, '#333333')
            ax.plot(seg_dates, curve, label=f'{name} (${final_val:,.0f})', color=color, linewidth=1, alpha=0.45)

    ax.axhline(y=1000, color='black', linestyle=':', linewidth=1, alpha=0.3)
    ax.set_title(f'Custom Period Backtest: {str(start_date)[:10]} → {str(end_date)[:10]}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (USD)')
    ax.legend(frameon=True, facecolor='white', fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    fig.autofmt_xdate()
    return fig


def get_date_range_info(dates, prices):
    full_bh = 1000.0 * prices / prices[0]
    return {
        'min_date': str(dates.min())[:10],
        'max_date': str(dates.max())[:10],
        'full_bh': float(full_bh[-1]),
    }
