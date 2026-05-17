import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from trading_engine import (
    load_and_split_data,
    get_bounds,
    evaluate_fitness,
    evaluate_fitness_multi,
    get_train_segments,
    run_comparison,
    evaluate_on_periods,
    plot_combined_convergence,
    plot_combined_backtest,
    plot_single_vs_multi,
    plot_stress_test,
    plot_bh_range,
    get_date_range_info,
    figure_to_base64,
)

app = Flask(__name__, template_folder='templates')

@app.route('/image/<path:filename>')
def image_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'image'), filename)

try:
    train_df, test_df = load_and_split_data()
    train_prices = train_df['close'].values.astype(float)
    test_prices = test_df['close'].values.astype(float)
    train_segments, seg_labels = get_train_segments(train_df, train_prices)
except Exception as e:
    print(f"Data loading error: {e}")
    train_df = test_df = train_prices = test_prices = train_segments = None

# In-memory store for cross-mode comparison
stored_results = {}


@app.route('/')
def home():
    return render_template('showcase.html')


@app.route('/demo')
def demo():
    date_info = None
    if train_df is not None and test_df is not None:
        full_prices = np.concatenate([train_prices, test_prices])
        full_dates = np.concatenate([train_df['date'].values, test_df['date'].values])
        date_info = get_date_range_info(full_dates, full_prices)
    return render_template('demo.html', date_info=date_info)


@app.route('/api/compare', methods=['POST'])
def api_compare():
    if train_prices is None:
        return jsonify({'error': 'Data not loaded'}), 500

    data = request.get_json()
    mode = data.get('mode', 'single')
    high_limit = data.get('high_limit', False)
    low_limit = data.get('low_limit', False)
    budget = int(data.get('budget', 3000))

    bounds = get_bounds(high_limit=high_limit, low_limit=low_limit)

    if mode == 'multi':
        def fitness_fn(p, prices):
            return evaluate_fitness_multi(p, prices, train_segments)
        mode_label = 'Multi-Segment Training'
    else:
        def fitness_fn(p, prices):
            return evaluate_fitness(p, prices)
        mode_label = 'Single-Segment Training'

    results = run_comparison(train_prices, bounds, budget, fitness_fn)

    # Evaluate on test set for each algorithm
    test_vals = {}
    for name, data in results.items():
        test_val = evaluate_fitness(np.array(data['params']), test_prices)
        test_vals[name] = float(test_val)
        data['test'] = float(test_val)
        data['gap_pct'] = round((data['train'] - test_val) / data['train'] * 100, 1) if data['train'] > 0 else 0

    bh_value = float(1000.0 * test_prices[-1] / test_prices[0])

    # Find best algorithm by test performance
    best_algo = max(results.items(), key=lambda x: x[1]['test'])
    best_algo_name = best_algo[0]

    # Store for cross-mode comparison
    store_key = 'single' if mode == 'single' else 'multi'
    stored_results[store_key] = {
        name: {'train': d['train'], 'test': d['test']} for name, d in results.items()
    }

    # Generate comparison charts
    conv_fig = plot_combined_convergence(results, budget, mode_label)
    conv_img = figure_to_base64(conv_fig)

    backtest_fig = plot_combined_backtest(results, test_df['date'].values, test_prices, bh_value, mode_label)
    backtest_img = figure_to_base64(backtest_fig)

    # Multi-period stress test
    period_defs = {
        '2014-2016\nSideways': ('2014-01-01', '2017-01-01'),
        '2017\nBull': ('2017-01-01', '2018-01-01'),
        '2018\nBear': ('2018-01-01', '2019-01-01'),
        '2019\nRecovery': ('2019-01-01', '2020-01-01'),
        '2020-2022\nBig Bull': ('2020-01-01', '2022-03-02'),
    }
    full_dates = np.concatenate([train_df['date'].values, test_df['date'].values])
    full_prices = np.concatenate([train_prices, test_prices])
    stress_data = {}
    for name, data in results.items():
        stress_data[name] = evaluate_on_periods(np.array(data['params']), full_prices, full_dates, period_defs)
    stress_fig = plot_stress_test(stress_data, mode_label)
    stress_img = figure_to_base64(stress_fig)

    # Single vs Multi comparison chart (if both modes exist)
    compare_img = None
    has_comparison = 'single' in stored_results and 'multi' in stored_results
    if has_comparison:
        compare_fig = plot_single_vs_multi(stored_results['single'], stored_results['multi'], bh_value)
        compare_img = figure_to_base64(compare_fig)

    return jsonify({
        'mode': mode,
        'mode_label': mode_label,
        'results': results,
        'test_vals': test_vals,
        'bh_value': round(bh_value, 2),
        'best_algo': best_algo_name,
        'convergence_image': conv_img,
        'backtest_image': backtest_img,
        'stress_test_image': stress_img,
        'compare_image': compare_img,
        'has_comparison': has_comparison,
    })


@app.route('/api/bh_range', methods=['POST'])
def api_bh_range():
    if train_prices is None:
        return jsonify({'error': 'Data not loaded'}), 500

    data = request.get_json()
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    algo_results = data.get('algo_results', {})
    best_algo = data.get('best_algo', '')

    full_dates = np.concatenate([train_df['date'].values, test_df['date'].values])
    full_prices = np.concatenate([train_prices, test_prices])
    start_dt = np.datetime64(start_date)
    end_dt = np.datetime64(end_date)

    mask = (full_dates >= start_dt) & (full_dates <= end_dt)
    if mask.sum() < 2:
        return jsonify({'error': 'Selected range too short'}), 400

    seg_prices = full_prices[mask]
    bh_final = float(1000.0 * seg_prices[-1] / seg_prices[0])

    # Calculate algo values for this range
    algo_range_vals = {}
    for name, d in algo_results.items():
        params = np.array(d['params'])
        val = evaluate_fitness(params, seg_prices)
        algo_range_vals[name] = float(val)

    fig = plot_bh_range(full_dates, full_prices, start_dt, end_dt, algo_results, best_algo)
    img = figure_to_base64(fig) if fig else None

    return jsonify({
        'bh_final': bh_final,
        'algo_range_vals': algo_range_vals,
        'image': img,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
