import os
from flask import Flask, render_template, request, send_from_directory
from trading_engine import (
    load_and_split_data,
    get_bounds,
    choose_algorithm,
    evaluate_fitness,
    plot_convergence,
    plot_portfolio_curve,
    plot_trade_signals,
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
except Exception as e:
    print(f"Data loading error: {e}")
    train_df = test_df = train_prices = test_prices = None

@app.route('/')
def home():
    """Main showcase page"""
    return render_template('showcase.html')

@app.route('/demo', methods=['GET', 'POST'])
def demo():
    """Interactive demo page"""
    result = None
    convergence_image = None
    portfolio_image = None
    signals_image = None
    error_msg = None

    form = {
        'method': 'CS',
        'high_limit': False,
        'low_limit': False,
        'budget': 600,
    }

    if request.method == 'POST':
        if train_prices is None or test_prices is None:
            error_msg = "Data not loaded. Please ensure BTC-Daily.csv is in the project directory."
            return render_template('demo.html', form=form, error_msg=error_msg)

        form['method'] = request.form.get('method', 'CS')
        form['high_limit'] = request.form.get('high_limit') == 'on'
        form['low_limit'] = request.form.get('low_limit') == 'on'
        form['budget'] = int(request.form.get('budget', 600))

        try:
            bounds = get_bounds(high_limit=form['high_limit'], low_limit=form['low_limit'])
            best_params, best_fitness, history = choose_algorithm(
                form['method'], train_prices, bounds, form['budget']
            )

            test_value = evaluate_fitness(best_params, test_prices)
            method_name = {
                'CS': 'Cuckoo Search',
                'SA': 'Simulated Annealing',
                'ALO': 'Antlion Optimizer',
                'RS': 'Random Search',
            }.get(form['method'], form['method'])

            result = {
                'algo_name': method_name,
                'best_fitness': f'${best_fitness:,.2f}',
                'test_value': f'${test_value:,.2f}',
                'budget': form['budget'],
                'high_limit': 'Enabled' if form['high_limit'] else 'Disabled',
                'low_limit': 'Enabled' if form['low_limit'] else 'Disabled',
            }

            convergence_image = figure_to_base64(plot_convergence(history, method_name))
            portfolio_image = figure_to_base64(
                plot_portfolio_curve(best_params, test_df['date'].values, test_prices, method_name)
            )
            signals_image = figure_to_base64(
                plot_trade_signals(best_params, test_df['date'].values, test_prices, method_name)
            )
        except Exception as e:
            error_msg = f"Algorithm execution error: {str(e)}"

    return render_template(
        'demo.html',
        form=form,
        result=result,
        convergence_image=convergence_image,
        portfolio_image=portfolio_image,
        signals_image=signals_image,
        error_msg=error_msg,
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
