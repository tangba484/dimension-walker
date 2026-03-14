from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from services.stock_service import get_or_create_chart_data
from data.ticker_map import TICKERS
app = Flask(__name__, instance_relative_config=True)
os.makedirs(app.instance_path, exist_ok=True)


@app.route("/")
def index():
    error = request.args.get("error")
    return render_template("index.html", error=error)


@app.route("/search", methods=["POST"])
def search():
    ticker = request.form["ticker"].strip().upper()
    return redirect(url_for("ticker_page", ticker=ticker))


@app.route("/ticker/<ticker>")
def ticker_page(ticker):
    
    try:
        chart_data = get_or_create_chart_data(ticker)
        return render_template(
            "ticker.html",
            ticker=ticker.upper(),
            dates=chart_data["dates"],
            actual_prices=chart_data["actual_prices"],
            no_pca_test=chart_data["no_pca_test"],
            no_pca_future=chart_data["no_pca_future"],
            pca_test=chart_data["pca_test"],
            pca_future=chart_data["pca_future"],
            run_info=chart_data["run_info"],
            message=None
        )
    except Exception as e:
        return redirect(url_for("index", error="invalid_ticker"))


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/api/ticker-suggest")
def ticker_suggest():
    query = request.args.get("q", "").strip().lower()

    if not query:
        return jsonify([])

    results = []

    for item in TICKERS:
        ticker = item["ticker"]
        name = item["name"]

        if query in ticker.lower() or query in name.lower():
            results.append({
                "ticker": ticker,
                "name": name
            })

    results = results[:10]
    return jsonify(results)

@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)