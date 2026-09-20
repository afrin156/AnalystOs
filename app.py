"""
AnalystOS - main Flask app.

Run with:
    pip install -r requirements.txt
    python app.py

Then open http://localhost:5000
"""

import os
import io
import pandas as pd
from flask import Flask, render_template, request, jsonify

from agents.supervisor import run_full_analysis
from agents.simulation import simulate_discount_change, simulate_customer_loss
from agents.next_questions import check_question_answerable
from agents.memory import get_history

app = Flask(__name__)

_current_df = None
_current_key_columns = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/load-sample", methods=["POST"])
def load_sample():
    global _current_df
    path = os.path.join(os.path.dirname(__file__), "sample_data", "company_sales.csv")
    _current_df = pd.read_csv(path)
    result = run_full_analysis(_current_df, "company_sales.csv (sample)")
    global _current_key_columns
    _current_key_columns = result["key_columns"]
    return jsonify(result)


@app.route("/upload", methods=["POST"])
def upload():
    global _current_df, _current_key_columns

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    try:
        _current_df = pd.read_csv(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Could not read this file as CSV: {e}"}), 400

    result = run_full_analysis(_current_df, file.filename)
    _current_key_columns = result["key_columns"]
    return jsonify(result)


@app.route("/simulate/discount", methods=["POST"])
def simulate_discount():
    if _current_df is None:
        return jsonify({"error": "No dataset loaded yet. Upload a CSV or load the sample first."}), 400

    data = request.get_json()
    change_pct = float(data.get("change_pct", 10))
    result = simulate_discount_change(_current_df, _current_key_columns, change_pct)
    return jsonify(result)


@app.route("/simulate/customer-loss", methods=["POST"])
def simulate_loss():
    if _current_df is None:
        return jsonify({"error": "No dataset loaded yet. Upload a CSV or load the sample first."}), 400

    data = request.get_json()
    top_n = int(data.get("top_n", 100))
    result = simulate_customer_loss(_current_df, _current_key_columns, top_n)
    return jsonify(result)


@app.route("/ask", methods=["POST"])
def ask():
    if _current_df is None:
        return jsonify({"error": "No dataset loaded yet."}), 400

    data = request.get_json()
    question = data.get("question", "")

    check = check_question_answerable(question, list(_current_df.columns))
    if not check["answerable"]:
        return jsonify({
            "answerable": False,
            "message": "INSUFFICIENT EVIDENCE",
            "reason": check["reason"],
            "available_columns": check["available_columns"],
        })

    return jsonify({
        "answerable": True,
        "message": "This question may be answerable - try using the Discovery findings and Investigate feature above, which search the data directly for relevant patterns.",
    })


@app.route("/history", methods=["GET"])
def history():
    return jsonify(get_history())


if __name__ == "__main__":
    app.run(debug=True, port=5000)
