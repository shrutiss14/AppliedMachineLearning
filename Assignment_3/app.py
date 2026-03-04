from flask import Flask, request, jsonify, render_template_string
import joblib
from score import score



app = Flask(__name__)

# Load model once when app starts
model = joblib.load("best_model.pkl")




@app.route("/", methods=["GET"])
def home():
    html_page = """
    <html>
        <head>
            <title>Spam Classifier</title>
        </head>
        <body>
            <h1>Spam Classifier</h1>
            <form method="post" action="/score">
                <input type="text" name="text" placeholder="Enter message"/>
                <button type="submit">Classify</button>
            </form>
        </body>
    </html>
    """
    return render_template_string(html_page)




@app.route("/score", methods=["POST"])
def score_endpoint():

    # Handle JSON requests
    if request.is_json:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"]

    # Handle form submission
    else:
        text = request.form.get("text")

        if text is None:
            return jsonify({"error": "Missing 'text' field"}), 400

    try:
        prediction, propensity = score(text, model, 0.5)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "prediction": prediction,
        "propensity": propensity
    })




if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)