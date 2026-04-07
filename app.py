from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    cleaned_news = clean_text(news)
    vector = vectorizer.transform([cleaned_news])

    prediction = model.predict(vector)[0]

    if prediction == "REAL":
        result_class = "real"
        emoji = "✅"
    else:
        result_class = "fake"
        emoji = "⚠️"

    return render_template(
        "index.html",
        prediction_text=f"{emoji} Prediction: {prediction}",
        result_class=result_class,
        input_news=news
    )

if __name__ == "__main__":
    app.run(debug=True)
