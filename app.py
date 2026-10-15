from flask import Flask, render_template, request
import joblib
import re
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Flask configuration
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "Disaster_tweet_Deployment" / "templates"),
    static_folder=str(BASE_DIR / "static")
)

# Load the trained model
MODEL_PATH = BASE_DIR / "validated_disaster_tweet_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("Disaster tweet model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    raise


def clean_text(text):
    """Clean tweet text before prediction."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    tweet_text = ""
    confidence = None
    error = None

    if request.method == "POST":
        tweet_text = request.form.get("tweet", "").strip()

        if not tweet_text:
            error = "Please enter a tweet before clicking Predict."
        else:
            try:
                cleaned_tweet = clean_text(tweet_text)

                # Predict using the complete TF-IDF + Logistic Regression pipeline
                result = int(model.predict([cleaned_tweet])[0])

                if result == 1:
                    prediction = "Disaster-Related Tweet"
                else:
                    prediction = "Non-Disaster Tweet"

                # Display confidence if supported by the model
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([cleaned_tweet])[0]
                    confidence = round(float(max(probabilities)) * 100, 2)

            except Exception as e:
                error = f"Prediction failed: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        tweet=tweet_text,
        confidence=confidence,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)

