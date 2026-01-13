from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["Hemoglobin"]),
        float(request.form["RBC"]),
        float(request.form["PCV"]),
        float(request.form["MCV"]),
        float(request.form["MCH"]),
        float(request.form["MCHC"])
    ]

    final_features = np.array([features])
    prediction = model.predict(final_features)

    result = "Anemia Detected" if prediction[0] == 1 else "No Anemia"

    return render_template("predict.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
