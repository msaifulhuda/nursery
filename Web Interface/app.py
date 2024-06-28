from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the StackingClassifier model
filename = "models/stacking_model-nursery.pkl"
with open(filename, "rb") as file:
    model = pickle.load(file)

# Define encoding dictionary
encoding_dict = {
    "parents": {"usual": 2, "pretentious": 1, "great_pret": 0},
    "has_nurs": {"proper": 3, "less_proper": 2, "improper": 1, "critical": 0, "very_crit": 4},
    "form": {"complete": 0, "completed": 1, "incomplete": 3, "foster": 2},
    "children": {1: 0, 2: 1, 3: 2, "more": 3},
    "housing": {"convenient": 0, "less_conv": 2, "critical": 1},
    "finance": {"convenient": 0, "inconv": 1},
    "social": {"nonprob": 0, "slightly_prob": 2, "problematic": 1},
    "health": {"recommended": 2, "priority": 1, "not_recom": 0}
}

# Define class names mapping
class_names = {
    0: "not_recom",
    1: "priority",
    2: "recommend",
    3: "spec_prior",
    4: "very_recom"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    parents = request.form["parents"]
    has_nurs = request.form["has_nurs"]
    form = request.form["form"]
    children = request.form["children"]
    housing = request.form["housing"]
    finance = request.form["finance"]
    social = request.form["social"]
    health = request.form["health"]

    # Create input data dictionary
    input_data = {
        "parents": parents,
        "has_nurs": has_nurs,
        "form": form,
        "children": children,
        "housing": housing,
        "finance": finance,
        "social": social,
        "health": health
    }

    # Convert input data to DataFrame
    mydata = pd.DataFrame([input_data])

    # Convert type numeric ke jenis data yang sesuai
    mydata["children"] = mydata["children"].replace({"1": 1, "2": 2, "3": 3, "more": "more"})

    # Apply encoding to DataFrame
    encoded_data = mydata.copy()
    for col, mapping in encoding_dict.items():
        encoded_data.loc[0, col] = mapping[encoded_data.loc[0, col]]

    # Make prediction using encoded data
    predictions = model.predict(encoded_data)

    # Get predicted class label and name
    predicted_class = predictions[0]
    predicted_class_name = class_names.get(predicted_class)

    return render_template("index.html", prediction=predicted_class_name)

if __name__ == "__main__":
    app.run(debug=True)
