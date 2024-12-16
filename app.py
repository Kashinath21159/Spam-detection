
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the DND model, callerid encoder, and spam model
with open('dnd_model.pkl', 'rb') as f:
    dnd_model = pickle.load(f)
with open('callerid_encoder.pkl', 'rb') as f:
    callerid_encoder = pickle.load(f)
with open('spam_model.pkl', 'rb') as f:  # Load the spam model
    spam_model = pickle.load(f)

# Manually define the urgency level mapping
urgency_level_map = {'Low': 0, 'Medium': 1, 'High': 2}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    category = None
    if request.method == "POST":
        # Get form data
        call_duration = int(request.form["call_duration"])
        call_frequency = int(request.form["call_frequency"])
        spam_keywords_count = int(request.form["spam_keywords_count"])

        # Spam detection
        features = np.array([[call_duration, call_frequency, spam_keywords_count]])
        is_spam = spam_model.predict(features)[0]  # Predict if the call is spam
        result = "Spam" if is_spam else "Not Spam"

        # DND management
        caller_id = request.form["caller_id"]
        call_time = request.form["call_time"]
        is_whitelisted = int(request.form["is_whitelisted"])
        urgency_level = request.form["urgency_level"]

        # Preprocess the features for DND prediction
        dnd_features = np.array([[caller_id, call_time, is_whitelisted, urgency_level]])

        try:
            # Try to encode the caller_id using the loaded encoder
            dnd_features[0, 0] = callerid_encoder.transform([caller_id])[0]  # Encoding caller_id
        except ValueError:  # Handle unseen caller ID gracefully
            dnd_features[0, 0] = -1  # Assign default value for unseen caller IDs

        # Convert call_time (assuming it's in HH:MM format) to numeric (hour of the day)
        dnd_features[0, 1] = int(call_time.split(":")[0])  # Extracting hour (e.g., "13:36" -> 13)

        # Convert urgency_level to a numeric value using the predefined map
        dnd_features[0, 3] = urgency_level_map.get(urgency_level, -1)  # Default to -1 for unknown levels

        # Make DND prediction
        action_taken = dnd_model.predict(dnd_features)[0]
        category = {0: "Blocked", 1: "Allowed"}[action_taken]

        return render_template("results.html", result=result, category=category)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
