import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # Import accuracy_score
import pickle

# Load the DND dataset
dnd_data = pd.read_csv('dnd_data.csv')

# Preprocessing: Encode categorical features (e.g., urgency_level, action_taken)
label_encoder = LabelEncoder()
dnd_data['urgency_level'] = label_encoder.fit_transform(dnd_data['urgency_level'])
dnd_data['action_taken'] = label_encoder.fit_transform(dnd_data['action_taken'])

# Separate features and labels
X = dnd_data[['caller_id', 'call_time', 'is_whitelisted', 'urgency_level']]
y = dnd_data['action_taken']

# Convert `call_time` to numeric format (e.g., extract hour of the day and convert minutes to seconds)
def convert_call_time_to_seconds(call_time):
    try:
        time_obj = pd.to_datetime(call_time, format='%H:%M')
        return time_obj.hour * 60 + time_obj.minute  # Convert to minutes
    except:
        return 0  # Default value if conversion fails

# Apply conversion function to `call_time`
X['call_time'] = X['call_time'].apply(convert_call_time_to_seconds)

# Encode `caller_id` if necessary (convert categorical string IDs to numerical)
X['caller_id'] = label_encoder.fit_transform(X['caller_id'])

# Ensure all columns are in numerical format (no strings left)
X = X.apply(pd.to_numeric, errors='coerce')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model (Random Forest Classifier)
dnd_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
dnd_model.fit(X_train, y_train)

# Make predictions
y_pred = dnd_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)  # Now we can use accuracy_score
print(f"DND management model accuracy: {accuracy * 100:.2f}%")

# Save the model
with open('dnd_model.pkl', 'wb') as f:
    pickle.dump(dnd_model, f)

