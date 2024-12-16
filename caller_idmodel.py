

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # Import accuracy_score
import pandas as pd
import pickle

# Load the Caller ID dataset
callerid_data = pd.read_csv('caller_id_data.csv')

# Preprocessing: Encode categorical features (e.g., urgency_level, action_taken)
label_encoder = LabelEncoder()
callerid_data['urgency_level'] = label_encoder.fit_transform(callerid_data['urgency_level'])
callerid_data['action_taken'] = label_encoder.fit_transform(callerid_data['action_taken'])

# Separate features and labels
X = callerid_data[['caller_id', 'urgency_level']]
y = callerid_data['action_taken']



# Encode `caller_id` if necessary
X['caller_id'] = label_encoder.fit_transform(callerid_data['caller_id'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model (Random Forest Classifier)
callerid_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
callerid_model.fit(X_train, y_train)

# Make predictions
y_pred = callerid_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Caller ID categorization model accuracy: {accuracy * 100:.2f}%")

# Save the model
with open('callerid_model.pkl', 'wb') as f:
    pickle.dump(callerid_model, f)
