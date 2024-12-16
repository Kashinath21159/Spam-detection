import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the spam dataset
spam_data = pd.read_csv('spam_data.csv')

# Preprocessing: Separate features and labels
X = spam_data[['call_duration', 'call_frequency', 'spam_keywords_count']]
y = spam_data['is_spam']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model (Random Forest Classifier)
spam_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
spam_model.fit(X_train, y_train)

# Make predictions
y_pred = spam_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Spam detection model accuracy: {accuracy * 100:.2f}%")

# Save the model
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(spam_model, f)
