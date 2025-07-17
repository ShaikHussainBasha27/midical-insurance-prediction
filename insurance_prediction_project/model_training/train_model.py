import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
data = pd.read_csv('data/insurance_data.csv')

# Preprocess the data (similar to what we did earlier)
# Assuming you have already encoded categorical variables as shown in previous examples

X = data[['Age', 'Gender', 'BMI', 'MaritalStatus', 'Children', 'Smoker', 'Region']]
y = data['InsuranceCost']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model to a pickle file
with open('app/model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
