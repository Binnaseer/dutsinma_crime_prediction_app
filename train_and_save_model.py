# train_and_save_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("dutsinma_crime_mock.csv", parse_dates=['Date'])
df['Hour'] = pd.to_datetime(df['Time']).dt.hour
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df = pd.get_dummies(df, columns=['Location_Desc'], drop_first=True)
features = ['Hour','Month','Day','Latitude','Longitude'] + [c for c in df.columns if c.startswith('Location_Desc_')]
X = df[features]
y = df['Crime_Type']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
joblib.dump({'model': clf, 'features': features}, "model.joblib")
print("Model trained and saved as model.joblib")