import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv("data.csv")

data = df[['Close']]

for i in range(1, 6):
    data[f'lag_{i}'] = data['Close'].shift(i)

data.dropna(inplace=True)

X = data.drop('Close', axis=1)
y = data['Close']

model = RandomForestRegressor(n_estimators=200)
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained successfully!")