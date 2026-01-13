import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# Sample anemia dataset
data = {
    "Hemoglobin": [11, 13, 10, 14, 9, 15],
    "RBC": [4.1, 4.8, 3.9, 5.0, 3.6, 5.2],
    "PCV": [33, 40, 30, 42, 28, 45],
    "MCV": [80, 88, 75, 90, 70, 92],
    "MCH": [26, 29, 24, 30, 22, 31],
    "MCHC": [31, 34, 30, 35, 29, 36],
    "Anemia": [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop("Anemia", axis=1)
y = df["Anemia"]

model = LogisticRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ model.pkl created successfully")
