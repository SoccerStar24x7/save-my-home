import numpy as np
import joblib as jb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold

from data_prep import off, on

# TRAINING:
# X:
# y:
X = off
y = on

selector = VarianceThreshold(threshold=0.1)
X = selector.fit_transform(X)

# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
print("Assigned training values")
model = DecisionTreeClassifier()
print("Model fit")
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

accuracy = round(float(accuracy), 4)

print(f"Training accuracy: {accuracy * 100}%")
print()

# save model

jb.dump(model, "model.pkl")

