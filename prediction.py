import joblib
import os

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "modelKNN1.pkl")

clf = joblib.load(MODEL_PATH)

def predict(data):
    return clf.predict(data)
