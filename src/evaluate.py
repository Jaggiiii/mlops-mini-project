import pandas as pd
import pickle, json
from sklearn.metrics import mean_absolute_error

test = pd.read_csv("outputs/test.csv")
model = pickle.load(open("outputs/model.pkl", "rb"))
preds = model.predict(test[["sqft", "bedrooms"]])

mae = mean_absolute_error(test["price"], preds)
json.dump({"mae": round(mae, 2)}, open("outputs/metrics.json", "w"), indent=2)
