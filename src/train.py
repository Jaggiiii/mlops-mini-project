import pandas as pd
import pickle, yaml
from sklearn.linear_model import LinearRegression

params = yaml.safe_load(open("params.yaml"))["model"]
train = pd.read_csv("outputs/train.csv")
X, y = train[["sqft", "bedrooms"]], train["price"]

model = LinearRegression(fit_intercept=params["fit_intercept"]).fit(X, y)
pickle.dump(model, open("outputs/model.pkl", "wb"))
