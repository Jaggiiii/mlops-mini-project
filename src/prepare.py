import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["split"]
df = pd.read_csv("data/housing.csv")
train, test = train_test_split(df, test_size=params["test_size"], random_state=params["random_state"])
train.to_csv("outputs/train.csv", index=False)
test.to_csv("outputs/test.csv", index=False)
