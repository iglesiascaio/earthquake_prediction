import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_parquet("data/features/earthquake_features.parquet")

# Convert datetime column to numerical features
df["time_utc"] = pd.to_datetime(df["time_utc"])
df["year"] = df["time_utc"].dt.year
df["month"] = df["time_utc"].dt.month
df["day"] = df["time_utc"].dt.day
df["hour"] = df["time_utc"].dt.hour
df["minute"] = df["time_utc"].dt.minute
df["second"] = df["time_utc"].dt.second

# Drop non-numeric identifier
df.drop(columns=["event_id"], inplace=True)

# Convert categorical variables to numeric using Label Encoding
for col in ["magnitude_type", "event_type"]:
    df[col] = LabelEncoder().fit_transform(df[col])

# df = df.query("time_utc >= '2022-01-01'")


import ipdb

ipdb.set_trace()

X = df.drop(columns=["max_mag_next_30d", "target_class"])
y = df["max_mag_next_30d"]

# Temporal train-test split (until 2024-01-01 is train, the rest is test)
X_train = X.loc[X.time_utc < "2024-01-01"]
X_test = X.loc[X.time_utc >= "2024-01-30"]

y_train = y.loc[X.time_utc < "2024-01-01"]
y_test = y.loc[X.time_utc >= "2024-01-30"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

X_train = X_train.drop(columns=["time_utc"])
X_test = X_test.drop(columns=["time_utc"])


model = XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
