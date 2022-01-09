# %%

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import pandas as pd

# %%

data = pd.read_csv("../data/iris.csv")
X = data.drop("class", axis=1)
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# %%

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# %%

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

# %%

model = XGBClassifier(
    max_depth=3,
    objective="multi:softprob",
    eval_metric="merror",
    use_label_encoder=False,
)
model.fit(X_train_scaled, y_train)
preds = model.predict(X_test_scaled)
print(model.score(X_test_scaled, y_test))
print(confusion_matrix(y_test, preds))

# %%
