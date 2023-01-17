from sklearn.datasets import load_iris
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as skRF
from classification.models import RandomForestClassifier
import pandas as pd

iris = load_iris(as_frame=True)
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
iris = pd.concat([X, y], axis=1).sample(frac=1.0, random_state=42)
iris.iloc[0:50, :].to_csv('../sample_data/3_clients/client1/data.csv', index=False)
iris.iloc[50:100, :].to_csv('../sample_data/3_clients/client2/data.csv', index=False)
iris.iloc[100:, :].to_csv('../sample_data/3_clients/client3/data.csv', index=False)

model = RandomForestClassifier(n_estimators=2, random_state=42, max_depth=1)

model.init()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("This:", matthews_corrcoef(y_test, y_pred))

sk_model = skRF(n_estimators=2, random_state=42, max_depth=1)
sk_model.fit(X_train, y_train)
y_pred = sk_model.predict(X_test)
print("sklearn:", matthews_corrcoef(y_test, y_pred))
