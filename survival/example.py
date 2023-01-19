from lifelines import datasets
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest as sksurvRSF
from survival.models import RandomForest
from survival.scoring import concordance_index
import pandas as pd


def map_to_scikit_surv(y):
    y = y.values.tolist()
    y1 = []
    y2 = []
    for w in y:
        y1.append(w[0])
        y2.append(w[1])
    y1 = list(map(bool, y1))
    ya = []
    for i in range(len(y1)):
        ya.append([y1[i], y2[i]])
    yy = pd.DataFrame(ya)
    y = yy.to_records(index=False)
    return y


rossi = datasets.load_rossi()
# Attention: duration column must be index 0, event column index 1 in y
y = rossi.loc[:, ["arrest", "week"]]
X = rossi.drop(["arrest", "week"], axis=1)
X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

rsf = RandomForest(n_estimators=10, n_jobs=-1, random_state=10)
rsf = rsf.fit(X, y)
y_pred = rsf.predict(X_test)
c_val = concordance_index(y_time=y_test["week"], y_pred=y_pred, y_event=y_test["arrest"])
print("This:", c_val)

rsf = sksurvRSF(n_estimators=10, n_jobs=-1, random_state=10)
rsf = rsf.fit(X, map_to_scikit_surv(y))
y_pred = rsf.predict(X_test)
c_val = concordance_index(y_time=y_test["week"], y_pred=y_pred, y_event=y_test["arrest"])
print("scikit-survival:", c_val)



