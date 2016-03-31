import gc
import SLP
import numpy as np
import pandas as pd
from sklearn import svm
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import accuracy_score

train = pd.read_csv("train.csv")
train = train.astype(np.float64)

features = list(train.columns.values)
features.remove("Id")
features.remove("Made Donation in March 2007")

X = train[features]
y = train["Made Donation in March 2007"]

X = X/np.amax(X)
y = y.reshape(len(y),1)

del train
gc.collect()

# limit = int(0.8*len(y))
# clf = RandomForestRegressor(n_jobs = -1,n_estimators = 100000,max_features=None)
# clf = make_pipeline(PolynomialFeatures(5), Ridge(alpha=1e-5,fit_intercept=True,normalize=True))
# clf = svm.SVR(kernel="poly",degree=3)
# clf = LogisticRegression(dual = False , C = 1.0 , n_jobs = -1)
# clf = LinearRegression(fit_intercept = True , normalize = True , n_jobs = -1) # Highest
clf = SLP.ANN(X.shape[1],100,y.shape[1])
# clf.fit(X[0:limit],y[0:limit])
clf.fit(X,y)
# print(clf.costFunction(X[limit:],y[limit:]))

test = pd.read_csv("test.csv")
Id = test["Id"]

X = test[features]

X = X/np.amax(X)

del test
gc.collect()

prediction = clf.predict(X)
prediction = prediction.reshape(len(prediction),)
# print(prediction[prediction > 0.95])

submission = pd.DataFrame({
				"":Id,
				"Made Donation in March 2007":prediction
			})


submission.to_csv("submission.csv",index=False)