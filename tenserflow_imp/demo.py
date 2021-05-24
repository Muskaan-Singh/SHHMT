import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
iris = load_iris()


FileName='/home/mandeep/PycharmProjects/samsadhni/data/human.txt'

data=pd.read_csv(FileName, sep=",")
data.fillna('id', inplace=True)

data.shape
data.head(10)
X = iris.data
y = iris.target
print(X.shape)
print(y.shape)

X = data.iloc[:,-1]
y = data.iloc[:,1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

vect = CountVectorizer()
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
# equivalently: combine fit and transform into a single step
X_train_dtm = vect.fit_transform(X_train)
# examine the document-term matrix
X_train_dtm
X_test_dtm = vect.transform(X_test)
X_test_dtm

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)
# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)
# example false negative



y_pred_prob = nb.predict_proba(X_test_dtm)[:,-1]

y_test= y_test.values
vect = CountVectorizer()
vect.fit(y_test)
vect.get_feature_names()
simple_train_dtm = vect.transform(y_test)
simple_train_dtm.toarray()
pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
print (simple_train_dtm.tocsc())
print type(y_pred_prob)
y_test=simple_train_dtm


metrics.roc_auc_score(y_test, y_pred_prob)
# import and instantiate a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
# make class predictions for X_test_dtm
y_pred_class = logreg.predict(X_test_dtm)
# calculate predicted probabilities for X_test_dtm (well calibrated)
y_pred_prob = logreg.predict_proba(X_test_dtm)[:, -1]
y_pred_prob
# calculate accuracy
metrics.accuracy_score(y_test, y_pred_class)
# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)
